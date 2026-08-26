import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments, )

from egs.transformer.local.model import MultiHeadTokenClassifier
from egs.transformer.local.morph import to_tags, feature_tags, feature_tags_loss_weights
from src.utils.logger import logger

MODEL_NAME = "VSSA-SDSA/LT-MLKM-modernBERT"

# our data uses a custom tag in the last (10th) CoNLL-U column instead of MISC
TAG_COLUMN = 9
EXPECTED_COLUMNS = 10


@dataclass
class DataCollatorForMultiHeadTokenClassification:
    tokenizer: Any
    padding: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract label columns dynamically (keys starting with 'labels_')
        label_keys = [k for k in features[0].keys() if k.startswith("labels_")]

        # Extract non-label inputs (input_ids, attention_mask)
        non_label_features = [
            {k: v for k, v in f.items() if not k.startswith("labels_")}
            for f in features
        ]

        # Pad standard tokenizer inputs
        batch = self.tokenizer.pad(non_label_features, padding=self.padding, return_tensors="pt")

        # Find batch maximum length
        batch_max_len = batch["input_ids"].shape[1]

        # Pad label sequences with -100 to match batch_max_len
        for label_key in label_keys:
            padded_labels = []
            for f in features:
                seq = f[label_key]
                padded = seq + [-100] * (batch_max_len - len(seq))
                padded_labels.append(padded)
            batch[label_key] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def read_conllu(path):
    """Reads a CoNLL-U-like file into a list of {"words": [...], "tags": [...]} sentences."""
    sentences = []
    words = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading CoNLL-U file", unit=" lines"):
            line = line.rstrip("\n")

            if not line:
                if words:
                    sentences.append({"words": words, "tags": tags})
                    words = []
                    tags = []
                continue

            if line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != EXPECTED_COLUMNS:
                raise ValueError("Expected {} columns, got {}: {}".format(EXPECTED_COLUMNS, len(columns), line))

            words.append(columns[1])
            tags.append(to_tags(columns[TAG_COLUMN]))

    if words:
        sentences.append({"words": words, "tags": tags})

    return sentences


def prepare_tags(sentences, features):
    """Builds a sorted tag vocabulary from the given sentences."""
    res = {}
    for feature in features:
        tag_set = set()
        for sentence in sentences:
            for tags in sentence["tags"]:
                f = tags.get(feature, "")
                if f:
                    tag_set.add(f)
        res[feature] = sorted(tag_set)
    return res


def tokenize_and_align_labels(words, tags, tokenizer, tag2id, feature_tags):
    encoding = tokenizer(words, is_split_into_words=True)

    labels_dict = {f"labels_{feat}": [] for feat in feature_tags}

    previous_word_id = None
    for word_id in encoding.word_ids():
        if word_id is None or word_id == previous_word_id:
            # Mask out non-first subwords and special tokens for all heads
            for feat in feature_tags:
                labels_dict[f"labels_{feat}"].append(-100)
        else:
            # Look up the target tag ID for each feature
            for feat in feature_tags:
                tag_str = tags[word_id].get(feat, "")
                if tag_str == "":
                    labels_dict[f"labels_{feat}"].append(-100)
                else:
                    labels_dict[f"labels_{feat}"].append(tag2id[feat][tag_str])

        previous_word_id = word_id

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        **labels_dict,
    }


def build_dataset(sentences, tokenizer, tag2id, feature_tags):
    """Converts sentences into a tokenized, multi-label aligned HuggingFace Dataset."""
    dataset = Dataset.from_dict({
        "words": [s["words"] for s in sentences],
        "tags": [s["tags"] for s in sentences],
    })

    def _process(batch):
        # Dynamically initialize output storage for input_ids, attention_mask, and all feature label columns
        batch_outputs = {
            "input_ids": [],
            "attention_mask": [],
            **{f"labels_{feat}": [] for feat in feature_tags}
        }

        for words, tags in zip(batch["words"], batch["tags"]):
            aligned = tokenize_and_align_labels(words, tags, tokenizer, tag2id, feature_tags)
            for key, value in aligned.items():
                batch_outputs[key].append(value)

        return batch_outputs

    return dataset.map(_process, batched=True, remove_columns=["words", "tags"])


class WeightedTrainer(Trainer):
    """Multi-task trainer computing weighted cross-entropy across feature heads."""

    def __init__(self, *args, feature_loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_loss_weights = feature_loss_weights or {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract dynamic target labels
        labels_dict = {
            k.replace("labels_", ""): inputs.pop(k)
            for k in list(inputs.keys())
            if k.startswith("labels_")
        }

        logits_dict = model(**inputs)

        losses = {}
        for head_name, logits in logits_dict.items():
            if head_name in labels_dict:
                labels = labels_dict[head_name]
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                head_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                losses[labels] = head_loss

        if not losses:
            raise ValueError(
                f"No matching label heads found! "
                f"Model heads: {list(logits_dict.keys())}, "
                f"Batch label keys: {list(labels_dict.keys())}"
            )

        # Sum losses into a PyTorch Tensor that tracks gradients
        total_loss = sum(
            self.feature_loss_weights.get(head_name, 1.0) * head_loss
            for head_name, head_loss in losses.items()
        )

        return (total_loss, logits_dict) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom evaluation step keeping PyTorch Tensors on CPU."""
        has_labels = any(k.startswith("labels_") for k in inputs.keys())

        # Pull out target labels
        labels_dict = {
            k.replace("labels_", ""): inputs[k]
            for k in list(inputs.keys())
            if k.startswith("labels_")
        }

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None
                logits_dict = model(**inputs)

        if prediction_loss_only:
            return (loss, None, None)

        # Move tensors to CPU, but DO NOT convert to numpy here
        logits_dict = {k: v.detach().cpu() for k, v in logits_dict.items()}
        labels_dict = {k: v.detach().cpu() for k, v in labels_dict.items()}

        return (loss, logits_dict, labels_dict)


def make_compute_metrics():
    """Computes token-level accuracy per head and overall mean accuracy."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        # Safety check if HF unpacked dicts as tuples
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0] if len(predictions) == 1 else predictions
        if isinstance(labels, (tuple, list)):
            labels = labels[0] if len(labels) == 1 else labels

        metrics = {}
        accuracies = []

        for head_name, pred_logits in predictions.items():
            if head_name not in labels:
                continue

            head_labels = labels[head_name]
            preds = np.argmax(pred_logits, axis=-1)

            true_preds, true_labels = [], []
            for pred_row, label_row in zip(preds, head_labels):
                for p, l in zip(pred_row, label_row):
                    if l != -100:
                        true_preds.append(p)
                        true_labels.append(l)

            acc = accuracy_score(true_labels, true_preds) if true_labels else 0.0
            metrics[f"{head_name}_accuracy"] = acc
            accuracies.append(acc)

        # Map overall accuracy -> Trainer turns this into 'eval_accuracy'
        metrics["accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
        return metrics

    return compute_metrics


def main(argv):
    parser = argparse.ArgumentParser(description="Trains transformer model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--out", nargs='?', required=True, help="Model output file")
    parser.add_argument("--epochs", type=float, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device train/eval batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients over, to simulate a larger batch size on limited GPU memory")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for memory by not storing all activations")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--val_size", type=float, default=0.02, help="Fraction of sentences held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--freeze_base", action="store_true",
                        help="Freeze the pretrained encoder and only train the classification head")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Stop after this many steps (overrides --epochs), useful for a quick smoke test")
    parser.add_argument("--tag_weights_file", nargs='?', default=None,
                        help="File with one problematic tag per line (optionally 'TAG<TAB>WEIGHT'); "
                             "listed tags get --tag_weights_default_weight (or their own value), all others 1.0")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("Loading: {}".format(args.input))
    logger.info(f"Epochs:  {args.epochs}")
    logger.info(f"Batch size:  {args.batch_size}")
    logger.info(f"Grad accum steps:  {args.grad_accum_steps}")
    logger.info(f"Gradient checkpointing:  {args.gradient_checkpointing}")
    logger.info(f"Max steps:  {args.max_steps}")

    sentences = read_conllu(args.input)
    tags = prepare_tags(sentences, feature_tags)

    # Nested dictionaries mapping each feature to its tag-to-id / id-to-tag dicts
    tag2id = {
        feature: {tag: i for i, tag in enumerate(f_tags)}
        for feature, f_tags in tags.items()
    }

    id2tag = {
        feature: {i: tag for i, tag in enumerate(f_tags)}
        for feature, f_tags in tags.items()
    }

    token_count = sum(len(s["words"]) for s in sentences)
    logger.info("Sentences: {}".format(len(sentences)))
    logger.info("Tokens:    {}".format(token_count))
    for feature, f_tags in tags.items():
        logger.info("Feature: {}  Tags: {}".format(feature, len(f_tags)))
        logger.info("Tags for {}: {}".format(feature, f_tags))
        logger.info("Loss weight for {}: {}".format(feature, feature_tags_loss_weights.get(feature, 1.0)))

    logger.info("Loading tokenizer: {}".format(MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if sentences:
        first = sentences[1]
        logger.info("First sentence:")
        for word, tag in zip(first["words"], first["tags"]):
            logger.info("{:30s} {}".format(word, tag))

        aligned = tokenize_and_align_labels(first["words"], first["tags"], tokenizer, tag2id, feature_tags)
        readable_tokens = [tokenizer.convert_tokens_to_string([t]).strip() or t
                           for t in tokenizer.convert_ids_to_tokens(aligned["input_ids"])]
        logger.info("Tokens:     {}".format(readable_tokens))
        for feature in feature_tags:
            logger.info(f"Labels {feature}:     {aligned["labels_{}".format(feature)]}")
            readable_tags = [id2tag[feature][t] if t != -100 else "-" for t in aligned[f"labels_{feature}"]]
            logger.info("Readable {}: {}".format(feature, readable_tags))

    logger.info("Building dataset")
    dataset = build_dataset(sentences, tokenizer, tag2id, feature_tags)
    split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)
    train_dataset, eval_dataset = split["train"], split["test"]
    logger.info("Train sentences: {}".format(len(train_dataset)))
    logger.info("Eval sentences:  {}".format(len(eval_dataset)))

    logger.info("Loading model: {}".format(MODEL_NAME))
    feature_num_labels = {feat: len(tags[feat]) for feat in feature_tags}
    model = MultiHeadTokenClassifier(
        model_name=MODEL_NAME,
        feature_num_labels=feature_num_labels,
    )
    logger.info("Loaded multi-head model with heads: {}".format(list(model.heads.keys())))
    for feat, head in model.heads.items():
        logger.info(f"  Head [{feat}]: {head}")

    if args.freeze_base:
        logger.info("Freezing base encoder, training classification head only")
        for param in model.base_model.parameters():
            param.requires_grad = False

    data_collator = DataCollatorForMultiHeadTokenClassification(tokenizer=tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable: {trainable:,}")
    logger.info(f"Total:     {total:,}")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Monitors average accuracy across all heads
        save_total_limit=10,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        max_steps=args.max_steps,
        remove_unused_columns=False,
    )

    # Note: Ensure class_weights is a dictionary mapping head names to weight tensors, e.g. {"pos": ..., "gender": ...}
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(),
        feature_loss_weights=feature_tags_loss_weights,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model to: {}".format(args.out))
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    model.config.save_pretrained(args.out)

    label_config = {
        "features": feature_tags,
        "tag2id": tag2id,
        "id2tag": id2tag,
    }

    # Save as labels.json
    labels_json_path = os.path.join(args.out, "labels.json")
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved label mappings to {labels_json_path}")
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
