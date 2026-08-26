import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.utils.logger import logger

MODEL_NAME = "VSSA-SDSA/LT-MLKM-modernBERT"

# our data uses a custom tag in the last (10th) CoNLL-U column instead of MISC
TAG_COLUMN = 9
EXPECTED_COLUMNS = 10


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
            tags.append(columns[TAG_COLUMN])

    if words:
        sentences.append({"words": words, "tags": tags})

    return sentences


def prepare_tags(sentences):
    """Builds a sorted tag vocabulary from the given sentences."""
    tag_set = set()
    for sentence in sentences:
        tag_set.update(sentence["tags"])
    return sorted(tag_set)


def extract_pos(tag):
    """Extracts POS (first character) from a Multext-East tag string."""
    return tag[0] if tag else "X"


def prepare_pos_tags(sentences):
    """Builds a sorted POS vocabulary from the given sentences."""
    pos_set = set()
    for sentence in sentences:
        for tag in sentence["tags"]:
            pos_set.add(extract_pos(tag))
    return sorted(pos_set)


def tokenize_and_align_labels(words, tags, tokenizer, tag2id, pos2id=None):
    """Tokenizes words and aligns each subword to its word's tag, using -100 for non-first subwords.
    Optionally also produces per-feature labels (e.g. pos) via feature2id dicts.
    """
    encoding = tokenizer(words, is_split_into_words=True)

    labels = []
    labels_pos = [] if pos2id else None
    previous_word_id = None
    for word_id in encoding.word_ids():
        if word_id is None or word_id == previous_word_id:
            labels.append(-100)
            if labels_pos is not None:
                labels_pos.append(-100)
        else:
            labels.append(tag2id[tags[word_id]])
            if labels_pos is not None:
                labels_pos.append(pos2id[extract_pos(tags[word_id])])
        previous_word_id = word_id

    result = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }
    if labels_pos is not None:
        result["labels_pos"] = labels_pos
    return result


def build_dataset(sentences, tokenizer, tag2id, pos2id=None):
    """Converts sentences into a tokenized, label-aligned HuggingFace Dataset.
    Optionally includes per-feature labels (e.g. pos) when a feature2id dict is provided.
    """
    dataset = Dataset.from_dict({
        "words": [s["words"] for s in sentences],
        "tags": [s["tags"] for s in sentences],
    })

    def _process(batch):
        input_ids, attention_masks, labels = [], [], []
        labels_pos_list = [] if pos2id else None
        for words, tags in zip(batch["words"], batch["tags"]):
            aligned = tokenize_and_align_labels(words, tags, tokenizer, tag2id, pos2id=pos2id)
            input_ids.append(aligned["input_ids"])
            attention_masks.append(aligned["attention_mask"])
            labels.append(aligned["labels"])
            if labels_pos_list is not None:
                labels_pos_list.append(aligned["labels_pos"])
        result = {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}
        if labels_pos_list is not None:
            result["labels_pos"] = labels_pos_list
        return result

    remove_cols = ["words", "tags"]
    return dataset.map(_process, batched=True, remove_columns=remove_cols)


class WeightedTrainer(Trainer):
    """Trainer using a weighted cross-entropy loss to up-weight critical/confusing tags.
    Supports multi-task learning with additional per-feature heads (e.g. pos).
    """

    def __init__(self, *args, class_weights=None, feature_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.feature_names = feature_names or []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Pop feature labels from inputs
        feature_labels = {}
        for name in self.feature_names:
            key = f"labels_{name}"
            if key in inputs:
                feature_labels[name] = inputs.pop(key)

        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits

        # Main tag loss
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Feature losses (multi-task)
        if self.feature_names and hasattr(model, "feature_heads"):
            hidden_states = outputs.hidden_states[-1]
            feature_logits = model.feature_heads(hidden_states)
            for name in self.feature_names:
                if name in feature_labels:
                    loss_fct_feat = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss += loss_fct_feat(
                        feature_logits[name].view(-1, feature_logits[name].size(-1)),
                        feature_labels[name].view(-1),
                    )

        return (loss, outputs) if return_outputs else loss


def init_tag_weights(tag2id):
    weights = torch.ones(len(tag2id))
    return weights


def parse_tag_weights_file(path, tag2id, default_weight: float = 2.0):
    """Reads one tag per line (optionally 'TAG\\tWEIGHT') and weights the rest at 1.0."""
    weights = torch.ones(len(tag2id))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split("=")
            tag = parts[0].strip()
            value = float(parts[1]) if len(parts) > 1 else default_weight
            if tag not in tag2id:
                raise ValueError("Unknown tag in --tag_weights_file: {}".format(tag))
            weights[tag2id[tag]] = value
            logger.info("Tag weight: {} = {}".format(tag, value))
    return weights


class MLPClassifierHead(nn.Module):
    """Two-layer GELU+LayerNorm head, more expressive than a single Linear projection."""

    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.out_proj(x)


class MultiFeatureDataCollator(DataCollatorForTokenClassification):
    """Extends token classification collator to also pad per-feature label fields with -100."""

    def __init__(self, feature_names, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = feature_names

    def torch_call(self, features):
        batch = super().torch_call(features)
        # Pad each feature label field with -100
        for name in self.feature_names:
            key = f"labels_{name}"
            if key not in features[0]:
                continue
            max_len = max(len(f[key]) for f in features)
            padded = []
            for f in features:
                cur = f[key]
                pad_len = max_len - len(cur)
                if pad_len > 0:
                    cur = cur + [-100] * pad_len
                padded.append(cur)
            batch[key] = torch.tensor(padded, dtype=torch.long)
        return batch


class MultiFeatureHead(nn.Module):
    """Container for multiple per-feature MLP heads for multi-task morphological tagging.

    Each feature (e.g. pos, gen, case) gets its own 2-layer MLP head.
    Easy to extend: just add new features to the feature_sizes dict.
    """

    def __init__(self, hidden_size, feature_sizes: dict[str, int], dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_sizes.keys())
        self.heads = nn.ModuleDict({
            name: MLPClassifierHead(hidden_size, num_classes, dropout)
            for name, num_classes in feature_sizes.items()
        })

    def forward(self, hidden_states):
        """Returns a dict of {feature_name: logits_tensor}."""
        return {name: head(hidden_states) for name, head in self.heads.items()}


def load_finetuned_model(model_dir):
    """Loads a fine-tuned model dir, rebuilding a custom classifier head (if used) before reloading its weights."""
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    head_path = os.path.join(model_dir, "classifier_head.txt")
    classifier_head = open(head_path, encoding="utf-8").read().strip() if os.path.exists(head_path) else "linear"

    if classifier_head == "mlp":
        dropout = getattr(model.config, "classifier_dropout", None) or 0.1
        model.classifier = MLPClassifierHead(model.config.hidden_size, model.config.num_labels, dropout=dropout)
        # from_pretrained already discarded the mismatched classifier.* weights above; reload them now
        from safetensors.torch import load_file
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Reloaded MLP classifier weights, missing={}, unexpected={}".format(missing, unexpected))
        # load_state_dict keeps the destination's dtype (float32); match the encoder's dtype (e.g. bf16)
        encoder_dtype = next(model.base_model.parameters()).dtype
        model.classifier = model.classifier.to(dtype=encoder_dtype)

    # Rebuild feature heads if feature configs exist
    features_dir = os.path.join(model_dir, "features")
    if os.path.exists(features_dir):
        feature_sizes = {}
        for fname in os.listdir(features_dir):
            if fname.endswith(".txt"):
                feature_name = fname.replace(".txt", "")
                vocab = open(os.path.join(features_dir, fname), encoding="utf-8").read().strip().splitlines()
                feature_sizes[feature_name] = len(vocab)
        if feature_sizes:
            dropout = getattr(model.config, "classifier_dropout", None) or 0.1
            model.feature_heads = MultiFeatureHead(model.config.hidden_size, feature_sizes, dropout=dropout)
            model.feature_heads = model.feature_heads.to(dtype=encoder_dtype)
            logger.info("Rebuilt feature heads: {}".format(feature_sizes))

    return model


def make_compute_metrics(feature_names=None):
    """Builds a compute_metrics function reporting token-level accuracy, ignoring -100 labels.
    When feature_names is provided, also reports per-feature accuracy (requires multi-output
    prediction_step override for full support — currently reports main tag accuracy only).
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        true_predictions = []
        true_labels = []
        for pred_row, label_row in zip(predictions, labels):
            for pred, label in zip(pred_row, label_row):
                if label != -100:
                    true_predictions.append(pred)
                    true_labels.append(label)

        return {"accuracy": accuracy_score(true_labels, true_predictions)}

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
    parser.add_argument("--classifier_head", choices=["linear", "mlp"], default="mlp",
                         help="Classification head on top of the encoder: plain Linear or a 2-layer MLP head")
    parser.add_argument("--max_steps", type=int, default=-1,
                         help="Stop after this many steps (overrides --epochs), useful for a quick smoke test")
    parser.add_argument("--tag_weights_file", nargs='?', default=None,
                         help="File with one problematic tag per line (optionally 'TAG<TAB>WEIGHT'); "
                              "listed tags get --tag_weights_default_weight (or their own value), all others 1.0")
    parser.add_argument("--features", nargs='?', default="pos",
                        help="Comma-separated list of feature heads for multi-task learning (e.g. 'pos,gen,case'). "
                             "Each feature gets its own MLP head. Feature values are extracted from tags via extract_<feature>(). "
                             "Default: pos only. Set to empty string to disable.")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("Loading: {}".format(args.input))
    logger.info(f"Epochs:  {args.epochs}")
    logger.info(f"Batch size:  {args.batch_size}")
    logger.info(f"Grad accum steps:  {args.grad_accum_steps}")
    logger.info(f"Gradient checkpointing:  {args.gradient_checkpointing}")
    logger.info(f"Max steps:  {args.max_steps}")

    sentences = read_conllu(args.input)
    tags = prepare_tags(sentences)
    tag2id = {tag: i for i, tag in enumerate(tags)}
    id2tag = {i: tag for i, tag in enumerate(tags)}

    # Build per-feature vocabularies (e.g. pos)
    feature_names = [f.strip() for f in args.features.split(",") if f.strip()] if args.features else []
    pos_tags = prepare_pos_tags(sentences) if "pos" in feature_names else []
    pos2id = {pos: i for i, pos in enumerate(pos_tags)} if pos_tags else None
    id2pos = {i: pos for i, pos in enumerate(pos_tags)} if pos_tags else None
    feature_vocabs = {}
    if pos_tags:
        feature_vocabs["pos"] = pos_tags

    token_count = sum(len(s["words"]) for s in sentences)
    logger.info("Sentences: {}".format(len(sentences)))
    logger.info("Tokens:    {}".format(token_count))
    logger.info("Tags:      {}".format(len(tags)))
    logger.info("First 20 tags:")
    for i, tag in enumerate(tags[:20]):
        logger.info("{:4d}  {}".format(i, tag))

    if sentences:
        first = sentences[1]
        logger.info("First sentence:")
        for word, tag in zip(first["words"], first["tags"]):
            logger.info("{:30s} {}".format(word, tag))

    logger.info("Loading tokenizer: {}".format(MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if sentences:
        aligned = tokenize_and_align_labels(first["words"], first["tags"], tokenizer, tag2id, pos2id=pos2id)
        readable_tokens = [tokenizer.convert_tokens_to_string([t]).strip() or t
                            for t in tokenizer.convert_ids_to_tokens(aligned["input_ids"])]
        readable_tags = [id2tag[t] if t != -100 else "-" for t in aligned["labels"]]
        logger.info("Tokens:     {}".format(readable_tokens))
        logger.info("Labels:     {}".format(aligned["labels"]))
        logger.info("Readable:   {}".format(readable_tags))
        if "labels_pos" in aligned:
            readable_pos = [id2pos[t] if t != -100 else "-" for t in aligned["labels_pos"]]
            logger.info("POS labels: {}".format(readable_pos))

    logger.info("Building dataset")
    dataset = build_dataset(sentences, tokenizer, tag2id, pos2id=pos2id)
    split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)
    train_dataset, eval_dataset = split["train"], split["test"]
    logger.info("Train sentences: {}".format(len(train_dataset)))
    logger.info("Eval sentences:  {}".format(len(eval_dataset)))

    logger.info("Loading model: {}".format(MODEL_NAME))
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(tags),
        id2label=id2tag,
        label2id=tag2id,
    )
    logger.info("Classifier: {}".format(model.classifier))

    if args.classifier_head == "mlp":
        dropout = getattr(model.config, "classifier_dropout", None) or 0.1
        model.classifier = MLPClassifierHead(model.config.hidden_size, len(tags), dropout=dropout)
        logger.info("Replaced classifier with MLP head: {}".format(model.classifier))

    # Initialize per-feature heads (multi-task learning)
    if feature_names:
        feature_sizes = {name: len(feature_vocabs[name]) for name in feature_names if name in feature_vocabs}
        if feature_sizes:
            dropout = getattr(model.config, "classifier_dropout", None) or 0.1
            model.feature_heads = MultiFeatureHead(model.config.hidden_size, feature_sizes, dropout=dropout)
            encoder_dtype = next(model.base_model.parameters()).dtype
            model.feature_heads = model.feature_heads.to(dtype=encoder_dtype)
            logger.info("Added feature heads: {}".format(feature_sizes))

    if args.freeze_base:
        logger.info("Freezing base encoder, training classification head only")
        for param in model.base_model.parameters():
            param.requires_grad = False

    data_collator = MultiFeatureDataCollator(feature_names, tokenizer=tokenizer) if feature_names else \
        DataCollatorForTokenClassification(tokenizer=tokenizer)

    class_weights = init_tag_weights(tag2id)
    if args.tag_weights_file and args.tag_weights_file.lower() != "none":
        logger.info("Tag weights file: {}".format(args.tag_weights_file))
        class_weights *= parse_tag_weights_file(args.tag_weights_file, tag2id)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    trainable = sum( p.numel() for p in model.parameters() if p.requires_grad )
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
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=10,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        max_steps=args.max_steps,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(feature_names=feature_names),
        class_weights=class_weights,
        feature_names=feature_names,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model to: {}".format(args.out))
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "tags.txt"), "w", encoding="utf-8") as f:
        for tag in tags:
            f.write(tag + "\n")
    with open(os.path.join(args.out, "classifier_head.txt"), "w", encoding="utf-8") as f:
        f.write(args.classifier_head)
    # Save per-feature vocabularies
    if feature_vocabs:
        features_dir = os.path.join(args.out, "features")
        os.makedirs(features_dir, exist_ok=True)
        for feat_name, vocab in feature_vocabs.items():
            with open(os.path.join(features_dir, "{}.txt".format(feat_name)), "w", encoding="utf-8") as f:
                for val in vocab:
                    f.write(val + "\n")
        logger.info("Saved feature vocabs: {}".format(list(feature_vocabs.keys())))

    logger.info("Done")


if __name__ == "__main__":

    main(sys.argv[1:])
