import argparse
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
)

from egs.transformer.local.model import MultiHeadTokenClassifier
from egs.transformer.local.morph import feature_tags, feature_tags_loss_weights
from egs.transformer.local.train import (
    build_dataset,
    read_conllu, DataCollatorForMultiHeadTokenClassification,
)
from egs.transformer.local.train import make_compute_metrics
from src.utils.logger import logger


class MultiHeadDistillationTrainer(Trainer):
    """Combines gold-label, multi-head logit, and layer-mapped hidden-state distillation losses."""

    def __init__(self, *args, teacher, temperature, alpha, hidden_alpha, feature_loss_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.hidden_alpha = hidden_alpha
        self.feature_loss_weights = feature_loss_weights or {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Extract dynamic target labels (`labels_<feature>`)
        labels_dict = {
            k.replace("labels_", ""): inputs.pop(k)
            for k in list(inputs.keys())
            if k.startswith("labels_")
        }

        # Ensure teacher model is on the right device
        teacher_device = next(model.parameters()).device
        if next(self.teacher.parameters()).device != teacher_device:
            self.teacher.to(teacher_device)

        # 2. Forward pass through Student
        student_outputs = model.base_model(**inputs, output_hidden_states=True)
        student_seq_output = student_outputs[0]
        student_hidden_states = student_outputs.hidden_states
        student_logits_dict = {
            feat: head(student_seq_output.to(dtype=next(head.parameters()).dtype))
            for feat, head in model.heads.items()
        }

        # 3. Forward pass through Teacher (No Gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher.base_model(**inputs, output_hidden_states=True)
            teacher_seq_output = teacher_outputs[0]
            teacher_hidden_states = teacher_outputs.hidden_states
            teacher_logits_dict = {
                feat: head(teacher_seq_output.to(dtype=next(head.parameters()).dtype))
                for feat, head in self.teacher.heads.items()
            }

        # 4. Compute Loss per Head (Cross-Entropy Hard Loss + Soft Logit KL-Divergence)
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        temp = self.temperature
        total_weight = 0.0

        for head_name, student_logits in student_logits_dict.items():
            if head_name not in labels_dict:
                continue

            labels = labels_dict[head_name]
            mask = labels.ne(-100)
            weight = self.feature_loss_weights.get(head_name, 1.0)
            total_weight += weight

            # Hard Loss (Cross Entropy)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            head_hard_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
            total_hard_loss += weight * head_hard_loss

            # Soft Loss (Teacher-Student KL Divergence over valid tokens)
            if head_name in teacher_logits_dict:
                teacher_logits = teacher_logits_dict[head_name]
                kl_per_token = F.kl_div(
                    F.log_softmax(student_logits.float() / temp, dim=-1),
                    F.softmax(teacher_logits.float() / temp, dim=-1),
                    reduction="none",
                ).sum(dim=-1)

                head_soft_loss = kl_per_token.masked_select(mask).mean() * (temp ** 2) if mask.any() else torch.tensor(
                    0.0, device=teacher_device)
                total_soft_loss += weight * head_soft_loss

        if total_weight > 0:
            total_hard_loss /= total_weight
            total_soft_loss /= total_weight

        # 5. Layer-Mapped Hidden State Distillation Loss
        hidden_loss = 0.0
        num_student_layers = len(student_hidden_states) - 1
        num_teacher_layers = len(teacher_hidden_states) - 1

        # Combine mask across all feature heads for hidden loss alignment
        combined_mask = torch.stack([l.ne(-100) for l in labels_dict.values()]).any(dim=0)

        proj_dtype = model.hidden_projection.weight.dtype
        for s_idx in range(1, len(student_hidden_states)):
            t_idx = int(s_idx * num_teacher_layers / num_student_layers)
            student_hidden = student_hidden_states[s_idx].to(dtype=proj_dtype)
            projected_student = model.hidden_projection(student_hidden)
            teacher_hidden = teacher_hidden_states[t_idx].to(dtype=projected_student.dtype)

            if combined_mask.any():
                hidden_loss += F.mse_loss(
                    projected_student[combined_mask],
                    teacher_hidden[combined_mask],
                )

        hidden_loss /= num_student_layers

        # 6. Combined Weighted Loss
        hard_alpha = 1.0 - self.alpha - self.hidden_alpha
        loss = (hard_alpha * total_hard_loss) + (self.alpha * total_soft_loss) + (self.hidden_alpha * hidden_loss)

        return (loss, student_logits_dict) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Custom evaluation step keeping PyTorch Tensors on CPU."""
        has_labels = any(k.startswith("labels_") for k in inputs.keys())

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

        logits_dict = {k: v.detach().cpu() for k, v in logits_dict.items()}
        labels_dict = {k: v.detach().cpu() for k, v in labels_dict.items()}

        return (loss, logits_dict, labels_dict)

def make_student_config(teacher_config, layers: int, hidden_size: int, heads: int):
    """Creates a smaller student AutoConfig derived from teacher's configuration."""
    config = copy.deepcopy(teacher_config)
    config.num_hidden_layers = layers
    if hasattr(config, "layer_types"):
        config.layer_types = config.layer_types[:layers]
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size * 4
    config.num_attention_heads = heads
    return config


def main(argv):
    parser = argparse.ArgumentParser(description="Distills a compact ModernBERT token tagger from a fine-tuned teacher")
    parser.add_argument("--input", required=True, help="Training CoNLL-U file")
    parser.add_argument("--teacher", required=True, help="Fine-tuned teacher model directory")
    parser.add_argument("--out", required=True, help="Student model output directory")
    parser.add_argument("--layers", type=int, default=4, help="Student encoder layer count")
    parser.add_argument("--hidden_size", type=int, default=384,
                        help="Student hidden size; must be divisible by --heads")
    parser.add_argument("--heads", type=int, default=6, help="Student attention head count")
    parser.add_argument("--epochs", type=float, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train/eval batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Student learning rate")
    parser.add_argument("--val_size", type=float, default=0.05, help="Validation sentence fraction")
    parser.add_argument("--temperature", type=float, default=2.0, help="Teacher-logit softening temperature")
    parser.add_argument("--alpha", type=float, default=0.3, help="Logit distillation loss proportion")
    parser.add_argument("--hidden_alpha", type=float, default=0.2, help="Hidden-state distillation loss proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    if args.hidden_size % args.heads:
        raise ValueError("--hidden_size must be divisible by --heads")
    if args.alpha < 0.0 or args.hidden_alpha < 0.0 or args.alpha + args.hidden_alpha > 1.0:
        raise ValueError("--alpha and --hidden_alpha must be non-negative and sum to at most 1")

    logger.info("Loading training data: {}".format(args.input))
    sentences = read_conllu(args.input)
    logger.info("Loading teacher and tokenizer: {}".format(args.teacher))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher, tokenizer, label_config = MultiHeadTokenClassifier.load(args.teacher, device=device)
    logger.info("Teacher parameters: {:,}".format(sum(parameter.numel() for parameter in teacher.parameters())))
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    logger.info("Building tokenized dataset")
    dataset = build_dataset(sentences, tokenizer, label_config["tag2id"], feature_tags=feature_tags)
    split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)

    student_config = make_student_config(
        teacher.config,
        args.layers,
        args.hidden_size,
        args.heads,
    )
    student = MultiHeadTokenClassifier(
        config=student_config,
        feature_num_labels=teacher.feature_num_labels,
        load_encoder_weights=False,
    )
    student.hidden_projection = torch.nn.Linear(args.hidden_size, teacher.config.hidden_size)
    logger.info("Student parameters: {:,}".format(sum(parameter.numel() for parameter in student.parameters())))

    data_collator = DataCollatorForMultiHeadTokenClassification(tokenizer=tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    updates_per_epoch = (len(split["train"]) + args.batch_size * args.grad_accum_steps - 1) // \
                        (args.batch_size * args.grad_accum_steps)
    warmup_steps = max(1, int(updates_per_epoch * args.epochs * 0.1))
    logger.info("Warmup steps: {}".format(warmup_steps))
    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
    )

    trainer = MultiHeadDistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        hidden_alpha=args.hidden_alpha,
        compute_metrics=make_compute_metrics(),
        feature_loss_weights=feature_tags_loss_weights,
    )

    logger.info("Training distilled student")
    trainer.train()

    logger.info("Saving model to: {}".format(args.out))
    del student.hidden_projection
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    student.config.save_pretrained(args.out)

    # Save as labels.json
    labels_json_path = os.path.join(args.out, "labels.json")
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved label mappings to {labels_json_path}")
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
