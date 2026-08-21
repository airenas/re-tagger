import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    ModernBertConfig,
    ModernBertForTokenClassification,
    Trainer,
    TrainingArguments,
)

from egs.transformer.local.train import (
    build_dataset,
    load_finetuned_model,
    prepare_tags,
    read_conllu,
)
from src.utils.logger import logger


class DistillationTrainer(Trainer):
    """Combines gold-label, logit, and layer-mapped hidden-state distillation losses."""

    def __init__(self, *args, teacher, temperature, alpha, hidden_alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.hidden_alpha = hidden_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)
        student_logits = outputs.logits

        teacher_device = student_logits.device
        if next(self.teacher.parameters()).device != teacher_device:
            self.teacher.to(teacher_device)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits

        mask = labels.ne(-100)
        hard_loss = F.cross_entropy(
            student_logits.float().view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        temperature = self.temperature
        kl_per_token = F.kl_div(
            F.log_softmax(student_logits.float() / temperature, dim=-1),
            F.softmax(teacher_logits.float() / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        soft_loss = kl_per_token.masked_select(mask).mean() * temperature ** 2
        hidden_loss = 0.0
        for student_index in range(1, len(outputs.hidden_states)):
            teacher_index = int(student_index * (len(teacher_outputs.hidden_states) - 1) /
                                (len(outputs.hidden_states) - 1))
            projected_student = model.hidden_projection(outputs.hidden_states[student_index].float())
            teacher_hidden = teacher_outputs.hidden_states[teacher_index].float()
            hidden_loss += F.mse_loss(
                projected_student[mask],
                teacher_hidden[mask],
            )
        hidden_loss /= len(outputs.hidden_states) - 1
        hard_alpha = 1.0 - self.alpha - self.hidden_alpha
        loss = hard_alpha * hard_loss + self.alpha * soft_loss + self.hidden_alpha * hidden_loss
        return (loss, outputs) if return_outputs else loss


def make_student_config(teacher_config, num_labels, id2label, label2id, layers, hidden_size, heads):
    """Creates a compact ModernBERT config that remains compatible with the teacher tokenizer."""
    config = ModernBertConfig.from_dict(copy.deepcopy(teacher_config.to_dict()))
    config.num_hidden_layers = layers
    config.layer_types = config.layer_types[:layers]
    config.hidden_size = hidden_size
    config.intermediate_size = hidden_size * 4
    config.num_attention_heads = heads
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id
    return config


def main(argv):
    parser = argparse.ArgumentParser(description="Distills a compact ModernBERT token tagger from a fine-tuned teacher")
    parser.add_argument("--input", required=True, help="Training CoNLL-U file")
    parser.add_argument("--teacher", required=True, help="Fine-tuned teacher model directory")
    parser.add_argument("--out", required=True, help="Student model output directory")
    parser.add_argument("--layers", type=int, default=4, help="Student encoder layer count")
    parser.add_argument("--hidden_size", type=int, default=384, help="Student hidden size; must be divisible by --heads")
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
    tags = prepare_tags(sentences)
    tag2id = {tag: index for index, tag in enumerate(tags)}
    id2tag = {index: tag for index, tag in enumerate(tags)}

    logger.info("Loading teacher and tokenizer: {}".format(args.teacher))
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = load_finetuned_model(args.teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    logger.info("Building tokenized dataset")
    dataset = build_dataset(sentences, tokenizer, tag2id)
    split = dataset.train_test_split(test_size=args.val_size, seed=args.seed)

    student_config = make_student_config(
        teacher.config,
        len(tags),
        id2tag,
        tag2id,
        args.layers,
        args.hidden_size,
        args.heads,
    )
    student = ModernBertForTokenClassification(student_config)
    student.hidden_projection = torch.nn.Linear(args.hidden_size, teacher.config.hidden_size)
    logger.info("Student parameters: {:,}".format(sum(parameter.numel() for parameter in student.parameters())))

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
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
    )
    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        processing_class=tokenizer,
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        hidden_alpha=args.hidden_alpha,
    )
    logger.info("Training distilled student")
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "tags.txt"), "w", encoding="utf-8") as output:
        output.write("\n".join(tags) + "\n")
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])