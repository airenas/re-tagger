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
    MLPClassifierHead,
    build_dataset,
    load_finetuned_model,
    prepare_tags,
    read_conllu,
)
from src.utils.logger import logger


class DistillationTrainer(Trainer):
    """
    Distillation trainer combining:

      1. Gold-label cross entropy
      2. Teacher-logit KL divergence
      3. Intermediate hidden-state MSE

    The hidden-state loss maps each student layer to the exact
    teacher layer that was used to initialize that student layer.
    """

    def __init__(
        self,
        *args,
        teacher,
        temperature,
        alpha,
        hidden_alpha,
        teacher_layer_indices,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.hidden_alpha = hidden_alpha
        self.teacher_layer_indices = teacher_layer_indices

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        # ---------------------------------------------------------
        # Student
        # ---------------------------------------------------------
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )

        student_logits = outputs.logits

        # ---------------------------------------------------------
        # Teacher
        # ---------------------------------------------------------
        teacher_device = student_logits.device

        if next(self.teacher.parameters()).device != teacher_device:
            self.teacher.to(teacher_device)

        with torch.no_grad():
            teacher_outputs = self.teacher(
                **inputs,
                output_hidden_states=True,
            )

        teacher_logits = teacher_outputs.logits

        # ---------------------------------------------------------
        # Token mask
        # ---------------------------------------------------------
        mask = labels.ne(-100)

        # ---------------------------------------------------------
        # 1. Hard / gold-label loss
        # ---------------------------------------------------------
        hard_loss = F.cross_entropy(
            student_logits.float().view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # ---------------------------------------------------------
        # 2. Logit distillation
        # ---------------------------------------------------------
        temperature = self.temperature

        student_log_probs = F.log_softmax(
            student_logits.float() / temperature,
            dim=-1,
        )

        teacher_probs = F.softmax(
            teacher_logits.float() / temperature,
            dim=-1,
        )

        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)

        soft_loss = (
            kl_per_token.masked_select(mask).mean()
            * temperature ** 2
        )

        # ---------------------------------------------------------
        # 3. Hidden-state distillation
        # ---------------------------------------------------------
        hidden_loss = student_logits.new_tensor(0.0)

        num_hidden_layers = len(outputs.hidden_states) - 1

        if num_hidden_layers > 0:
            for student_layer_number in range(1, num_hidden_layers + 1):
                if student_layer_number > len(self.teacher_layer_indices):
                    break

                # hidden_states[0] = embeddings
                #
                # hidden_states[1] = Transformer layer 0
                # hidden_states[2] = Transformer layer 1
                # etc.
                student_hidden = outputs.hidden_states[
                    student_layer_number
                ].float()

                teacher_layer_index = (
                    self.teacher_layer_indices[student_layer_number - 1]
                )

                teacher_hidden = teacher_outputs.hidden_states[
                    teacher_layer_index + 1
                ].float()

                # If student and teacher widths are identical,
                # hidden_projection is not needed.
                if hasattr(model, "hidden_projection"):
                    student_hidden = model.hidden_projection(
                        student_hidden
                    )

                hidden_loss = hidden_loss + F.mse_loss(
                    student_hidden[mask],
                    teacher_hidden[mask],
                )

            hidden_loss = hidden_loss / min(
                num_hidden_layers,
                len(self.teacher_layer_indices),
            )

        # ---------------------------------------------------------
        # Combine losses
        # ---------------------------------------------------------
        hard_alpha = (
            1.0
            - self.alpha
            - self.hidden_alpha
        )

        loss = (
            hard_alpha * hard_loss
            + self.alpha * soft_loss
            + self.hidden_alpha * hidden_loss
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def parse_layer_indices(value):
    """
    Parse:

        "0,4,8,12,17,21"

    into:

        [0, 4, 8, 12, 17, 21]
    """

    if not value:
        return None

    try:
        indices = [
            int(x.strip())
            for x in value.split(",")
            if x.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "--layer_indices must be comma-separated integers, "
            "for example: 0,4,8,12,17,21"
        ) from exc

    if not indices:
        raise ValueError("--layer_indices cannot be empty")

    if len(set(indices)) != len(indices):
        raise ValueError(
            "--layer_indices contains duplicate layer indices"
        )

    return indices


def make_student_config(
    teacher_config,
    num_labels,
    id2label,
    label2id,
    layers,
    hidden_size,
    heads,
    layer_indices,
):
    """
    Create a compact ModernBERT config.

    For layer pruning, layer_types must correspond to the
    actual teacher layers that will be copied.
    """

    config = ModernBertConfig.from_dict(
        copy.deepcopy(teacher_config.to_dict())
    )

    config.num_hidden_layers = layers

    # Preserve the teacher's layer type for each selected layer.
    #
    # ModernBERT can have different types of attention layers,
    # so don't simply take teacher layer_types[:layers] when
    # selecting arbitrary layers.
    teacher_layer_types = getattr(
        teacher_config,
        "layer_types",
        None,
    )

    if teacher_layer_types is not None:
        config.layer_types = [
            teacher_layer_types[index]
            for index in layer_indices
        ]

    config.hidden_size = hidden_size
    # config.intermediate_size = hidden_size * 4
    config.num_attention_heads = heads

    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id

    return config


def initialize_student_from_teacher(
    student,
    teacher,
    layer_indices,
    copy_classifier=True,
):
    """
    Initialize a compact student from a fine-tuned teacher.

    This function copies weights only when the corresponding
    dimensions are compatible.

    For the recommended 6x768 student:

        Teacher: 22 x 768
        Student:  6 x 768

    we copy:

        embeddings
        selected Transformer layers
        final norm
        classifier

    Returns:
        True if the encoder was successfully initialized
        from the teacher.
    """

    teacher_hidden_size = teacher.config.hidden_size
    student_hidden_size = student.config.hidden_size

    if teacher_hidden_size != student_hidden_size:
        logger.warning(
            "Teacher hidden size=%d, student hidden size=%d. "
            "Cannot directly copy Transformer weights.",
            teacher_hidden_size,
            student_hidden_size,
        )

        if copy_classifier:
            logger.warning(
                "Classifier head will remain randomly initialized "
                "because hidden dimensions differ."
            )

        return False

    # ---------------------------------------------------------
    # Find model components
    # ---------------------------------------------------------
    teacher_model = teacher.model
    student_model = student.model

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------
    logger.info("Copying teacher embeddings")

    student_model.embeddings.load_state_dict(
        teacher_model.embeddings.state_dict()
    )

    # ---------------------------------------------------------
    # Transformer layers
    # ---------------------------------------------------------
    teacher_layers = teacher_model.layers
    student_layers = student_model.layers

    if len(student_layers) != len(layer_indices):
        raise ValueError(
            "Student has {} layers but {} layer indices were supplied".format(
                len(student_layers),
                len(layer_indices),
            )
        )

    for student_index, teacher_index in enumerate(layer_indices):
        logger.info(
            "Copying teacher layer %d -> student layer %d",
            teacher_index,
            student_index,
        )

        if teacher_index < 0 or teacher_index >= len(teacher_layers):
            raise ValueError(
                "Teacher layer index {} is invalid. "
                "Teacher has {} layers.".format(
                    teacher_index,
                    len(teacher_layers),
                )
            )

        student_layers[student_index].load_state_dict(
            teacher_layers[teacher_index].state_dict()
        )

    # ---------------------------------------------------------
    # Final normalization
    # ---------------------------------------------------------
    if hasattr(student_model, "final_norm") and hasattr(
        teacher_model,
        "final_norm",
    ):
        logger.info("Copying teacher final_norm")

        student_model.final_norm.load_state_dict(
            teacher_model.final_norm.state_dict()
        )

    # ---------------------------------------------------------
    # Classifier
    # ---------------------------------------------------------
    if copy_classifier:
        if (
            student.classifier.in_features
            if hasattr(student.classifier, "in_features")
            else None
        ):
            # This branch is mainly for a standard Linear head.
            pass

        try:
            student.classifier.load_state_dict(
                teacher.classifier.state_dict()
            )

            logger.info(
                "Copied teacher classifier head"
            )

        except RuntimeError as exc:
            logger.warning(
                "Could not copy teacher classifier: %s",
                exc,
            )

    return True


def initialize_classifier(
    student,
    teacher,
    num_labels,
    dropout,
):
    """
    Create the requested MLP classifier.

    If teacher and student hidden sizes match, copy the complete
    fine-tuned teacher MLP classifier.

    Otherwise initialize the student classifier randomly.
    """

    student_hidden_size = student.config.hidden_size
    teacher_hidden_size = teacher.config.hidden_size

    student.classifier = MLPClassifierHead(
        student_hidden_size,
        num_labels,
        dropout=dropout,
    )

    if student_hidden_size != teacher_hidden_size:
        logger.info(
            "Student/teacher hidden sizes differ "
            "(%d vs %d); classifier initialized randomly.",
            student_hidden_size,
            teacher_hidden_size,
        )
        return

    try:
        student.classifier.load_state_dict(
            teacher.classifier.state_dict()
        )

        logger.info(
            "Initialized student MLP classifier "
            "from fine-tuned teacher classifier."
        )

    except RuntimeError as exc:
        logger.warning(
            "Could not copy teacher classifier. "
            "Using random initialization. Error: %s",
            exc,
        )


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Distills a compact ModernBERT token tagger "
            "from a fine-tuned teacher."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Training CoNLL-U file",
    )

    parser.add_argument(
        "--teacher",
        required=True,
        help="Fine-tuned teacher model directory",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Student model output directory",
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help=(
            "Student encoder layer count. "
            "Recommended first experiment: 6"
        ),
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help=(
            "Student hidden size. "
            "Use 768 for teacher-weight initialization. "
            "Use 512 for a smaller randomly initialized student."
        ),
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=12,
        help="Student attention head count.",
    )

    parser.add_argument(
        "--layer_indices",
        type=str,
        default="0,4,8,12,17,21",
        help=(
            "Teacher layers to copy, comma separated. "
            "Example: 0,4,8,12,17,21. "
            "If omitted, layers are selected automatically."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=10,
        help="Training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device train/eval batch size",
    )

    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Student learning rate",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.05,
        help="Validation sentence fraction",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Teacher-logit distillation temperature",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Logit distillation loss weight",
    )

    parser.add_argument(
        "--hidden_alpha",
        type=float,
        default=0.2,
        help="Hidden-state distillation loss weight",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args(argv)

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------
    if args.layers <= 0:
        raise ValueError(
            "--layers must be greater than zero"
        )

    if args.hidden_size % args.heads:
        raise ValueError(
            "--hidden_size must be divisible by --heads"
        )

    if args.alpha < 0.0:
        raise ValueError(
            "--alpha must be non-negative"
        )

    if args.hidden_alpha < 0.0:
        raise ValueError(
            "--hidden_alpha must be non-negative"
        )

    if args.alpha + args.hidden_alpha > 1.0:
        raise ValueError(
            "--alpha and --hidden_alpha must sum to at most 1"
        )

    requested_layer_indices = parse_layer_indices(
        args.layer_indices
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    logger.info(
        "Loading training data: %s",
        args.input,
    )

    sentences = read_conllu(args.input)

    tags = prepare_tags(sentences)

    tag2id = {
        tag: index
        for index, tag in enumerate(tags)
    }

    id2tag = {
        index: tag
        for index, tag in enumerate(tags)
    }

    logger.info(
        "Training sentences: %d",
        len(sentences),
    )

    logger.info(
        "Number of labels: %d",
        len(tags),
    )

    # ---------------------------------------------------------
    # Load teacher
    # ---------------------------------------------------------
    logger.info(
        "Loading teacher and tokenizer: %s",
        args.teacher,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher
    )

    teacher = load_finetuned_model(
        args.teacher
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    teacher_layers_count = (
        teacher.config.num_hidden_layers
    )

    teacher_hidden_size = (
        teacher.config.hidden_size
    )

    teacher_heads = (
        teacher.config.num_attention_heads
    )

    logger.info(
        "Teacher layers: %d",
        teacher_layers_count,
    )

    logger.info(
        "Teacher hidden size: %d",
        teacher_hidden_size,
    )

    logger.info(
        "Teacher attention heads: %d",
        teacher_heads,
    )

    # ---------------------------------------------------------
    # Select teacher layers
    # ---------------------------------------------------------
    if requested_layer_indices is None:
        # Evenly distribute selected layers over the teacher.
        #
        # For 22 -> 6 this gives approximately:
        #
        #   0, 4, 8, 13, 17, 21
        #
        layer_indices = [
            round(
                i
                * (teacher_layers_count - 1)
                / (args.layers - 1)
            )
            for i in range(args.layers)
        ]

    else:
        layer_indices = requested_layer_indices

    if len(layer_indices) != args.layers:
        raise ValueError(
            "Number of --layer_indices ({}) must equal "
            "--layers ({}).".format(
                len(layer_indices),
                args.layers,
            )
        )

    for index in layer_indices:
        if index < 0 or index >= teacher_layers_count:
            raise ValueError(
                "Teacher layer index {} is outside "
                "0..{}".format(
                    index,
                    teacher_layers_count - 1,
                )
            )

    logger.info(
        "Teacher layers selected: %s",
        layer_indices,
    )

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------
    logger.info("Building tokenized dataset")

    dataset = build_dataset(
        sentences,
        tokenizer,
        tag2id,
    )

    split = dataset.train_test_split(
        test_size=args.val_size,
        seed=args.seed,
    )

    logger.info(
        "Train sentences: %d",
        len(split["train"]),
    )

    logger.info(
        "Eval sentences: %d",
        len(split["test"]),
    )

    # ---------------------------------------------------------
    # Build student config
    # ---------------------------------------------------------
    student_config = make_student_config(
        teacher.config,
        len(tags),
        id2tag,
        tag2id,
        args.layers,
        args.hidden_size,
        args.heads,
        layer_indices,
    )

    logger.info(
        "Creating student: %d layers x %d hidden",
        args.layers,
        args.hidden_size,
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # This creates the architecture, but initially all weights
    # are random.
    #
    # We immediately replace compatible weights below with
    # weights copied from the teacher.
    # ---------------------------------------------------------
    student = ModernBertForTokenClassification(
        student_config
    )

    classifier_dropout = (
        getattr(
            student.config,
            "classifier_dropout",
            None,
        )
        or 0.1
    )

    # ---------------------------------------------------------
    # Initialize classifier
    # ---------------------------------------------------------
    initialize_classifier(
        student,
        teacher,
        len(tags),
        classifier_dropout,
    )

    # ---------------------------------------------------------
    # Initialize encoder from teacher
    # ---------------------------------------------------------
    encoder_initialized = (
        initialize_student_from_teacher(
            student,
            teacher,
            layer_indices,
            copy_classifier=False,
        )
    )

    # ---------------------------------------------------------
    # Hidden projection
    #
    # Only needed if student hidden size differs from teacher.
    #
    # For recommended 6x768:
    #
    #   student = 768
    #   teacher = 768
    #
    # so there is NO projection.
    # ---------------------------------------------------------
    if args.hidden_size != teacher_hidden_size:
        logger.info(
            "Creating hidden projection: %d -> %d",
            args.hidden_size,
            teacher_hidden_size,
        )

        student.hidden_projection = torch.nn.Linear(
            args.hidden_size,
            teacher_hidden_size,
        )
    else:
        logger.info(
            "Student and teacher have identical hidden size "
            "(%d); no hidden projection required.",
            args.hidden_size,
        )

    # ---------------------------------------------------------
    # Parameter counts
    # ---------------------------------------------------------
    trainable = sum(
        parameter.numel()
        for parameter in student.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in student.parameters()
    )

    logger.info(
        "Student encoder initialized from teacher: %s",
        encoder_initialized,
    )

    logger.info(
        "Student parameters: %s",
        format(total, ","),
    )

    logger.info(
        "Student trainable parameters: %s",
        format(trainable, ","),
    )

    # ---------------------------------------------------------
    # Training precision
    # ---------------------------------------------------------
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        torch.cuda.is_available()
        and not use_bf16
    )

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------
    effective_batch_size = (
        args.batch_size
        * args.grad_accum_steps
    )

    updates_per_epoch = (
        len(split["train"])
        + effective_batch_size
        - 1
    ) // effective_batch_size

    warmup_steps = max(
        1,
        int(
            updates_per_epoch
            * args.epochs
            * 0.1
        ),
    )

    logger.info(
        "Warmup steps: %d",
        warmup_steps,
    )

    # ---------------------------------------------------------
    # Training arguments
    # ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=os.path.join(
            args.out,
            "checkpoints",
        ),
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

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer
        ),
        processing_class=tokenizer,
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        hidden_alpha=args.hidden_alpha,
        teacher_layer_indices=layer_indices,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    logger.info(
        "Starting distilled student training"
    )

    logger.info(
        "Teacher layer mapping: %s",
        layer_indices,
    )

    trainer.train()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    logger.info(
        "Saving student model to: %s",
        args.out,
    )

    trainer.save_model(args.out)

    tokenizer.save_pretrained(args.out)

    with open(
        os.path.join(args.out, "tags.txt"),
        "w",
        encoding="utf-8",
    ) as output:
        output.write(
            "\n".join(tags) + "\n"
        )

    with open(
        os.path.join(
            args.out,
            "classifier_head.txt",
        ),
        "w",
        encoding="utf-8",
    ) as output:
        output.write("mlp")

    with open(
        os.path.join(
            args.out,
            "teacher_layer_indices.txt",
        ),
        "w",
        encoding="utf-8",
    ) as output:
        output.write(
            ",".join(
                str(index)
                for index in layer_indices
            )
        )

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
