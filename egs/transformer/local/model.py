import json
import os
from typing import Dict

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


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


class MultiHeadTokenClassifier(nn.Module):
    """Transformer base encoder with multiple classification heads (one per feature)."""

    def __init__(self, model_name: str = "", config: AutoConfig = None, feature_num_labels: Dict[str, int] = None,
                 dropout=0.1,
                 load_encoder_weights=True, use_lora=True):
        super().__init__()

        if config:
            self.config = config
        else:
            self.config = AutoConfig.from_pretrained(model_name)

        if load_encoder_weights:
            self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.base_model = AutoModel.from_config(self.config)

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                target_modules=["Wqkv", "out_proj", "in_proj"],
            )
            self.base_model = get_peft_model(
                self.base_model,
                lora_config,
            )

        self.heads = nn.ModuleDict()
        self.feature_num_labels = feature_num_labels or {}
        hidden_size = self.config.hidden_size

        for feat, num_labels in feature_num_labels.items():
            self.heads[feat] = MLPClassifierHead(hidden_size, num_labels, dropout=dropout)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        base_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("labels_")}
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **base_kwargs)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        return {feat: head(sequence_output) for feat, head in self.heads.items()}

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu"):
        """
        Class method to load model, tokenizer, and metadata from a saved checkpoint folder.
        """
        # 1. Load label configuration metadata
        labels_json_path = os.path.join(model_dir, "labels.json")
        if not os.path.exists(labels_json_path):
            raise FileNotFoundError(f"Missing labels.json in {model_dir}")

        with open(labels_json_path, "r", encoding="utf-8") as f:
            label_config = json.load(f)

        id2label = label_config.get("id2tag", {})
        id2label = {head: {int(k): v for k, v in head_map.items()} for head, head_map in id2label.items()}
        label_config["id2tag"] = id2label
        feature_num_labels = {head: len(head_map) for head, head_map in id2label.items()}

        # 2. Load Config (Fallback to base_model_name if config.json missing in directory)
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            config = AutoConfig.from_pretrained(model_dir)
        else:
            raise FileNotFoundError(
                f"No config.json found in {model_dir}. "
                "Pass `base_model_name='...'` to from_pretrained() or save config.json in training."
            )

        # 3. Instantiate model architecture using config
        model = cls(
            config=config,
            feature_num_labels=feature_num_labels,
            load_encoder_weights=False,
        )

        # 4. Load state dict weights (safetensors or bin)
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location=device)
        else:
            raise FileNotFoundError(f"No valid weights file found in {model_dir}")

        model.load_state_dict(state_dict, strict=True)
        encoder_dtype = next(model.base_model.parameters()).dtype
        model = model.to(device=device, dtype=encoder_dtype)
        model.eval()

        # 5. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        return model, tokenizer, label_config
