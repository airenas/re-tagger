import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

from egs.transformer.local.train import load_finetuned_model
from src.utils.logger import logger


class TokenClassifierExportWrapper(torch.nn.Module):
    """Exports only the inference inputs and logits, without Hugging Face output objects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main(argv):
    parser = argparse.ArgumentParser(description="Exports a fine-tuned transformer tagger to ONNX")
    parser.add_argument("--model", required=True, help="Fine-tuned model directory")
    parser.add_argument("--out", required=True, help="Output ONNX file")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args(argv)

    logger.info("Loading tokenizer: {}".format(args.model))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger.info("Loading model: {}".format(args.model))
    model = load_finetuned_model(args.model).cpu().eval()
    wrapper = TokenClassifierExportWrapper(model)

    encoded = tokenizer("ONNX export sample", return_tensors="pt")
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Exporting ONNX: {}".format(args.out))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (encoded["input_ids"], encoded["attention_mask"]),
            args.out,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=args.opset,
        )
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])