import argparse
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from egs.transformer.local.train import load_finetuned_model, read_conllu
from src.utils.logger import logger


def predict(text, tokenizer, model, device, is_words):
    """Predicts tags for a batch of sentences, returning a list of per-word tag lists."""
    words_batch = [text]
    encoding = tokenizer(words_batch, is_split_into_words=is_words, padding=True, truncation=True, return_tensors="pt",
                         return_offsets_mapping=True)
    logger.info(f"Encoding: {encoding}")
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    logger.info(f"Tokens: {tokens}")

    word_ids = encoding.word_ids(batch_index=0)
    offsets = encoding["offset_mapping"][0]

    for token, word_id, offset in zip(tokens, word_ids, offsets):
        start, end = offset.tolist()
        print(
            f"{token:10} "
            f"word={str(word_id):6} "
            f"chars=({start:2}, {end:2}) "
            f"text={text[start:end]!r}"
        )

    with torch.no_grad():
        logits = model(**encoding.to(device)).logits
    predicted_ids = logits.argmax(dim=-1).cpu()


    results = []
    previous_word_id = None
    for token_pos, word_id in enumerate(encoding.word_ids(batch_index=0)):
        if word_id is None or word_id == previous_word_id:
            previous_word_id = word_id
            continue
        tag = model.config.id2label[predicted_ids[0, token_pos].item()]
        previous_word_id = word_id
        results.append(tag)

    return results


def main(argv):
    parser = argparse.ArgumentParser(description="Predicts tags for text with a fine-tuned transformer model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--text", nargs='?', required=True, help="Input text. '-' is a sep between words. REQUIRED")
    parser.add_argument("--model", nargs='?', required=True, help="Trained model directory")
    parser.add_argument("--split", nargs='?', type=int, required=False, default=1, help="split text by -")

    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("Loading model: {}".format(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_finetuned_model(args.model).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params:
            logger.debug(f"{name:40} {params:,}")

    text = args.text
    if args.split == 1:
        text = text.split("-")
    logger.info(f"Predicting: {text}")
    predicted_tags = predict(text, tokenizer, model, device, args.split == 1)
    logger.info(f"Predicted: {predicted_tags}")

    logger.info("Done")


if __name__ == "__main__":

    main(sys.argv[1:])
