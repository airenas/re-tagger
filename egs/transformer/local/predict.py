import argparse
import sys

import torch
from tqdm import tqdm

from egs.transformer.local.model import MultiHeadTokenClassifier
from egs.transformer.local.morph import to_full
from egs.transformer.local.train import read_conllu
from src.utils.logger import logger


def predict_batch(sentences, tokenizer, model, device, head_names=None, id2label=None):
    """Predicts tags for a batch of sentences, returning a list of per-word tag lists."""
    words_batch = [s["words"] for s in sentences]
    encoding = tokenizer(words_batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits_dict = model(**encoding.to(device))

    predicted_ids_dict = {
        head: logits.argmax(dim=-1).cpu()
        for head, logits in logits_dict.items()
    }

    results = []
    for i, words in enumerate(words_batch):
        # List of dictionaries, one per word: [{'pos': 'NOUN', 'full': '...'}, ...]
        word_predictions = [{} for _ in words]
        previous_word_id = None

        for token_pos, word_id in enumerate(encoding.word_ids(batch_index=i)):
            # Skip special tokens (None) and subword tokens (only map the first subword token of each word)
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue

            # Decode each head for the first subword of the current word
            for head in head_names:
                pred_id = predicted_ids_dict[head][i, token_pos].item()
                head_id2label = id2label[head]
                word_predictions[word_id][head] = head_id2label.get(pred_id, "<UNK>")

            previous_word_id = word_id

        results.append([to_full(word_pred) for word_pred in word_predictions])

    return results


def main(argv):
    parser = argparse.ArgumentParser(description="Predicts tags with a fine-tuned transformer model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Input conllu file")
    parser.add_argument("--model", nargs='?', required=True, help="Trained model directory")
    parser.add_argument("--out", nargs='?', required=True, help="Prediction output file")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("Loading model: {}".format(args.model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: {}".format(device))
    model, tokenizer, label_config = MultiHeadTokenClassifier.load(args.model, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params:
            logger.info(f"{name:40} {params:,}")

    logger.info("Loading: {}".format(args.input))
    sentences = read_conllu(args.input)
    logger.info("Sentences: {}".format(len(sentences)))

    with open(args.out, "w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(sentences), args.batch_size), desc="Predicting", unit=" batches"):
            batch = sentences[start:start + args.batch_size]
            predicted_tags = predict_batch(batch, tokenizer, model, device, head_names=label_config["features"], id2label=label_config["id2tag"])
            for sentence, tags in zip(batch, predicted_tags):
                for word, tag in zip(sentence["words"], tags):
                    out_f.write("{}\t{}\n".format(word, tag))

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
