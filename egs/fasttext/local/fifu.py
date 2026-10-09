import argparse
import io
import logging
import os
import sys

import finalfusion
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="""Input fifu file.
        """,
    )
    
    args = parser.parse_args()
    
        # Load .fifu model (supports both subword-enabled and word-only formats)
    embeds = finalfusion.load_finalfusion(args.input)

    for line in sys.stdin:
        word = line.strip()
        if not word:
            continue
        in_vocab = word in embeds
        # subword models compose a vector from char n-grams for words outside the vocabulary
        vector = embeds.embedding(word)
        if vector is None:
            vector = np.zeros(embeds.storage.shape[1], dtype=np.float32)
            print(f"Word '{word}' has no vector. Using zero vector.")
        elif in_vocab:
            print(f"Found vector for '{word}', shape: {vector.shape}")
        else:
            print(f"OOV '{word}', vector from subwords, shape: {vector.shape}")
        print(f"Vector for '{word}': {vector}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")