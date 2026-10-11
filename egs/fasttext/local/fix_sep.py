#!/usr/bin/env python3
import argparse
import logging
import os

from tqdm import tqdm

from egs.fasttext.local.clean import KEEP_PUNCT, SHORT_FORMS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="""Input text file.
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="""Output file
            """,
    )

    return parser.parse_args()

def one_upper_letter(word):
    if word in SHORT_FORMS or (len(word) == 2 and word[0].isupper() and word[1] == "."):
        return True
    return False


def fix_sep(line):
    """Fix sep - add space before trailing sep"""
    res = []
    for word in line.split():
        pos = len(word)
        while (pos - 1) > -1 and word[pos - 1] in KEEP_PUNCT and not one_upper_letter(word[: pos]):
            pos -= 1
        if 0 < pos < len(word):
            res.append(word[:pos])
            res.append(word[pos:])
            continue
        res.append(word)
    return " ".join(res)


def main():
    args = get_args()

    logging.info(f"fixing separated {args.input}")

    read, wrote = 0, 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading file"):
                line = line.rstrip("\n")
                read += 1
                fixed = fix_sep(line)
                wrote += 1
                f_out.write(fixed + "\n")
    logging.info(f"read {read}, wrote {wrote}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
