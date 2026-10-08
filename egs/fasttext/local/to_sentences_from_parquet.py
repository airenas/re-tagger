import argparse
import logging
import os
import random
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


class Word:
    def __init__(self, string):
        strs = string.split("(=", 1)
        if len(strs) == 2:
            self.word = strs[0]
            strs_mi = strs[1].split(")", 1)
            if len(strs_mi) == 2:
                self.mi = strs_mi[0]
                self.word, self.punct = split_word_punctuation(self.word + strs_mi[1])  # add punctuation to word
                return
        self.word, self.punct = split_word_punctuation(string)
        self.mi = ""

    def to_str(self):
        if self.mi:
            return f"{self.word}{self.punct}(={self.mi})"
        else:
            return f"{self.word}{self.punct}"


def split_word_punctuation(word):
    w = ""
    for c in word:
        if c.isalpha() or c.isdigit():
            w += c
        else:
            return w, word[len(w):]
    return w, ""


def _resolve_parquet_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise ValueError(f"No parquet files found in '{input_path}'")


def iter_text_rows(input_path: str, text_field: str = "text"):
    for parquet_file in _resolve_parquet_files(input_path):
        pf = pq.ParquetFile(parquet_file)
        if text_field not in set(pf.schema_arrow.names):
            raise ValueError(
                f"Column '{text_field}' not found in {parquet_file}. Available: {', '.join(pf.schema_arrow.names)}"
            )
        for batch in pf.iter_batches(columns=[text_field]):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str):
                    yield value


def count_rows(input_path: str) -> int:
    total = 0
    for parquet_file in _resolve_parquet_files(input_path):
        total += pq.ParquetFile(parquet_file).metadata.num_rows
    return total


def clean(line):
    ## remove morf info if any
    line = line.strip()
    words = [Word(s) for s in line.split()]
    line = " ".join([w.word + w.punct for w in words])
    line = " ".join(line.split())
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="""Parquet files dir.
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output txt file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="how many rows to export, 0 means no limit (default: 0)."
    )
    args = parser.parse_args()

    logging.info(f"Extract sentenced from {args.input}")
    logging.info(f"Extract sentenced to {args.output}")

    random.seed(42)

    total = count_rows(args.input)
    c = 0
    with tqdm(total=total, unit="rows", desc="Saving") as pbar:
        with open(args.output, "w", encoding="utf-8") as f_out:
            for line in iter_text_rows(args.input):
                line = clean(line)
                f_out.write(line + "\n")
                c += 1
                if args.limit > 0 and c >= args.limit:
                    logging.info(f"Reached limit of {args.limit} lines, stopping")
                    break
                pbar.update(1)
    logging.info(f"Extracted {c} lines")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")

    main()

    logging.info("Done")
