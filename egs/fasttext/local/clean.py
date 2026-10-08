#!/usr/bin/env python3
import argparse
import logging
import os
import re
import unicodedata

from tqdm import tqdm

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


LT_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyząčęėįšųūž")

URL_RE = re.compile(
    r"^(?:(?:[a-z][a-z0-9+.-]*://|www\.)[^\s/?#:]+|[^\s/?#:@]+\.[a-z]{2,}(?=[/?#:]))",
    re.IGNORECASE)
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")
TRAILING_PUNCT = ".,;:!?)]}»”\"'"


def replace_url_email(word):
    """Replace an email with <email> and a url with <url>, keeping trailing punctuation."""
    core = word.rstrip(TRAILING_PUNCT)
    tail = word[len(core):]
    if EMAIL_RE.match(core):
        return "<email>" + tail
    if URL_RE.match(core):
        return "<url>" + tail
    return word


MAX_PUNCT_RUN = 3


def has_punct_run(line):
    """True if the line has more than MAX_PUNCT_RUN adjacent punctuation characters."""
    run = 0
    for c in line:
        if unicodedata.category(c).startswith("P"):
            run += 1
            if run > MAX_PUNCT_RUN:
                return True
        else:
            run = 0
    return False


def skip(line):
    """Skip the line if it has no letters (only punctuation, digits, spaces),
    has a long run of punctuation, or contains a letter that is not Lithuanian.
    Digits are allowed."""
    if not any(c.isalpha() for c in line):
        return True
    if sum(1 for c in line if c.isalpha()) < 3:
        return True    
    if has_punct_run(line):
        return True
    return any(c.isalpha() and c.lower() not in LT_LETTERS for c in line)


def clean_text(line):
    """Clean the line by normalizing whitespace and punctuation."""
    # Normalize whitespace
    words = line.split()
    res = []
    for word in words:
        w = word.split("(=", 1)
        res.append(replace_url_email(w[0]))

    line = " ".join(res)
    # Normalize punctuation
    line = unicodedata.normalize("NFKC", line)
    if skip(line):
        return line, False
    return line, True


def main():
    args = get_args()

    logging.info(f"skipping and normalizing {args.input}")

    read, wrote, skipped = 0, 0, 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading file"):
                line = line.rstrip("\n")
                read += 1
                cleaned, ok = clean_text(line)
                if not ok or len(cleaned) < 2: # skip too short sentences, perhaps bad splitting into sentences
                    skipped += 1
                    continue
                wrote += 1
                f_out.write(cleaned + "\n")
    logging.info(f"read {read}, wrote {wrote}, skipped {skipped} sentences")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter,
                        level=getattr(logging, os.environ.get("LOGLEVEL", "WARNING").upper(), logging.WARNING))

    logging.info(f"Starting")
    main()
    logging.info(f"Done")
