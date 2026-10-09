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
MAX_WORD_LEN = 30
MIN_ALNUM_RATIO = 2/3
# letter next to a digit, e.g. abc123 or 12abc
MIXED_LETTER_DIGIT_RE = re.compile(r"[^\W\d_]\d|\d[^\W\d_]")
# punctuation between letters, e.g. pvz.lt, t.y; hyphens and dashes are allowed
INNER_PUNCT_RE = re.compile(r"[^\W\d_][^\w\s\-‐‑‒–—―−]+[^\W\d_]")
REPEAT_LETTER_RE = re.compile(r"([^\W\d_])\1{3,}", re.IGNORECASE)


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


def alnum_ratio(line):
    """Share of letters and digits among non-space characters."""
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isalnum()) / len(chars)


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
    if alnum_ratio(line) < MIN_ALNUM_RATIO:
        return True
    if MIXED_LETTER_DIGIT_RE.search(line) or INNER_PUNCT_RE.search(line):
        return True
    if "|" in line:
        return True
    if REPEAT_LETTER_RE.search(line):
        return True
    if any(len(w) > MAX_WORD_LEN for w in line.split()):
        return True
    return any(c.isalpha() and c.lower() not in LT_LETTERS for c in line)


KEEP_PUNCT = frozenset("-,.!?+*/")
DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})


def replace_punct(line):
    """Replace punctuation not in KEEP_PUNCT with a space and collapse spaces."""
    line = line.translate(DASHES)
    line = "".join(" " if unicodedata.category(c).startswith("P") and c not in KEEP_PUNCT else c for c in line)
    return " ".join(line.split())


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
    return replace_punct(line), True


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
