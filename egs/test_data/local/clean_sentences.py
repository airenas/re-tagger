import argparse
import logging
import sys

from tqdm import tqdm

from src.utils.conllu import ConlluReader


def has_alpha(sent):
    return any(char.isalpha() for word in sent.words() for char in word)

def has_mi(sent):
    for line in sent.lines:
        if not line.startswith("#"):
            wrds = line.split("\t")
            if len(wrds) < 10:
                return False
            if "Multext=" not in wrds[9]:
                return False
            tag = wrds[9].split("Multext=", 1)[1]
            if tag == "":
                return False
            if tag[0] == "N" and len(tag) < 7:
                return False
            if tag[0] == "V" and len(tag) < 14:
                return False
            if "=" in tag: # smth wrong
                return False
            if ":" in tag: # smth wrong
                return False    
    return True


def main(argv):
    parser = argparse.ArgumentParser(
        description="Drop CoNLL-U sentences without alphabetic characters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Initial conllu file")
    parser.add_argument("--output", required=True, help="Cleaned conllu file")
    args = parser.parse_args(argv)

    read_count = 0
    kept_count = 0
    with ConlluReader(args.input) as reader, open(args.output, "w") as output:
        for sent in tqdm(reader, desc="Filtering..."):
            read_count += 1
            if not has_alpha(sent):
                logging.info(f"drop: {sent.sentence()}")
                continue
            if not has_mi(sent):
                logging.info(f"drop: {sent.sentence()}")
                continue    
            kept_count += 1
            output.write("\n".join(sent.lines))
            output.write("\n\n")

    logging.info("Read %d sentences, kept %d, dropped %d" % (read_count, kept_count, read_count - kept_count))


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main(sys.argv[1:])