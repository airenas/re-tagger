import argparse
import sys

from tqdm import tqdm

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Find all punctuations in conllu",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, pc = 0, 0
    res = set()
    with ConlluReader(args.input) as cr:
        with tqdm(desc="reading") as pbar:
            for sent in cr:
                pbar.update(1)
                sc += 1
                words = sent.words()
                for word in words:
                    if len(word) == 1 and not word.isalnum():
                        pc += 1
                        res.add(word)
    print("{}".format("".join(res)))

    print("Read %d sentences, %d puncts" % (sc, pc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
