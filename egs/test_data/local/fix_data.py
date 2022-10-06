import argparse
import sys

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Fix conllu file",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, ws = 0, 0
    with ConlluReader(args.input) as cr:
        for sent in cr:
            sc += 1
            for line in sent.lines:
                if "<g/>" in line:
                    ws += 1
                else:
                    print(line)
            print()

    print("Read %d sentences, %d skipped g's" % (sc, ws), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
