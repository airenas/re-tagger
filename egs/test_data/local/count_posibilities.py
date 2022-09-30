import argparse
import sys

from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Count distribution of lemma count possibilities for each word",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial words file with lemmas")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    i, ps = 0, dict()
    with open(args.input, 'r') as file:
        for line in file:
            i += 1
            line = line.strip()
            words = line.split("\t")
            v = 0
            if len(words) > 1:
                v = len(words[1].split(";"))
            ps[v] = ps.get(v, 0) + 1

    logger.info("Read %d lines" % (i))
    for k, v in ps.items():
        logger.info(" %d -> %d, %.2f" % (k, v, v / i))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
