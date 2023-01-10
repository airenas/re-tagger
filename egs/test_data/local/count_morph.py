import argparse
import sys

from src.utils.compare import drop_non_important
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial words file")
    parser.add_argument("--show", default=False, action=argparse.BooleanOptionalAction,
                        help="Do show wrong lines")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    a, e = 0, 0
    with open(args.input, 'r') as file:
        for line in file:
            a += 1
            line = line.strip()
            words = line.split("\t")
            if drop_non_important(words[1]) != drop_non_important(words[2]):
                e += 1
                if args.show:
                    logger.info(line)

    logger.info("Read %d lines, errors %d, %.4f, %.4f" % (a, e, e / a, 1 - e / a))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
