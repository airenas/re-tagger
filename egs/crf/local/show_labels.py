import argparse
import csv
import sys

import pandas as pd

from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Show labels",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)
    res = dict()
    for d in data.values:
        res[d[9]] = res.get(d[9], 0) + 1
    i = 0
    for k, v in res.items():
        i += 1
        logger.info(" %s -> %d" % (k, v))
    logger.info("Count %d" % i)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
