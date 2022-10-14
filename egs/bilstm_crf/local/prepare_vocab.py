import argparse
import csv
import sys

import pandas as pd

from egs.bilstm_crf.local.format_data import ending
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Prepares vocabulary",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--out_v", nargs='?', required=True, help="Vocabulary output file")
    parser.add_argument("--out_t", nargs='?', required=True, help="Tags vocabulary output file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)
    tags = list(data[9].unique())
    tags = ['<PAD>'] + tags
    logger.info("tags count {}".format(len(tags)))
    words = list(data[1].unique())
    logger.info("words count: {}".format(len(words)))
    ends = list(set([ending(w) for w in words]))
    logger.info("ends count: {}".format(len(ends)))

    logger.info("saving {}".format(args.out_v))
    with open(args.out_v, 'w') as f:
        for w in words:
            print(w, file=f)

    logger.info("saving {}".format(args.out_t))
    with open(args.out_t, 'w') as f:
        for w in tags:
            print(w, file=f)
    logger.info("saving {}".format(args.out_v + ".end"))
    with open(args.out_v + ".end", 'w') as f:
        for w in ends:
            print(w, file=f)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
