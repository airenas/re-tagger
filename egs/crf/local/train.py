import argparse
import csv
import pickle
import sys

import pandas as pd
import sklearn_crfsuite
from sklearn import metrics
from sklearn_crfsuite import metrics as crfmetrics
from sklearn_crfsuite.utils import flatten

from egs.crf.local.features import format_data, sent2features, sent2labels
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--out", nargs='?', required=True, help="Model output file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    data = {}
    logger.info("loading data {}".format(args.input))
    data['train'] = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)

    print(data['train'], sep='\n\n')
    logger.info("preparing data")
    train_sents = format_data(data['train'])
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    logger.info("training crf")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.25,
        c2=0.3,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True
    )
    crf.fit(x_train, y_train)
    logger.info("done")
    with open(args.out, "wb") as f:
        pickle.dump(crf, f)

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
