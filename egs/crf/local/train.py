import argparse
import csv
import pickle
import sys

import pandas as pd
import sklearn_crfsuite

from egs.crf.local.features import format_data, sent2features, sent2labels
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--out", nargs='?', required=True, help="Model output file")
    parser.add_argument("--f_before", nargs='?', default=2, help="How many words use for features")
    parser.add_argument("--f_after", nargs='?', default=2, help="How many words use for features after")
    parser.add_argument("--f_func", nargs='?', default="get_word_feat", help="Features function")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    data = {}
    logger.info("loading data {}".format(args.input))
    data['train'] = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)

    print(data['train'], sep='\n\n')
    logger.info("Features: [-{},w,{}], func: {}".format(args.f_before, args.f_after, args.f_func))
    logger.info("preparing data")
    train_sents = format_data(data['train'])
    x_train = [sent2features(s, words_before=int(args.f_before), words_after=int(args.f_after), method=args.f_func) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    print(x_train[0][:], sep='\n\n')

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
