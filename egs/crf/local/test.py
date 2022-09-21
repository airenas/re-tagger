import argparse
import csv
import pickle
import sys

import pandas as pd
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
    parser.add_argument("--model", nargs='?', required=True, help="Model file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    data = {}
    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)

    print(data, sep='\n\n')
    logger.info("preparing data")
    test_sents = format_data(data)
    x_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    logger.info("loading crf from {}".format(args.model))
    with open(args.model, "rb") as f:
        crf = pickle.load(f)

    # obtaining metrics such as accuracy, etc. on the train set
    labels = list(crf.classes_)
    # labels.remove('X')

    y_pred = crf.predict(x_test)
    print('F1 score on the {} = {}\n'.format(args.input,
                                             crfmetrics.flat_f1_score(y_test, y_pred, average='weighted',
                                                                      labels=labels)))
    print('Accuracy on the {} = {}\n'.format(args.input, crfmetrics.flat_accuracy_score(y_test, y_pred)))

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    y_train_flat = flatten(y_test)
    y_pred_flat = flatten(y_pred)
    print('Train set classification report: \n\n{}'
          .format(metrics.classification_report(y_train_flat, y_pred_flat, labels=sorted_labels, digits=3)))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
