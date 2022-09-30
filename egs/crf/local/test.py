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
    parser.add_argument("--out", nargs='?', required=False, help="Writes out predictions to tab separated file")
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
    logger.info("loaded")

    # obtaining metrics such as accuracy, etc. on the train set
    labels = list(crf.classes_)
    # labels.remove('X')

    logger.info("predict {}".format(args.input))
    y_pred = crf.predict(x_test)
    logger.info("prediction done")
    if args.out:
        with open(args.out, 'w') as f:
            for i, x in enumerate(x_test):
                y = y_pred[i]
                for j, xw in enumerate(x):
                    print("{}\t{}".format(xw.get("word"), y[j]), file=f)

    print('F1 score on the {} = {}\n'.format(args.input,
                                             crfmetrics.flat_f1_score(y_test, y_pred, average='weighted',
                                                                      labels=labels, zero_division=0)))
    print('Accuracy on the {} = {}\n'.format(args.input, crfmetrics.flat_accuracy_score(y_test, y_pred)))

    # sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    # y_train_flat = flatten(y_test)
    # y_pred_flat = flatten(y_pred)
    # print('Data set classification report: \n\n{}'
    #       .format(metrics.classification_report(y_train_flat, y_pred_flat, labels=sorted_labels, digits=3,
    #                                             zero_division=0)))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
