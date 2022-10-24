import argparse
import csv
import pickle
import sys

import pandas as pd
from sklearn_crfsuite import metrics as crfmetrics
from tqdm import tqdm

from egs.crf.local.features import format_data, sent2features, sent2labels, drop_puncts
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--model", nargs='?', required=True, help="Model file")
    parser.add_argument("--out", nargs='?', required=False, help="Writes out predictions to tab separated file")
    parser.add_argument("--no_punct", default=False, action=argparse.BooleanOptionalAction,
                        help="Do not use punctuation in training/testing")
    parser.add_argument("--f_before", nargs='?', default=2, help="How many words use for features")
    parser.add_argument("--f_after", nargs='?', default=2, help="How many words use for features after")
    parser.add_argument("--f_func", nargs='?', default="get_word_feat", help="Features function")
    parser.add_argument("--f_skip_punct", default=False, action=argparse.BooleanOptionalAction,
                        help="Skip punctuation in features")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    data = {}
    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)

    print(data, sep='\n\n')

    logger.info(
        "Features: [-{},w,{}], func: {}, skip punct: {}".format(args.f_before, args.f_after, args.f_func,
                                                                args.f_skip_punct))
    logger.info("preparing data")

    test_sents = format_data(data)

    if args.no_punct:
        logger.info("Drop punctuations")
        test_sents = [drop_puncts(s) for s in tqdm(test_sents, "dropping punctuations", len(test_sents))]

    params = {"words_before": int(args.f_before), "words_after": int(args.f_after), "method": args.f_func,
              "skip_punct": args.f_skip_punct}
    x_test = [sent2features(s, params=params) for s in test_sents]
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
