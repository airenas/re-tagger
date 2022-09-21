import argparse
import csv
import math
import re
import string
import sys

import pandas as pd
import sklearn_crfsuite
from sklearn import metrics
from sklearn_crfsuite import metrics as crfmetrics
from sklearn_crfsuite.utils import flatten

from src.utils.logger import logger


def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': str(word[-3:]),
        'word[-2:]': str(word[-2:]),
        'word[-4:]': str(word[-4:]),
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)', r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)', r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': str(word1[-3:]),
            '-1:word[-2:]': str(word1[-2:]),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation),
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i - 2][0]
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': str(word2[-3:]),
            '-2:word[-2:]': str(word2[-2:]),
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation),
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': str(word1[-3:]),
            '+1:word[-2:]': str(word1[-2:]),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation),
        })
    else:
        features['EOS'] = True

    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)', r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': str(word2[-3:]),
            '+2:word[-2:]': str(word2[-2:]),
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation),
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [word[1] for word in sent]


def sent2tokens(sent):
    return [word[0] for word in sent]


# formatting the data into sentences
def format_data(csv_data):
    sents = []
    pos_id = 9
    for i in range(len(csv_data)):
        w_num = csv_data.iloc[i, 0]
        if math.isnan(w_num):
            continue
        elif w_num == 1.0 or w_num < w_prev:
            sents.append([[csv_data.iloc[i, 1], csv_data.iloc[i, pos_id]]])
        else:
            sents[-1].append([csv_data.iloc[i, 1], csv_data.iloc[i, pos_id]])
        w_prev = w_num
    for sent in sents:
        for i, word in enumerate(sent):
            if type(word[0]) != str:
                del sent[i]
    return sents


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--test", nargs='?', required=True, help="Test conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    data = {}
    logger.info("loading data {}".format(args.input))
    data['train'] = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)
    logger.info("loading data {}".format(args.test))
    data['test'] = pd.read_csv(args.test, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)

    print(data['train'], sep='\n\n')
    print(data['test'], sep='\n\n')
    logger.info("preparing data")
    train_sents = format_data(data['train'])
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    test_sents = format_data(data['test'])
    x_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    logger.info("training crf")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.25,
        c2=0.3,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, y_train)
    logger.info("done")
    # obtaining metrics such as accuracy, etc. on the train set
    labels = list(crf.classes_)
    # labels.remove('X')

    y_pred = crf.predict(x_train)
    print('F1 score on the train set = {}\n'.format(
        crfmetrics.flat_f1_score(y_train, y_pred, average='weighted', labels=labels)))
    print('Accuracy on the train set = {}\n'.format(crfmetrics.flat_accuracy_score(y_train, y_pred)))

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    y_train_flat = flatten(y_train)
    y_pred_flat = flatten(y_pred)
    print('Train set classification report: \n\n{}'
          .format(metrics.classification_report(y_train_flat, y_pred_flat, labels=sorted_labels, digits=3)))

    y_pred = crf.predict(x_test)
    print('F1 score on the test set = {}\n'.format(crfmetrics.flat_f1_score(y_test, y_pred,
                                                                            average='weighted', labels=labels)))
    print('Accuracy on the test set = {}\n'.format(crfmetrics.flat_accuracy_score(y_test, y_pred)))

    y_test_flat = flatten(y_test)
    y_pred_flat = flatten(y_pred)
    print('Test set classification report: \n\n{}'.format(
        metrics.classification_report(y_test_flat, y_pred_flat, labels=sorted_labels, digits=3)))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
