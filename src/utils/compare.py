import argparse
import sys

from src.utils.logger import logger
from sklearn.metrics import accuracy_score, f1_score


def main(argv):
    parser = argparse.ArgumentParser(description="Compares two files",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--f1", nargs='?', required=True, help="File 1")
    parser.add_argument("--f2", nargs='?', required=True, help="File 2")
    parser.add_argument("--p1", nargs='?', default=1, help="pos tab separate column for compare for f1")
    parser.add_argument("--p2", nargs='?', default=1, help="pos tab separate column for compare for f2")
    parser.add_argument("--diff_sym", nargs='?', default="<--diff-->", help="Add symbols to lines that differs")

    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("File 1: {}".format(args.f1))
    logger.info("File 2: {}".format(args.f2))
    wc, errc, mwc, errmvc = 0, 0, 0, 0
    p1, p2 = int(args.p1), int(args.p2)
    y_pred, y_true = [], []
    with open(args.f1, 'r') as f1:
        with open(args.f2, 'r') as f2:
            f2i = iter(f2)
            for l1 in f1:
                wc += 1
                l2 = next(f2i)
                l1 = l1.strip()
                l2 = l2.strip()
                w1 = l1.split("\t")
                w2 = l2.split("\t")
                if len(w1) < p1:
                    raise Exception("problem at {}, {}: {}".format(args.f1, wc, l1))
                if len(w2) < p2:
                    raise Exception("problem at {}, {}: {}".format(args.f2, wc, l2))
                if w1[0] != w2[0]:
                    raise Exception("problem at {}, {} != {}".format(wc, w1[0], w2[0]))
                y_true.append(w1[p1])
                y_pred.append(w2[p2])
                if " " in w1[0]:
                    mwc += 1
                if w1[p1] != w2[p2]:
                    print("{}\t{}\t{}\t{}".format(w1[0], w1[p1], w2[p2], args.diff_sym))
                    errc += 1
                    if " " in w1[0]:
                        errmvc += 1
                else:
                    print("{}\t{}".format(w1[0], w1[args.p1]))
    logger.info("Results: all: {}, err: {}, {}".format(wc, errc, errc / wc))
    logger.info("Results: multiple: {}, err: {}, {}".format(mwc, errmvc, errmvc / mwc))
    logger.info("Acc: {}".format(accuracy_score(y_true, y_pred)))
    labels = set()
    for la in y_true:
        labels.add(la)
    logger.info("F1 : {}".format(f1_score(y_true, y_pred, labels=list(labels), average='weighted', zero_division=0)))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])