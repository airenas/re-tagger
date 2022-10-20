import argparse
import sys

from src.utils.compare import show_compare_results
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Compares two files",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--f1", nargs='?', required=True, help="File 1")
    parser.add_argument("--f2", nargs='?', required=True, help="File 2")
    parser.add_argument("--diff_sym", nargs='?', default="<--diff-->", help="Add symbols to lines that differs")

    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("File 1: {}".format(args.f1))
    logger.info("File 2: {}".format(args.f2))
    wc, errc, mwc, errmvc = 0, 0, 0, 0
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
                w2t = []
                if len(w2) > 1:
                    w2t = w2[1].split(":")
                if w1[0] != w2[0]:
                    w1[0] = w1[0].replace("#", "_").strip()
                if w1[0] != w2[0]:
                    raise Exception("problem at {}, '{}' != '{}'".format(wc, w1[0], w2[0]))
                y_true.append(w1[1])
                if w1[1] in w2t:
                    y_pred.append(w1[1])
                    print("{}\t{}".format(w1[0], w1[1]))
                else:
                    errc += 1
                    y_pred.append(':'.join(w2t))
                    print("{}\t{}\t{}\t{}".format(w1[0], w1[1], ':'.join(w2t), args.diff_sym))
    logger.info("Results: all: {}, err: {}, {}".format(wc, errc, errc / wc))
    show_compare_results(y_true, y_pred)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
