import argparse
import sys

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

from src.utils.logger import logger
from src.utils.punct import is_punctuation


def drop_non_important(w, m):
    m = list(m)
    if w.lower() == 'ir' and m[0] == "Q":
        m[0] = 'C'
    if m[0] in "NS":
        m[-1] = "-"
    if m[0] == "N" and m[1] in "nx":
        m[1] = "c"
    if m[0] == "M" and m[1] in "c":
        m[3] = "-"
    if m[0] in "XY":
        m[1] = "-"
    if m[0] in "OIQSRAVC":
        m[1] = "g"
    return "".join(m)


def show_res(res, n=15):
    footers = ["micro avg", "macro avg", "weighted avg"]
    headers = ["precision", "recall", "f1-score", "support"]
    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in res.keys())
    width = max(name_width, len(longest_last_line_heading), 2)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for i, (key, values) in enumerate(res.items()):
        if i > n:
            break
        if key not in footers and type(values) is dict:
            report += row_fmt.format(key, *values.values(), width=width, digits=2)
    report += "\n\n\n"
    for key in footers:
        if type(res.get(key, None)) is dict:
            report += row_fmt.format(key, *res[key].values(), width=width, digits=2)

    return report


def show_compare_results(y_true, y_pred):
    labels = set()
    for la in y_true:
        labels.add(la)
    logger.info("Acc: {}".format(accuracy_score(y_true, y_pred)))
    logger.info("F1 : {}".format(f1_score(y_true, y_pred, labels=list(labels), average='weighted', zero_division=0)))
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    res = metrics.classification_report(y_true, y_pred, labels=sorted_labels, digits=3, zero_division=0,
                                        output_dict=True)
    try:
        res = dict(sorted(res.items(), key=lambda item: - (1.0 - item[1]['precision']) * item[1]['support']), )
    except BaseException as err:
        logger.warning(err)
    logger.info('Data set classification report: \n\n{}'
                .format(show_res(res)))


def first_word(l1):
    w1 = l1.split("\t")
    return w1[0].strip()


def main(argv):
    parser = argparse.ArgumentParser(description="Compares two files",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--f1", nargs='?', required=True, help="File 1")
    parser.add_argument("--f2", nargs='?', required=True, help="File 2")
    parser.add_argument("--p1", nargs='?', default=1, help="pos tab separate column for compare for f1")
    parser.add_argument("--p2", nargs='?', default=1, help="pos tab separate column for compare for f2")
    parser.add_argument("--diff_sym", nargs='?', default="<--diff-->", help="Add symbols to lines that differs")
    parser.add_argument("--ip1", default=False, action=argparse.BooleanOptionalAction,
                        help="Ignore punctuation for file 1")
    parser.add_argument("--ip2", default=False, action=argparse.BooleanOptionalAction,
                        help="Ignore punctuation for file 2")

    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("File 1: {}".format(args.f1))
    logger.info("File 2: {}".format(args.f2))
    wc, errc, mwc, errmvc, err_not_imp = 0, 0, 0, 0, 0
    p1, p2 = int(args.p1), int(args.p2)
    y_pred, y_true = [], []
    with open(args.f1, 'r') as f1:
        with open(args.f2, 'r') as f2:
            f2i = iter(f2)
            for l1 in f1:
                if args.ip1 and is_punctuation(first_word(l1)):
                    continue
                try:
                    l2 = next(f2i)
                    while args.ip2 and is_punctuation(first_word(l2)):
                        l2 = next(f2i)
                except StopIteration:
                    logger.warning("Files not match")
                    break
                wc += 1
                l1 = l1.strip()
                l2 = l2.strip()
                w1 = l1.split("\t")
                w2 = l2.split("\t")
                if len(w1) < p1:
                    raise Exception("problem at {}, {}: {}".format(args.f1, wc, l1))
                if len(w2) < p2:
                    raise Exception("problem at {}, {}: {}".format(args.f2, wc, l2))
                if w1[0] != w2[0]:
                    w1[0] = w1[0].replace("#", "_").strip()
                if w1[0] != w2[0]:
                    raise Exception("problem at {}, '{}' != '{}'".format(wc, w1[0], w2[0]))
                w1[p1] = drop_non_important(w1[0], w1[p1])
                w2[p2] = drop_non_important(w2[0], w2[p2])
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
    if mwc > 0:
        logger.info("Results: multiple: {}, err: {}, {}".format(mwc, errmvc, errmvc / mwc))
    show_compare_results(y_true, y_pred)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
