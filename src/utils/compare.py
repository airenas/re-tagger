import argparse
import sys

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

from src.utils.logger import logger


def not_important(w, t1, t2):
    t1l = list(t1)
    t2l = list(t2)
    if w.lower() == 'ir' and t1l[0] in "CQ" and t2l[0] in "CQ":
        return True
    if t1l[0] == "Q" and t2l[0] == "Q":
        return True
    if t1l[0] == "N" and t2l[0] == "N" and t1l[0:-1] == t2l[0:-1]:
        return True
    if t1l[0] == "S" and t2l[0] == "S" and t1l[0:-1] == t2l[0:-1]:
        return True
    return False


def show_res(res):
    footers = ["micro avg", "macro avg", "weighted avg"]
    headers = ["precision", "recall", "f1-score", "support"]
    longest_last_line_heading = "weighted avg"
    name_width = max(len(cn) for cn in res.keys())
    width = max(name_width, len(longest_last_line_heading), 2)
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for key, values in res.items():
        if key not in footers:
            report += row_fmt.format(key, *values.values(), width=width, digits=2)
    report += "\n\n\n"
    for key in footers:
        report += row_fmt.format(key, *res[key].values(), width=width, digits=2)

    return report


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
    wc, errc, mwc, errmvc, err_not_imp = 0, 0, 0, 0, 0
    p1, p2 = int(args.p1), int(args.p2)
    y_pred, y_pred_not_imp, y_true = [], [], []
    with open(args.f1, 'r') as f1:
        with open(args.f2, 'r') as f2:
            f2i = iter(f2)
            for l1 in f1:
                try:
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
                y_true.append(w1[p1])
                y_pred.append(w2[p2])
                y_pred_not_imp.append(w2[p2])
                if " " in w1[0]:
                    mwc += 1
                if w1[p1] != w2[p2]:
                    print("{}\t{}\t{}\t{}".format(w1[0], w1[p1], w2[p2], args.diff_sym))
                    errc += 1
                    if not_important(w1[0], w1[p1], w2[p2]):
                        err_not_imp += 1
                        y_pred_not_imp[-1] = w1[p1]
                    if " " in w1[0]:
                        errmvc += 1
                else:
                    print("{}\t{}".format(w1[0], w1[args.p1]))
    logger.info("Results: all: {}, err: {}, {}, not important: {}".format(wc, errc, errc / wc, err_not_imp))
    if mwc > 0:
        logger.info("Results: multiple: {}, err: {}, {}".format(mwc, errmvc, errmvc / mwc))
    logger.info("Acc: {}".format(accuracy_score(y_true, y_pred)))
    labels = set()
    for la in y_true:
        labels.add(la)
    logger.info("F1 : {}".format(f1_score(y_true, y_pred, labels=list(labels), average='weighted', zero_division=0)))
    logger.info("Acc not important: {}".format(accuracy_score(y_true, y_pred_not_imp)))

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    res = metrics.classification_report(y_true, y_pred, labels=sorted_labels, digits=3, zero_division=0,
                                        output_dict=True)
    res = dict(sorted(res.items(), key=lambda item: - (1.0 - item[1]['precision']) * item[1]['support']), )
    logger.info('Data set classification report: \n\n{}'
                .format(show_res(res)))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
