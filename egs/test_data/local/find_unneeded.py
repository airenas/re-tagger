import argparse
import sys

from src.utils.logger import logger


def diff(labelsN, pos):
    s = set()
    for _, l in enumerate(labelsN):
        p = l[pos]
        if p != "-":
            s.add(p)
    lr = len(s) - 1
    if lr > 0:
        return 1
    return 0


def analyze(data, labels, word):
    for pos in ["N", "V", "A", "P", "M"]:
        labelsN = list(filter(lambda l: (l.startswith(pos)), labels))
        if len(labelsN) > 1:
            l = labelsN[0]
            if not pos in data:
                data[pos] = [0] * (len(l))
            n = data[pos]
            for i in range(len(n)):
                c = diff(labelsN, i)
                if n[i] == 0 and c > 0:
                    logger.info("{}{}: {} {}".format(pos, i, word, labels))
                n[i] += c


def main(argv):
    parser = argparse.ArgumentParser(description="Find unneeded params in labels",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial tab separated file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    data = dict()
    logger.info("loading data {}".format(args.input))
    with open(args.input, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split("\t")
            try:
                if len(parts) > 1 and ":" in parts[1]:
                    labels = list(map(lambda l: (l.split(":")[1]), parts[1].split(";")))
                    analyze(data, labels, parts[0])
            except BaseException as err:
                raise Exception("problem at " + line + "\n" + repr(err))

    for k, v in data.items():
        logger.info(" {} -> {}".format(k, v))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
