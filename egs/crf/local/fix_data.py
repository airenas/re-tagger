import argparse
import sys

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def lemmas(l, w):
    return l.get(w)


def fix_label(param):
    s = list(param)
    if s[0] == 'N':
        s[3] = "-"

        s[5] = "-"
        s[6] = "-"
    elif s[0] == 'A':
        s[4] = "-"

        s[1] = "-"
        s[2] = "-"
        s[6] = "-"
    elif s[0] == 'V':
        s[5] = "-"

        s[9] = "-"
        s[13] = "-"
    elif s[0] == 'P':
        s[1] = "-"
        s[5] = "-"
    elif s[0] == 'M':
        s[5] = "-"
        s[6] = "-"
    return "".join(s)


def main(argv):
    parser = argparse.ArgumentParser(description="Outputs all possible lemma variants for each word",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, ws = 0, 0
    with ConlluReader(args.input) as cr:
        for sent in cr:
            sc += 1
            for line in sent.lines:
                if "<g/>" in line:
                    ws += 1
                else:
                    if not line.startswith("#"):
                        line = line.replace("#", "_")
                        wrds = line.split("\t")
                        wrds[9] = wrds[9].partition("Multext=")[2]
                        wrds[9] = fix_label(wrds[9])
                        line = "\t".join(wrds)
                    print(line)
            print()

    print("Read %d sentences, %d skipped g's" % (sc, ws), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
