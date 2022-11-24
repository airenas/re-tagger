import argparse
import sys

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Join sentences to results",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial results file")
    parser.add_argument("--in_s", nargs='?', required=True, help="Original conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    wc, li = 0, 0
    with ConlluReader(args.in_s) as cr:
        with open(args.input, 'r') as fl:
            fli = iter(fl)
            for sent in cr:
                sent_id = ""
                for line in sent.lines:
                    li += 1
                    if line.startswith("#"):
                        if "sent_id" in line:
                            sent_id = line.split("=")[1].strip()
                    else:
                        if not sent_id:
                            raise Exception("no sentence at {}".format(li))
                        line = line.replace("#", "_")
                        wrds = line.split("\t")
                        lw, wc = next(fli).strip(), wc + 1
                        lwp = lw.split("\t")
                        wrds[1] = wrds[1].strip()
                        if lwp[0] != wrds[1]:
                            lwp[0] = lwp[0].replace("#", "_").strip()
                        if lwp[0] != wrds[1]:
                            raise Exception("problem at {}, '{}' != '{}'".format(li, wrds[1], lwp[0]))
                        print("{}\t{}".format(sent_id, lw))

    print("Read %d lines, %d words" % (li, wc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
