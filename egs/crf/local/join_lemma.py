import argparse
import sys

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Outputs all possible lemma variants for each word",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--lemmas", nargs='?', required=True, help="File with all possible lemmas")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    li = 0
    with ConlluReader(args.input) as cr:
        with open(args.lemmas, 'r') as fl:
            fli = iter(fl)
            for sent in cr:
                for line in sent.lines:
                    li += 1
                    if not line.startswith("#"):
                        line = line.replace("#", "_")
                        wrds = line.split("\t")
                        lw = next(fli).strip()
                        if "<g/>" in lw:
                            lw = next(fli).strip()
                        lwp = lw.split("\t")
                        wrds[1] = wrds[1].strip()
                        if lwp[0] != wrds[1]:
                            lwp[0] = lwp[0].replace("#", "_").strip()
                        if lwp[0] != wrds[1]:
                            raise Exception("problem at {}, '{}' != '{}'".format(li, wrds[1], lwp[0]))
                        if len(lwp) > 1:
                            wrds.append(lwp[1])
                        else:
                            wrds.append("_")
                        line = "\t".join(wrds)
                    print(line)
            print()

    print("Read %d lines" % li, file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
