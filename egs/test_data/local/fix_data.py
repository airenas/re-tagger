import argparse
import sys

import regex
from tqdm import tqdm

from src.utils.conllu import ConlluReader
from src.utils.logger import logger

re_num_dash = regex.compile("[0-9]+\-\p{Letter}+")
re_num_dash_replace = regex.compile("[0-9]+\-")


def is_number_with_dash(w):
    return re_num_dash.fullmatch(w) is not None


def replace_number_with_dash(w):
    return re_num_dash_replace.sub("n-", w)


def fix_dash_in_word(l):
    if l.startswith('#'):
        return l
    tags = l.split("\t")
    if len(tags) < 2:
        return l
    w = tags[1]
    if len(w) > 3 and "-" in w and is_number_with_dash(w):
        tags[1] = replace_number_with_dash(w)
        # logger.info("{} - {}".format(w, tags[1]))
        return "\t".join(tags)
    return l


def main(argv):
    parser = argparse.ArgumentParser(description="Fix conllu file",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, ws = 0, 0
    with ConlluReader(args.input) as cr:
        with tqdm(desc="fixing") as pbar:
            for sent in cr:
                pbar.update(1)
                sc += 1
                for line in sent.lines:
                    if "<g/>" in line:
                        ws += 1
                    else:
                        print(fix_dash_in_word(line))
                print()

    print("Read %d sentences, %d skipped g's" % (sc, ws), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
