import argparse
import sys

from tqdm import tqdm

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def is_noun(param):
    p = list(param)
    return (p[0] == 'N' and p[4] == 'n') or (p[0] == 'P' and p[4] == 'n')


def main(argv):
    parser = argparse.ArgumentParser(description="Find nominatives in sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--split-words", nargs='?', required=False, default="", help="Split words separated by '_'")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, wc, cp, phc = 0, 0, 0, 0
    res = dict()
    split_words = args.split_words.split("_")
    logger.info("Split words {}".format(split_words))
    with ConlluReader(args.input) as cr:
        with tqdm(desc="reading") as pbar:
            for sent in cr:
                pbar.update(1)
                sc += 1
                words = list(sent.words())
                tags = list(sent.tags())
                c, w = 0, 0

                def add():
                    nonlocal cp, w, c, phc
                    if w > 0:
                        res[c] = res.get(c, 0) + 1
                        if c == 0 and cp < 100:
                            cp += 1
                            print(" ".join(words))
                        c, w = 0, 0
                        phc += 1

                for i, word in enumerate(words):
                    if is_noun(tags[i]):
                        c += 1
                    if word in split_words:
                        add()
                    else:
                        w += 1
                add()

    for k, v in sorted(res.items()):
        print("{}\t{}".format(k, v))

    print("Read %d sentences, %d phrases" % (sc, phc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
