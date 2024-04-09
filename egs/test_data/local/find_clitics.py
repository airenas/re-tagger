import argparse
import sys

from tqdm import tqdm

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Find clitics in conllu",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, wc = 0, 0
    res = dict()
    with ConlluReader(args.input) as cr:
        with tqdm(desc="reading") as pbar:
            for sent in cr:
                pbar.update(1)
                sc += 1
                words = sent.words()
                try:
                    tags = list(sent.tags())
                except BaseException as err:
                    raise err

                for i, word in enumerate(words):
                    if " " in word:
                        if not tags[i].startswith('X'):
                            wc += 1
                            res[word.lower()] = res.get(word.lower(), set())
                            res[word.lower()].add(tags[i])
    for k, v in res.items():
        print("{}\t{}".format(k, ";".join([":" + s for s in v])))

    print("Read %d sentences, %d clitics" % (sc, wc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
