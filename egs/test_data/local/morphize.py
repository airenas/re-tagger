import argparse
import sys

from src.utils.conllu import ConlluReader
from src.utils.logger import logger
from src.utils.morph import Morphizer


def lemmas(l, w):
    return l.get(w)


def main(argv):
    parser = argparse.ArgumentParser(description="Invokes morph for each sentence in dataset",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--url", nargs='?', default="http://localhost:8002/morphology", help="Morphizer url")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, wc = 0, 0
    morph = Morphizer(args.url)
    with ConlluReader(args.input) as cr:
        for sent in cr:
            sc += 1
            words = list(sent.words())
            expected = list(sent.tags())
            morphs = morph.invoke(words)
            # logger.info(words)
            for i in range(len(words)):
                wc += 1
                print("%s\t%s\t%s" % (words[i], expected[i], morphs[i]))

    logger.info("Read %d sentences, %d words" % (sc, wc))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
