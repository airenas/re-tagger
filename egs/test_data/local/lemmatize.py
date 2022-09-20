import argparse
import sys

from src.utils.conllu import ConlluReader
from src.utils.lemma import Lemmatizer
from src.utils.logger import logger


def lemmas(l, w):
    return l.get(w)


def main(argv):
    parser = argparse.ArgumentParser(description="Outputs all possible lemma variants for each word",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--url", nargs='?', default="http://localhost:8001", help="Lemmatizer url")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, wc = 0, 0
    with ConlluReader(args.input) as cr:
        with Lemmatizer(args.url) as lemma:
            for sent in cr:
                sc += 1
                wrds = sent.words()
                for w in wrds:
                    wc += 1
                    print("%s\t%s" % (w, lemmas(lemma, w)))

    print("Read %d sentences, %d words" % (sc, wc), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])