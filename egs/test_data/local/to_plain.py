import argparse
import logging
import sys

from src.utils.compare import drop_non_important
from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Prepares plain result file with word\ttag",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logging.info("Starting")

    # read test sentences
    sc, wc = 0, 0
    with ConlluReader(args.input) as cr:
        for sent in cr:
            sc += 1
            words = list(sent.words())
            try:
                expected = list(sent.tags())
            except BaseException as err:
                raise Exception("problem at {}, '{}', err: {}".format(sc, ' '.join(words), err))
            for i in range(len(words)):
                wc += 1
                try:
                    print("%s\t%s" % (words[i], drop_non_important(words[i], expected[i])))
                except BaseException as ex:
                    logging.error(f"In: '{words[i], expected[i]}', {ex}")
                    raise ex

    logging.info("Read %d sentences, %d words" % (sc, wc))
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main(sys.argv[1:])
