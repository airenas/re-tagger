import argparse
import sys

from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Collect all possible tags for each word",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial file with lemma info")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading data {}".format(args.input))
    with open(args.input, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split("\t")
            try:
                if len(parts) > 1:
                    if ":" in parts[1]:
                        labels = list(map(lambda l: (l.split(":")[1]), parts[1].split(";")))
                    else:
                        labels = [parts[1]]
                    print("%s\t%s" % (parts[0], ":".join(labels)))
                else:
                    print("%s\t%s" % (parts[0], ""))
            except BaseException as err:
                raise Exception("problem at " + line + "\n" + repr(err))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
