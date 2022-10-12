import argparse
import sys

from gensim.models.fasttext import load_facebook_vectors


def main(argv):
    parser = argparse.ArgumentParser(description="Inspect fasttext facebook model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Model file")
    parser.add_argument("--word", nargs='?', required=True, help="Word to inspect")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    model = load_facebook_vectors(args.input)
    print("Loaded %s" % args.input, file=sys.stderr)
    print("Word %s" % args.word, file=sys.stderr)
    vector = model[args.word]
    print("Vector {}".format(vector), file=sys.stderr)
    sims = model.most_similar(args.word, topn=10)
    print("Similar {}".format(sims), file=sys.stderr)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
