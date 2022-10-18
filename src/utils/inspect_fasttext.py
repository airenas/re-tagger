import argparse
import sys

from gensim.models.fasttext import load_facebook_vectors


def main(argv):
    parser = argparse.ArgumentParser(description="Inspect fasttext facebook model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Model file")
    parser.add_argument("--word", nargs='?', required=False, help="Word to inspect")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    model = load_facebook_vectors(args.input)
    print("Loaded %s" % args.input, file=sys.stderr)
    words = sys.stdin
    if args.word:
        print("Word %s" % args.word, file=sys.stderr)
        words = [args.word]
    else:
        print("Use stdin - enter words", file=sys.stderr)
    for word in words:
        word = word.strip()
        print("Word %s" % word, file=sys.stderr)
        print("Index %s" % model.key_to_index.get(word, -1), file=sys.stderr)
        vector = model[word]
        print("Vector {}".format(vector), file=sys.stderr)
        sims = model.most_similar(word, topn=10)
        print("Similars 10 {}".format(sims), file=sys.stderr)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
