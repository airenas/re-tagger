from gensim.models import Word2Vec, FastText
import argparse
import sys

def main(argv):
    parser = argparse.ArgumentParser(description="Inspect word2vec",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Model file")                                 
    parser.add_argument("--word", nargs='?', required=True, help="Word to inspect")    
    parser.add_argument("--fast", default=False, action=argparse.BooleanOptionalAction,
                        help="Do use fasttext")                                                                                             
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    if args.fast:
        model = FastText.load(args.input)
    else:     
        model = Word2Vec.load(args.input)
    print("Loaded %s" % args.input, file=sys.stderr)
    print("Word %s" % args.word, file=sys.stderr)
    vector = model.wv[args.word]
    print("Vector {}".format(vector), file=sys.stderr)
    sims = model.wv.most_similar(args.word, topn=10) 
    print("Similar {}".format(sims), file=sys.stderr)
    
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])