import argparse
import sys
import regex
from gensim.models import Word2Vec, FastText
from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, path):
        self.path = path
        self.read = 0

    def __iter__(self):
        corpus_path = datapath(self.path)
        for line in open(corpus_path):
            line = line.strip()
            tokens = regex.sub(r"[^\p{L}0-9]+", " ", line.lower()).split()
            self.read +=1
            if self.read % 1000000 == 0:
                print("Read %d" % self.read, file=sys.stderr)
            yield tokens

def main(argv):
    parser = argparse.ArgumentParser(description="Creates word2vec model",
                                     epilog="E.g. cat in.txt | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Input file")  
    parser.add_argument("--output", nargs='?', required=True, help="File for writing model result")  
    parser.add_argument("--fast", default=False, action=argparse.BooleanOptionalAction,
                        help="Do use fasttext")                               
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    print("Training model", file=sys.stderr)
    if args.fast:
        model = FastText(sentences=MyCorpus(args.input), vector_size=100, window=5, min_count=1, epochs=10, workers=12)
    else:
        model = Word2Vec(sentences=MyCorpus(args.input), vector_size=100, window=5, min_count=1, workers=12)
    model.save(args.output)
    print("Saved model %s" % args.output, file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])