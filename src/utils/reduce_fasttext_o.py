import argparse
import sys

import fasttext
import fasttext.util


def main(argv):
    parser = argparse.ArgumentParser(description="Reduce fasttext facebook models vectors",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Model file")
    parser.add_argument("--output", nargs='?', required=True, help="Output model file")
    parser.add_argument("--dim", nargs='?', type=int, required=True, help="Dimension to reduce to")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    model = fasttext.load_model(args.input)
    print("Loaded %s" % args.input, file=sys.stderr)
    print("Dim %d" % model.get_dimension(), file=sys.stderr)
    print("Is quantized %d" % model.is_quantized(), file=sys.stderr)

    fasttext.util.reduce_model(model, args.dim)
    print("Is quantized %d" % model.is_quantized(), file=sys.stderr)
    print("Dim %d" % model.get_dimension(), file=sys.stderr)

    model.save_model(args.output)

    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
