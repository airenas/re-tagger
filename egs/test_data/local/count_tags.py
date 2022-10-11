import argparse
import sys


def do_output(return_train, v):
    if return_train:
        return v == 0
    return v > 0


def main(argv):
    parser = argparse.ArgumentParser(description="Converts tags count",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial tab separated file")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    # read test sentences
    tags = dict()
    rl = 0
    with open(args.input, 'r') as file:
        for line in file:
            wrds = line.strip().split('\t')
            tag = wrds[1]
            tags[tag] = tags.get(tag, 0) + 1
            rl += 1
    ct = 0
    tags_sorted = {k: v for k, v in sorted(tags.items(), key=lambda item: -item[1])}
    for key, value in tags_sorted.items():
        print("{}\t{}".format(key, value))
        ct += 1
    print("Read %d lines. Wrote %d tags" % (rl, ct), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
