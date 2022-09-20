import argparse
import sys

from src.utils.conllu import ConlluReader


def do_output(return_train, v):
    if return_train:
        return v == 0
    return v > 0


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--test_file", nargs='?', required=True, help="File with test sentences")
    parser.add_argument("--return_train", default=False, action=argparse.BooleanOptionalAction,
                        help="Do return train dataset. If not set then returns test dataset")
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    # read test sentences
    tests = dict()
    ts, ta = 0, 0
    with open(args.test_file, 'r') as file:
        for line in file:
            line = line.strip()
            tests[line] = tests.get(line, 0) + 1
            ta += 1
    for key, value in tests.items():
        if value > 1:
            ts += value
            print("WARN: Test sentence was not found(%d): %s" % (value, key), file=sys.stderr)
    print("Read %d test sentences. Same %d" % (ta, ts), file=sys.stderr)

    i, w, skip = 0, 0, 0
    written = set()
    used = dict()
    with ConlluReader(args.input) as cr:
        for sent in cr:
            i += 1
            s = sent.sentence()
            out = args.return_train
            if s in tests:
                v = tests[s]
                if v > 0:
                    tests[s] = v - 1
                else:
                    used[s] = used.get(s, 0) + 1
                out = do_output(args.return_train, v)
            if out:
                w += 1
                for l in sent.lines:
                    print(l)
                print()
                written.add(s)
            else:
                skip += 1

    # warn on usages                
    for key, value in used.items():
        print("WARN: Sentence already was used(%d): %s" % (value, key), file=sys.stderr)

        # test all used
    nf = 0
    print("WARN: Test sentences were not found:", file=sys.stderr)
    for key, value in tests.items():
        if value > 0:
            print("%s" % key, file=sys.stderr)
            nf += 1
    print("========", file=sys.stderr)

    print("Read %d sentences. Wrote %d sentences, skip: %d, not found: %d" % (i, w, skip, nf), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
