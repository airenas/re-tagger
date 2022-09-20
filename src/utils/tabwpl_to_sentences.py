import argparse
import sys


def main(argv):
    parser = argparse.ArgumentParser(description="Changes tab-wpl file to plain text sentences",
                                     epilog="E.g. cat in.txt | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    i, s, line = 0, 0, ""
    skip = {"<p/>", "<doc/>", "<g/>"}
    for d in sys.stdin.readlines():
        i += 1
        data = d.replace('\ufeff', '').split("\t")
        word = data[0].strip()
        if word == "<s>":
            if len(line) > 0:
                print("Hmm line expected to be empty, was: '%s'" % line, file=sys.stderr)
                line = ""
        elif word == "</s>":
            if len(line) > 0:
                print(line)
                line = ""
                s += 1
        elif word in skip:
            pass
        else:
            if len(line) > 0:
                line = line + " "
            line = line + word            

    print("Read %d lines. Wrote %d sentences" % (i, s), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])