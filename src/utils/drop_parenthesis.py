import argparse
import sys


def main(argv):
    parser = argparse.ArgumentParser(description="Changes tab-wpl file to plain text sentences",
                                     epilog="E.g. cat in.txt | " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    i, d = 0, 0
    for line in sys.stdin.readlines():
        i += 1
        line = line.strip()
        if "<" in line:
            d +=1
        else:
            print(line)

    print("Read %d lines. Dropped %d sentences" % (i, d), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])