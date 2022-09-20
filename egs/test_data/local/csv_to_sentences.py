import argparse
import sys
import csv


def main(argv):
    parser = argparse.ArgumentParser(description="Converts csv to sentences",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial csv file") 
    args = parser.parse_args(args=argv)

    print("Starting", file=sys.stderr)

    i, s, line, last_s = 0, 0, "", ""
    with open(args.input, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        print(header, file=sys.stderr)
        for row in csvreader:
            i += 1
            si = row[1]
            w = row[4]
            if si != last_s:
                last_s = si
                if line:
                    print(line)
                    s += 1
                    line = ""
            if line:
                line = line + " "
            line = line + w
        if line:
            print(line)
            s += 1    

    print("Read %d lines. Wrote %d sentences" % (i, s), file=sys.stderr)
    print("Done", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
