import argparse
import sys

from src.utils.logger import logger


def calc(p, t):
    if p[0] != t[0]:
        return 50
    res = 0.0
    for i, v in enumerate(p):
        if v != t[i]:
            if v != '-':
                res += 1
            else:
                res += .01
    return res


def restore(all, pred):
    if len(all) == 0:
        return pred, False, True, False
    bv = 1000
    pl = list(pred)
    mult = 0
    res = ""
    for t in all:
        v = calc(pl, list(t))
        if v < 1:
            mult += 1
        if v < bv:
            bv = v
            res = t
    return res, bv < 1, bv == 50, mult > 1


def main(argv):
    parser = argparse.ArgumentParser(description="Restores (selects one of possible lemmas) from predicted values",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lemmas", nargs='?', required=True, help="File with all possible lemmas")
    parser.add_argument("--pred", nargs='?', required=True, help="Prediction file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("File lemmas: {}".format(args.lemmas))
    logger.info("File pred  : {}".format(args.pred))
    wc, rc, mpc, nopc, multi_wc = 0, 0, 0, 0, 0
    with open(args.pred, 'r') as fp:
        with open(args.lemmas, 'r') as fl:
            fli = iter(fl)
            for lp in fp:
                wc += 1
                ll = next(fli)
                lp = lp.strip()
                ll = ll.strip()
                wp = lp.split("\t")
                wl = ll.split("\t")
                if len(wl) == 1:
                    all_tags = []
                else:
                    all_tags = wl[1].split(":")
                if wp[0] != wl[0]:
                    raise Exception("problem at {}, {} != {}".format(wc, wp[0], wl[0]))
                try:
                    fp, match, no_pos, several_match = restore(all_tags, wp[1])
                    print("{}\t{}\t{}\t{}".format(wl[0], fp, ":".join(all_tags), wp[1]))
                    if match:
                        rc += 1
                    if no_pos:
                        nopc += 1
                    if several_match:
                        mpc += 1
                    if " " in wp[0]:
                        multi_wc += 1
                except BaseException as err:
                    raise Exception("err at: {}. {}".format(wc, err))
    logger.info("Read: {}, restored: {}, multiple possible: {}, other pos: {}".format(wc, rc, mpc, nopc))
    logger.info("Multiple words: {}".format(multi_wc))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
