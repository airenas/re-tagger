import argparse
import sys

from src.utils.compare import drop_non_important
from src.utils.logger import logger
from src.utils.punct import is_punctuation


def half_change(pos, predited, t, i):
    if pos == 'N':
        if (i == 2 and predited == 'f' and t == 'c') or (i == 3 and predited == 'p' and t == 'd'):
            return True
    if pos == 'A':
        if (i == 3 and predited == 'f' and t == 'n') or (i == 4 and predited == 'p' and t == 'd'):
            return True
    if pos == 'P':
        if i == 3 and predited == 'p' and t == 'd':
            return True

    return False


def calc(p, t):
    if p[0] != t[0]:
        return 50
    res = 0.0
    for i, v in enumerate(p):
        if v != t[i]:
            if half_change(p[0], v, t[i], i):
                res += .03
            elif v != '-':
                res += 1
            else:
                res += .01
    return res


def restore(all, pred, tags):
    if len(all) == 0:
        return pred, False, True, False
    bv = 1000
    pl = list(pred)
    mult = set()
    res = ""
    for t in all:
        v = calc(pl, list(t))
        freq_p = 0.001 / (tags.get(t, 0) + 1.0)
        v += freq_p
        if v < 1:
            mult.add(t)
            if len(mult) > 1:
                pass
        if v < bv:
            bv = v
            res = t
    return res, bv < 1, bv == 50, len(mult) > 1


def main(argv):
    parser = argparse.ArgumentParser(description="Restores (selects one of possible lemmas) from predicted values",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lemmas", nargs='?', required=True, help="File with all possible lemmas")
    parser.add_argument("--pred", nargs='?', required=True, help="Prediction file")
    parser.add_argument("--tags", nargs='?', required=True, help="Tags frequency file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    with open(args.tags, 'r') as f:
        tags = {it[0]: {mi.split(":")[0]: int(mi.split(":")[1]) for mi in it[1].strip().split(" ")} for it in
                [w.strip().split("\t") for w in f]}
    logger.info("File lemmas: {}".format(args.lemmas))
    logger.info("File pred  : {}".format(args.pred))
    wc, rc, mpc, nopc, multi_wc = 0, 0, 0, 0, 0
    with open(args.pred, 'r') as fp:
        with open(args.lemmas, 'r') as fl:
            fli = iter(fl)
            for lp in fp:
                wc += 1
                lp = lp.strip()
                wp = lp.split("\t")

                def next_lw():
                    _ll = next(fli)
                    _ll = _ll.strip()
                    _wl = _ll.split("\t")
                    _wl[0] = _wl[0].replace("#", "_").strip()
                    if len(_wl) == 1:
                        _all_tags = []
                    else:
                        _all_tags = [drop_non_important(mi) for mi in _wl[1].split(":")]
                    return _wl, _all_tags

                wl, all_tags = next_lw()

                wp[0] = wp[0].replace("#", "_").strip()
                while wp[0] != wl[0]:
                    if is_punctuation(wl[0]):
                        print("{}\t{}".format(wl[0], wl[1]))
                        wl, all_tags = next_lw()
                    else:
                        raise Exception("problem at {}, '{}' != '{}'".format(wc, wp[0], wl[0]))
                try:
                    fp, match, no_pos, several_match = restore(all_tags, wp[1], tags.get(wp[0], {}))
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
