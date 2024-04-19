import argparse
import sys
from collections import Counter

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def to_str(cnt: Counter):
    res = ""
    prefix = ""
    for v, key in cnt.items():
        res += f"{prefix}{v}:{key}"
        prefix = ";"
    return res


def fix(mf):
    if mf == "(":
        return mf
    index = mf.find('(')
    if index != -1:
        return mf[:index].strip()
    return mf


def main(argv):
    parser = argparse.ArgumentParser(description="Prepares lemma:tag frequencies for each word",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    res, sc, wc, skc = {}, 0, 0, 0
    with ConlluReader(args.input) as cr:
        for sent in cr:
            sc += 1
            words = list(sent.words())
            try:
                tags = list(sent.tags())
            except BaseException as err:
                raise Exception("problem at {}, '{}', err: {}".format(sc, ' '.join(words), err))
            try:
                main_forms = list(sent.main_forms())
            except BaseException as err:
                raise Exception("problem at {}, '{}', err: {}".format(sc, ' '.join(words), err))
            for i in range(len(words)):
                wc += 1
                w, mf, t = words[i].strip(), fix(main_forms[i].strip()), tags[i].strip()
                if ';' in w or ':' in w or not mf or not t or not w:
                    skc += 1
                    continue
                cnt = res.get(w)
                if not cnt:
                    cnt = Counter()
                    res[w] = cnt
                cnt.update([f"{mf}:{t}"])

    logger.info("Read %d sentences, %d words, skipped: %d" % (sc, wc, skc))
    wc = 0
    for v, key in res.items():
        val_str = to_str(key)
        print(f"{v}\t{val_str}")

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
