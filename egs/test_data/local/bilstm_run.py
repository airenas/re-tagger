import argparse
import sys
from typing import List

import requests
from tqdm import tqdm

from src.utils.conllu import ConlluReader
from src.utils.logger import logger


def lemmas(l, w):
    return l.get(w)



def map_res(param):
    res = []
    for w in param:
        tp = w.get('type', '')
        if tp != 'SPACE' and tp != 'SENTENCE_END':
            res.append(w.get('mi', ''))
    # logger.info(res)
    return res


def prepare_input(words):
    res = []
    for w in words:
        res.append(w)
    return [res]


class LSTMServer:
    def __init__(self, url: str):
        logger.info("Init LSTMServer at: %s" % url)
        self.__url = url

    def invoke(self, words: List[str]) -> List[str]:
        inp = prepare_input(words)
        try:
            # logger.info(json.dumps(inp))
            x = requests.post(self.__url, json=inp, timeout=10)
            if x.status_code != 200:
                raise Exception("Can't morphize '{}', {}".format(x.status_code, x.text))
            return map_res(x.json())
        except BaseException as err:
            raise err

def main(argv):
    parser = argparse.ArgumentParser(description="Invokes Bi-LSTMServer service for each sentence in dataset",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--url", nargs='?', default="http://localhost:8000/tag-parsed", help="BI-LSTM server url")
    args = parser.parse_args(args=argv)

    logger.info("Starting")

    # read test sentences
    sc, wc = 0, 0
    morph = LSTMServer(args.url)
    with ConlluReader(args.input) as cr:
        with tqdm(desc="tagging") as pbar:
            for sent in cr:
                pbar.update(1)
                sc += 1
                words = list(sent.words())
                morphs = morph.invoke(words)
                for i in range(len(words)):
                    wc += 1
                    print("%s\t%s" % (words[i], morphs[i]))

    logger.info("Read %d sentences, %d words" % (sc, wc))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
