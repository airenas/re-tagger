import json
from typing import List

import requests

from src.utils.logger import logger


def map_res(param):
    res = []
    # logger.info(json.dumps(param))
    for msd in param['msd']:
        res.append(msd[0][1])
    # logger.info(res)
    return res


def prepare(words):
    res = ""
    segs = []
    pos = 0
    for w in words:
        if res:
            res += " "
            pos += 1
        res += w
        segs.append([pos, len(w)])
        pos += len(w)
    anot = {"lex": {"p": [[0, len(res)]], "s": [[0, len(res)]], "seg": segs}}
    return res, anot


def prepare_input(words):
    text, anot = prepare(words)
    return {"scope": "all", "body": text, "annotations": anot}


class Morphizer:
    def __init__(self, url: str):
        logger.info("Init morphizer at: %s" % url)
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
