import os.path

import requests

from src.utils.logger import logger

_path_t = "~/.cache/lemma/cache"
_path = os.path.expanduser(_path_t)


# {'ending': '', 'mi': [{'mf': 'Lietuva', 'mi': 'I000000003120I0', 'mi_vdu': 'Npfsgng', 'mis': ''}]

def map_res(param):
    res = ""
    for mi in param['mi']:
        if res:
            res += ";"
        res += mi["mf"] + ":" + mi["mi_vdu"]
    return res


def add_initial():
    res = {
        ".": "Tp",
        ",": "Tc",
        ";": "Ts",
        ":": "Tn",
        "?": "Tq", "?..": "Tq",
        "!": "Te",
        "...": "Ti", "…": "Ti",
        "-": "Th", "–": "Th", "—": "Th",
        "(": "Tl", "[": "Tl", "{": "Tl",
        ")": "Tr", "]": "Tr", "}": "Tr",
        "/": "Tt",
        "'": "Tu", "\"": "Tu", "„": "Tu", "“": "Tu", "‘": "Tu"
    }
    for o in ['|', '\\', '*', '%', '^', '$', '•', '+']:
        res[o] = 'Tx'

    return res


def fix(txt):
    return txt.replace("à", "a")


def fix_empty_tag(txt, res):
    if len(txt) == 1 and txt.isalpha():
        return ":Xr"
    return ":X-"


class Lemmatizer:
    def __init__(self, url: str, clitics_file: str):
        logger.info("Init lemma at: %s" % (url))
        self.__url = url
        self.cache = dict()
        for v, k in add_initial().items():
            self.cache[v] = ":" + k
        if os.path.exists(_path):
            with open(_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    wrds = line.split("\t")
                    self.cache[wrds[0]] = wrds[1]
        with open(clitics_file, 'r') as file:
            for line in file:
                line = line.strip()
                wrds = line.split("\t")
                self.cache[wrds[0]] = wrds[1]

    def get(self, txt: str) -> str:
        if txt.isdigit():
            return ":M----d-"
        if txt in self.cache:
            return self.cache[txt]
        if " " in txt:
            if txt.lower() in self.cache:
                return self.cache[txt.lower()]
            return ":X-"
        try:
            res = self.call_service(txt)
            if not res:
                res = fix_empty_tag(txt, res)
            self.cache[txt] = res
            if res:
                self.fd.write("{}\t{}\n".format(txt, res))
        except BaseException as err:
            logger.error(err)
            res = ":X-"
        # if not res:
        #     logger.warn("no lemma '%s'" % txt)
        return res

    def call_service(self, txt: str) -> str:
        txt = fix(txt)
        url = "%s/analyze/%s" % (self.__url, txt)
        # logger.info("Call '%s'" % url)
        x = requests.get(url, timeout=10)
        if x.status_code != 200:
            raise Exception("Can't lemmatize '{}'".format(txt))
        return map_res(x.json())

    def __enter__(self):
        if not os.path.exists(_path):
            logger.info("Creating '%s'" % os.path.dirname(_path))
            os.makedirs(os.path.dirname(_path), exist_ok=True)
        self.fd = open(_path, 'a+')
        logger.info("Opened %s" % _path)
        return self

    def __exit__(self, type, value, traceback):

        self.fd.close()
