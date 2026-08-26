# feature_tags = ["pos",  "gender", "case", "number", "person", "tense", "mood", "voice", "degree"]
from src.utils.logger import logger

feature_tags = ["pos", "gender", "case", "number"]
feature_tags_loss_weights = {"pos": 5.0, "gender": 1.0, "case": 2.0, "number": 1.0}


def get_gender(pos, param):
    if pos == 'N':
        return param[2]
    if pos == 'A':
        return param[3]
    if pos == 'P':
        return param[2]
    if pos == 'M':
        return param[2]
    if pos == 'V':
        return param[6]
    return '-'


def get_case(pos, param):
    if pos == 'N':
        return param[4]
    if pos == 'A':
        return param[5]
    if pos == 'P':
        return param[4]
    if pos == 'M':
        return param[4]
    if pos == 'V':
        return param[10]
    if pos == 'S':
        return param[2]
    return '-'


def get_number(pos, param):
    if pos == 'N':
        return param[3]
    if pos == 'A':
        return param[4]
    if pos == 'P':
        return param[3]
    if pos == 'M':
        return param[3]
    if pos == 'V':
        return param[5]
    return '-'


def to_tags(param: str):
    pos = param[0]
    res = {"pos": pos}
    res["full"] = param
    res["gender"] = get_gender(pos, param)
    res["case"] = get_case(pos, param)
    res["number"] = get_number(pos, param)

    if res["gender"] == 's':
        logger.warning(f"gender is 's' in param: {param}")
    return res


def to_full(word_pred):
    pos = word_pred.get("pos", "")
    if pos == "N":
        return f"N-{word_pred.get("gender", "-")}{word_pred.get("number", "-")}{word_pred.get("case", "-")}--"
    if pos == "V":
        return f"V----{word_pred.get("number", "-")}{word_pred.get("gender", "-")}---{word_pred.get("case", "-")}---"
    if pos == "A":
        return f"A--{word_pred.get("gender", "-")}{word_pred.get("number", "-")}{word_pred.get("case", "-")}-"
    if pos == "P":
        return f"P-{word_pred.get("gender", "-")}{word_pred.get("number", "-")}{word_pred.get("case", "-")}-"
    if pos == "M":
        return f"M-{word_pred.get("gender", "-")}{word_pred.get("number", "-")}{word_pred.get("case", "-")}--"
    if pos == "R":
        return f"R--"
    if pos == "S":
        return f"S-{word_pred.get("case", "-")}"
    if pos == "C":
        return f"C-"
    if pos == "Q":
        return f"Q-"
    if pos == "I":
        return f"I-"
    if pos == "O":
        return f"O-"
    if pos == "Y":
        return f"Y-"
    if pos == "D":
        return f"D-"
    if pos == "T":
        return f"T-"
    if pos == "X":
        return f"X-"
    raise NotImplementedError("to_full not implemented for pos: {}".format(pos))
