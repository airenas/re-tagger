import math
import string

from tqdm import tqdm

from src.utils.punct import is_punctuation


def feat_word_v0(word, prefix, lemma):
    wl = word.lower()
    res = {
        prefix + 'word': word,
        prefix + 'len(word)': len(word),
        prefix + 'word[:4]': wl[:4],
        prefix + 'word[:3]': wl[:3],
        prefix + 'word[:2]': wl[:2],
        prefix + 'word[-4:]': str(wl[-4:]),
        prefix + 'word[-3:]': str(wl[-3:]),
        prefix + 'word[-2:]': str(wl[-2:]),
        prefix + 'word.lower()': wl,
        prefix + 'word.ispunctuation': is_punctuation(word),
        prefix + 'word.isdigit()': word.isdigit(),
    }
    return res


def f_punct(word, prefix):
    return {
        prefix + 'word': word,
        prefix + 'len(word)': 1,
        prefix + 'word.ispunctuation': True,
    }


def feat_word_v0_punct(word, prefix, lemma):
    if is_punctuation(word):
        res = f_punct(word, prefix)
    else:
        res = feat_word_v0(word, prefix, lemma)
    return res


def feat_word_v1(word, prefix, lemma):
    if word in string.punctuation:
        res = f_punct(word, prefix)
    else:
        wl = word.lower()
        res = {
            prefix + 'word': word,
            prefix + 'len(word)': len(word),
            prefix + 'word[:4]': wl[:4],
            prefix + 'word[-4:]': str(wl[-4:]),
            prefix + 'word[-3:]': str(wl[-3:]),
            prefix + 'word[-2:]': str(wl[-2:]),
            prefix + 'word.lower()': wl,
            prefix + 'word.ispunctuation': False,
            prefix + 'word.isdigit()': word.isdigit(),
        }
    return res


def lemma_mf(lemma):
    if len(lemma) == 0:
        return ""
    res = list(set(lemma[0]))
    res.sort()
    return ",".join(res)


def to_int_feat(ind, other):
    if ind == 0:
        return 0
    if other == 0:
        return 2
    return 1


def l_verb(tag):
    return tag[0] == 'V' and (tag[2] == 'i' or tag[2] == 'm' or tag[2] == 'b')


def l_feat(lemma, f):
    if len(lemma) == 0:
        return -1
    ind, other = 0, 0
    for r in lemma[1]:
        if f(r):
            ind += 1
        else:
            other += 1
    return to_int_feat(ind, other)


def l_is_case(tag):
    return tag[0] == 'N' or tag[0] == 'A' or tag[0] == 'P' or (tag[0] == 'M' and tag[5] == 'l') or (
            tag[0] == 'V' and (tag[2] == 'p' or tag[2] == 'a' or tag[2] == 'h'))


def l_adverb(tag):
    return tag[0] == 'R'


def l_preposition(tag):
    return tag[0] == 'S'


def l_proper(tag):
    return tag[0] == 'N' and tag[1] == 'p'


def l_number_s(tag):
    return (tag[0] == 'N' and tag[3] == 's') or (tag[0] == 'V' and tag[5] == 's') or (
            tag[0] == 'A' and tag[4] == 's') or (tag[0] == 'P' and tag[3] == 's') or (tag[0] == 'M' and tag[3] == 's')


def is_plural(p):
    return p == 'p' or p == 'd'


def l_number_p(tag):
    return (tag[0] == 'N' and is_plural(tag[3])) or (tag[0] == 'V' and is_plural([5])) or (
            tag[0] == 'A' and is_plural(tag[4])) or (tag[0] == 'P' and is_plural(tag[3])) or (
                   tag[0] == 'M' and is_plural(tag[3]))


def l_gender_m(tag):
    return (tag[0] == 'N' and tag[2] == 'm') or (tag[0] == 'V' and tag[6] == 'm') or (
            tag[0] == 'A' and tag[3] == 'm') or (tag[0] == 'P' and tag[2] == 'm') or (tag[0] == 'M' and tag[2] == 'm')


def l_gender_f(tag):
    return (tag[0] == 'N' and tag[2] == 'f') or (tag[0] == 'V' and tag[6] == 'f') or (
            tag[0] == 'A' and tag[3] == 'f') or (tag[0] == 'P' and tag[2] == 'f') or (tag[0] == 'M' and tag[2] == 'f')


def feat_lemma(word, prefix, lemma):
    wl = word.lower()
    if word in string.punctuation:
        res = f_punct(word, prefix)
    else:
        res = {
            prefix + 'word': word,
            prefix + 'len(word)': len(word),
            prefix + 'word[-4:]': str(wl[-4:]),
            prefix + 'word[-3:]': str(wl[-3:]),
            prefix + 'word[-2:]': str(wl[-2:]),
            prefix + 'word.lower()': wl,
            prefix + 'word.ispunctuation': False,
            prefix + 'word.isdigit()': word.isdigit(),
            prefix + 'mf': lemma_mf(lemma),
            prefix + 'verb': l_feat(lemma, l_verb),
            prefix + 'is_case': l_feat(lemma, l_is_case),
            prefix + 'adverb': l_feat(lemma, l_adverb),
            prefix + 'preposition': l_feat(lemma, l_preposition),
            prefix + 'proper': l_feat(lemma, l_proper),
            prefix + 'number_s': l_feat(lemma, l_number_s),
            prefix + 'number_p': l_feat(lemma, l_number_p),
            prefix + 'gender_f': l_feat(lemma, l_gender_f),
            prefix + 'gender_m': l_feat(lemma, l_gender_m),
        }
    return res


def get_word(sent, i):
    return sent[i][0]


def get_lemma(sent, i):
    return sent[i][2]


def get_next_index(sent, ci, skip_punct, move):
    while True:
        ci += move
        if ci < 0 or ci >= len(sent):
            break
        if skip_punct and get_word(sent, ci) in string.punctuation:
            continue
        return ci
    return -1


def word2features(sent, i, params, extract_func):
    words_before = params["words_before"]
    words_after = params["words_after"]
    skip_punct = params["skip_punct"]
    try:
        w, le = get_word(sent, i), get_lemma(sent, i)
        features = dict()
        features.update(extract_func(w, '', le))
        if i == 0:
            features['BOS'] = True
        if i == len(sent) - 1:
            features['EOS'] = True

        ci = i
        for ib in range(1, words_before + 1):
            ci = get_next_index(sent, ci, skip_punct, -1)
            if ci > -1:
                w, le = get_word(sent, ci), get_lemma(sent, ci)
                features.update(extract_func(w, '-{}:'.format(ib), le))
            else:
                break

        ci = i
        for ia in range(1, words_after + 1):
            ci = get_next_index(sent, ci, skip_punct, 1)
            if ci > -1:
                w, le = get_word(sent, ci), get_lemma(sent, ci)
                features.update(extract_func(w, '+{}:'.format(ia), le))
            else:
                break
    except BaseException as err:
        raise Exception("Word: {}, lemma: {}, err: {}".format(w, le, err))
    return features


def sent2features(sent, params):
    f = globals()[params["method"]]
    return [word2features(sent, i, params, f) for i in range(len(sent))]


def sent2labels(sent):
    return [word[1] for word in sent]


def sent2tokens(sent):
    return [word[0] for word in sent]


# formatting the data into sentences
def parse_lemma(lemma):
    if not lemma or lemma == "_":
        return []
    v = lemma.split(";")
    mf = list(map(lambda l: l.split(":")[0], v))
    tags = list(map(lambda l: list(l.split(":")[1]), v))
    return [mf, tags]


def format_data(csv_data):
    sents = []
    pos_id = 9
    pos_lemma = 10
    w_prev = 0
    with tqdm("format data", total=len(csv_data)) as pbar:
        for i in range(len(csv_data)):
            pbar.update(1)
            w_num = csv_data.iloc[i, 0]
            if math.isnan(w_num):
                continue
            elif w_num == 1.0 or w_num < w_prev:
                sents.append(
                    [[csv_data.iloc[i, 1], csv_data.iloc[i, pos_id], parse_lemma(csv_data.iloc[i, pos_lemma])]])
            else:
                sents[-1].append(
                    [csv_data.iloc[i, 1], csv_data.iloc[i, pos_id], parse_lemma(csv_data.iloc[i, pos_lemma])])
            w_prev = w_num
    return sents


def drop_puncts(s):
    res = []
    for wd in s:
        if not is_punctuation(wd[0]):
            res.append(wd)
    return res
