import math
import string


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
        prefix + 'word.ispunctuation': (word in string.punctuation),
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
    if word in string.punctuation:
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


def word2features(sent, i, words_before, words_after, extract_func):
    try:
        w, le = get_word(sent, i), get_lemma(sent, i)
        features = dict()
        features.update(extract_func(w, '', le))
        if i > 0:
            if words_before > 0:
                w, le = get_word(sent, i - 1), get_lemma(sent, i - 1)
                features.update(extract_func(w, '-1:', le))
        else:
            features['BOS'] = True

        for ib in range(2, words_before + 1):
            if (i - ib) >= 0:
                w, le = get_word(sent, i - ib), get_lemma(sent, i - ib)
                features.update(extract_func(w, '-{}:'.format(ib), le))

        if i < len(sent) - 1:
            if words_after > 0:
                w, le = get_word(sent, i + 1), get_lemma(sent, i + 1)
                features.update(extract_func(w, '+1:', le))
        else:
            features['EOS'] = True

        for ia in range(2, words_after + 1):
            if (i + ia) < len(sent):
                w, le = get_word(sent, i + ia), get_lemma(sent, i + ia)
                features.update(extract_func(w, '+{}:'.format(ia), le))
    except BaseException as err:
        raise Exception("Word: {}, lemma: {}, err: {}".format(w, le, err))
    return features


def sent2features(sent, words_before=2, words_after=2, method="get_word_feat"):
    f = globals()[method]
    return [word2features(sent, i, words_before, words_after, f) for i in range(len(sent))]


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
    for i in range(len(csv_data)):
        w_num = csv_data.iloc[i, 0]
        if math.isnan(w_num):
            continue
        elif w_num == 1.0 or w_num < w_prev:
            sents.append([[csv_data.iloc[i, 1], csv_data.iloc[i, pos_id], parse_lemma(csv_data.iloc[i, pos_lemma])]])
        else:
            sents[-1].append([csv_data.iloc[i, 1], csv_data.iloc[i, pos_id], parse_lemma(csv_data.iloc[i, pos_lemma])])
        w_prev = w_num
    for sent in sents:
        for i, word in enumerate(sent):
            if type(word[0]) != str:
                del sent[i]
    return sents
