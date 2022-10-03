import math
import string


def get_word_feat_v0(word, prefix):
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


def get_word_feat_v0_punct(word, prefix):
    wl = word.lower()
    if word in string.punctuation:
        res = {
            prefix + 'word': word,
            prefix + 'word.lower()': wl,
            prefix + 'len(word)': len(word),
            prefix + 'word.ispunctuation': True,
        }
    else:
        res = get_word_feat_v0(word, prefix)
    return res


def get_word_feat(word, prefix):
    wl = word.lower()
    if word in string.punctuation:
        res = {
            prefix + 'word': word,
            prefix + 'word.lower()': wl,
            prefix + 'len(word)': len(word),
            prefix + 'word.ispunctuation': True,
        }
    else:
        res = {
            prefix + 'word': word,
            prefix + 'len(word)': len(word),
            prefix + 'word[-3:]': str(wl[-3:]),
            prefix + 'word[-2:]': str(wl[-2:]),
            prefix + 'word.lower()': wl,
            prefix + 'word.ispunctuation': False,
            prefix + 'word.isdigit()': word.isdigit(),
        }
    return res


def word2features(sent, i, words_before, words_after, extract_func):
    word = sent[i][0]
    features = dict()
    features.update(extract_func(word, ''))
    if i > 0:
        if words_before > 0:
            features.update(extract_func(sent[i - 1][0], '-1:'))
    else:
        features['BOS'] = True

    for ib in range(2, words_before + 1):
        if (i - ib) >= 0:
            features.update(extract_func(sent[i - ib][0], '-{}:'.format(ib)))

    if i < len(sent) - 1:
        if words_after > 0:
            features.update(extract_func(sent[i + 1][0], '+1:'))
    else:
        features['EOS'] = True

    for ia in range(2, words_after + 1):
        if (i + ia) < len(sent):
            features.update(extract_func(sent[i + ia][0], '+{}:'.format(ia)))

    return features


def sent2features(sent, words_before=2, words_after=2, method="get_word_feat"):
    f = globals()[method]
    return [word2features(sent, i, words_before, words_after, f) for i in range(len(sent))]


def sent2labels(sent):
    return [word[1] for word in sent]


def sent2tokens(sent):
    return [word[0] for word in sent]


# formatting the data into sentences
def format_data(csv_data):
    sents = []
    pos_id = 9
    w_prev = 0
    for i in range(len(csv_data)):
        w_num = csv_data.iloc[i, 0]
        if math.isnan(w_num):
            continue
        elif w_num == 1.0 or w_num < w_prev:
            sents.append([[csv_data.iloc[i, 1], csv_data.iloc[i, pos_id]]])
        else:
            sents[-1].append([csv_data.iloc[i, 1], csv_data.iloc[i, pos_id]])
        w_prev = w_num
    for sent in sents:
        for i, word in enumerate(sent):
            if type(word[0]) != str:
                del sent[i]
    return sents
