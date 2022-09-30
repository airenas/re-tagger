import math
import string


def get_word_feat(word, prefix):
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


def word2features(sent, i):
    word = sent[i][0]
    features = dict()
    features.update(get_word_feat(word, ''))
    if i > 0:
        features.update(get_word_feat(sent[i - 1][0], '-1:'))
    else:
        features['BOS'] = True

    if i > 1:
        features.update(get_word_feat(sent[i - 2][0], '-2:'))
    # if i > 2:
    #     features.update(get_word_feat(sent[i - 3][0], '-3:'))

    if i < len(sent) - 1:
        features.update(get_word_feat(sent[i + 1][0], '+1:'))
    else:
        features['EOS'] = True

    if i < len(sent) - 2:
        features.update(get_word_feat(sent[i + 2][0], '+2:'))
    # if i < len(sent) - 3:
    #     features.update(get_word_feat(sent[i + 3][0], '+3:'))

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [word[1] for word in sent]


def sent2tokens(sent):
    return [word[0] for word in sent]


# formatting the data into sentences
def format_data(csv_data):
    sents = []
    pos_id = 9
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
