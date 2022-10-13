import math

from tqdm import tqdm


def format_data(csv_data, max_len=60):
    pos_id = 9
    w_prev = 0
    res = []
    words, tags, sl, pw = [], [], 0, ""
    with tqdm(total=len(csv_data), desc="prepare data") as pbar:
        for i in range(len(csv_data)):
            pbar.update(1)
            w_num = csv_data.iloc[i, 0]
            if math.isnan(w_num):
                continue
            if (w_num == 1.0 or w_num < w_prev) and len(words) > 0:
                res.append({"tokens": words, "tags": tags})
                words, tags, sl = [], [], 0
            # split if very long sentence
            if sl > max_len and pw == ',':
                res.append({"tokens": words, "tags": tags})
                words, tags, sl = [], [], 0
            pw = csv_data.iloc[i, 1]
            words.append(pw)
            tags.append(csv_data.iloc[i, pos_id])
            sl += 1
            w_prev = w_num
    if len(words) > 0:
        res.append({"tokens": words, "tags": tags})
    return res


def ending(w):
    return str(w[-4:]).lower()
