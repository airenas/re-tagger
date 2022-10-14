import math

import numpy as np
import tensorflow as tf
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm

from src.utils.logger import logger


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
    return str(w[-4:]).lower().strip()


def prepare_fasttext_matrix_emb_layer(ft_model_file, words):
    logger.info("loading {}".format(ft_model_file))
    ft_model = load_facebook_vectors(ft_model_file)
    logger.info("loaded {}".format(ft_model_file))
    dim = ft_model[ft_model.index_to_key[0]].shape[0]
    matrix = np.zeros((len(words), dim))
    for i, w in enumerate(words):
        ev = ft_model[w]
        if ev is not None:
            matrix[i] = ev
        if w not in ft_model.key_to_index:
            logger.debug("not found word '{}'".format(w))
    return tf.keras.layers.Embedding(input_dim=matrix.shape[0],
                                     output_dim=matrix.shape[1], weights=[matrix]), matrix.shape[1]
