import argparse
import csv
import math
import sys

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.utils.logger import logger


def format_data(csv_data):
    pos_id = 9
    w_prev = 0
    res = []
    words, tags = [], []
    with tqdm(total=len(csv_data), desc="prepare data") as pbar:
        for i in range(len(csv_data)):
            pbar.update(1)
            w_num = csv_data.iloc[i, 0]
            if math.isnan(w_num):
                continue
            if (w_num == 1.0 or w_num < w_prev) and len(words) > 0:
                res.append({"tokens": words, "tags": tags})
                words, tags = [], []
            words.append(csv_data.iloc[i, 1])
            tags.append(csv_data.iloc[i, pos_id])
            w_prev = w_num
    if len(words) > 0:
        res.append({"tokens": words, "tags": tags})
    return res


def main(argv):
    parser = argparse.ArgumentParser(description="Predicts with bilstm-crf model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Test conllu file")
    parser.add_argument("--model", nargs='?', required=True, help="Input model")
    parser.add_argument("--in_v", nargs='?', required=True, help="Input vocab")
    parser.add_argument("--in_t", nargs='?', required=True, help="Input tags vocab")
    parser.add_argument("--out", nargs='?', required=True, help="Prediction output file")
    parser.add_argument("--use_ends", default=False, action=argparse.BooleanOptionalAction,
                        help="Use endings")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading model {}".format(args.in_v))
    model = tf.keras.models.load_model(args.model)
    model.summary()
    model.base_model.summary()

    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)
    logger.info("sample data")
    print(data.head(10), sep='\n\n')
    logger.info("loading vocab {}".format(args.in_v))
    with open(args.in_v, 'r') as f:
        words = [w.strip() for w in f]
    logger.info("words count: {}".format(len(words)))
    logger.info("loading ends {}".format(args.in_v + '.end'))
    with open(args.in_v + '.end', 'r') as f:
        ends = [w.strip() for w in f]
    logger.info("ends count: {}".format(len(ends)))
    logger.info("loading tags {}".format(args.in_t))
    with open(args.in_t, 'r') as f:
        tags = [w.strip() for w in f]
    logger.info("tags count {}".format(len(tags)))
    logger.info("preparing data")
    data_test = format_data(data)
    # Model architecture
    num_tags = len(tags)

    train_tokens = tf.ragged.constant(words)
    # train_tokens = tf.map_fn(tf.strings.lower, train_tokens)
    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]", mask_token="[MASK]")
    lookup_layer.adapt(train_tokens)

    e_lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(ends), oov_token="[UNK]", mask_token="[MASK]")
    e_lookup_layer.adapt(tf.ragged.constant(ends))

    logger.info("lookup vocab size {}".format(len(lookup_layer.get_vocabulary())))
    logger.info("first ten words: {}".format(lookup_layer.get_vocabulary()[:10]))
    logger.info("lookup ends size {}".format(len(e_lookup_layer.get_vocabulary())))
    logger.info("first ten ends: {}".format(e_lookup_layer.get_vocabulary()[:10]))

    def preprocess_tokens(tokens):
        # tokens = tf.strings.lower(tokens)
        return lookup_layer(tokens)

    def preprocess_e_tokens(tokens):
        # tokens = tf.strings.lower(tokens)
        return e_lookup_layer(tokens)

    with open(args.out, 'w') as f:
        with tqdm(total=len(data_test), desc="predicting") as pbar:
            for item in data_test:
                pbar.update(1)
                tokens = item['tokens']
                preprocessed_inputs = preprocess_tokens(tokens)
                inputs = tf.reshape(preprocessed_inputs, shape=[1, -1])
                if args.use_ends:
                    preprocessed_ends = preprocess_e_tokens([str(w[-4:]).lower() for w in tokens])
                    ends = tf.reshape(preprocessed_ends, shape=[1, -1])
                    inputs = (inputs, ends)
                outputs = model(inputs)
                prediction = [tags[i] for i in outputs[0]]
                # print("raw tokens: ", tokens)
                # print("raw inputs: ", inputs)
                # print("raw outputs: ", outputs)
                # print("prediction: ", prediction)
                for i, w in enumerate(tokens):
                    print("{}\t{}".format(w, prediction[i]), file=f)

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
