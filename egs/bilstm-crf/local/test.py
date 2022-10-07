import argparse
import csv
import math
import sys

import pandas as pd
import tensorflow as tf

from src.utils.logger import logger


def format_data(csv_data):
    pos_id = 9
    w_prev = 0
    res = []
    words, tags = [], []
    for i in range(len(csv_data)):
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
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading model {}".format(args.in_v))
    model = tf.keras.models.load_model(args.model)
    model.summary()
    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)
    logger.info("sample data")
    print(data.head(10), sep='\n\n')
    logger.info("loading vocab {}".format(args.in_v))
    with open(args.in_v, 'r') as f:
        words = [w.strip() for w in f]
    logger.info("words count: {}".format(len(words)))
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
    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]")
    lookup_layer.adapt(train_tokens)

    logger.info("lookup vocab size {}".format(len(lookup_layer.get_vocabulary())))
    logger.info("first ten words: {}".format(lookup_layer.get_vocabulary()[:10]))

    data_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    def dataset_preprocess(tokens, tag_ids):
        preprocessed_tokens = preprecess_tokens(tokens)
        # increase by 1 for all tag_ids,
        # because `<PAD>` is added as the first element in tags list
        # preprocessed_tag_ids = tag_ids + 1
        preprocessed_tag_ids = tag_ids

        return preprocessed_tokens, preprocessed_tag_ids

    def preprecess_tokens(tokens):
        # tokens = tf.strings.lower(tokens)
        return lookup_layer(tokens)

    with open(args.out, 'w') as f:
        for item in data_test:
            tokens = item['tokens']
            print("raw inputs: ", tokens)
            # preprocess
            preprocessed_inputs = preprecess_tokens(tokens)
            inputs = tf.reshape(preprocessed_inputs, shape=[1, -1])
            outputs = model(inputs)
            print("raw outputs: ", outputs)
            prediction = [tags[i] for i in outputs[0]]
            print("prediction: ", prediction)
            for i, w in enumerate(tokens):
                print("{}\t{}".format(w, prediction[i], file=f))

    #     prediction = [tags[i] for i in outputs[0]]
    #
    # sample_tags = [raw_tags[i] for i in sample_tag_ids]

    # # Keypoint: EU -> B-ORG, German -> B-MISC, British -> B-MISC
    # print("ground true tags: ", sample_tags)
    # print("predicted tags: ", prediction)

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
