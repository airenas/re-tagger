import argparse
import csv
import math
import sys

import pandas as pd
import tensorflow as tf
from tensorflow_addons.text import CRFModelWrapper

from src.utils.logger import logger


def format_data(csv_data, tag_to_index):
    pos_id = 9
    w_prev = 0
    res = []
    words, tags = [], []
    for i in range(len(csv_data)):
        w_num = csv_data.iloc[i, 0]
        if math.isnan(w_num):
            continue
        if (w_num == 1.0 or w_num < w_prev) and len(words) > 0:
            res.append({"tokens": words, "tags": [tag_to_index[t] for t in tags]})
            words, tags = [], []
        words.append(csv_data.iloc[i, 1])
        tags.append(csv_data.iloc[i, pos_id])
        w_prev = w_num
    if len(words) > 0:
        res.append({"tokens": words, "tags": [tag_to_index[t] for t in tags]})
    return res


def main(argv):
    parser = argparse.ArgumentParser(description="Trains bilstm-crf model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--in_v", nargs='?', required=True, help="Input vocab")
    parser.add_argument("--in_t", nargs='?', required=True, help="Input tags vocab")
    parser.add_argument("--out", nargs='?', required=True, help="Model output file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
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
    tag_to_index = {w: i for i, w in enumerate(tags)}
    logger.info("preparing data")
    data_train = format_data(data, tag_to_index)
    # Model architecture
    num_tags = len(tags)
    embedding = 100
    batch_size = 32

    input = tf.keras.layers.Input(shape=(None,))
    output = tf.keras.layers.Embedding(input_dim=len(words) + 2, output_dim=embedding, mask_zero=True)(input)
    output = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation="relu"))(output)
    m1 = tf.keras.Model(input, output)
    m1.summary()
    model = CRFModelWrapper(m1, num_tags)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.02))

    train_tokens = tf.ragged.constant(words)
    # train_tokens = tf.map_fn(tf.strings.lower, train_tokens)
    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]")
    lookup_layer.adapt(train_tokens)

    print(len(lookup_layer.get_vocabulary()))
    print(lookup_layer.get_vocabulary()[:10])

    def create_data_generator(dataset):
        def data_generator():
            for item in dataset:
                yield item['tokens'], item['tags']

        return data_generator

    data_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    train_data = tf.data.Dataset.from_generator(
        create_data_generator(data_train),
        output_signature=data_signature
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

    # With `padded_batch()`, each batch may have different length
    # shape: (batch_size, None)
    train_dataset = (
        train_data.map(dataset_preprocess)
            .padded_batch(batch_size=batch_size).cache()
    )

    # checkpointer = ModelCheckpoint(filepath='model.tf',
    #                                verbose=2,
    #                                mode='auto',
    #                                save_best_only=True,
    #                                monitor='crf_loss')

    model.fit(train_dataset, epochs=10, verbose=1, callbacks=[])
    model.summary(150)
    logger.info('Saving tf model ...')
    tf.keras.models.save_model(model, args.out + "/1/")
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
