import argparse
import csv
import math
import sys

import pandas as pd
import tensorflow as tf
from tensorflow_addons.text import CRFModelWrapper
from tqdm import tqdm

from src.utils.logger import logger


def format_data(csv_data, tag_to_index, max_len):
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
                res.append({"tokens": words, "tags": [tag_to_index[t] for t in tags]})
                words, tags, sl = [], [], 0
            # split if very long sentence
            if sl > max_len and pw == ',':
                res.append({"tokens": words, "tags": [tag_to_index[t] for t in tags]})
                words, tags, sl = [], [], 0
            pw = csv_data.iloc[i, 1]
            words.append(pw)
            tags.append(csv_data.iloc[i, pos_id])
            sl += 1
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
    parser.add_argument("--use_ends", default=False, action=argparse.BooleanOptionalAction,
                        help="Use endings")
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
    logger.info("loading ends {}".format(args.in_v + '.end'))
    with open(args.in_v + '.end', 'r') as f:
        ends = [w.strip() for w in f]
    logger.info("ends count: {}".format(len(ends)))
    logger.info("loading tags {}".format(args.in_t))
    with open(args.in_t, 'r') as f:
        tags = [w.strip() for w in f]
    logger.info("tags count {}".format(len(tags)))
    tag_to_index = {w: i for i, w in enumerate(tags)}
    logger.info("preparing data")
    data_train = format_data(data, tag_to_index, 100)
    # Model architecture
    num_tags = len(tags)
    embedding = 150
    hidden = 300
    batch_size = 32

    w_input = tf.keras.layers.Input(shape=(None,))
    w_output = tf.keras.layers.Embedding(input_dim=len(words) + 2, output_dim=embedding, mask_zero=True)(w_input)
    input, output = w_input, w_output
    use_ends = args.use_ends
    if use_ends:
        e_input = tf.keras.layers.Input(shape=(None,))
        e_output = tf.keras.layers.Embedding(input_dim=len(words) + 2, output_dim=50, mask_zero=True)(e_input)
        input = [w_input, e_input]
        output = tf.keras.layers.concatenate([w_output, e_output])
    output = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=hidden, return_sequences=True, recurrent_dropout=0.1))(output)
    # output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation="relu"))(output)
    m1 = tf.keras.Model(input, output)
    m1.summary()
    model = CRFModelWrapper(m1, num_tags)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.02))
    # model.build(input_shape=[(None, 20), (None, 20)])
    # model.summary()

    train_tokens = tf.ragged.constant(words)
    # train_tokens = tf.map_fn(tf.strings.lower, train_tokens)
    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]", mask_token="[MASK]")
    lookup_layer.adapt(train_tokens)
    e_lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(ends), oov_token="[UNK]", mask_token="[MASK]")
    e_lookup_layer.adapt(tf.ragged.constant(ends))

    print(len(lookup_layer.get_vocabulary()))
    print(lookup_layer.get_vocabulary()[:10])

    print(len(e_lookup_layer.get_vocabulary()))
    print(e_lookup_layer.get_vocabulary()[:10])

    def create_data_generator(dataset):
        def data_generator():
            for item in dataset:
                # print(len(item['tokens']))
                yield item['tokens'], [str(w[-4:]).lower() for w in item['tokens']], item['tags']

        return data_generator

    data_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    train_data = tf.data.Dataset.from_generator(
        create_data_generator(data_train),
        output_signature=data_signature
    )

    def dataset_preprocess(tokens, ends, tag_ids):
        preprocessed_tokens = preprocess_tokens(tokens)
        if use_ends:
            e_preprocessed_tokens = preprocess_e_tokens(ends)
            return (preprocessed_tokens, e_preprocessed_tokens), tag_ids
        return preprocessed_tokens, tag_ids

    def preprocess_tokens(tokens):
        # tokens = tf.strings.lower(tokens)
        return lookup_layer(tokens)

    def preprocess_e_tokens(tokens):
        # tokens = tf.strings.lower(tokens)
        return e_lookup_layer(tokens)

    # With `padded_batch()`, each batch may have different length
    # shape: (batch_size, None)
    train_dataset = (
        train_data.map(dataset_preprocess)
            .padded_batch(batch_size=batch_size)
    )

    # checkpointer = ModelCheckpoint(filepath='model.tf',
    #                                verbose=2,
    #                                mode='auto',
    #                                save_best_only=True,
    #                                monitor='crf_loss')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1,
                                                  restore_best_weights=True, min_delta=0)

    model.fit(train_dataset, epochs=10, verbose=1, callbacks=[early_stop])
    model.summary(150)
    logger.info('Saving tf model ...')
    tf.keras.models.save_model(model, args.out)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
