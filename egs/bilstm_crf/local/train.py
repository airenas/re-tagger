import argparse
import csv
import sys

import pandas as pd
import tensorflow as tf
from tensorflow_addons.text import CRFModelWrapper

from egs.bilstm_crf.local.format_data import format_data, ending
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Trains bilstm_crf model",
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
    data_train = format_data(data)
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

    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]", mask_token="[MASK]")
    lookup_layer.adapt(tf.ragged.constant(words))
    e_lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(ends), oov_token="[UNK]", mask_token="[MASK]")
    e_lookup_layer.adapt(tf.ragged.constant(ends))
    t_lookup_layer = tf.keras.layers.StringLookup(vocabulary=tags, num_oov_indices=0)

    logger.info(
        "Words: {}, first 10: {}".format(len(lookup_layer.get_vocabulary()), lookup_layer.get_vocabulary()[:10]))
    logger.info(
        "Ends: {}, first 10: {}".format(len(e_lookup_layer.get_vocabulary()), e_lookup_layer.get_vocabulary()[:10]))
    logger.info(
        "Tags: {}, first 10: {}".format(len(t_lookup_layer.get_vocabulary()), t_lookup_layer.get_vocabulary()[:10]))

    def create_data_generator(dataset):
        def data_generator():
            for item in dataset:
                # print(len(item['tokens']))
                yield item['tokens'], [ending(w) for w in item['tokens']], item['tags']

        return data_generator

    data_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string)
    )

    train_data = tf.data.Dataset.from_generator(
        create_data_generator(data_train),
        output_signature=data_signature
    )

    def dataset_preprocess(tokens, _ends, _tags):
        r_tokens = lookup_layer(tokens)
        r_tags = t_lookup_layer(_tags)
        if use_ends:
            r_ends = e_lookup_layer(_ends)
            return (r_tokens, r_ends), r_tags
        return r_tokens, r_tags

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
