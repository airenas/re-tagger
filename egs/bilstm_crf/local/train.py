import argparse
import csv
import sys

import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow_addons.text import CRFModelWrapper

from egs.bilstm_crf.local.format_data import format_data
from egs.bilstm_crf.local.load import load_vocab
from egs.bilstm_crf.local.prepare_data import make_train_dataset
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Trains bilstm_crf model",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", nargs='?', required=True, help="Initial conllu file")
    parser.add_argument("--in_v", nargs='?', required=True, help="Input vocab")
    parser.add_argument("--in_t", nargs='?', required=True, help="Input tags vocab")
    parser.add_argument("--out", nargs='?', required=True, help="Model output file")
    parser.add_argument("--hidden", nargs='?', default=200, help="Hidden layer size")
    parser.add_argument("--batch", nargs='?', default=32, help="Batch size")
    parser.add_argument("--use_ends", default=False, action=argparse.BooleanOptionalAction,
                        help="Use endings")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading data {}".format(args.input))
    data = pd.read_csv(args.input, sep='\t', comment='#', header=None, quotechar=None, quoting=csv.QUOTE_NONE)
    logger.info("sample data")
    print(data.head(10), sep='\n\n')
    lookup_layer = load_vocab(args.in_v, "words")
    e_lookup_layer = load_vocab(args.in_v + ".end", "endings")

    logger.info("loading tags {}".format(args.in_t))
    with open(args.in_t, 'r') as f:
        tags = [w.strip() for w in f]
    logger.info("tags count {}".format(len(tags)))
    t_lookup_layer = tf.keras.layers.StringLookup(vocabulary=tags, num_oov_indices=0)

    logger.info("preparing data")
    data_sent = format_data(data)
    data_train, data_val = train_test_split(data_sent, test_size=0.1, shuffle=True, random_state=1)
    logger.info("Data len train: {}".format(len(data_train)))
    logger.info("Data len val  : {}".format(len(data_val)))

    # Model architecture
    num_tags = len(tags)
    embedding = 150
    hidden = int(args.hidden)
    batch_size = int(args.batch)
    logger.info("Hidden      : {}".format(hidden))
    logger.info("Batch size: : {}".format(batch_size))

    w_input = tf.keras.layers.Input(shape=(None,))
    w_output = tf.keras.layers.Embedding(input_dim=len(lookup_layer.get_vocabulary()), output_dim=embedding,
                                         mask_zero=True)(w_input)
    input, output = w_input, w_output
    use_ends = args.use_ends
    if use_ends:
        e_input = tf.keras.layers.Input(shape=(None,))
        e_output = tf.keras.layers.Embedding(input_dim=len(e_lookup_layer.get_vocabulary()), output_dim=50,
                                             mask_zero=True)(e_input)
        input = [w_input, e_input]
        output = tf.keras.layers.concatenate([w_output, e_output])
    output = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=hidden, return_sequences=True, recurrent_dropout=0.1))(output)
    # output = tf.keras.layers.LSTM(units=hidden, return_sequences=True, recurrent_dropout=0.1)(output)
    # output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation="relu"))(output)
    m1 = tf.keras.Model(input, output)
    m1.summary()
    model = CRFModelWrapper(m1, num_tags)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    logger.info(
        "Tags: {}, first 10: {}".format(len(t_lookup_layer.get_vocabulary()), t_lookup_layer.get_vocabulary()[:10]))

    def dataset_preprocess(_tokens, _ends, _tags):
        r_tokens = lookup_layer(_tokens)
        r_tags = t_lookup_layer(_tags)
        if use_ends:
            r_ends = e_lookup_layer(_ends)
            return (r_tokens, r_ends), r_tags
        return r_tokens, r_tags

    train_ds = (make_train_dataset(data_train).map(dataset_preprocess).padded_batch(batch_size=batch_size))
    val_ds = (make_train_dataset(data_val).map(dataset_preprocess).padded_batch(batch_size=batch_size))

    checkpoint = ModelCheckpoint(filepath=args.out + ".tmp",
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=False,
                                 mode='min',
                                 period=5)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=1, callbacks=[checkpoint, es])
    model.summary(150)
    logger.info('Saving tf model ...')
    tf.keras.models.save_model(model, args.out)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
