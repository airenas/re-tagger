import argparse
import csv
import sys

import pandas as pd
import tensorflow as tf

from egs.bilstm_crf.local.format_data import format_data, ending
from egs.bilstm_crf.local.load import load_vocab
from egs.bilstm_crf.local.predict import predict_ds
from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Predicts with bilstm_crf model",
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

    lookup_layer = load_vocab(args.in_v, "words")
    e_lookup_layer = load_vocab(args.in_v + '.end', "endings")

    logger.info("loading tags {}".format(args.in_t))
    with open(args.in_t, 'r') as f:
        tags = [w.strip() for w in f]
    logger.info("tags count {}".format(len(tags)))
    t_lookup_layer = tf.keras.layers.StringLookup(vocabulary=tags, num_oov_indices=0, invert=True)
    logger.info(
        "Tags: {}, first 10: {}".format(len(t_lookup_layer.get_vocabulary()), t_lookup_layer.get_vocabulary()[:10]))

    logger.info("preparing data")
    data_test = format_data(data)

    def create_data_generator(dataset):
        def data_generator():
            for item in dataset:
                # print(len(item['tokens']))
                yield item['tokens'], [ending(w) for w in item['tokens']]

        return data_generator

    data_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.string),
    )

    train_data = tf.data.Dataset.from_generator(
        create_data_generator(data_test),
        output_signature=data_signature
    )

    def dataset_preprocess(_tokens, _ends):
        r_tokens = lookup_layer(_tokens)
        if args.use_ends:
            r_ends = e_lookup_layer(_ends)
            return (r_tokens, r_ends),
        return r_tokens

    tests_dataset = (
        train_data.map(dataset_preprocess)
            .padded_batch(batch_size=32)
    )

    def predictions():
        for _outputs in model.predict(tests_dataset, steps=None, verbose=1):
            yield t_lookup_layer(_outputs)

    with open(args.out, 'w') as f:
        predict_ds(f, data_test, iter(predictions()))

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
