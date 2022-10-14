import argparse
import csv
import sys

import pandas as pd
import tensorflow as tf

from egs.bilstm_crf.local.format_data import format_data, ending
from egs.bilstm_crf.local.predict import predict
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

    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]", mask_token="[MASK]")
    lookup_layer.adapt(tf.ragged.constant(words))
    e_lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(ends), oov_token="[UNK]", mask_token="[MASK]")
    e_lookup_layer.adapt(tf.ragged.constant(ends))
    t_lookup_layer = tf.keras.layers.StringLookup(vocabulary=tags, num_oov_indices=0, invert=True)
    logger.info(
        "Words: {}, first 10: {}".format(len(lookup_layer.get_vocabulary()), lookup_layer.get_vocabulary()[:10]))
    logger.info(
        "Ends: {}, first 10: {}".format(len(e_lookup_layer.get_vocabulary()), e_lookup_layer.get_vocabulary()[:10]))
    logger.info(
        "Tags: {}, first 10: {}".format(len(t_lookup_layer.get_vocabulary()), t_lookup_layer.get_vocabulary()[:10]))

    uc, uec = 0, 0

    def in_f(_tokens):
        nonlocal uc, uec
        out = lookup_layer(_tokens)
        for i, pi in enumerate(out):
            if pi == 1:
                uc += 1
                logger.debug("no word {}".format(_tokens[i]))
        out = tf.reshape(out, shape=[1, -1])
        if args.use_ends:
            _ends = e_lookup_layer([ending(w) for w in _tokens])
            for i, pi in enumerate(_ends):
                if pi == 1:
                    uec += 1
                    logger.debug("no end '{}'".format(ending(_tokens[i])))
            _ends = tf.reshape(_ends, shape=[1, -1])
            out = (out, _ends)
        return out

    def out_f(_outputs):
        return t_lookup_layer(_outputs)

    with open(args.out, 'w') as f:
        predict(model, f, data_test, in_func=in_f, out_func=out_f)

    logger.info("Unknown w: {}, e: {}".format(uc, uec))
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
