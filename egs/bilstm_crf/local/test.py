import argparse
import csv
import sys

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from egs.bilstm_crf.local.format_data import format_data, ending
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

    with open(args.out, 'w') as f:
        with tqdm(total=len(data_test), desc="predicting") as pbar:
            for item in data_test:
                pbar.update(1)
                tokens = item['tokens']
                preprocessed_inputs = lookup_layer(tokens)
                inputs = tf.reshape(preprocessed_inputs, shape=[1, -1])
                if args.use_ends:
                    preprocessed_ends = e_lookup_layer([ending(w) for w in tokens])
                    ends = tf.reshape(preprocessed_ends, shape=[1, -1])
                    inputs = (inputs, ends)
                outputs = model(inputs)
                prediction = t_lookup_layer(outputs[0])
                # print("raw tokens: ", tokens)
                # print("raw inputs: ", inputs)
                # print("raw outputs: ", outputs)
                # print("prediction: ", prediction)
                for i, w in enumerate(tokens):
                    print("{}\t{}".format(w, prediction[i]), file=f)

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
