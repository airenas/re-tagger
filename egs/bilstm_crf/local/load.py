import tensorflow as tf

from src.utils.logger import logger


def load_vocab(file_name, name):
    logger.info("loading {} {}".format(name, file_name))
    with open(file_name, 'r') as f:
        words = [w.strip() for w in f]
    logger.info("{} count: {}".format(name, len(words)))
    lookup_layer = tf.keras.layers.StringLookup(max_tokens=len(words), oov_token="[UNK]", mask_token="[MASK]")
    lookup_layer.adapt(tf.ragged.constant(words))
    logger.info(
        "{} layer: {}, first 10: {}".format(name, len(lookup_layer.get_vocabulary()),
                                            lookup_layer.get_vocabulary()[:10]))
    return lookup_layer
