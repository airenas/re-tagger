import argparse
import sys

import onnx
import tensorflow as tf
import tf2onnx

from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Export to ONNX",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", nargs='?', required=True, help="Input model")
    parser.add_argument("--out", nargs='?', required=True, help="Output onnx file")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("loading model {}".format(args.model))
    model = tf.keras.models.load_model(args.model)
    model.summary()
    model.base_model.summary()

    logger.info("converting")
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=18)
    logger.info(f"saving {args.out}")
    onnx.save(onnx_model, args.out)

    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])
