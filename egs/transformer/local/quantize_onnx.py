import argparse
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.utils.logger import logger


def main(argv):
    parser = argparse.ArgumentParser(description="Applies dynamic INT8 weight quantization to an ONNX tagger")
    parser.add_argument("--input", required=True, help="Input floating-point ONNX model")
    parser.add_argument("--out", required=True, help="Output dynamically quantized INT8 ONNX model")
    args = parser.parse_args(argv)

    logger.info("Quantizing {} to {}".format(args.input, args.out))
    quantize_dynamic(args.input, args.out, weight_type=QuantType.QInt8)
    logger.info("Done")


if __name__ == "__main__":
    main(sys.argv[1:])