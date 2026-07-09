"""Helper script to convert a Keras model (.keras) to TFLite format (.tflite).

Example:
    python training/convert_to_tflite.py --input models/ai_detector_mobilenetv3.keras --output models/ai_detector_mobilenetv3.tflite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set TF logging level to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Keras Model to TFLite Model")
    parser.add_argument("--input", default="models/ai_detector_mobilenetv3.keras", help="Path to input Keras model")
    parser.add_argument("--output", default="models/ai_detector_mobilenetv3.tflite", help="Path to output TFLite model")
    parser.add_argument("--quantize", action="store_true", help="Apply post-training dynamic range quantization")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input Keras model file does not exist: {input_path}")
        return

    print(f"Loading Keras model from {input_path}...")
    model = tf.keras.models.load_model(str(input_path))

    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantize:
        print("Applying post-training dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing TFLite model to {output_path}...")
    output_path.write_bytes(tflite_model)
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()
