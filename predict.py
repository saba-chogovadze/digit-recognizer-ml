"""Predict a digit with the trained model.

Examples:
  python predict.py --index 5                 # predict sample #5 of the dataset
  python predict.py --image my_digit.png      # white digit on dark background
  python predict.py --image my_digit.png --invert   # dark digit on white paper
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

MODEL_PATH = Path("outputs/model.joblib")


def image_to_features(path, invert=False):
    """Convert any image to the 8x8, 0-16 grayscale format the model expects."""
    img = Image.open(path).convert("L").resize((8, 8), Image.LANCZOS)
    pixels = np.asarray(img, dtype=float)
    if invert:
        pixels = 255 - pixels
    return (pixels / 255 * 16).reshape(1, -1)


def predict_digit(model, features):
    """Return (digit, confidence) for a (1, 64) feature array."""
    probabilities = model.predict_proba(features)[0]
    digit = int(np.argmax(probabilities))
    return digit, float(probabilities[digit])


def main():
    parser = argparse.ArgumentParser(description="Digit recognizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="index of a sample in the digits dataset")
    group.add_argument("--image", help="path to an image file")
    parser.add_argument("--invert", action="store_true", help="invert image colors")
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    if args.index is not None:
        digits = load_digits()
        features = digits.data[args.index].reshape(1, -1)
        true_label = digits.target[args.index]
    else:
        features = image_to_features(args.image, args.invert)
        true_label = None

    digit, confidence = predict_digit(model, features)
    print(f"Predicted digit: {digit}  (confidence {confidence:.1%})")
    if true_label is not None:
        print(f"True label:      {true_label}")


if __name__ == "__main__":
    main()
