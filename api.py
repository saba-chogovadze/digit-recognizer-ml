"""Small REST API that serves the trained digit recognizer.

Run:  python api.py
Then: curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
        -d '{"pixels": [0, 0, 5, 13, ... 64 numbers between 0 and 16 ...]}'
"""
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = Path("outputs/model.joblib")
N_PIXELS = 64


def create_app(model_path=MODEL_PATH):
    app = Flask(__name__)
    model = joblib.load(model_path)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if not data or "pixels" not in data:
            return jsonify(error="JSON body with a 'pixels' list is required"), 400
        pixels = data["pixels"]
        if not isinstance(pixels, list) or len(pixels) != N_PIXELS:
            return jsonify(error=f"'pixels' must be a list of {N_PIXELS} numbers"), 400
        try:
            features = np.array(pixels, dtype=float).reshape(1, -1)
        except (TypeError, ValueError):
            return jsonify(error="'pixels' must contain only numbers"), 400
        if features.min() < 0 or features.max() > 16:
            return jsonify(error="pixel values must be between 0 and 16"), 400

        probabilities = model.predict_proba(features)[0]
        digit = int(np.argmax(probabilities))
        return jsonify(digit=digit, confidence=round(float(probabilities[digit]), 4))

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
