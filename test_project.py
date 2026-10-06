import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import predict
import train
from api import create_app


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a quick model once so tests do not depend on committed files."""
    x_train, _, y_train, _ = train.load_data()
    model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))
    model.fit(x_train, y_train)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path


def test_load_data_shapes():
    x_train, x_test, y_train, y_test = train.load_data()
    assert x_train.shape[1] == 64
    assert len(x_train) + len(x_test) == 1797
    assert len(y_train) == len(x_train)


def test_compare_models_returns_all_candidates():
    x_train, _, y_train, _ = train.load_data()
    results = train.compare_models(x_train, y_train, cv=2)
    assert set(results) == set(train.candidate_models())
    assert all(0.8 < mean <= 1.0 for mean, _ in results.values())


def test_image_to_features_range(tmp_path):
    path = tmp_path / "digit.png"
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(path)
    features = predict.image_to_features(path)
    assert features.shape == (1, 64)
    assert features.max() <= 16 and features.min() >= 0


def test_predict_digit_on_dataset_sample(model_path):
    model = joblib.load(model_path)
    digits = load_digits()
    correct = sum(
        predict.predict_digit(model, digits.data[i].reshape(1, -1))[0] == digits.target[i]
        for i in range(50)
    )
    assert correct >= 45


def test_api_health_and_predict(model_path):
    client = create_app(model_path).test_client()
    assert client.get("/health").get_json() == {"status": "ok"}

    sample = load_digits().data[3].tolist()
    response = client.post("/predict", json={"pixels": sample})
    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["digit"] <= 9
    assert 0 < body["confidence"] <= 1


@pytest.mark.parametrize(
    "payload",
    [{}, {"pixels": [1, 2, 3]}, {"pixels": ["a"] * 64}, {"pixels": [99] * 64}],
)
def test_api_rejects_bad_input(model_path, payload):
    client = create_app(model_path).test_client()
    assert client.post("/predict", json=payload).status_code == 400
