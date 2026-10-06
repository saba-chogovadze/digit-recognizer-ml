# Digit Recognizer ML

An end-to-end machine learning project: it compares several classifiers on the
handwritten digits dataset, tunes the best one, evaluates it, and serves it
through a small REST API.

## What it does
- Loads the scikit-learn digits dataset (1,797 images of 8x8 pixels, classes 0-9)
- Compares **Logistic Regression, KNN, Random Forest and SVM** with 5-fold cross-validation
- Tunes the SVM (`C`, `gamma`) with **grid search**
- Evaluates on a held-out test set (confusion matrix, precision/recall per digit)
- Saves the trained model with `joblib`
- Predicts from a dataset sample or from your own image (`predict.py`)
- Serves predictions through a **Flask REST API** with input validation
- Includes automated tests (pytest)

## Results

| Model               | CV accuracy (5-fold) |
|---------------------|----------------------|
| Logistic Regression | 0.970                |
| KNN                 | 0.975                |
| Random Forest       | 0.975                |
| **SVM (RBF)**       | **0.983**            |

Best SVM after grid search: `C=100`, `gamma=0.01`. **Test accuracy: 98.3%** on 360 unseen images.

![Model comparison](outputs/model_comparison.png)
![Confusion matrix](outputs/confusion_matrix.png)

## Setup
```bash
pip install -r requirements.txt
python train.py          # trains, evaluates, saves model and plots to outputs/
```

## Predict
```bash
python predict.py --index 5                      # a sample from the dataset
python predict.py --image my_digit.png           # white digit on dark background
python predict.py --image my_digit.png --invert  # dark digit on white paper
```

## REST API
```bash
python api.py
```
```bash
curl http://127.0.0.1:5000/health

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [<64 numbers between 0 and 16>]}'
# {"digit": 3, "confidence": 0.9936}
```
Invalid input (wrong length, non-numbers, out-of-range values) returns HTTP 400 with an error message.

## Tests
```bash
pytest
```
Covers data loading, model comparison, image conversion, prediction quality and the API (including bad input).

## Project structure
```
train.py          training, comparison, tuning, evaluation
predict.py        command-line prediction (dataset sample or image file)
api.py            Flask REST API
test_project.py   automated tests
outputs/          saved model, metrics.json and plots
```

## Limitations
- The images are only 8x8 pixels, so photos or drawings of digits are shrunk heavily and
  predictions on them are less reliable than on the dataset itself.
- Next steps: train on the full MNIST dataset (28x28) with a small neural network,
  add a drawing canvas web page, and containerize the API with Docker.

## What I learned
- Comparing models fairly with cross-validation and pipelines (no data leakage)
- Hyperparameter tuning with `GridSearchCV`
- Evaluating classifiers beyond accuracy (confusion matrix, per-class metrics)
- Saving models and serving them through an API with input validation
