"""Train, compare, tune and evaluate classifiers on the scikit-learn digits dataset.

Run:  python train.py
Outputs (in ./outputs): model.joblib, metrics.json, confusion_matrix.png, model_comparison.png
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
OUTPUT_DIR = Path("outputs")


def load_data(seed=SEED):
    """Return x_train, x_test, y_train, y_test (80/20 stratified split)."""
    digits = load_digits()
    return train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=seed, stratify=digits.target
    )


def candidate_models():
    """Models to compare. Scaling is part of each pipeline to avoid data leakage."""
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    }


def compare_models(x_train, y_train, cv=5):
    """Cross-validate every candidate. Returns {name: (mean_accuracy, std)}."""
    results = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, x_train, y_train, cv=cv)
        results[name] = (float(scores.mean()), float(scores.std()))
    return results


def tune_svm(x_train, y_train, cv=5):
    """Grid search over C and gamma for the SVM pipeline."""
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))
    grid = {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01]}
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    return search


def plot_comparison(results, path):
    names = list(results)
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, means, yerr=stds, capsize=4, color="#4C78A8")
    plt.ylim(0.9, 1.0)
    plt.ylabel("Cross-validated accuracy")
    plt.title("Model comparison (5-fold CV)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x_train, x_test, y_train, y_test = load_data()

    print("Comparing models with 5-fold cross-validation...")
    results = compare_models(x_train, y_train)
    for name, (mean, std) in results.items():
        print(f"  {name:<20} {mean:.3f} (+/- {std:.3f})")
    plot_comparison(results, OUTPUT_DIR / "model_comparison.png")

    print("Tuning SVM with grid search...")
    search = tune_svm(x_train, y_train)
    best = search.best_estimator_
    print(f"  best params: {search.best_params_}  (CV accuracy {search.best_score_:.3f})")

    predictions = best.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(best, OUTPUT_DIR / "model.joblib")
    metrics = {
        "cv_comparison": {n: {"mean": m, "std": s} for n, (m, s) in results.items()},
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_accuracy": float(search.best_score_),
        "test_accuracy": float(test_accuracy),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and reports to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
