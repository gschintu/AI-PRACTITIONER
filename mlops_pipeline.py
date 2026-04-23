import datetime
import json
import os
import pickle
import random

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_and_prepare_data(test_size=0.2, seed=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=seed, stratify=iris.target
    )
    print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, C=1.0, seed=42):
    model = LogisticRegression(solver="liblinear", C=C, random_state=seed, max_iter=200)
    model.fit(X_train, y_train)
    print("Model training complete.")
    print(f"  Coefficients: {model.coef_.shape}, intercept: {model.intercept_.shape}")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Evaluation accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    return acc


def save_artifacts(model, config, experiment_dir="experiments"):
    os.makedirs(experiment_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(experiment_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "iris_model.joblib")
    config_path = os.path.join(run_dir, "run_config.json")
    joblib.dump(model, model_path)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved artifacts to {run_dir}")
    return run_dir


def log_experiment(run_dir, metrics):
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Logged metrics to {metrics_path}")


def main():
    set_seeds(42)

    stage_config = {
        "data_stage": "load_and_prepare",
        "training_stage": "logistic_regression",
        "evaluation_stage": "accuracy_and_report",
        "hyperparameters": {"C": 1.0, "test_size": 0.2},
        "seed": 42,
    }

    X_train, X_test, y_train, y_test = load_and_prepare_data(
        test_size=stage_config["hyperparameters"]["test_size"], seed=stage_config["seed"]
    )

    model = train_model(X_train, y_train, C=stage_config["hyperparameters"]["C"], seed=stage_config["seed"])
    accuracy = evaluate_model(model, X_test, y_test)

    run_dir = save_artifacts(model, stage_config)
    log_experiment(run_dir, {"accuracy": accuracy})

    print("Pipeline complete. You can inspect the run directory for reproducibility and versioning.")


if __name__ == "__main__":
    main()
