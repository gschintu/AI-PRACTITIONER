import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError:
    shap = None


def build_synthetic_dataset(n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.normal(35, 10, size=n_samples)
    income = rng.normal(50000, 15000, size=n_samples)
    protected = rng.binomial(1, 0.35, size=n_samples)

    linear_score = 0.03 * age + 0.00004 * income - 0.5 * protected
    logits = 1 / (1 + np.exp(-linear_score))
    label = (logits + rng.normal(0, 0.1, size=n_samples)) > 0.6

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "protected": protected,
        "label": label.astype(int),
    })
    return df


def compute_fairness(df, predictions, protected_attr="protected"):
    df = df.copy()
    df["prediction"] = predictions
    groups = df[protected_attr].unique()
    metrics = {}

    for group in groups:
        subset = df[df[protected_attr] == group]
        positives = subset["prediction"].mean()
        true_positive = ((subset["prediction"] == 1) & (subset["label"] == 1)).sum()
        actual_positive = (subset["label"] == 1).sum()
        tpr = true_positive / actual_positive if actual_positive > 0 else 0.0
        metrics[f"group_{group}_positive_rate"] = positives
        metrics[f"group_{group}_tpr"] = tpr

    metrics["statistical_parity_difference"] = metrics["group_1_positive_rate"] - metrics["group_0_positive_rate"]
    metrics["equal_opportunity_difference"] = metrics["group_1_tpr"] - metrics["group_0_tpr"]
    return metrics


def explain_model(model, X_train, X_test):
    if shap is None:
        print("SHAP is not installed. Using model coefficients for interpretability.")
        coefs = pd.Series(model.coef_[0], index=["age", "income", "protected"])
        print("Feature importance (logistic regression coefficients):")
        print(coefs.sort_values(ascending=False).round(4))
        return

    print("Computing SHAP explanations. This may take a moment...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    print("SHAP summary completed. Inspect plotted feature importances.")


def main():
    df = build_synthetic_dataset(n_samples=1200)
    X = df[["age", "income", "protected"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    fairness = compute_fairness(
        pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
        predictions,
    )

    print("Model performance")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Confusion matrix:\n{cm}\n")
    print("Fairness metrics")
    for name, value in fairness.items():
        print(f"  {name}: {value:.4f}")

    print("\nInterpretability and explainability")
    explain_model(model, X_train, X_test)

    print("\nEthics note: protect personal data and avoid using sensitive attributes for direct decision-making. Use protected attributes only for detection and mitigation.")


if __name__ == "__main__":
    main()
