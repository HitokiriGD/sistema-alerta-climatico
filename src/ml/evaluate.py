from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_classification(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calcula metricas basicas de classificacao."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
