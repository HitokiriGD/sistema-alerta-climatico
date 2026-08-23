from typing import Any

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def evaluate_classifier(
    model: Any,
    X_test: Any,
    y_test: Any,
    labels: list[str],
) -> dict[str, Any]:
    """Avalia um classificador supervisionado com metricas principais."""
    predictions = model.predict(X_test)
    return build_metrics_summary(y_test, predictions, labels)


def build_metrics_summary(
    y_true: Any,
    y_pred: Any,
    labels: list[str],
) -> dict[str, Any]:
    """Monta resumo serializavel das metricas de classificacao."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
    }


def select_best_model(
    results: dict[str, dict[str, Any]],
    metric: str = "f1_macro",
) -> str:
    """Seleciona o modelo com maior valor na metrica informada."""
    if not results:
        raise ValueError("Nenhum resultado de modelo foi informado.")

    missing_metric = [
        model_name for model_name, result in results.items() if metric not in result
    ]
    if missing_metric:
        raise ValueError(f"Metrica ausente nos resultados: {metric}")

    return max(results, key=lambda model_name: float(results[model_name][metric]))


def evaluate_classification(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calcula metricas antigas de classificacao ponderada."""
    summary = build_metrics_summary(
        y_true,
        y_pred,
        labels=sorted(set(y_true) | set(y_pred)),
    )
    return {
        "accuracy": summary["accuracy"],
        "precision": summary["precision_weighted"],
        "recall": summary["recall_weighted"],
        "f1_score": summary["f1_weighted"],
    }
