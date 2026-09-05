import json
from pathlib import Path
from typing import Any

import pandas as pd
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
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
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
        "classification_report": report,
        "metrics_by_class": extract_metrics_by_class(report, labels),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
    }


def extract_metrics_by_class(
    classification_report_data: dict[str, Any],
    labels: list[str],
) -> dict[str, dict[str, float | int]]:
    """Extrai precision, recall, f1-score e support por classe."""
    metrics_by_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        raw_metrics = classification_report_data.get(label, {})
        metrics_by_class[label] = {
            "precision": float(raw_metrics.get("precision", 0.0)),
            "recall": float(raw_metrics.get("recall", 0.0)),
            "f1-score": float(raw_metrics.get("f1-score", 0.0)),
            "support": int(raw_metrics.get("support", 0)),
        }
    return metrics_by_class


def build_baseline_comparison(
    metrics_by_model: dict[str, dict[str, Any]],
    best_model_name: str,
    selection_metric: str,
    baseline_model_name: str = "baseline_most_frequent",
) -> dict[str, Any]:
    """Compara o melhor modelo com o baseline de classe majoritaria."""
    if baseline_model_name not in metrics_by_model:
        raise ValueError(f"Baseline ausente nos resultados: {baseline_model_name}")
    if best_model_name not in metrics_by_model:
        raise ValueError(f"Melhor modelo ausente nos resultados: {best_model_name}")

    baseline_score = float(
        metrics_by_model[baseline_model_name][selection_metric]
    )
    best_score = float(metrics_by_model[best_model_name][selection_metric])
    return {
        "baseline_model_name": baseline_model_name,
        "best_model_name": best_model_name,
        "selection_metric": selection_metric,
        "baseline_score": baseline_score,
        "best_model_score": best_score,
        "absolute_difference": best_score - baseline_score,
        "best_model_outperforms_baseline": best_score > baseline_score,
    }


def build_robust_evaluation_report(
    split_type: str,
    train_period: dict[str, Any],
    test_period: dict[str, Any],
    total_records: int,
    train_records: int,
    test_records: int,
    train_class_distribution: dict[str, int],
    test_class_distribution: dict[str, int],
    features_used: list[str],
    labels: list[str],
    metrics_by_model: dict[str, dict[str, Any]],
    best_model_name: str,
    selection_metric: str,
) -> dict[str, Any]:
    """Monta relatorio completo da avaliacao robusta."""
    metrics_by_class = {
        model_name: metrics.get("metrics_by_class")
        or extract_metrics_by_class(metrics["classification_report"], labels)
        for model_name, metrics in metrics_by_model.items()
    }
    return {
        "split_type": split_type,
        "train_period": train_period,
        "test_period": test_period,
        "total_records": int(total_records),
        "train_records": int(train_records),
        "test_records": int(test_records),
        "train_class_distribution": train_class_distribution,
        "test_class_distribution": test_class_distribution,
        "features_used": features_used,
        "labels": labels,
        "metrics_by_model": metrics_by_model,
        "metrics_by_class": metrics_by_class,
        "priority_class_metrics": {
            label: metrics_by_class[best_model_name].get(label)
            for label in ["alto", "critico"]
            if label in metrics_by_class[best_model_name]
        },
        "best_model_name": best_model_name,
        "selection_metric": selection_metric,
        "baseline_comparison": build_baseline_comparison(
            metrics_by_model,
            best_model_name,
            selection_metric,
        ),
        "confusion_matrix": {
            "model_name": best_model_name,
            "labels": labels,
            "matrix": metrics_by_model[best_model_name]["confusion_matrix"],
        },
        "methodological_note": (
            "Os rotulos risk_level sao derivados do RiskClassifier. "
            "Assim, o modelo aprende a reproduzir uma classificacao tecnica "
            "baseada em regras. As metricas nao representam validacao contra "
            "eventos reais oficiais de desastre."
        ),
    }


def save_json_report(path: str | Path, report: dict[str, Any]) -> None:
    """Salva relatorio JSON com indentacao para leitura humana."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_confusion_matrix_csv(
    path: str | Path,
    matrix: list[list[int]],
    labels: list[str],
) -> None:
    """Salva matriz de confusao usando os labels como linhas e colunas."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(matrix, index=labels, columns=labels)
    dataframe.index.name = "actual"
    dataframe.to_csv(csv_path)


def build_robust_evaluation_output_paths(
    report_dir: str | Path,
    split_type: str,
) -> tuple[Path, Path]:
    """Monta caminhos de relatorio separados por tipo de split."""
    normalized_split = split_type.strip().lower()
    if normalized_split not in {"temporal", "random"}:
        raise ValueError(f"Tipo de split nao suportado: {split_type}")

    report_path = (
        Path(report_dir)
        / f"risk_level_robust_evaluation_{normalized_split}_report.json"
    )
    confusion_matrix_path = (
        Path(report_dir)
        / f"risk_level_robust_confusion_matrix_{normalized_split}.csv"
    )
    return report_path, confusion_matrix_path


def save_robust_evaluation_outputs(
    report_dir: str | Path,
    split_type: str,
    report: dict[str, Any],
    labels: list[str],
) -> tuple[Path, Path]:
    """Salva relatorio robusto e matriz usando nomes por split."""
    report_path, confusion_matrix_path = build_robust_evaluation_output_paths(
        report_dir,
        split_type,
    )
    save_json_report(report_path, report)
    save_confusion_matrix_csv(
        confusion_matrix_path,
        report["confusion_matrix"]["matrix"],
        labels,
    )
    return report_path, confusion_matrix_path


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
