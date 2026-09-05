import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.evaluate import build_robust_evaluation_report
from src.ml.evaluate import evaluate_classifier
from src.ml.evaluate import save_robust_evaluation_outputs
from src.ml.evaluate import select_best_model
from src.ml.train import RISK_LEVEL_LABELS
from src.ml.train import load_training_dataset
from src.ml.train import prepare_features_target_with_metadata
from src.ml.train import split_dataset
from src.ml.train import split_dataset_temporal
from src.ml.train import train_candidate_models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Le parametros para avaliacao robusta dos modelos de risk_level."""
    parser = argparse.ArgumentParser(
        description="Avalia modelos de ML para prever risk_level.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/ml_training_dataset.parquet"),
        help="Dataset rotulado gerado a partir do INMET DuckDB.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "temporal"],
        default="temporal",
        help="Tipo de divisao treino/teste.",
    )
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-start-year", type=int, default=2025)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metric", default="f1_macro")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/reports"),
        help="Diretorio para salvar relatorios de avaliacao.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dataset.exists():
        print(
            "Dataset rotulado nao encontrado. Gere primeiro com "
            "python scripts/build_ml_dataset.py."
        )
        return 1

    try:
        result = evaluate_ml_models(
            dataset_path=args.dataset,
            split_type=args.split,
            train_end_year=args.train_end_year,
            test_start_year=args.test_start_year,
            test_size=args.test_size,
            random_state=args.random_state,
            metric=args.metric,
            report_dir=args.report_dir,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print_evaluation_summary(result)
    return 0


def evaluate_ml_models(
    dataset_path: str | Path,
    split_type: str = "temporal",
    train_end_year: int = 2024,
    test_start_year: int = 2025,
    test_size: float = 0.2,
    random_state: int = 42,
    metric: str = "f1_macro",
    report_dir: str | Path = "data/reports",
) -> dict[str, Any]:
    """Executa treino, avaliacao robusta e escrita dos relatorios."""
    dataset = load_training_dataset(dataset_path)
    split_data = _build_split(
        dataset=dataset,
        split_type=split_type,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
        test_size=test_size,
        random_state=random_state,
    )

    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    _validate_training_classes(y_train)

    labels = _ordered_labels(pd.concat([y_train, y_test], ignore_index=True))
    trained_models = train_candidate_models(X_train, y_train)
    metrics_by_model = {
        model_name: evaluate_classifier(model, X_test, y_test, labels)
        for model_name, model in trained_models.items()
    }
    best_model_name = select_best_model(metrics_by_model, metric)

    report = build_robust_evaluation_report(
        split_type=split_data["split_info"]["split_type"],
        train_period=split_data["split_info"]["train_period"],
        test_period=split_data["split_info"]["test_period"],
        total_records=split_data["split_info"]["total_records"],
        train_records=len(X_train),
        test_records=len(X_test),
        train_class_distribution=_class_distribution(y_train),
        test_class_distribution=_class_distribution(y_test),
        features_used=split_data["split_info"]["features_used"],
        labels=labels,
        metrics_by_model=metrics_by_model,
        best_model_name=best_model_name,
        selection_metric=metric,
    )

    report_path, confusion_matrix_path = save_robust_evaluation_outputs(
        report_dir=report_dir,
        split_type=report["split_type"],
        report=report,
        labels=labels,
    )

    return {
        "report_path": str(report_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "split_type": report["split_type"],
        "train_records": report["train_records"],
        "test_records": report["test_records"],
        "train_class_distribution": report["train_class_distribution"],
        "test_class_distribution": report["test_class_distribution"],
        "metrics_by_model": report["metrics_by_model"],
        "best_model_name": report["best_model_name"],
        "selection_metric": report["selection_metric"],
        "baseline_comparison": report["baseline_comparison"],
    }


def print_evaluation_summary(result: dict[str, Any]) -> None:
    """Imprime resumo da avaliacao robusta no terminal."""
    print(f"Tipo de split: {result['split_type']}")
    print(f"Registros de treino: {result['train_records']}")
    print(f"Registros de teste: {result['test_records']}")
    print("Distribuicao das classes no treino:")
    _print_distribution(result["train_class_distribution"])
    print("Distribuicao das classes no teste:")
    _print_distribution(result["test_class_distribution"])
    print("Metricas por modelo:")
    for model_name, metrics in result["metrics_by_model"].items():
        print(
            "- "
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}, "
            f"recall_macro={metrics['recall_macro']:.4f}"
        )
    print(
        f"Melhor modelo: {result['best_model_name']} "
        f"pela metrica {result['selection_metric']}"
    )
    baseline = result["baseline_comparison"]
    print(
        "Comparacao com baseline: "
        f"{baseline['best_model_name']}={baseline['best_model_score']:.4f}, "
        f"{baseline['baseline_model_name']}={baseline['baseline_score']:.4f}, "
        f"diferenca={baseline['absolute_difference']:.4f}"
    )
    print(f"Relatorio salvo em: {result['report_path']}")
    print(
        "Matriz de confusao salva em: "
        f"{result['confusion_matrix_path']}"
    )


def _build_split(
    dataset: pd.DataFrame,
    split_type: str,
    train_end_year: int,
    test_start_year: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    if split_type == "temporal":
        X_train, X_test, y_train, y_test, split_info = split_dataset_temporal(
            dataset,
            train_end_year=train_end_year,
            test_start_year=test_start_year,
        )
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "split_info": split_info,
        }

    X, y, features, metadata = prepare_features_target_with_metadata(
        dataset,
        preserved_columns=["year"],
    )
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    split_info = {
        "split_type": "random",
        "train_period": _period_from_metadata(metadata.loc[X_train.index]),
        "test_period": _period_from_metadata(metadata.loc[X_test.index]),
        "total_records": int(len(X)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "features_used": features,
    }
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "split_info": split_info,
    }


def _validate_training_classes(y_train: pd.Series) -> None:
    if y_train.nunique() < 2:
        raise ValueError(
            "Treinamento requer pelo menos duas classes de risk_level "
            "na parte de treino."
        )


def _ordered_labels(y: pd.Series) -> list[str]:
    observed = [str(label) for label in y.dropna().unique()]
    ordered = [label for label in RISK_LEVEL_LABELS if label in observed]
    ordered.extend(sorted(set(observed) - set(ordered)))
    return ordered


def _class_distribution(y: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in y.value_counts().items()}


def _period_from_metadata(metadata: pd.DataFrame) -> dict[str, Any]:
    if "year" not in metadata.columns:
        return {"start_year": None, "end_year": None}

    years = pd.to_numeric(metadata["year"], errors="coerce").dropna()
    if years.empty:
        return {"start_year": None, "end_year": None}
    return {
        "start_year": int(years.min()),
        "end_year": int(years.max()),
    }


def _print_distribution(distribution: dict[str, int]) -> None:
    for label, count in distribution.items():
        print(f"- {label}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
