import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.train import train_and_evaluate_models


def parse_args() -> argparse.Namespace:
    """Le parametros para treinamento supervisionado de risk_level."""
    parser = argparse.ArgumentParser(
        description="Treina e avalia modelos de ML para prever risk_level.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/ml_training_dataset.parquet"),
        help="Dataset rotulado gerado a partir do INMET DuckDB.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models"),
        help="Diretorio para salvar o melhor modelo e metadados.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/reports"),
        help="Diretorio para salvar relatorios de avaliacao.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metric", default="f1_macro")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        print(
            "Dataset rotulado nao encontrado. Gere primeiro com "
            "python scripts/build_ml_dataset.py."
        )
        return 1

    try:
        result = train_and_evaluate_models(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            metric=args.metric,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    print_training_summary(result)
    return 0


def print_training_summary(result: dict[str, Any]) -> None:
    """Imprime resumo do treinamento para uso local e academico."""
    print(f"Total de registros: {result['total_records']}")
    print(f"Registros de treino: {result['train_records']}")
    print(f"Registros de teste: {result['test_records']}")
    print("Distribuicao das classes:")
    _print_distribution(result["class_distribution"])
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
        f"Melhor modelo: {result['selected_model_name']} "
        f"pela metrica {result['metric_used']}"
    )
    print(f"Modelo salvo em: {result['model_path']}")
    print(f"Metadados salvos em: {result['metadata_path']}")
    print(f"Relatorio salvo em: {result['report_path']}")
    print(f"Matriz de confusao salva em: {result['confusion_matrix_path']}")


def _print_distribution(distribution: dict[str, int]) -> None:
    for label, count in distribution.items():
        print(f"- {label}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
