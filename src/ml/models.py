from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_model(model_name: str) -> Any:
    """Cria um modelo supervisionado a partir do nome informado."""
    normalized_name = model_name.strip().lower()

    if normalized_name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if normalized_name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    if normalized_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(eval_metric="logloss", random_state=42)

    raise ValueError(f"Modelo nao suportado: {model_name}")
