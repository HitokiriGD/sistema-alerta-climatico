from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """Adapta XGBClassifier para alvos textuais como baixo/moderado/alto."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "LabelEncodedXGBClassifier":
        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(y)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
        )
        self.model_.fit(X, encoded_y)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded_predictions = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(
            np.asarray(encoded_predictions, dtype=int)
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(X)


def build_logistic_regression_model() -> Pipeline:
    """Cria baseline interpretavel com padronizacao das features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def build_baseline_most_frequent_model() -> DummyClassifier:
    """Cria baseline simples que sempre prediz a classe majoritaria."""
    return DummyClassifier(strategy="most_frequent")


def build_random_forest_model() -> RandomForestClassifier:
    """Cria Random Forest simples com pesos balanceados."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )


def build_xgboost_model() -> LabelEncodedXGBClassifier:
    """Cria XGBoost com codificacao interna dos rotulos textuais."""
    return LabelEncodedXGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )


def get_candidate_models() -> dict[str, Any]:
    """Retorna os modelos candidatos avaliados no TCC."""
    return {
        "baseline_most_frequent": build_baseline_most_frequent_model(),
        "logistic_regression": build_logistic_regression_model(),
        "random_forest": build_random_forest_model(),
        "xgboost": build_xgboost_model(),
    }


def create_model(model_name: str) -> Any:
    """Cria um modelo supervisionado a partir do nome informado."""
    normalized_name = model_name.strip().lower()
    models = get_candidate_models()

    if normalized_name in models:
        return models[normalized_name]

    raise ValueError(f"Modelo nao suportado: {model_name}")
