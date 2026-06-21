from typing import Any

import pandas as pd

from src.ml.models import create_model


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = "logistic_regression",
) -> Any:
    """Treina um modelo supervisionado simples."""
    model = create_model(model_name)
    return model.fit(features, target)
