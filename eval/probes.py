"""Linear probes: how well does a frozen/aligned embedding predict a physical property?

Redshift is the primary target (a Ridge regression). The probe is trained on the train split's
embeddings and scored on the test split's, so it measures linearly decodable information in the
representation, not memorization.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def probe_regression(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, alpha: float = 1.0
) -> Dict[str, float]:
    """Ridge-regress ``y`` from embeddings; return R2 and MSE on the test split. NaN targets dropped."""
    train_mask = np.isfinite(train_y)
    test_mask = np.isfinite(test_y)
    if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 1:
        return {
            "r2": float("nan"),
            "mse": float("nan"),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
    model = Ridge(alpha=alpha)
    model.fit(train_x[train_mask], train_y[train_mask])
    predictions = model.predict(test_x[test_mask])
    return {
        "r2": float(r2_score(test_y[test_mask], predictions)),
        "mse": float(mean_squared_error(test_y[test_mask], predictions)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
