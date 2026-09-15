"""Stacked-ensemble meta-learner (Wolpert 1992).

The base models are the three frozen, already-trained fusion predictors
(`fusion-concat`, `fusion-xattn`, `fusion-moe`). Their scalar predictions on a
holdout set are the meta-features; a small MLP learns to combine them.

**Where the meta-learner is trained:** on the base models' predictions for the
**validation** split. The base models used `val` only for early stopping, not for
weight updates, so their `val` predictions are a legitimate (if mildly optimistic)
holdout -- far less leaky than reusing `train`, where the base models have largely
memorised the targets. A strict out-of-fold scheme is stronger but costs 3-5x more
base-model training; it's a documented future option, not the default.

Meta-features are the 3 predictions only (not internal representations): with
64-3400 val molecules a 3->8->1 MLP is already near its capacity limit, and a
1280-d representation stack would just overfit. `run_baselines.py` also reports the
plain unweighted average (`ensemble-avg`) so it's visible whether the learned
stacker actually beats it.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_HIDDEN = (8,)
_ALPHA = 1.0          # strong L2: degrades gracefully toward a plain average
_MAX_ITER = 1000


class Stacker:
    def __init__(self, hidden=_HIDDEN, alpha=_ALPHA, max_iter=_MAX_ITER, seed: int = 0) -> None:
        self.scaler = StandardScaler()
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden, alpha=alpha, activation="relu",
            solver="lbfgs", max_iter=max_iter, random_state=seed,
        )

    def fit(self, base_preds, y) -> "Stacker":
        X = self.scaler.fit_transform(np.asarray(base_preds, dtype=np.float64))
        self.mlp.fit(X, np.asarray(y, dtype=np.float64).ravel())
        return self

    def predict(self, base_preds) -> np.ndarray:
        return self.mlp.predict(self.scaler.transform(np.asarray(base_preds, dtype=np.float64)))


def stacked_prediction(val_base, val_y, test_base, seed: int = 0, **kw) -> np.ndarray:
    """Fit the meta-learner on (val_base -> val_y), return its test predictions.

    val_base / test_base: [N, K] arrays of the K base models' predictions.
    """
    return Stacker(seed=seed, **kw).fit(val_base, val_y).predict(test_base)
