"""Frozen-readout statistics with the historical NumPy metric definitions."""
from __future__ import annotations
from typing import Any
import numpy as np

def metrics(scores: np.ndarray, truth: np.ndarray, nodes: list[int]) -> dict[str, Any]:
    selected = np.asarray(scores[:, nodes], dtype=np.float64)
    labels = truth[:, nodes]
    errors = selected - labels.astype(np.float64)
    predicted = selected >= .5
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    tn = int(np.count_nonzero(~predicted & ~labels))
    return {
        "states": len(nodes), "positive_labels": tp + fn, "negative_labels": fp + tn,
        "rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
        "positive_rmse": float(np.sqrt(np.mean(errors[labels] ** 2))) if labels.any() else None,
        "negative_rmse": float(np.sqrt(np.mean(errors[~labels] ** 2))) if (~labels).any() else None,
        "max_absolute_error": float(np.max(np.abs(errors))) if errors.size else None,
        "positive_score_min": float(selected[labels].min()) if labels.any() else None,
        "positive_score_mean": float(selected[labels].mean()) if labels.any() else None,
        "negative_score_max": float(selected[~labels].max()) if (~labels).any() else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "empty_predicted_states": int(np.count_nonzero(predicted.sum(axis=0) == 0)),
        "states_with_no_true_positive": int(np.count_nonzero((predicted & labels).sum(axis=0) == 0)),
        "exact_set_states": int(np.count_nonzero(np.all(predicted == labels, axis=0))),
        "threshold": .5,
    }


def grouped(scores: np.ndarray, truth: np.ndarray, sources: list[int], sinks: list[int]) -> dict[str, Any]:
    return {"sources": metrics(scores, truth, sources), "sinks": metrics(scores, truth, sinks),
            "all_states": metrics(scores, truth, list(range(truth.shape[1])))}


def readout_norms(g: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(np.asarray(g, dtype=np.float64), compute_uv=False)
    return {"frobenius": float(np.linalg.norm(g)), "operator_2": float(singular[0])}
