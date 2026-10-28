"""Shared numerical summaries with explicit historical diagnostic definitions."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

def classification(raw: np.ndarray, truth: np.ndarray, nodes: list[int], threshold: float) -> dict[str, Any]:
    selected = raw[:, nodes]
    expected = truth[:, nodes]
    predicted = selected >= threshold
    tp = int(np.count_nonzero(predicted & expected))
    fp = int(np.count_nonzero(predicted & ~expected))
    fn = int(np.count_nonzero(~predicted & expected))
    tn = int(np.count_nonzero(~predicted & ~expected))
    return {
        "states": len(nodes), "state_ids": nodes, "threshold": threshold,
        "positive_labels": tp + fn, "negative_labels": tn + fp,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "empty_predicted_states": int(np.count_nonzero(predicted.sum(axis=0) == 0)),
        "states_with_false_positive": int(np.count_nonzero((predicted & ~expected).any(axis=0))),
        "mse": float(np.mean((selected - expected.astype(np.float32)) ** 2)) if selected.size else None,
        "raw_min": float(selected.min()) if selected.size else None,
        "raw_max": float(selected.max()) if selected.size else None,
        "false_positive_by_state": [
            {"state": node, "false_positive": int(np.count_nonzero(predicted[:, index] & ~expected[:, index]))}
            for index, node in enumerate(nodes)
        ],
    }

def target_distribution(values: np.ndarray, target: float) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return {"count": 0}
    error = x - target
    return {
        "count": len(x), "target": target, "mean": float(x.mean()),
        "std": float(x.std()), "minimum": float(x.min()), "maximum": float(x.max()),
        "quantiles": {str(p): float(np.quantile(x, p)) for p in (.01, .05, .1, .25, .5, .75, .9, .95, .99)},
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "below_zero": int((x < 0).sum()), "above_one": int((x > 1).sum()),
    }

def finite(value):
    return float(value) if math.isfinite(value) else None

def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)

def distribution(values):
    return {"count": len(values), "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "median": quantile(values, 0.5), "p95": quantile(values, 0.95),
            "max": max(values) if values else None}

def longest_run(values):
    best_start, best_length = None, 0
    current_start, current_length = None, 0
    for index, value in enumerate(values):
        if value:
            if current_start is None:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_start, best_length = current_start, current_length
        else:
            current_start, current_length = None, 0
    return {"start_transition_index": best_start, "length_actions": best_length}
