"""Fit a linear availability readout to frozen state prototypes on CUDA."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import torch
import numpy as np


@torch.no_grad()
def availability_statistics(scores: torch.Tensor, labels: torch.Tensor,
                            columns: torch.Tensor, threshold: float = .5) -> dict[str, Any]:
    observed, expected = scores[:, columns], labels[:, columns].bool()
    predicted = observed >= threshold
    positive, negative = observed[expected], observed[~expected]
    return {
        "states": int(columns.sum().item()), "entries": observed.numel(),
        "tp": int((predicted & expected).sum().item()),
        "tn": int((~predicted & ~expected).sum().item()),
        "fp": int((predicted & ~expected).sum().item()),
        "fn": int((~predicted & expected).sum().item()),
        "rmse": float((observed - expected.float()).square().mean().sqrt().item()),
        "exact_outgoing_sets": int((predicted == expected).all(dim=0).sum().item()),
        "positive_min": float(positive.min().item()) if positive.numel() else None,
        "negative_max": float(negative.max().item()) if negative.numel() else None,
        "threshold": threshold,
    }


@torch.no_grad()
def fit_frozen_availability(Q: torch.Tensor, G: torch.Tensor, labels: torch.Tensor,
                            *, rcond: float = 1e-12) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return minimum-change and minimum-norm solutions without mutating inputs.

    Labels describe all observed nodes, including zero-outdegree nodes. The
    minimum-change solution preserves G's component orthogonal to span(Q).
    The solve uses CUDA float64 SVD; stored/inference readouts use float32.
    """
    if Q.device.type != "cuda" or G.device != Q.device or labels.device != Q.device:
        raise ValueError("Frozen readout fitting requires all tensors on the same CUDA device")
    if Q.ndim != 2 or G.ndim != 2 or labels.shape != (G.shape[0], Q.shape[1]) or G.shape[1] != Q.shape[0]:
        raise ValueError("Inconsistent Q/G/availability dimensions")
    if not 0 < rcond < 1:
        raise ValueError("rcond must be between zero and one")
    torch.cuda.synchronize(Q.device)
    started = perf_counter()
    q64, g64, a64 = Q.double(), G.double(), labels.double()
    U, singular, Vh = torch.linalg.svd(q64, full_matrices=False, driver="gesvd")
    cutoff = singular.max() * rcond
    retained = singular > cutoff
    inverse = torch.where(retained, singular.reciprocal(), torch.zeros_like(singular))
    pseudoinverse = (Vh.T * inverse) @ U.T
    norm = a64 @ pseudoinverse
    change = g64 + (a64 - g64 @ q64) @ pseudoinverse
    variants = {"minimum_change": change.float(), "minimum_norm": norm.float()}
    torch.cuda.synchronize(Q.device)
    solve_seconds = perf_counter() - started
    sources = labels.bool().any(dim=0)
    groups = {"sources": sources, "sinks": ~sources, "all_states": torch.ones_like(sources)}
    statistics = {}
    for name, readout in {"raw": G, **variants}.items():
        scores = readout @ Q
        if not bool(torch.isfinite(scores).all().item()):
            raise FloatingPointError("Nonfinite fitted G scores")
        statistics[name] = {group: availability_statistics(scores, labels, mask)
                            for group, mask in groups.items()}
        statistics[name].update(frobenius_norm=float(readout.norm().item()),
                                distance_from_raw=float((readout - G).norm().item()))
    return variants, {
        "solve": {"device": str(Q.device), "dtype": "float64", "inference_dtype": "float32",
                  "method": "reduced_svd_gesvd", "rcond": rcond,
                  "rank": int(retained.sum().item()), "columns": Q.shape[1],
                  "condition": float((singular.max() / singular.min()).item()),
                  "cutoff": float(cutoff.item()), "singular_values": singular.cpu().tolist(),
                  "seconds": solve_seconds,
                  "minimum_change_formula": "G + (A - GQ) Q+",
                  "minimum_norm_formula": "A Q+"},
        "variants": statistics,
        "supervision": "all observed graph nodes; no task goals or shortest paths",
    }


def weighted_numpy_reference(q64: np.ndarray, g0: np.ndarray, truth: np.ndarray,
                  nodes: list[int], weights: np.ndarray, *, rcond: float,
                  minimum_change: bool) -> tuple[np.ndarray, dict[str, Any]]:
    import numpy as np

    design = q64[:, nodes].T
    desired = truth[:, nodes].T.astype(np.float64)
    residual = desired - design @ g0.T if minimum_change else desired
    root_weights = np.sqrt(weights)
    weighted = design * root_weights[:, None]
    weighted_target = residual * root_weights[:, None]
    u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    cutoff = rcond * singular[0]
    retained = singular > cutoff
    delta = (vt[retained].T / singular[retained]) @ (u[:, retained].T @ weighted_target)
    result = g0 + delta.T if minimum_change else delta.T
    return result, {
        "solver": "numpy_float64_explicit_reduced_SVD_pseudoinverse",
        "design_shape": list(weighted.shape), "rank": int(retained.sum()),
        "rcond": rcond, "absolute_singular_cutoff": float(cutoff),
        "singular_values": singular.tolist(),
        "largest_singular": float(singular[0]), "smallest_singular": float(singular[-1]),
        "condition_largest_over_smallest": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "retained_condition": float(singular[0] / singular[retained][-1]) if retained.any() else None,
        "minimum_change_from_original_G": minimum_change,
        "solution_selection": "minimum_Frobenius_change_from_original_G" if minimum_change else "minimum_Frobenius_norm_G",
        "weighted_training_objective": float(np.sum(weights[:, None] * (design @ result.T - desired) ** 2) / (2 * weights.sum())),
        "state_weights": {str(node): float(weight) for node, weight in zip(nodes, weights)},
    }
