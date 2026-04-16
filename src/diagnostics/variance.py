"""Variance-retention diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics import variance_ratio_alpha


def variance_retention(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Alias for variance-retention ratio alpha."""
    return variance_ratio_alpha(y_true=y_true, y_pred=y_pred)


def detect_variance_collapse(alpha: float, collapse_threshold: float = 0.5) -> bool:
    """Flag substantial variance attenuation."""
    if collapse_threshold <= 0.0:
        raise ValueError("collapse_threshold must be strictly positive.")
    return float(alpha) < collapse_threshold


@dataclass(frozen=True)
class VarianceFlagSummary:
    """Threshold-based variance diagnostic flags."""

    alpha: float
    collapse_flag: bool
    inflation_flag: bool
    near_ideal_flag: bool


def variance_diagnostic_flags(
    alpha: float,
    collapse_threshold: float = 0.5,
    inflation_threshold: float = 1.5,
    tolerance: float = 0.15,
) -> VarianceFlagSummary:
    """Return simple threshold-based variance flags."""
    if collapse_threshold <= 0.0:
        raise ValueError("collapse_threshold must be strictly positive.")
    if inflation_threshold <= 1.0:
        raise ValueError("inflation_threshold must be greater than 1.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    alpha_value = float(alpha)
    return VarianceFlagSummary(
        alpha=alpha_value,
        collapse_flag=alpha_value < collapse_threshold,
        inflation_flag=alpha_value > inflation_threshold,
        near_ideal_flag=abs(alpha_value - 1.0) <= tolerance,
    )
