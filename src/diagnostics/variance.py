"""Variance-retention diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import skill_vp as _skill_vp
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


def build_variance_retention_table(
    fold_metrics_df: pd.DataFrame,
    skill_df: pd.DataFrame,
    collapse_threshold: float = 0.5,
    inflation_threshold: float = 1.5,
    tolerance: float = 0.15,
) -> pd.DataFrame:
    """Compute per-(dataset, model, horizon) variance retention and Skill_VP.

    Parameters
    ----------
    fold_metrics_df:
        Per-fold rows with at least ``dataset``, ``model``, ``horizon``,
        ``y_true``, and ``y_pred`` columns.
    skill_df:
        Aggregate skill table from ``build_skill_table``, used to pull
        ``skill_vs_persistence`` per (dataset, model, horizon).
    collapse_threshold, inflation_threshold, tolerance:
        Passed to ``variance_diagnostic_flags``.
    """
    required_fold = {"dataset", "model", "horizon", "y_true", "y_pred"}
    required_skill = {"dataset", "model", "horizon", "skill_vs_persistence"}
    for required, label in [
        (required_fold, "fold_metrics_df"),
        (required_skill, "skill_df"),
    ]:
        missing = required - set(
            fold_metrics_df.columns if label == "fold_metrics_df" else skill_df.columns
        )
        if missing:
            raise ValueError(f"Missing required columns in {label}: {sorted(missing)}")

    skill_lookup = skill_df.set_index(["dataset", "model", "horizon"])[
        "skill_vs_persistence"
    ].to_dict()

    rows = []
    for (dataset, model, horizon), group in fold_metrics_df.groupby(
        ["dataset", "model", "horizon"], sort=True
    ):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)

        alpha = variance_retention(y_true, y_pred)
        flags = variance_diagnostic_flags(
            alpha,
            collapse_threshold=collapse_threshold,
            inflation_threshold=inflation_threshold,
            tolerance=tolerance,
        )
        skill_val = float(
            skill_lookup.get((dataset, model, int(horizon)), float("nan"))
        )
        svp = float(_skill_vp(skill_val, alpha)) if not np.isnan(skill_val) else float("nan")

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "skill": skill_val,
                "alpha": alpha,
                "skill_vp": svp,
                "collapse_flag": flags.collapse_flag,
                "inflation_flag": flags.inflation_flag,
                "near_ideal_flag": flags.near_ideal_flag,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model", "horizon"])
        .reset_index(drop=True)
    )


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
