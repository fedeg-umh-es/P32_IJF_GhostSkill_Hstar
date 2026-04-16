"""Core metrics for multi-horizon forecast evaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_1d_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("Input array must not be empty.")
    return array


def rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """Compute root mean squared error for aligned vectors."""
    true = _as_1d_float_array(y_true)
    pred = _as_1d_float_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def relative_skill_vs_persistence(
    model_rmse: float | np.ndarray,
    persistence_rmse: float | np.ndarray,
) -> float | np.ndarray:
    """Compute persistence-relative skill as 1 - RMSE_model / RMSE_persistence."""
    model = np.asarray(model_rmse, dtype=float)
    baseline = np.asarray(persistence_rmse, dtype=float)
    if model.shape != baseline.shape:
        raise ValueError("model_rmse and persistence_rmse must have matching shapes.")

    with np.errstate(divide="ignore", invalid="ignore"):
        skill = 1.0 - (model / baseline)

    zero_mask = baseline == 0.0
    if np.any(zero_mask):
        skill = np.where(zero_mask & (model == 0.0), 0.0, skill)
        skill = np.where(zero_mask & (model != 0.0), -np.inf, skill)

    if skill.ndim == 0:
        return float(skill)
    return skill


def variance_ratio_alpha(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    ddof: int = 1,
) -> float:
    """Return the variance-retention ratio alpha = Var(pred) / Var(true)."""
    true = _as_1d_float_array(y_true)
    pred = _as_1d_float_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    true_var = float(np.var(true, ddof=ddof))
    pred_var = float(np.var(pred, ddof=ddof))
    if np.isclose(true_var, 0.0):
        raise ValueError("Variance of y_true is zero; alpha is undefined.")
    return pred_var / true_var


def skill_vp(
    skill: float | np.ndarray,
    alpha: float | np.ndarray,
) -> float | np.ndarray:
    """Combine skill and variance retention into a diagnostic quantity.

    Defined here as:
    Skill_VP = skill * min(alpha, 1 / alpha)

    This keeps the sign of skill while penalizing both variance collapse
    (alpha << 1) and variance inflation (alpha >> 1).
    """
    skill_arr = np.asarray(skill, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    if skill_arr.shape != alpha_arr.shape:
        raise ValueError("skill and alpha must have matching shapes.")
    if np.any(alpha_arr <= 0.0):
        raise ValueError("alpha must be strictly positive.")

    penalty = np.minimum(alpha_arr, 1.0 / alpha_arr)
    out = skill_arr * penalty
    if out.ndim == 0:
        return float(out)
    return out


def ranking_from_metric_dict(
    metric_by_model: Mapping[str, float],
    ascending: bool = True,
) -> list[tuple[str, float]]:
    """Return a stable ranking from a metric dictionary."""
    if not metric_by_model:
        raise ValueError("metric_by_model must not be empty.")
    return sorted(metric_by_model.items(), key=lambda item: (item[1], item[0]), reverse=not ascending)
