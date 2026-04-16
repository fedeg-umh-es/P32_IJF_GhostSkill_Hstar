"""Operational forecast skill horizon utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def compute_hstar(skill_by_horizon: Sequence[float], criterion: str = "strict") -> int:
    """Compute H* from a horizon-wise skill sequence.

    Parameters
    ----------
    skill_by_horizon:
        Ordered horizon-wise skill values starting at h=1.
    criterion:
        `strict` returns the last consecutive horizon from h=1 with positive skill.
        `relax` returns the last horizon anywhere with positive skill.
        `nonnegative` uses skill >= 0 instead of skill > 0.
    """
    skill = np.asarray(skill_by_horizon, dtype=float)
    if skill.ndim != 1 or skill.size == 0:
        raise ValueError("skill_by_horizon must be a non-empty 1D sequence.")

    if criterion == "strict":
        hstar = 0
        for value in skill:
            if value > 0.0:
                hstar += 1
            else:
                break
        return hstar

    if criterion == "relax":
        positive = np.where(skill > 0.0)[0]
        return 0 if positive.size == 0 else int(positive[-1] + 1)

    if criterion == "nonnegative":
        valid = np.where(skill >= 0.0)[0]
        return 0 if valid.size == 0 else int(valid[-1] + 1)

    raise ValueError("criterion must be one of: 'strict', 'relax', 'nonnegative'.")


def build_skill_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "horizon", "model", "rmse"}
    missing = required - set(metrics_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    baseline_df = (
        metrics_df.loc[metrics_df["model"] == "persistence", ["dataset", "horizon", "rmse"]]
        .rename(columns={"rmse": "rmse_persistence"})
        .copy()
    )

    if (baseline_df["rmse_persistence"] <= 0).any():
        raise ValueError("Persistence RMSE must be strictly positive to compute skill.")

    skill_df = metrics_df.merge(
        baseline_df,
        on=["dataset", "horizon"],
        how="left",
        validate="many_to_one",
    )

    if skill_df["rmse_persistence"].isna().any():
        raise ValueError("Missing persistence RMSE for at least one dataset/horizon pair.")

    skill_df["skill_vs_persistence"] = 1.0 - (
        skill_df["rmse"] / skill_df["rmse_persistence"]
    )
    skill_df.loc[skill_df["model"] == "persistence", "skill_vs_persistence"] = 0.0

    return (
        skill_df[
            [
                "dataset",
                "horizon",
                "model",
                "rmse",
                "rmse_persistence",
                "skill_vs_persistence",
            ]
        ]
        .sort_values(["dataset", "horizon", "model"])
        .reset_index(drop=True)
    )


def build_horizons_summary(skill_df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "horizon", "model", "skill_vs_persistence"}
    missing = required - set(skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for (dataset, model), group in skill_df.groupby(["dataset", "model"], sort=True):
        group = group.sort_values("horizon").reset_index(drop=True)

        if model == "persistence":
            h = 0
            h_star_relax = 0
            h_star_strict = 0
        else:
            skill_values = group["skill_vs_persistence"].to_list()
            h = compute_hstar(skill_values, criterion="relax")
            h_star_relax = h
            h_star_strict = compute_hstar(skill_values, criterion="strict")

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "H": int(h),
                "H_star_relax": int(h_star_relax),
                "H_star_strict": int(h_star_strict),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
