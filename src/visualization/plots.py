"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_skill_by_horizon(
    skill_df: pd.DataFrame,
    output_path: str | Path,
    dataset_name: str | None = None,
) -> None:
    required = {"dataset", "horizon", "model", "skill_vs_persistence"}
    missing = required - set(skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_df = skill_df.copy()

    if dataset_name is not None:
        plot_df = plot_df.loc[plot_df["dataset"] == dataset_name].copy()

    if plot_df.empty:
        raise ValueError("No rows available for plotting skill by horizon.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, group in plot_df.groupby("model", sort=True):
        group = group.sort_values("horizon")
        ax.plot(
            group["horizon"],
            group["skill_vs_persistence"],
            marker="o",
            label=model,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Skill vs persistence")
    ax.set_title("Skill by horizon")
    ax.set_xticks(sorted(plot_df["horizon"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
