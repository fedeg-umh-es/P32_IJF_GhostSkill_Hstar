"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_skill_by_horizon(
    skill_df: pd.DataFrame,
    output_path: str | Path,
    dataset_name: str | None = None,
    exclude_baseline: bool = True,
    baseline_model: str = "persistence",
) -> None:
    required = {"dataset", "horizon", "model", "skill_vs_baseline"}
    missing = required - set(skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_df = skill_df.copy()

    if dataset_name is not None:
        plot_df = plot_df.loc[plot_df["dataset"] == dataset_name].copy()

    if exclude_baseline:
        plot_df = plot_df.loc[plot_df["model"] != baseline_model].copy()

    if plot_df.empty:
        raise ValueError("No rows available for plotting skill by horizon.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, group in plot_df.groupby("model", sort=True):
        group = group.sort_values("horizon")
        ax.plot(
            group["horizon"],
            group["skill_vs_baseline"],
            marker="o",
            label=model,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(f"Skill vs {baseline_model}")
    ax.set_title("Skill by horizon")
    ax.set_xticks(sorted(plot_df["horizon"].unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fold_skill_boxplot(
    fold_skill_df: pd.DataFrame,
    output_path: str | Path,
    dataset_name: str | None = None,
    exclude_baseline: bool = True,
    baseline_model: str = "persistence",
) -> None:
    required = {"dataset", "fold", "horizon", "model", "skill_vs_baseline"}
    missing = required - set(fold_skill_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_df = fold_skill_df.copy()
    if dataset_name is not None:
        plot_df = plot_df.loc[plot_df["dataset"] == dataset_name].copy()
    if exclude_baseline:
        plot_df = plot_df.loc[plot_df["model"] != baseline_model].copy()
    if plot_df.empty:
        raise ValueError("No rows available for plotting fold-wise skill distributions.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.boxplot(
        data=plot_df,
        x="horizon",
        y="skill_vs_baseline",
        hue="model",
        ax=ax,
        showfliers=False,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(f"Fold-wise skill vs {baseline_model}")
    ax.set_title("Fold-wise Skill Distribution by Horizon")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_skill_sensitivity_grid(
    skill_frames: list[pd.DataFrame],
    output_path: str | Path,
    datasets: list[str],
    baseline_order: list[str],
    exclude_baseline: bool = True,
) -> None:
    required = {"dataset", "horizon", "model"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_frames: list[pd.DataFrame] = []
    for frame in skill_frames:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        plot_df = frame.copy()
        if "skill_vs_baseline" not in plot_df.columns:
            if "skill_vs_persistence" not in plot_df.columns:
                raise ValueError("Skill dataframe must include skill_vs_baseline or skill_vs_persistence.")
            plot_df["skill_vs_baseline"] = plot_df["skill_vs_persistence"]
        if "baseline_model" not in plot_df.columns:
            plot_df["baseline_model"] = "persistence"
        normalized_frames.append(plot_df)

    plot_df = pd.concat(normalized_frames, ignore_index=True)
    plot_df = plot_df.loc[plot_df["dataset"].isin(datasets)].copy()
    plot_df["baseline_model"] = pd.Categorical(
        plot_df["baseline_model"],
        categories=baseline_order,
        ordered=True,
    )

    n_rows = len(datasets)
    n_cols = len(baseline_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, baseline_model in enumerate(baseline_order):
            ax = axes[row_idx][col_idx]
            panel_df = plot_df.loc[
                (plot_df["dataset"] == dataset_name)
                & (plot_df["baseline_model"] == baseline_model)
            ].copy()
            if exclude_baseline:
                panel_df = panel_df.loc[panel_df["model"] != baseline_model].copy()

            if panel_df.empty:
                ax.set_visible(False)
                continue

            for model, group in panel_df.groupby("model", sort=True):
                group = group.sort_values("horizon")
                ax.plot(
                    group["horizon"],
                    group["skill_vs_baseline"],
                    marker="o",
                    linewidth=2,
                    label=model,
                )

            ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
            ax.set_title(f"{dataset_name} | vs {baseline_model}")
            ax.set_xticks(sorted(panel_df["horizon"].unique()))
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Skill")
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(title="Model", loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fold_skill_sensitivity_grid(
    fold_skill_frames: list[pd.DataFrame],
    output_path: str | Path,
    datasets: list[str],
    baseline_order: list[str],
    exclude_baseline: bool = True,
) -> None:
    required = {"dataset", "fold", "horizon", "model"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_frames: list[pd.DataFrame] = []
    for frame in fold_skill_frames:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        plot_df = frame.copy()
        if "skill_vs_baseline" not in plot_df.columns:
            if "skill_vs_persistence" not in plot_df.columns:
                raise ValueError(
                    "Fold skill dataframe must include skill_vs_baseline or skill_vs_persistence."
                )
            plot_df["skill_vs_baseline"] = plot_df["skill_vs_persistence"]
        if "baseline_model" not in plot_df.columns:
            plot_df["baseline_model"] = "persistence"
        normalized_frames.append(plot_df)

    plot_df = pd.concat(normalized_frames, ignore_index=True)
    plot_df = plot_df.loc[plot_df["dataset"].isin(datasets)].copy()
    plot_df["baseline_model"] = pd.Categorical(
        plot_df["baseline_model"],
        categories=baseline_order,
        ordered=True,
    )

    n_rows = len(datasets)
    n_cols = len(baseline_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 4.2 * n_rows),
        sharex=True,
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, baseline_model in enumerate(baseline_order):
            ax = axes[row_idx][col_idx]
            panel_df = plot_df.loc[
                (plot_df["dataset"] == dataset_name)
                & (plot_df["baseline_model"] == baseline_model)
            ].copy()
            if exclude_baseline:
                panel_df = panel_df.loc[panel_df["model"] != baseline_model].copy()

            if panel_df.empty:
                ax.set_visible(False)
                continue

            sns.boxplot(
                data=panel_df,
                x="horizon",
                y="skill_vs_baseline",
                hue="model",
                ax=ax,
                showfliers=False,
            )
            ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
            ax.set_title(f"{dataset_name} | vs {baseline_model}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Fold-wise skill")
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(title="Model", loc="best")
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_variance_sensitivity_grid(
    variance_frames: list[pd.DataFrame],
    output_path: str | Path,
    datasets: list[str],
    baseline_order: list[str],
) -> None:
    required = {"dataset", "horizon", "model", "alpha", "skill_vp"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_frames: list[pd.DataFrame] = []
    for frame in variance_frames:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        plot_df = frame.copy()
        if "baseline_model" not in plot_df.columns:
            plot_df["baseline_model"] = "persistence"
        normalized_frames.append(plot_df)

    plot_df = pd.concat(normalized_frames, ignore_index=True)
    plot_df = plot_df.loc[plot_df["dataset"].isin(datasets)].copy()
    plot_df["baseline_model"] = pd.Categorical(
        plot_df["baseline_model"],
        categories=baseline_order,
        ordered=True,
    )

    n_rows = len(datasets)
    n_cols = len(baseline_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.8 * n_cols, 4.0 * n_rows),
        sharex=True,
        sharey=False,
    )
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, baseline_model in enumerate(baseline_order):
            ax = axes[row_idx][col_idx]
            panel_df = plot_df.loc[
                (plot_df["dataset"] == dataset_name)
                & (plot_df["baseline_model"] == baseline_model)
            ].copy()

            if panel_df.empty:
                ax.set_visible(False)
                continue

            for model, group in panel_df.groupby("model", sort=True):
                group = group.sort_values("horizon")
                ax.plot(
                    group["horizon"],
                    group["alpha"],
                    marker="o",
                    linewidth=2,
                    label=f"{model} alpha",
                )

            ax.axhline(1.0, linestyle="-", linewidth=1, color="black")
            ax.axhline(0.5, linestyle="--", linewidth=1, color="gray")
            ax.set_title(f"{dataset_name} | vs {baseline_model}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Variance retention (alpha)")
            ax.set_xticks(sorted(panel_df["horizon"].unique()))
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(title="Series", loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
