from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.plots import (
    plot_fold_skill_sensitivity_grid,
    plot_skill_sensitivity_grid,
    plot_variance_sensitivity_grid,
)


TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

DATASETS = ["valencia_pm10", "madrid_pm10_rank"]
BASELINES = ["persistence", "seasonal_persistence_7"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected input table: {path}")
    return pd.read_csv(path)


def _skill_table_path(dataset: str, baseline: str) -> Path:
    if baseline == "persistence":
        prefix = "" if dataset == "pm10_example" else f"{dataset}_"
        return TABLES_DIR / f"{prefix}skill_by_horizon.csv"
    return TABLES_DIR / f"{dataset}_vs_{baseline}_skill_by_horizon.csv"


def _fold_skill_table_path(dataset: str, baseline: str) -> Path:
    if baseline == "persistence":
        prefix = "" if dataset == "pm10_example" else f"{dataset}_"
        return TABLES_DIR / f"{prefix}skill_by_fold.csv"
    return TABLES_DIR / f"{dataset}_vs_{baseline}_skill_by_fold.csv"


def _variance_table_path(dataset: str, baseline: str) -> Path:
    if baseline == "persistence":
        prefix = "" if dataset == "pm10_example" else f"{dataset}_"
        return TABLES_DIR / f"{prefix}variance_retention_summary.csv"
    return TABLES_DIR / f"{dataset}_vs_{baseline}_variance_retention_summary.csv"


def main() -> None:
    skill_frames = [
        _read_csv(_skill_table_path(dataset, baseline))
        for dataset in DATASETS
        for baseline in BASELINES
    ]
    fold_skill_frames = [
        _read_csv(_fold_skill_table_path(dataset, baseline))
        for dataset in DATASETS
        for baseline in BASELINES
    ]
    variance_frames = [
        _read_csv(_variance_table_path(dataset, baseline))
        for dataset in DATASETS
        for baseline in BASELINES
    ]

    plot_skill_sensitivity_grid(
        skill_frames=skill_frames,
        output_path=FIGURES_DIR / "real_cases_skill_sensitivity_grid.png",
        datasets=DATASETS,
        baseline_order=BASELINES,
    )
    plot_fold_skill_sensitivity_grid(
        fold_skill_frames=fold_skill_frames,
        output_path=FIGURES_DIR / "real_cases_foldwise_robustness_grid.png",
        datasets=DATASETS,
        baseline_order=BASELINES,
    )
    plot_variance_sensitivity_grid(
        variance_frames=variance_frames,
        output_path=FIGURES_DIR / "real_cases_variance_retention_grid.png",
        datasets=DATASETS,
        baseline_order=BASELINES,
    )


if __name__ == "__main__":
    main()
