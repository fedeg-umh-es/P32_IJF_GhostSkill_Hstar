from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data.loaders import load_dataset
from src.diagnostics.hstar import build_horizons_summary, build_skill_table, build_trajectory_summary
from src.evaluation.metrics import rmse
from src.evaluation.rolling_origin import generate_rolling_origin_folds
from src.models.baselines import persistence_forecast
from src.visualization.plots import plot_skill_by_horizon


DEFAULT_CONFIG = Path("configs/datasets/pm10_example.yaml")
DEFAULT_METRICS_OUTPUT = Path("outputs/tables/metrics_by_horizon.csv")
DEFAULT_LOG_OUTPUT = Path("outputs/logs/run_summary.txt")
DEFAULT_SKILL_OUTPUT = Path("outputs/tables/skill_by_horizon.csv")
DEFAULT_HORIZONS_OUTPUT = Path("outputs/tables/horizons_summary.csv")
DEFAULT_TRAJECTORY_OUTPUT = Path("outputs/tables/trajectory_summary.csv")
DEFAULT_SKILL_PLOT_OUTPUT = Path("outputs/figures/skill_by_horizon.png")


def build_direct_lagged_samples(y: np.ndarray, horizon: int, n_lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    origin_rows: list[int] = []

    for origin_end in range(n_lags - 1, len(y) - horizon):
        window = y[origin_end - n_lags + 1 : origin_end + 1]
        target = y[origin_end + horizon]
        X_rows.append(window.astype(float))
        y_rows.append(float(target))
        origin_rows.append(origin_end)

    if not X_rows:
        raise ValueError(
            f"Not enough observations to build lagged samples for horizon={horizon} and n_lags={n_lags}."
        )

    return np.vstack(X_rows), np.asarray(y_rows, dtype=float), np.asarray(origin_rows, dtype=int)


def evaluate_persistence(y: np.ndarray, horizons: list[int], min_train_size: int, step_size: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    n_obs = len(y)

    for horizon in horizons:
        folds = generate_rolling_origin_folds(
            n_obs=n_obs,
            min_train_size=min_train_size,
            horizon=horizon,
            step=step_size,
        )
        y_true: list[float] = []
        y_pred: list[float] = []

        for fold in folds:
            pred = persistence_forecast(last_observation=y[fold.train_end], horizon=horizon)[-1]
            y_true.append(float(y[fold.test_end]))
            y_pred.append(float(pred))

        rows.append(
            {
                "model": "persistence",
                "horizon": horizon,
                "rmse": rmse(y_true, y_pred),
                "n_folds": len(folds),
            }
        )

    return rows


def evaluate_ridge(
    y: np.ndarray,
    horizons: list[int],
    min_train_size: int,
    step_size: int,
    n_lags: int,
    alpha: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    n_obs = len(y)

    for horizon in horizons:
        folds = generate_rolling_origin_folds(
            n_obs=n_obs,
            min_train_size=min_train_size,
            horizon=horizon,
            step=step_size,
        )
        X_all, y_all, origin_all = build_direct_lagged_samples(y=y, horizon=horizon, n_lags=n_lags)

        y_true: list[float] = []
        y_pred: list[float] = []

        for fold in folds:
            train_mask = origin_all + horizon <= fold.train_end
            if not np.any(train_mask):
                raise ValueError(
                    f"No training samples available for horizon={horizon} in fold ending at index {fold.train_end}."
                )

            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            X_test = y[fold.train_end - n_lags + 1 : fold.train_end + 1].reshape(1, -1)

            if X_test.shape[1] != n_lags:
                raise ValueError(
                    f"Insufficient lag context for horizon={horizon} at test origin ending at index {fold.train_end}."
                )

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            y_true.append(float(y[fold.test_end]))
            y_pred.append(float(model.predict(X_test)[0]))

        rows.append(
            {
                "model": "ridge",
                "horizon": horizon,
                "rmse": rmse(y_true, y_pred),
                "n_folds": len(folds),
            }
        )

    return rows


def run_benchmark(
    config_path: Path,
    metrics_output: Path,
    log_output: Path,
    skill_output: Path,
    horizons_output: Path,
    trajectory_output: Path,
    skill_plot_output: Path,
    n_lags: int,
    ridge_alpha: float,
) -> pd.DataFrame:
    dataset = load_dataset(config_path)
    if dataset["name"] != "pm10_example":
        raise ValueError(f"This runner is currently wired only for pm10_example, got {dataset['name']}.")

    split_cfg = dataset["metadata"]["split"]
    if split_cfg.get("scheme") != "rolling_origin":
        raise ValueError("Only rolling_origin split is supported in this runner.")

    y = dataset["y"].to_numpy(dtype=float)
    horizons = list(dataset["horizons"])
    min_train_size = int(split_cfg["min_train_size"])
    step_size = int(split_cfg["step_size"])

    if min_train_size < n_lags:
        raise ValueError("min_train_size must be at least as large as n_lags.")

    results = evaluate_persistence(
        y=y,
        horizons=horizons,
        min_train_size=min_train_size,
        step_size=step_size,
    )
    results.extend(
        evaluate_ridge(
            y=y,
            horizons=horizons,
            min_train_size=min_train_size,
            step_size=step_size,
            n_lags=n_lags,
            alpha=ridge_alpha,
        )
    )

    metrics_df = pd.DataFrame(results)
    metrics_df.insert(0, "dataset", dataset["name"])
    metrics_df = metrics_df.sort_values(["horizon", "model"]).reset_index(drop=True)
    n_folds_by_horizon = (
        metrics_df.loc[:, ["horizon", "n_folds"]]
        .drop_duplicates()
        .sort_values("horizon")
    )
    n_folds_summary = ",".join(
        f"{int(row.horizon)}:{int(row.n_folds)}" for row in n_folds_by_horizon.itertuples(index=False)
    )

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_output, index=False)
    skill_df = build_skill_table(metrics_df)
    skill_output.parent.mkdir(parents=True, exist_ok=True)
    skill_df.to_csv(skill_output, index=False)
    horizons_df = build_horizons_summary(skill_df)
    horizons_output.parent.mkdir(parents=True, exist_ok=True)
    horizons_df.to_csv(horizons_output, index=False)
    trajectory_df = build_trajectory_summary(skill_df)
    trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_df.to_csv(trajectory_output, index=False)
    plot_skill_by_horizon(skill_df, skill_plot_output, dataset_name=dataset["name"])

    log_output.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"dataset={dataset['name']}",
        f"config={config_path.resolve()}",
        f"raw_data={dataset['metadata']['raw_data_path']}",
        f"n_obs={len(dataset['data'])}",
        f"target_column={dataset['target_column']}",
        f"horizons={horizons}",
        f"split_scheme={split_cfg['scheme']}",
        f"min_train_size={min_train_size}",
        f"step_size={step_size}",
        f"n_folds_by_horizon={n_folds_summary}",
        f"models=persistence,ridge",
        f"ridge_alpha={ridge_alpha}",
        f"n_lags={n_lags}",
        f"metrics_output={metrics_output.resolve()}",
        f"skill_table_path={skill_output.resolve()}",
        f"horizons_summary_path={horizons_output.resolve()}",
        f"trajectory_summary_path={trajectory_output.resolve()}",
        f"skill_plot_path={skill_plot_output.resolve()}",
        f"log_output={log_output.resolve()}",
    ]
    log_output.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal core benchmark for pm10_example.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--log-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--skill-output", type=Path, default=DEFAULT_SKILL_OUTPUT)
    parser.add_argument("--horizons-output", type=Path, default=DEFAULT_HORIZONS_OUTPUT)
    parser.add_argument("--trajectory-output", type=Path, default=DEFAULT_TRAJECTORY_OUTPUT)
    parser.add_argument("--skill-plot-output", type=Path, default=DEFAULT_SKILL_PLOT_OUTPUT)
    parser.add_argument("--n-lags", type=int, default=7)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_df = run_benchmark(
        config_path=args.config,
        metrics_output=args.metrics_output,
        log_output=args.log_output,
        skill_output=args.skill_output,
        horizons_output=args.horizons_output,
        trajectory_output=args.trajectory_output,
        skill_plot_output=args.skill_plot_output,
        n_lags=args.n_lags,
        ridge_alpha=args.ridge_alpha,
    )
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
