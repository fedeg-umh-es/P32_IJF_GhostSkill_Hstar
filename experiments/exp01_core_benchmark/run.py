from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data.loaders import load_dataset
from src.diagnostics.hstar import (
    build_fold_dispersion_summary,
    build_fold_skill_table,
    build_horizons_summary,
    build_robust_horizons_summary,
    build_skill_table,
    build_trajectory_summary,
)
from src.diagnostics.variance import build_variance_retention_table
from src.evaluation.rolling_origin import generate_rolling_origin_folds
from src.models.baselines import persistence_forecast
from src.visualization.plots import plot_fold_skill_boxplot, plot_skill_by_horizon


DEFAULT_CONFIG = Path("configs/datasets/pm10_example.yaml")
DEFAULT_METRICS_OUTPUT = Path("outputs/tables/metrics_by_horizon.csv")
DEFAULT_LOG_OUTPUT = Path("outputs/logs/run_summary.txt")
DEFAULT_SKILL_OUTPUT = Path("outputs/tables/skill_by_horizon.csv")
DEFAULT_HORIZONS_OUTPUT = Path("outputs/tables/horizons_summary.csv")
DEFAULT_TRAJECTORY_OUTPUT = Path("outputs/tables/trajectory_summary.csv")
DEFAULT_FOLD_SKILL_OUTPUT = Path("outputs/tables/skill_by_fold.csv")
DEFAULT_FOLD_DISPERSION_OUTPUT = Path("outputs/tables/skill_fold_dispersion_summary.csv")
DEFAULT_SKILL_PLOT_OUTPUT = Path("outputs/figures/skill_by_horizon.png")
DEFAULT_FOLD_SKILL_BOXPLOT_OUTPUT = Path("outputs/figures/skill_by_fold_boxplot.png")
DEFAULT_ROBUST_HORIZONS_OUTPUT = Path("outputs/tables/robust_horizons_summary.csv")
DEFAULT_VARIANCE_RETENTION_OUTPUT = Path("outputs/tables/variance_retention_summary.csv")


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


def evaluate_persistence_folds(
    y: np.ndarray,
    horizons: list[int],
    min_train_size: int,
    step_size: int,
    dataset_name: str,
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
        for fold_idx, fold in enumerate(folds):
            pred = persistence_forecast(last_observation=y[fold.train_end], horizon=horizon)[-1]
            yt = float(y[fold.test_end])
            yp = float(pred)
            rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_idx,
                    "horizon": horizon,
                    "model": "persistence",
                    "rmse": abs(yt - yp),
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    return rows


def evaluate_ridge_folds(
    y: np.ndarray,
    horizons: list[int],
    min_train_size: int,
    step_size: int,
    n_lags: int,
    alpha: float,
    dataset_name: str,
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

        for fold_idx, fold in enumerate(folds):
            train_mask = origin_all + horizon <= fold.train_end
            if not np.any(train_mask):
                raise ValueError(
                    f"No training samples for horizon={horizon} in fold {fold_idx}."
                )

            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            X_test = y[fold.train_end - n_lags + 1 : fold.train_end + 1].reshape(1, -1)

            if X_test.shape[1] != n_lags:
                raise ValueError(
                    f"Insufficient lag context for horizon={horizon} at fold {fold_idx}."
                )

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)
            yt = float(y[fold.test_end])
            yp = float(ridge.predict(X_test)[0])
            rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_idx,
                    "horizon": horizon,
                    "model": "ridge",
                    "rmse": abs(yt - yp),
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    return rows


def evaluate_sarima_folds(
    y: np.ndarray,
    horizons: list[int],
    min_train_size: int,
    step_size: int,
    dataset_name: str,
    order: tuple[int, int, int] = (1, 1, 0),
) -> list[dict[str, float | int | str]]:
    from statsmodels.tsa.arima.model import ARIMA

    model_name = "arima_" + "".join(str(o) for o in order)
    rows: list[dict[str, float | int | str]] = []
    n_obs = len(y)

    for horizon in horizons:
        folds = generate_rolling_origin_folds(
            n_obs=n_obs,
            min_train_size=min_train_size,
            horizon=horizon,
            step=step_size,
        )
        for fold_idx, fold in enumerate(folds):
            train_y = y[fold.train_start : fold.train_end + 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(train_y, order=order).fit()
            yt = float(y[fold.test_end])
            yp = float(np.asarray(fit.forecast(steps=horizon))[-1])
            rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_idx,
                    "horizon": horizon,
                    "model": model_name,
                    "rmse": abs(yt - yp),
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    return rows


def _fold_rows_to_aggregate(
    fold_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Derive aggregate RMSE rows from per-fold absolute errors.

    Per-fold ``rmse`` values are absolute errors |y_true - y_pred|.
    Aggregate RMSE = sqrt(mean(e_i^2)), which matches the standard formula.
    """
    from collections import defaultdict

    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in fold_rows:
        key = (str(row["dataset"]), str(row["model"]), int(row["horizon"]))
        groups[key].append(float(row["rmse"]))

    result: list[dict[str, float | int | str]] = []
    for (dataset, model, horizon), errors in sorted(groups.items()):
        errors_arr = np.asarray(errors)
        result.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "rmse": float(np.sqrt(np.mean(errors_arr**2))),
                "n_folds": len(errors),
            }
        )
    return result


def run_benchmark(
    config_path: Path,
    metrics_output: Path,
    log_output: Path,
    skill_output: Path,
    horizons_output: Path,
    trajectory_output: Path,
    fold_skill_output: Path,
    fold_dispersion_output: Path,
    robust_horizons_output: Path,
    variance_retention_output: Path,
    skill_plot_output: Path,
    fold_skill_boxplot_output: Path,
    n_lags: int,
    ridge_alpha: float,
    arima_order: tuple[int, int, int] = (1, 1, 0),
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

    fold_rows = evaluate_persistence_folds(
        y=y,
        horizons=horizons,
        min_train_size=min_train_size,
        step_size=step_size,
        dataset_name=dataset["name"],
    )
    fold_rows.extend(
        evaluate_ridge_folds(
            y=y,
            horizons=horizons,
            min_train_size=min_train_size,
            step_size=step_size,
            n_lags=n_lags,
            alpha=ridge_alpha,
            dataset_name=dataset["name"],
        )
    )
    fold_rows.extend(
        evaluate_sarima_folds(
            y=y,
            horizons=horizons,
            min_train_size=min_train_size,
            step_size=step_size,
            dataset_name=dataset["name"],
            order=arima_order,
        )
    )

    metrics_df = pd.DataFrame(_fold_rows_to_aggregate(fold_rows))
    metrics_df = metrics_df.sort_values(["dataset", "horizon", "model"]).reset_index(drop=True)
    n_folds_by_horizon = (
        metrics_df.loc[metrics_df["model"] == "persistence", ["horizon", "n_folds"]]
        .sort_values("horizon")
    )
    n_folds_summary = ",".join(
        f"{int(row.horizon)}:{int(row.n_folds)}" for row in n_folds_by_horizon.itertuples(index=False)
    )
    arima_model_name = "arima_" + "".join(str(o) for o in arima_order)

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

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_skill_df = build_fold_skill_table(fold_metrics_df)
    fold_skill_output.parent.mkdir(parents=True, exist_ok=True)
    fold_skill_df.to_csv(fold_skill_output, index=False)
    fold_dispersion_df = build_fold_dispersion_summary(fold_skill_df)
    fold_dispersion_output.parent.mkdir(parents=True, exist_ok=True)
    fold_dispersion_df.to_csv(fold_dispersion_output, index=False)
    robust_horizons_df = build_robust_horizons_summary(fold_dispersion_df)
    robust_horizons_output.parent.mkdir(parents=True, exist_ok=True)
    robust_horizons_df.to_csv(robust_horizons_output, index=False)
    variance_df = build_variance_retention_table(fold_metrics_df, skill_df)
    variance_retention_output.parent.mkdir(parents=True, exist_ok=True)
    variance_df.to_csv(variance_retention_output, index=False)
    plot_skill_by_horizon(skill_df, skill_plot_output, dataset_name=dataset["name"])
    plot_fold_skill_boxplot(
        fold_skill_df,
        fold_skill_boxplot_output,
        dataset_name=dataset["name"],
        exclude_baseline=True,
    )


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
        f"models=persistence,ridge,{arima_model_name}",
        f"ridge_alpha={ridge_alpha}",
        f"n_lags={n_lags}",
        f"arima_order={arima_order}",
        f"metrics_output={metrics_output.resolve()}",
        f"skill_table_path={skill_output.resolve()}",
        f"horizons_summary_path={horizons_output.resolve()}",
        f"trajectory_summary_path={trajectory_output.resolve()}",
        f"fold_skill_path={fold_skill_output.resolve()}",
        f"fold_dispersion_path={fold_dispersion_output.resolve()}",
        f"robust_horizons_path={robust_horizons_output.resolve()}",
        f"variance_retention_path={variance_retention_output.resolve()}",
        f"skill_plot_path={skill_plot_output.resolve()}",
        f"fold_skill_boxplot_path={fold_skill_boxplot_output.resolve()}",
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
    parser.add_argument("--fold-skill-output", type=Path, default=DEFAULT_FOLD_SKILL_OUTPUT)
    parser.add_argument("--fold-dispersion-output", type=Path, default=DEFAULT_FOLD_DISPERSION_OUTPUT)
    parser.add_argument("--robust-horizons-output", type=Path, default=DEFAULT_ROBUST_HORIZONS_OUTPUT)
    parser.add_argument("--variance-retention-output", type=Path, default=DEFAULT_VARIANCE_RETENTION_OUTPUT)
    parser.add_argument("--skill-plot-output", type=Path, default=DEFAULT_SKILL_PLOT_OUTPUT)
    parser.add_argument("--fold-skill-boxplot-output", type=Path, default=DEFAULT_FOLD_SKILL_BOXPLOT_OUTPUT)
    parser.add_argument("--n-lags", type=int, default=7)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--arima-p", type=int, default=1)
    parser.add_argument("--arima-d", type=int, default=1)
    parser.add_argument("--arima-q", type=int, default=0)
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
        fold_skill_output=args.fold_skill_output,
        fold_dispersion_output=args.fold_dispersion_output,
        robust_horizons_output=args.robust_horizons_output,
        variance_retention_output=args.variance_retention_output,
        skill_plot_output=args.skill_plot_output,
        fold_skill_boxplot_output=args.fold_skill_boxplot_output,
        n_lags=args.n_lags,
        ridge_alpha=args.ridge_alpha,
        arima_order=(args.arima_p, args.arima_d, args.arima_q),
    )
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
