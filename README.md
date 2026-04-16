# P32 IJF Ghost Skill Hstar

## Provisional title
Operational Multi-Horizon Forecast Evaluation Under Leakage-Free Rolling-Origin: H*, Skill_VP, and Variance-Retention Diagnostics

## Central research question
When multi-horizon forecasts are evaluated under leakage-free rolling-origin design, to what extent can RMSE-based rankings be operationally misleading, and how much additional clarity is gained by combining forecast skill horizons with variance-retention diagnostics?

## Dominant story
This project develops an operational evaluation framework for multi-horizon forecasting under leakage-free rolling-origin evaluation, separating statistical skill from dynamic realism through joint profiling of persistence-relative skill and variance retention.

## Recalibrated claims
1. H* is not introduced as a new metric but as an operationalization of the forecast skill horizon under rolling-origin, leakage-free evaluation against persistence.
2. Skill_VP is not positioned as a universal replacement for skill scores but as an auxiliary diagnostic metric.
3. Ghost Skill is not framed as a discovery of the underlying mathematical mechanism but as an empirical finding in PM10 and machine-learning forecasting settings.
4. The main contribution is an integrated operational evaluation protocol combining rolling-origin evaluation, train-only preprocessing, persistence baselines, and joint profiling of skill and variance retention.
5. LRS is framed as an applied instrument for auditing temporal leakage risk.

## Planned datasets
- Urban air-quality series with operational multi-horizon forecasting relevance.
- PM10 and related benchmark settings where persistence is meaningful and rolling-origin evaluation is feasible.
- Cross-dataset extensions only when provenance and temporal integrity are documented.
- No dataset is assumed to be present until documented under `data/raw/` and referenced by config.

## Minimum baselines
- Naive persistence.
- Seasonal persistence when the sampling structure justifies it.
- ARIMA/SARIMA.
- One tabular boosting family model for lag-based forecasting.
- Deep learning models are kept available as controlled comparators, not default claims drivers.

## Planned experiments
- `exp01_core_benchmark`: core leakage-free benchmark with persistence-relative skill curves.
- `exp02_variance_retention`: variance-retention profiling and dynamic-collapse flags.
- `exp03_ranking_reversal`: RMSE ranking reversals versus operational skill interpretation.
- `exp04_cross_dataset`: external consistency checks across datasets with aligned protocol.
- `exp05_ablation`: targeted ablations on preprocessing, baselines, and audit components.

## Non-negotiable methodological principles
- Rolling-origin evaluation only.
- Train-only preprocessing at every fold.
- Explicit persistence baseline for every horizon.
- Horizon-wise reporting, not pooled-only reporting.
- No hidden future information in scaling, imputation, feature engineering, or selection.
- Diagnostics must distinguish error reduction from variance collapse.
- Manuscript claims must remain consistent with available evidence.

## Traceability rules
- Raw data stay in `data/raw/`; derived artifacts move to `data/interim/` and `data/processed/`.
- All configurable choices should be represented in `configs/`.
- Manuscript-facing tables and figures must be reproducible from scripts, not edited by hand.
- Outputs belong in `outputs/` and should be disposable/rebuildable.
- Tests must cover core evaluation primitives before empirical claims are advanced.
- No notebook is treated as the canonical source of logic.

## Next operational step
Document the first target dataset in `configs/datasets/`, implement the corresponding loader in `src/data/loaders.py`, and run a smoke benchmark through the rolling-origin engine before drafting results text.
