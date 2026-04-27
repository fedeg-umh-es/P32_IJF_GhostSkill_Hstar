# Audit — Hstar_PM10_PM25_Madrid_Valencia empirical outputs

## Executive verdict

The Madrid-Valencia repository is a valid empirical nucleus for the future P32 line, but it is not yet ready to feed the variance-retention / SkillVP diagnostic layer without a normalization step.

The repository contains a strong declared protocol and a script that generates row-level rolling-origin predictions. However, the stored remote repository does not expose committed prediction CSVs or final figure outputs. The reproducibility status is therefore **code-reconstructable but output-not-audited**.

Current classification: **A-minus empirical nucleus**.

It remains A only if row-level outputs can be regenerated locally and normalized to the diagnostic contract used by `p34-variance-retention-api`.

## What was audited remotely

Target repository:

```text
fedeg-umh-es/Hstar_PM10_PM25_Madrid_Valencia
```

Inspected assets:

- `README.md`
- `00_ANCLA_PROYECTO.md`
- `.gitignore`
- `main.tex`
- `code/run_rolling_skill.py`
- `code/figure_01_skill_curves.py`
- `code/figure_02_hstar_heatmap.py`
- `code/figure_03_city_comparison.py`
- `code/build_daily_pm_series.py`
- `code/query_eea_stations_v2.py`
- `code/convert_madrid_to_long.py`
- `code/convert_wide_to_long.py`

No code was executed during this remote audit.

## Repository role

The repository should be treated as the **primary empirical evidence repository** for P32, not as the conceptual controller and not as the diagnostic library.

It is valuable because it already declares:

- Madrid + Valencia;
- PM10 + PM2.5;
- daily frequency;
- h = 1..7 days;
- rolling-origin expanding-window validation;
- train-only preprocessing;
- simple persistence, seasonal persistence, SARIMA, and XGBoost;
- 46 prepared series and 43 final valid series.

## Critical finding 1 — outputs are generated but not committed

The `.gitignore` file excludes:

```text
data_pm/
data_pm_daily/
data_processed/
results/
```

Therefore the remote repository does not contain the main generated empirical artifacts needed for forensic verification.

Implication:

- existing committed scripts can regenerate outputs;
- remote GitHub cannot currently prove that manuscript tables/figures match generated CSVs;
- the P32 line cannot yet inherit numerical claims from this repository without local regeneration or artifact publication.

## Critical finding 2 — row-level predictions exist in script output, but not in P32 diagnostic contract

`code/run_rolling_skill.py` writes per-series prediction files named:

```text
rolling_origin_predictions_{pollutant}_{city}_{station}.csv
```

The prediction file has this wide format:

```text
origin_date, horizon, actual, persist_simple, persist_seasonal, sarima, boosting
```

This is useful but does not directly satisfy the P32 / p34 diagnostic contract.

Missing or implicit fields:

| Required field | Status in current script output |
|---|---|
| dataset | missing; reconstructable from file name |
| city | missing inside file; reconstructable from file name |
| pollutant | missing inside file; reconstructable from file name |
| station | missing inside file; reconstructable from file name |
| model | wide columns; must be melted to long format |
| fold | missing; can be generated as origin index per series |
| origin_date | present |
| train_end | missing; equals origin_date by protocol |
| test_start | missing; equals origin_date + 1 day |
| date | missing; equals origin_date + horizon days |
| y_true | present as `actual` |
| y_pred | present in model-specific wide columns |

Conclusion:

The repository is close to compatible with the variance-retention API, but needs an adapter script.

## Critical finding 3 — protocol / implementation mismatch to verify

The manuscript and anchor describe a rolling-origin expanding-window protocol. The script implements expanding training windows, but the defaults are:

```text
min_train_days = 365
origin_stride = 7
max_origins = 120
hmax = 7
```

This means the practical implementation is weekly-origin evaluation capped at the last 120 origins unless overridden.

The manuscript text should be checked for consistency if it states or implies a one-day chronological advance or `T - w0 - 7` folds per horizon.

Risk:

- not a fatal methodological flaw;
- but a Q1 reviewer could flag inconsistency between written protocol and executable protocol.

Required resolution:

Document explicitly whether the canonical experiment uses:

- daily origins, or
- weekly stride origins, or
- capped last-120 rolling origins.

## Critical finding 4 — daily aggregation coverage threshold mismatch

The README and manuscript state that daily means require at least 18 valid hourly observations.

However, `code/build_daily_pm_series.py` currently resamples hourly values to daily mean and drops only all-NaN days. It does not enforce a minimum count of 18 valid hourly observations per day.

Risk:

High for strict reproducibility and reviewer scrutiny.

This must be resolved before using the repository as a Q1 empirical nucleus.

Required resolution:

Either:

1. update the daily aggregation script to enforce `count >= 18`; or
2. revise the manuscript/README/ancla to remove the 18-hour threshold claim; or
3. document that upstream data already pre-filtered this condition, if true and verifiable.

## Critical finding 5 — train-only preprocessing is partially supported, but text overstates some operations

The rolling script uses train-only imputation inside each origin via `impute_train_only(y_train_raw)`, so the core claim is partly supported.

However, the manuscript/ancla mention feature scaling and possibly broader train-only preprocessing. The inspected script does not implement feature scaling for the XGBoost lag model.

Risk:

Medium. It is safer to write that missing-value imputation and lag construction are train-window based, rather than implying scaling if no scaling exists.

## Critical finding 6 — figure scripts depend on generated results directories

The figure scripts read from:

```text
results/
results_valencia/
```

They generate:

```text
figures/fig01_skill_curves.pdf/png
figures/fig02_hstar_heatmap.pdf/png
figures/fig03_city_comparison.pdf/png
```

Because `results/` is gitignored and generated figures were not verified as committed artifacts, the figures are reproducible in principle but not forensically traceable from the remote repository alone.

## Minimum adapter needed for P32

Create an adapter script later, not now, that transforms each wide prediction CSV into long diagnostic format:

```text
dataset, city, pollutant, station, model, fold, origin_date, train_end, test_start, horizon, date, y_true, y_pred
```

Mapping:

```text
dataset = pollutant + '_' + city + '_' + station
city = parsed from filename
pollutant = parsed from filename
station = parsed from filename
model = one of persist_simple, persist_seasonal, sarima, boosting
fold = sequential index by origin_date within dataset
origin_date = origin_date
train_end = origin_date
test_start = origin_date + 1 day
horizon = horizon
date = origin_date + horizon days
y_true = actual
y_pred = value from selected model column
```

## Go / no-go decision

### Go conditions

This asset can remain A-nucleus if all are satisfied:

1. regenerate `results/` and `results_valencia/` locally;
2. verify all 43 expected valid series exist;
3. verify prediction CSVs exist for each valid series;
4. confirm row-level predictions contain non-null `actual` and model predictions;
5. normalize outputs to the p34 diagnostic contract;
6. enforce or correct the 18-hour daily coverage threshold claim;
7. resolve the origin-stride / max-origins documentation mismatch;
8. verify that at least one main table and one main figure can be regenerated from scripts.

### No-go conditions

Do not use this asset as empirical nucleus if:

- prediction CSVs cannot be regenerated;
- y_true/y_pred cannot be recovered for all valid series;
- daily aggregation does not match the claimed coverage policy and cannot be corrected;
- manuscript claims rely on figures/tables not traceable to scripts;
- the already-submitted Madrid-Valencia manuscript creates unmanageable overlap.

## Recoverable claims

| Claim | Status | Required evidence |
|---|---|---|
| H* can be estimated across Madrid/Valencia PM10/PM2.5 under rolling-origin validation | Recoverable | regenerated per-series metrics and H* summaries |
| Multi-series evidence reduces single-case fragility | Recoverable | 43 verified series with traceable outputs |
| Persistence is a valid zero-skill reference | Recoverable | per-horizon metrics showing simple persistence skill = 0 by construction |
| SARIMA and XGBoost differ in H*_strict and fragmentation | Recoverable with caution | regenerated hstar summaries and figures |
| Dynamic fidelity / variance retention can be studied from this repo | Not directly recoverable yet | requires adapter to row-level long predictions and p34 calculations |
| 18-hour daily coverage was enforced | Not recoverable from inspected script | requires code change or documentation correction |

## Immediate action list

1. Regenerate outputs locally from raw data.
2. Inspect file counts in `results/` and `results_valencia/`.
3. Confirm expected prediction, metric, and H* files exist per series.
4. Build a manifest with file hashes and row counts.
5. Create an adapter to the p34 prediction contract.
6. Run variance-retention diagnostics only after contract normalization.

## No-do list

1. Do not write a P32 manuscript from this asset yet.
2. Do not inherit numerical claims from `main.tex` without regenerated CSV verification.
3. Do not treat README/ancla statements as empirical proof.
4. Do not mix this daily h=1..7 asset with hourly event-based repositories in the same claim.
5. Do not reuse the Madrid-Valencia dataset for a new manuscript without an explicit overlap matrix against the already submitted AQAH manuscript.
6. Do not promote this repository to final Q1 evidence until the 18-hour coverage issue is resolved.

## Final classification

**A-minus — empirical nucleus under conditional acceptance.**

The repository is strategically valuable and technically close to usable. Its main weakness is not conceptual. The weakness is forensic: outputs are not committed, row-level predictions need normalization, and there are two documentation/code consistency issues that must be resolved before claims can be inherited safely.
