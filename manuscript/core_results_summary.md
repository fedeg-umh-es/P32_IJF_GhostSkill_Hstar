# Core Results Summary — exp01

Dataset: `pm10_example` | Models: persistence (baseline), Ridge, ARIMA(1,1,0) | Horizons: 1–7

---

## Aggregate horizon summary

Source: `outputs/tables/horizons_summary.csv`

| Model | H* (relax) | H* (strict) |
|-------|-----------|------------|
| persistence | 0 | 0 |
| ridge | **7** | **7** |
| arima_110 | 1 | 1 |

---

## Robust horizon summary (fold-wise criteria)

Source: `outputs/tables/robust_horizons_summary.csv`

| Model | last h: positive median | last h: share ≥ 0.50 | last h: share ≥ 0.75 | last h: Q25 ≥ 0 |
|-------|------------------------|---------------------|---------------------|----------------|
| ridge | **6** | **6** | **6** | **6** |
| arima_110 | 2 | 2 | 0 | 0 |

**Key gap:** Ridge H* aggregate = 7, robust horizon = 6. One step, consistent across all fold-wise criteria.

---

## Fold-wise skill dispersion at critical horizons

Source: `outputs/tables/skill_fold_dispersion_summary.csv`

| Model | h | mean skill | std | median | share positive |
|-------|---|-----------|-----|--------|----------------|
| ridge | 5 | 0.527 | 1.52 | 0.846 | 0.924 |
| ridge | 6 | −0.501 | 5.12 | 0.714 | 0.792 |
| ridge | 7 | −0.522 | 2.28 | −0.091 | 0.372 |
| arima_110 | 1 | 0.085 | 0.59 | 0.190 | 0.689 |
| arima_110 | 2 | −0.014 | 0.56 | 0.056 | 0.582 |
| arima_110 | 3 | −0.028 | 0.45 | −0.047 | 0.451 |

Ridge h=6: mean negative, median positive — driven by extreme outlier folds where persistence error is near zero.  
Ridge h=7: mean negative, median negative, minority of folds positive — structural collapse.

---

## Figures

- `outputs/figures/skill_by_horizon.png` — aggregate skill curve (persistence excluded)
- `outputs/figures/skill_by_fold_boxplot.png` — fold-wise distribution by horizon and model

Primary figure for manuscript: **boxplot** (shows dispersion, not just aggregate).

---

## Three claims backed by these results

**E1.** Aggregate H* = 7 overstates operational predictability for Ridge. Fold-wise robust horizon = 6.

**E2.** Ridge maintains consistent fold-wise skill advantage at h = 1–6 under all four criteria simultaneously.

**E3.** ARIMA(1,1,0) collapses at h = 2–3, earlier than Ridge, ruling out a Ridge-specific model failure as the explanation for late-horizon deterioration.

---

## What is not yet claimed

- Whether this pattern generalizes beyond pm10_example.
- Whether the gap H* − robust_horizon is a systematic bias or dataset-specific.
- Formal statistical significance of the horizon difference.
- Ghost skill (variance collapse) — not yet measured on this dataset.
