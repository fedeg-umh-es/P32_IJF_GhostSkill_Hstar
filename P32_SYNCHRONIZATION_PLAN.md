# P32 Synchronization Plan — Rescue Audit for Future Environmental Forecasting Line

## Executive decision

P32 is the conceptual control repository for the future H* / Ghost Skill / Dynamic Fidelity research line. It should not absorb data, code, or historical manuscripts at this stage.

The recoverable architecture is role-based:

| Role | Repository / asset | Decision |
|---|---|---|
| Conceptual core | `fedeg-umh-es/P32_IJF_GhostSkill_Hstar` | A — nucleus |
| Empirical core | `fedeg-umh-es/Hstar_PM10_PM25_Madrid_Valencia` | A — empirical nucleus |
| Methodological support | `fedeg-umh-es/p34-variance-retention-api` | B — reusable diagnostic library |
| Single-case historical asset | `fedeg-umh-es/P33_variance_collapse` | C — quarry/archive |
| Technical variance-retention module | `fedeg-umh-es/P33_variance_retention_paper` | B/C — support or quarry |
| Hourly/event-based support | `fedeg-umh-es/e2-met-validation` | B/C — pending support |
| Historical H* documentation | `fedeg-umh-es/PM10-Horizons-Diagnostic` | C — methodological quarry |
| Rank-reversal antecedent | `fedeg-umh-es/madrid-pm10-rank-reversal` | C — historical antecedent |

## Non-negotiable working rule

Do not write a new manuscript yet.

The next stage is forensic consolidation:

1. define repository roles;
2. audit empirical outputs in the Madrid-Valencia repository;
3. verify row-level prediction availability;
4. normalize empirical outputs to the variance-retention API contract;
5. only then decide which claims are defensible.

## Strategic rationale

The previous single-case Ghost Skill / E2-MET manuscript is provisionally closed as a submission target. Its desk-reject sequence is interpreted as editorial evidence, not as a technical failure. The main lesson is that future work in this line needs broader empirical support and explicit traceability before any Q1-oriented manuscript is drafted.

The new line must not be built by cosmetic reframing of old submissions. It must be built from recoverable assets with explicit data-script-output-claim traceability.

## Canonical repository roles

### 1. P32_IJF_GhostSkill_Hstar

Function: conceptual control repository.

Use for:

- dominant research story;
- claim discipline;
- H* / Ghost Skill / Dynamic Fidelity definitions;
- synchronization plan;
- claim-to-evidence matrix;
- overlap control with already submitted manuscripts.

Do not use for:

- raw data storage;
- heavy outputs;
- duplicated pipelines;
- exploratory notebooks without traceability.

### 2. Hstar_PM10_PM25_Madrid_Valencia

Function: primary empirical evidence repository.

Use for:

- Madrid + Valencia evidence;
- PM10 + PM2.5 evidence;
- multi-station / multi-pollutant validation;
- row-level predictions if available;
- empirical audit of H*, skill, variance retention, and dynamic fidelity.

Critical verification required:

The repository must be audited to determine whether outputs contain at least:

- `dataset`
- `city`
- `pollutant`
- `station`
- `model`
- `horizon`
- `fold`
- `origin_date`
- `train_end`
- `test_start`
- `date`
- `y_true`
- `y_pred`

If row-level predictions are not available, P32 cannot yet support a robust Q1-level dynamic-fidelity claim.

### 3. p34-variance-retention-api

Function: reusable methodological support.

Use for:

- prediction-table contract;
- variance retention / alpha;
- SkillVP calculation;
- collapse / inflation / near-ideal flags;
- common diagnostic tables.

Target prediction contract:

```text
dataset, model, fold, origin_date, train_end, test_start, horizon, date, y_true, y_pred
```

Target diagnostic output:

```text
dataset, model, horizon, skill, alpha, skill_vp, collapse_flag, inflation_flag, near_ideal_flag
```

### 4. Historical and quarry repositories

The following repositories are not current submission nuclei:

- `P33_variance_collapse`
- `P33_variance_retention_paper`
- `e2-met-validation`
- `PM10-Horizons-Diagnostic`
- `madrid-pm10-rank-reversal`

They may provide:

- methodology blocks;
- diagnostic scripts;
- figure design;
- event/exceedance logic;
- historical decisions;
- cautionary examples of editorial over-narrowing.

They must not provide unverified general claims.

## Forensic traceability matrix required

Every future claim must be traceable as:

```text
claim -> table/figure -> csv/parquet -> script -> commit -> dataset
```

No claim should be inherited from an old manuscript unless this chain can be reconstructed.

## Go / no-go thresholds by asset class

### A — nucleus

An asset can remain A only if it can support at least one of these:

- a complete conceptual framework with explicit claim boundaries; or
- empirical outputs sufficient to regenerate at least one main table and one main figure; or
- a documented protocol with reproducible row-level predictions.

### B — support

An asset can remain B if it provides reusable code, metrics, schemas, tests, or documented intermediate outputs, even if it is not independently publishable.

### C — quarry

An asset is C if it has useful text, figures, scripts, or methodological decisions, but its claims are not independently defensible or its editorial framing has already proved weak.

### D — discard / freeze

An asset is D if it lacks recoverable data, lacks runnable code, lacks traceability, duplicates stronger assets, or introduces excessive editorial risk.

## Phase plan

### Phase 0 — Freeze and synchronize

Actions:

1. Maintain P32 as conceptual control repository.
2. Do not create new repositories.
3. Do not move data between repositories.
4. Do not resume tactical resubmission of single-case manuscripts.
5. Use `AUDIT_MASTER_TABLE.csv` as the current asset inventory.

Exit condition:

- repository roles documented;
- all known assets classified A/B/C/D;
- no active manuscript writing from inherited material.

### Phase 1 — Empirical output audit of Madrid-Valencia

Target repository:

```text
fedeg-umh-es/Hstar_PM10_PM25_Madrid_Valencia
```

Audit questions:

1. Are row-level predictions available?
2. Do they include `y_true` and `y_pred`?
3. Are folds and horizons explicit?
4. Are dates/origins/train-test boundaries explicit?
5. Are baseline predictions available?
6. Can at least one table and one figure be regenerated?
7. Are outputs tied to scripts and commits?

Exit condition:

A table can be produced with:

```text
dataset | city | pollutant | station | model | horizon | fold | origin_date | train_end | test_start | date | y_true | y_pred
```

### Phase 2 — Normalize to diagnostic contract

Target library:

```text
fedeg-umh-es/p34-variance-retention-api
```

Actions:

1. Map Madrid-Valencia outputs to the prediction contract.
2. Verify `test_start > train_end` for all rows.
3. Compute variance retention / alpha.
4. Compute SkillVP as an auxiliary diagnostic.
5. Generate internal diagnostic summaries.

Exit condition:

A diagnostic table exists with:

```text
dataset | city | pollutant | station | model | horizon | skill | alpha | skill_vp | collapse_flag
```

### Phase 3 — Claim recovery audit

Actions:

1. Build a claim inventory.
2. Mark each claim as recoverable, recoverable with reservations, or not recoverable.
3. Check overlap with submitted manuscripts.
4. Remove all claims not supported by traceable outputs.

Exit condition:

Every retained claim has a complete traceability chain.

## Claims that are potentially recoverable

| Claim | Status | Condition |
|---|---|---|
| Persistence-relative skill and dynamic fidelity can decouple | Recoverable with reservations | Needs multi-series evidence from Madrid-Valencia |
| Variance retention is useful as an auxiliary diagnostic | Recoverable | Needs row-level predictions and clear definition |
| SkillVP is an auxiliary adjustment, not a new standard metric | Recoverable | Must remain secondary to skill and variance retention |
| Single-case Ghost Skill evidence is illustrative | Recoverable only as historical evidence | Must not be framed as general result |
| Multi-series evidence reduces single-case fragility | Recoverable | Requires verified outputs across stations/pollutants |
| Rank reversal under rolling-origin is a useful warning pattern | Recoverable with high caution | Needs replication beyond Madrid single-case |

## Overlap control

Before drafting anything from these assets, explicitly check overlap with:

- already submitted Madrid/Valencia operational predictability manuscripts;
- E2-MET / Ghost Skill single-case material;
- P33 variance-retention drafts;
- event/exceedance analyses from hourly PM10 repositories.

Overlap dimensions:

- data;
- time period;
- models;
- horizons;
- metrics;
- figures;
- narrative claims;
- wording.

## No-do list

1. Do not submit the old single-case Ghost Skill manuscript again by changing only the target journal.
2. Do not write a new paper before the Madrid-Valencia row-level output audit.
3. Do not merge repositories physically at this stage.
4. Do not create additional repositories for this line.
5. Do not promote SkillVP as a universal metric.
6. Do not mix daily and hourly evidence in a single claim without explicit design separation.
7. Do not mix RMSE-H* and MAE-H* without declaring the primary metric.
8. Do not inherit claims from old manuscripts unless the full traceability chain exists.
9. Do not treat README-level documentation as empirical verification.
10. Do not ignore overlap with manuscripts already submitted or under review.

## Immediate next action

Audit `fedeg-umh-es/Hstar_PM10_PM25_Madrid_Valencia` for row-level prediction outputs and traceability.

No manuscript drafting should start until this audit is complete.
