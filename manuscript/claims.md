# Recalibrated Claims

## Claim set for manuscript drafting

1. **Operationalization rather than invention of H\***  
   The paper does not introduce H* as a fundamentally new forecast metric. Instead, it operationalizes the forecast skill horizon within a leakage-free rolling-origin framework, using persistence as the explicit baseline and horizon-wise skill curves as the decision object.

2. **Diagnostic role of Skill_VP**  
   Skill_VP is not proposed as a universal replacement for conventional skill scores. It is presented as an auxiliary diagnostic quantity that becomes informative when error-based skill must be interpreted jointly with dynamic realism and variance retention.

3. **Empirical framing of Ghost Skill**  
   Ghost Skill is not claimed as a discovery of the mathematical mechanism behind forecast degradation. It is framed as an empirical pattern observed in PM10 and machine-learning forecasting settings, where apparent gains in RMSE may coexist with implausible attenuation of predictive variability.

4. **Primary contribution as integrated protocol**  
   The main contribution is an integrated operational evaluation protocol that combines leakage-free rolling-origin assessment, train-only preprocessing, explicit persistence baselines, and joint horizon-wise profiling of skill and robustness. The paper’s value lies in showing how this protocol changes the interpretation of operational predictability relative to aggregate summaries alone.

5. **LRS as an audit instrument**  
   The Leakage Risk Score (LRS) is introduced as an applied audit instrument for temporal-validation design. Its role is to structure leakage-risk inspection across preprocessing, feature construction, and validation boundaries, not to certify validity by itself.

---

## Empirically grounded claims (exp01 — pm10_example, Ridge vs ARIMA(1,1,0))

These claims are backed by fold-wise evidence from `outputs/tables/robust_horizons_summary.csv` and `outputs/tables/skill_fold_dispersion_summary.csv`. They are not general theoretical claims — they are scoped to this dataset and these models.

**Claim E1 — Aggregate horizon overestimates operational predictability**  
Ridge achieves H* = 7 under aggregate skill (1 − RMSE_model/RMSE_persistence). However, under fold-wise criteria — positive median skill, majority of folds positive, non-negative Q25 — the robust operational horizon is h = 6 across all four metrics. The one-step gap between H* = 7 and robust horizon = 6 is not a rounding artifact; h = 7 fails every fold-wise robustness criterion applied.

> Manuscript phrase: "Aggregate horizon summaries can overstate operational predictability when fold-wise robustness is ignored."

**Claim E2 — Ridge maintains robust skill advantage up to h = 6**  
At horizons 1–6, Ridge shows positive median fold-wise skill, a majority of folds with positive skill, and non-negative Q25 under the current fold-wise criteria. The resulting profile is non-monotone, with a peak around h = 3–4, indicating that forecasting gains are not uniformly distributed across horizons but remain consistently supported up to h = 6 in this experiment.

> Manuscript phrase: "Under the fold-wise robustness criteria used here, Ridge reaches a robust operational horizon of 6, one step shorter than the aggregate H* = 7."

**Claim E3 — Late-horizon deterioration is not a Ridge-specific pathology**  
ARIMA(1,1,0) fails fold-wise robustness criteria much earlier than Ridge. Since late-horizon deterioration is also observed under this classical temporal model, the evidence does not support interpreting the h = 6–7 decline as a Ridge-specific artifact. Instead, it is more consistent with late-horizon difficulty in the forecasting task under this evaluation setting.

> Manuscript phrase: "ARIMA(1,1,0) fails robust fold-wise criteria much earlier than Ridge, supporting the interpretation that late-horizon deterioration is not explained solely by Ridge-specific modeling choices."
