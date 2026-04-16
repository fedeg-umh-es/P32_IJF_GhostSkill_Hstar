# IJF Paper Outline

## 1. Introduction
- Operational relevance of multi-horizon forecasting under real deployment constraints.
- Why low aggregate RMSE does not necessarily imply operationally credible forecasts.
- Motivation for leakage-free rolling-origin evaluation and persistence-relative interpretation.
- Paper objective and contribution preview.

## 2. Related Evaluation Pitfalls
- Temporal leakage and optimistic bias in forecasting pipelines.
- Misleading ranking stability under horizon aggregation.
- Limits of error-only assessment when dynamic realism deteriorates.
- Positioning relative to persistence-based forecast evaluation and operational predictability work.

## 3. Methods
- Leakage-free rolling-origin protocol.
- Train-only preprocessing rules.
- Baselines, candidate models, and horizon-wise evaluation.
- Definition of persistence-relative skill.
- Operationalization of H*.
- Variance-retention diagnostics and Skill_VP.
- Leakage Risk Score as an audit layer.

## 4. Experimental Design
- Dataset selection criteria and provenance rules.
- Forecast horizons, fold construction, and reproducibility controls.
- Experiment matrix for benchmark, ranking reversal, variance-retention, and ablation studies.
- Statistical summaries, visual outputs, and reporting standards.

## 5. Results
- Core benchmark results by dataset, model, and horizon.
- RMSE rankings versus persistence-relative skill interpretation.
- Cases where nominal ranking gains do not translate into operational skill.

## 6. Diagnostic Analysis
- Variance-retention profiles.
- Ghost Skill cases: low error with dynamic collapse.
- Skill_VP interpretation as an auxiliary diagnostic.
- LRS-based audit discussion for sensitive pipelines.

## 7. Discussion
- What constitutes usable operational improvement in multi-horizon forecasting.
- When RMSE rankings remain informative and when they become misleading.
- Implications for applied forecasting practice and benchmark design.

## 8. Limitations
- Dataset dependence and domain specificity.
- Imperfect identifiability of mechanism from empirical diagnostics alone.
- Sensitivity of thresholds in diagnostic interpretation.

## 9. Conclusion
- Summary of the integrated evaluation framework.
- Practical implications for operational forecast assessment.
- Future extensions for broader forecasting domains.
