from __future__ import annotations

from src.diagnostics.hstar import compute_hstar
from src.diagnostics.lrs import LeakageRiskComponents, leakage_risk_label, leakage_risk_score
from src.diagnostics.skill_vp import summarize_skill_vp
from src.diagnostics.variance import detect_variance_collapse, variance_diagnostic_flags


def test_compute_hstar_strict() -> None:
    assert compute_hstar([0.3, 0.2, -0.1, 0.1], criterion="strict") == 2


def test_compute_hstar_relax() -> None:
    assert compute_hstar([0.3, -0.2, 0.1], criterion="relax") == 3


def test_summarize_skill_vp_labels_collapse() -> None:
    result = summarize_skill_vp(skill=0.2, alpha=0.3)
    assert result.skill_vp < result.skill
    assert "collapse" in result.interpretation


def test_variance_flags_detect_collapse() -> None:
    flags = variance_diagnostic_flags(alpha=0.4)
    assert detect_variance_collapse(0.4) is True
    assert flags.collapse_flag is True
    assert flags.inflation_flag is False


def test_lrs_score_and_label() -> None:
    components = LeakageRiskComponents(0.1, 0.2, 0.3, 0.4)
    score = leakage_risk_score(components)
    assert score == 0.25
    assert leakage_risk_label(score) == "moderate"
