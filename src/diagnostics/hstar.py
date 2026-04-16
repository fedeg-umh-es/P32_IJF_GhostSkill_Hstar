"""Operational forecast skill horizon utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_hstar(skill_by_horizon: Sequence[float], criterion: str = "strict") -> int:
    """Compute H* from a horizon-wise skill sequence.

    Parameters
    ----------
    skill_by_horizon:
        Ordered horizon-wise skill values starting at h=1.
    criterion:
        `strict` returns the last consecutive horizon from h=1 with positive skill.
        `relax` returns the last horizon anywhere with positive skill.
        `nonnegative` uses skill >= 0 instead of skill > 0.
    """
    skill = np.asarray(skill_by_horizon, dtype=float)
    if skill.ndim != 1 or skill.size == 0:
        raise ValueError("skill_by_horizon must be a non-empty 1D sequence.")

    if criterion == "strict":
        hstar = 0
        for value in skill:
            if value > 0.0:
                hstar += 1
            else:
                break
        return hstar

    if criterion == "relax":
        positive = np.where(skill > 0.0)[0]
        return 0 if positive.size == 0 else int(positive[-1] + 1)

    if criterion == "nonnegative":
        valid = np.where(skill >= 0.0)[0]
        return 0 if valid.size == 0 else int(valid[-1] + 1)

    raise ValueError("criterion must be one of: 'strict', 'relax', 'nonnegative'.")
