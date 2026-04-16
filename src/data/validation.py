"""Validation helpers for time series input integrity."""

from __future__ import annotations

import pandas as pd


def assert_monotonic_time_index(df: pd.DataFrame, timestamp_col: str) -> None:
    """Raise if the timestamp column is not strictly increasing."""
    ts = pd.to_datetime(df[timestamp_col], errors="raise")
    if not ts.is_monotonic_increasing:
        raise ValueError(f"{timestamp_col} must be monotonic increasing.")
    if ts.duplicated().any():
        raise ValueError(f"{timestamp_col} contains duplicated timestamps.")
