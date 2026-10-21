"""Shared pytest fixtures."""

import numpy as np
import pytest

from pandoravisibility import Visibility


@pytest.fixture(autouse=True)
def empty_trace_field(monkeypatch):
    """Give every target an empty NIRDA star field unless a test sets its own.

    The trace check (on by default) would otherwise need the gaiaoffline
    database, which CI does not have. With no neighbors every roll is clean,
    so the roll search returns exactly what it did before the check existed.
    Tests of the check itself replace ``Visibility._trace_field`` again.
    """
    empty = {
        "target_source_id": 0,
        "target_jmag": np.nan,
        "xi": np.array([]),
        "eta": np.array([]),
        "jmag": np.array([]),
        "flux_ratio": np.array([]),
    }
    monkeypatch.setattr(Visibility, "_trace_field", lambda self, coord, time: empty)
