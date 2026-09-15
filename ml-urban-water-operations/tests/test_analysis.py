"""Small unit tests for deterministic helper logic.

The tests use tiny synthetic fixtures only to test software behaviour. The analysis
itself never substitutes synthetic data for a missing source dataset.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "analysis.py"
spec = importlib.util.spec_from_file_location("analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analysis)


def test_jaccard_flags():
    a = pd.Series([1, 1, 0, 0])
    b = pd.Series([1, 0, 1, 0])
    assert analysis.jaccard_flags(a, b) == pytest.approx(1 / 3)


def test_prepare_base_features_excludes_object_id_and_target():
    df = pd.DataFrame(
        {
            "LSOA_CODE": ["E01017001", "E01017002", "E01018001", "E01018002"],
            "TOTAL_CONSUMPTION": [10.0, 20.0, 30.0, 40.0],
            "ObjectId": [1, 2, 3, 4],
            "context_value": [2.0, 3.0, 5.0, 8.0],
        }
    )
    out = analysis.prepare_base_features(df)
    exogenous = out.attrs["exogenous_numeric"]
    assert "ObjectId" not in exogenous
    assert "TOTAL_CONSUMPTION" not in exogenous
    assert "context_value" in exogenous


def test_leave_one_out_mean_does_not_use_current_target():
    df = pd.DataFrame(
        {
            "LSOA_CODE": ["E01017001", "E01017002", "E01017003"],
            "LSOA_District": ["E010170", "E010170", "E010170"],
            "TOTAL_CONSUMPTION": [10.0, 20.0, 40.0],
        }
    )
    encoded = analysis.make_leave_one_out_training_features(df)
    expected = np.array([30.0, 25.0, 15.0])
    assert np.allclose(encoded["district_mean_consumption"].to_numpy(), expected)


def test_detect_anomalies_rejects_invalid_contamination():
    df = pd.DataFrame(
        {
            "LSOA_CODE": [f"E010170{i:02d}" for i in range(10)],
            "TOTAL_CONSUMPTION": np.arange(10, dtype=float),
            "LSOA_District": ["E010170"] * 10,
            "district_record_count": [10.0] * 10,
        }
    )
    with pytest.raises(ValueError):
        analysis.detect_anomalies(df, ["district_record_count"], 0.5, 42)
