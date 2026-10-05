from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irrigation.data import features
from irrigation.train import evaluate, split_data


def calendar_frame():
    d = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=15), "station": "test"})
    for n in [
        "SOIL_MOISTURE_5_DAILY",
        "SOIL_MOISTURE_10_DAILY",
        "P_DAILY_CALC",
        "T_DAILY_AVG",
        "RH_DAILY_AVG",
        "SOLARAD_DAILY",
    ]:
        d[n] = np.arange(15) / 100
    return d


def test_target_is_next_day_and_rolling_is_past_only():
    d = calendar_frame()
    a = features(d, 1)
    assert a.target.iloc[7] == d.SOIL_MOISTURE_10_DAILY.iloc[8]
    assert a.SOIL_MOISTURE_10_DAILY_mean7.iloc[7] == pytest.approx(
        d.SOIL_MOISTURE_10_DAILY.iloc[1:8].mean()
    )
    changed = d.copy()
    changed.loc[10:, "SOIL_MOISTURE_10_DAILY"] = 0.9
    b = features(changed, 1)
    assert a.SOIL_MOISTURE_10_DAILY_mean7.iloc[7] == b.SOIL_MOISTURE_10_DAILY_mean7.iloc[7]


def test_calendar_gap_rejected():
    with pytest.raises(ValueError):
        features(calendar_frame().drop(index=3))


def test_metric_false_alarm_contract():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([0.9, 0.9, 0.1, 0.1]), 0.5)
    assert m["false_alarms"] == 1 and m["positive_f1"] == 0.5


def test_station_overlap_rejected():
    config = {
        "train_stations": ["x"],
        "holdout_station": "x",
        "train_target_end": "2021-12-31",
        "validation_target_year": 2022,
        "test_target_start": "2023-01-01",
    }
    d = pd.DataFrame(
        {
            "station": ["x"] * 3,
            "target_date": pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="station"):
        split_data(d, config)


def test_actual_model_input_contract():
    import json

    from irrigation.predict import predict

    root = Path(__file__).resolve().parents[1]
    path = root / "models/selected.joblib"
    if not path.exists():
        pytest.skip("Locally trained artifact not in Git")
    sample = json.loads((root / "configs/example-input.json").read_text())
    assert 0 <= predict(sample, path)["next_day_below_soil_proxy_probability"] <= 1
    with pytest.raises(ValueError):
        predict({**sample, "extra": 1}, path)
