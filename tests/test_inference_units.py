"""Physical-unit rejection must occur before classification; no synthetic study labels."""

import json
from pathlib import Path

import pytest

from irrigation.data import FEATURES
from irrigation.predict import validate_payload


def sample():
    return json.loads((Path(__file__).parents[1] / "configs/example-input.json").read_text())


@pytest.mark.parametrize(
    "name,value",
    [
        ("soil10_lag1", 1.1),
        ("soil10_lag7", -0.1),
        ("RH_DAILY_AVG_mean7", 101),
        ("P_DAILY_CALC_mean7", -1),
        ("SOLARAD_DAILY", -1),
        ("T_DAILY_AVG", 100),
        ("T_DAILY_AVG_mean7", -273),
        ("doy_sin", 2),
        ("doy_cos", True),
    ],
)
def test_wrong_unit_or_physical_feature_rejected(name, value):
    with pytest.raises(ValueError):
        validate_payload(sample() | {name: value}, FEATURES)


def test_missing_values_remain_explicit_and_real_example_valid():
    payload = sample()
    validate_payload(payload, FEATURES)
    validate_payload(payload | {"soil10_lag1": None, "RH_DAILY_AVG": None}, FEATURES)


def test_seasonal_pair_must_be_coherent():
    with pytest.raises(ValueError, match="same day"):
        validate_payload(sample() | {"doy_sin": 0.0, "doy_cos": 0.0}, FEATURES)
