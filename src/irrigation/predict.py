"""Trusted local model inference; no irrigation prescription."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def validate_payload(payload: dict, required_features: list[str]) -> None:
    """Validate observed/lag/rolling units before a model can score the proxy."""
    if set(payload) != set(required_features):
        raise ValueError("Exact feature schema required")
    for name, value in payload.items():
        if value is None:
            continue  # Missingness stays explicit for the training-only imputer.
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Nonfinite or nonnumeric feature: {name}")
        if (
            name.startswith("SOIL_MOISTURE") or name.startswith("soil10_lag")
        ) and not 0 <= value <= 1:
            raise ValueError("Current, rolling and lagged soil moisture must be m3/m3 in [0,1]")
        if name.startswith("RH_DAILY_AVG") and not 0 <= value <= 100:
            raise ValueError("Relative humidity must be percent in [0,100]")
        if name.startswith(("P_DAILY_CALC", "SOLARAD_DAILY")) and value < 0:
            raise ValueError("Precipitation and radiation must be nonnegative")
        if name.startswith("T_DAILY_AVG") and not -90 <= value <= 65:
            raise ValueError(
                "Daily/rolling temperature must use Celsius in the supported [-90,65] range"
            )
        if name in {"doy_sin", "doy_cos"} and not -1 <= value <= 1:
            raise ValueError("Seasonal sine/cosine must lie in [-1,1]")
    seasonal = [payload.get(name) for name in ("doy_sin", "doy_cos")]
    if all(value is not None for value in seasonal) and not np.isclose(
        sum(value * value for value in seasonal), 1.0, rtol=0, atol=1e-3
    ):
        raise ValueError("Seasonal sine/cosine must represent the same day")


def predict(payload: dict, artifact: Path) -> dict:
    state = joblib.load(artifact)  # Load only the artifact trained locally by this repository.
    validate_payload(payload, state["features"])
    x = pd.DataFrame([payload], columns=state["features"]).astype(float)
    if state["kind"] == "persistence":
        if payload["SOIL_MOISTURE_10_DAILY"] is None:
            raise ValueError("Current soil measurement required for persistence")
        probability = float(payload["SOIL_MOISTURE_10_DAILY"] < state["soil_proxy_threshold"])
    else:
        probability = float(state["model"].predict_proba(x)[0, 1])
    return {
        "next_day_below_soil_proxy_probability": probability,
        "screening_flag": probability >= state["probability_threshold"],
        "soil_proxy_threshold_m3_m3": state["soil_proxy_threshold"],
        "scope": "Exploratory shallow-soil sensor alert; no crop stress or irrigation recommendation validated",
    }


if __name__ == "__main__":
    print(
        json.dumps(
            predict(
                json.loads(Path("configs/example-input.json").read_text()),
                Path("models/selected.joblib"),
            ),
            indent=2,
        )
    )
