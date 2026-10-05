"""Trusted local model inference; no irrigation prescription."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict(payload: dict, artifact: Path) -> dict:
    state = joblib.load(artifact)  # Load only the artifact trained locally by this repository.
    if set(payload) != set(state["features"]):
        raise ValueError("Exact feature schema required")
    for name, value in payload.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
        ):
            raise ValueError(f"Nonfinite or nonnumeric feature: {name}")
        if name.startswith("SOIL_MOISTURE") and value is not None and not 0 <= value <= 1:
            raise ValueError("Soil moisture must be m3/m3 in [0,1]")
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
