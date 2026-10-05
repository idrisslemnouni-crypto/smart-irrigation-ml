# Local verification — 5 October 2026

Twenty-seven real NOAA files were acquired, hash verified and processed. Models were actually trained; selection used only validation, before unseen-station test metrics. Five tests passed with the actual artifact. Ruff lint/format and the executed notebook passed. The held-out matrix was visually inspected: 437 true negatives, 13 false alarms, 2 misses and 100 detected below-proxy events.

A separate clean Python 3.12 environment and local Git clone were installed from scoped dependencies. The clone independently downloaded the 27 public NOAA source files and reran feature construction/training/evaluation. Selection, thresholds and all test metrics match; per-day probabilities reproduce within atol=rtol=1e-10. No source cache or trained artifact was copied into the clone. pip check passed and five contract tests passed after reproduction.

Saved local raw data and selected model remain ignored. Source ZIP contains reproducible code, provenance and reports. No actual irrigation labels, water-saving measurement, external deployment or GitHub CI run is claimed. Verify remote CI only after the scheduled daily publication.
