# Release gate: acceptance criteria for v1.0

No public claim of a working system until every row passes. Status reflects the
synthetic benchmark build of 2026-11-16; the two open rows are the real-data gate.

| # | Criterion | How it is checked | Status |
|---|---|---|---|
| 1 | All tests pass | `pytest -q` (works from a bare unzip via `conftest.py`, or after `pip install -e ".[dev]"`) | Passing (13) |
| 2 | Synthetic pipeline completes reproducibly | `make run` twice; seeds fixed in `config.yaml`; metrics identical | Passing |
| 3 | Dashboard loads without missing files | `streamlit run app/streamlit_app.py` straight after `make run`; headless smoke test in CI | Passing |
| 4 | At least one real Australian dataset runs through the complete pipeline | `make run-real` on East Tennant (gravity P201901 + GA magnetics + NTGS geochem/deposits) | **Open** |
| 5 | Synthetic and real results clearly distinguished | Report and app header carry `data source:`; README quotes real numbers separately | Passing (synthetic labelled; real pending) |
| 6 | Every public map shows source, date and uncertainty | Report header has region/source/date; `prospectivity_std` layer + per-target CV spread shipped | Passing |
| 7 | Deployable without retraining on the host | App reads `outputs/` only; `app/requirements.txt` has no torch/xgboost; Docker builds outputs at image build | Passing |
| 8 | A geologist can export a target report and reproduce its evidence | Targets GPKG/GeoJSON + per-target SHAP explanation + REPORT.md + seeds in config | Passing |

Open items 4 (and the real-data half of 5) are Phase 8 of the roadmap. When they
pass, tag `v1.0` and update the README results table with the real-data numbers.
