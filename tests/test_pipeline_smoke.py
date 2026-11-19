"""Full pipeline on a small, fast configuration."""
import json

from geoai_prospector.config import load_config
from geoai_prospector.pipeline import run


def test_pipeline_end_to_end(tmp_path):
    cfg = load_config()
    cfg["_root"] = str(tmp_path)
    cfg["source"] = "synthetic"
    cfg["grid"]["bounds"] = [520000, 7820000, 570000, 7870000]  # 50 x 50 km
    cfg["grid"]["resolution"] = 1000
    cfg["sources"]["synthetic"].update(n_samples=800, n_deposits=12)
    cfg["structure"]["cnn"].update(epochs=2, stride=16, patch=16)
    cfg["model"].update(n_folds=3, block_size_m=10000)
    cfg["model"]["xgb"]["n_estimators"] = 60
    m = run(cfg)
    out = tmp_path / "outputs"
    for f in ["prospectivity.tif", "prospectivity_uncertainty.tif", "targets.geojson", "metrics.json",
              "REPORT.md", "web/layers.json", "shap_beeswarm.png"]:
        assert (out / f).exists(), f
    assert m["spatial_cv"]["roc_auc"] > 0.5
    assert "uncertainty" in m and m["uncertainty"]["mean_cv_std"] >= 0
    assert json.loads((out / "metrics.json").read_text())["features"]
