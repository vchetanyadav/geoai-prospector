"""Explainable AI: SHAP values for every pixel, plain-English reasons per target,
and target polygons a geologist can load into QGIS."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
import xarray as xr  # noqa: E402
from rasterio.features import shapes  # noqa: E402
from scipy.ndimage import label  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from ..config import Grid  # noqa: E402
from .fusion import FEATURES, to_matrix  # noqa: E402

log = logging.getLogger(__name__)


def shap_values(model, ds: xr.Dataset, feats: list[str]) -> tuple[np.ndarray, float]:
    """TreeSHAP for every pixel, in log-odds units. Returns ((F, H, W), base value)."""
    X = to_matrix(ds, feats)
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    H, W = ds.sizes["y"], ds.sizes["x"]
    base = float(np.atleast_1d(ex.expected_value)[0])
    return sv.T.reshape(len(feats), H, W).astype("float32"), base


def _fmt(feat: str, value: float) -> str:
    name, unit = FEATURES.get(feat, (feat, ""))
    if not np.isfinite(value):
        return f"{name}: no data"
    if unit == "m":
        return f"{name} = {value / 1000:.1f} km"
    if unit in ("nT/km", "mGal/km"):
        return f"{name} = {value * 1000:.2f} {unit}"
    return f"{name} = {value:.2f} {unit}".rstrip()


def explain_pixel(ds: xr.Dataset, sv: np.ndarray, feats: list[str], r: int, c: int, top=3) -> dict:
    contrib = sv[:, r, c]
    order = np.argsort(-contrib)
    def row(i):
        return feats[i], float(contrib[i]), float(ds[feats[i]].values[r, c])

    drivers = [row(i) for i in order[:top] if contrib[i] > 0]
    against = [row(i) for i in order[::-1][:2] if contrib[i] < 0]
    pct = float(ds["prospectivity_pct"].values[r, c])
    prob = float(ds["prospectivity"].values[r, c])
    text = (f"Score {prob:.2f}; ranked in the top {max(0.1, 100 * (1 - pct)):.1f}% of the area. "
            + ("Main reasons: " + "; ".join(f"{_fmt(f, v)} (+{s:.2f})" for f, s, v in drivers) + "."
               if drivers else "No strong positive drivers.")
            + (" Held back by: " + "; ".join(f"{_fmt(f, v)} ({s:.2f})" for f, s, v in against) + "."
               if against else ""))
    return {"text": text, "drivers": drivers, "against": against, "score": prob, "percentile": pct}


def plots(model, ds: xr.Dataset, feats: list[str], out_dir: Path, n=4000, seed=0):
    X = to_matrix(ds, feats)
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    ex = shap.TreeExplainer(model)
    exp = ex(X[idx])
    exp.feature_names = [FEATURES[f][0] for f in feats]
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("What drives prospectivity (SHAP, log-odds)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    plt.figure()
    shap.plots.bar(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def waterfall(model, ds, feats, r, c, path: Path):
    X = to_matrix(ds, feats)[[r * ds.sizes["x"] + c]]
    ex = shap.TreeExplainer(model)
    e = ex(X)[0]
    e.feature_names = [FEATURES[f][0] for f in feats]
    plt.figure()
    shap.plots.waterfall(e, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def targets(ds: xr.Dataset, sv: np.ndarray, feats: list[str], grid: Grid, top_frac=0.02) -> gpd.GeoDataFrame:
    """Top `top_frac` of the area -> connected polygons, ranked, each with its SHAP story."""
    pct = ds["prospectivity_pct"].values
    mask = pct >= 1 - top_frac
    lab, n = label(mask)
    rows = []
    for geom, val in shapes(lab.astype("int32"), mask=mask, transform=grid.transform):
        k = int(val)
        rr, cc = np.where(lab == k)
        best = np.argmax(ds["prospectivity"].values[rr, cc])
        r, c = int(rr[best]), int(cc[best])
        info = explain_pixel(ds, sv, feats, r, c)
        known = float(ds["dist_known_deposit_m"].values[rr, cc].min())
        peak_std = float(ds["prospectivity_std"].values[r, c]) if "prospectivity_std" in ds else float("nan")
        rows.append({"geometry": shape(geom), "peak_score": info["score"], "peak_std": peak_std,
                     "area_km2": len(rr) * grid.res ** 2 / 1e6,
                     "status": "known deposit" if known <= 1000 else "greenfield",
                     "peak_x": float(grid.x[c]), "peak_y": float(grid.y[r]), "row": r, "col": c,
                     "top_driver": FEATURES[info["drivers"][0][0]][0] if info["drivers"] else "",
                     "explanation": info["text"]})
    gdf = gpd.GeoDataFrame(rows, crs=grid.crs)
    if len(gdf):
        gdf = gdf.dissolve(by=["row", "col"], aggfunc="first").reset_index()
        gdf = gdf.sort_values("peak_score", ascending=False).reset_index(drop=True)
        gdf.insert(0, "rank", np.arange(1, len(gdf) + 1))
    log.info("targets: %d polygons (%d greenfield)", len(gdf), int((gdf.get("status") == "greenfield").sum()))
    return gdf
