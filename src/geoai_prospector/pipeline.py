"""End-to-end orchestrator.

    python -m geoai_prospector.pipeline                 # uses config/config.yaml (synthetic by default)
    python -m geoai_prospector.pipeline --source real   # public NT / GA data
    python -m geoai_prospector.pipeline --if-changed    # scheduled mode: only run if upstream data changed
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from .config import Grid, load_config, resolve
from .etl import points, raster, vector
from .export import report, web_layers, write_geotiff
from .features import geochem, structure
from .io import download, synthetic
from .model import explain, fusion, train

log = logging.getLogger("geoai")


def _inputs(cfg: dict, grid: Grid) -> dict:
    if cfg["source"] == "synthetic":
        s = cfg["sources"]["synthetic"]
        paths = synthetic.generate(grid, resolve(cfg, "raw") / "synthetic", s["seed"], s["n_samples"], s["n_deposits"])
        r = cfg["sources"]["real"]  # reuse column specs; synthetic writes the same schema
        return {"paths": paths, "geochem_spec": r["geochem"], "deposit_spec": r["deposits"]}
    download.download_all(cfg)
    r = cfg["sources"]["real"]
    root = Path(cfg["_root"])
    return {"paths": {k: root / v["path"] for k, v in r.items()}, "geochem_spec": r["geochem"],
            "deposit_spec": r["deposits"]}


def structure_recovery(cfg: dict, grid: Grid, st: dict, tol_m=1500.0) -> dict:
    """Synthetic mode only: how well did Model A recover the planted faults?
    precision = share of detected deep-lineament pixels within tol of a planted fault;
    recall = share of planted deep-fault length (inside the grid) within tol of a detection."""
    truth = json.loads((resolve(cfg, "raw") / "synthetic" / "truth.json").read_text())
    X, Y = np.meshgrid(grid.x, grid.y)
    d_any = np.full(grid.shape, np.inf)
    d_deep = np.full(grid.shape, np.inf)
    for f in truth["faults"]:
        n = np.array([-f["d"][1], f["d"][0]])
        d = np.abs((X - f["p"][0]) * n[0] + (Y - f["p"][1]) * n[1])
        d_any = np.minimum(d_any, d)
        if f["deep"]:
            d_deep = np.minimum(d_deep, d)
    det = st["lineament_deep"]
    on_true = d_deep <= grid.res / 2 * 1.5
    return {"deep_lineament_precision": float((d_any[det] <= tol_m).mean()) if det.any() else 0.0,
            "deep_fault_recall": float((st["dist_deep_fault_m"][on_true] <= tol_m).mean()),
            "tolerance_m": tol_m}


def run(cfg: dict) -> dict:
    t0 = time.time()
    grid = Grid.from_config(cfg)
    out = resolve(cfg, "outputs")
    proc = resolve(cfg, "processed")
    log.info("grid %s, %dx%d px @ %.0f m (%.0f km2)", grid.crs.to_string(), grid.width, grid.height, grid.res,
             grid.width * grid.height * grid.res ** 2 / 1e6)
    inp = _inputs(cfg, grid)
    P = inp["paths"]

    # ---- 1. ETL ----
    e = cfg["etl"]
    mag, mag_valid = raster.to_grid(P["magnetics"], grid, e["resampling"], e["fill_max_search_px"])
    grav, grav_valid = raster.to_grid(P["gravity"], grid, e["resampling"], e["fill_max_search_px"])
    valid = np.isfinite(mag) & np.isfinite(grav)
    mag_f = np.where(np.isfinite(mag), mag, np.nanmedian(mag))
    grav_f = np.where(np.isfinite(grav), grav, np.nanmedian(grav))
    layers = {
        "mag_rtp": mag, "grav": grav,
        "mag_tilt": raster.tilt(mag_f, grid.res), "mag_hgm": raster.hgm(mag_f, grid.res),
        "grav_hgm": raster.hgm(grav_f, grid.res), "grav_1vd": raster.vertical_derivative(grav_f, grid.res),
    }
    gc = points.load_geochem(P["geochem"], inp["geochem_spec"], cfg["geochem"]["elements"], grid,
                             cfg["geochem"]["max_missing_frac"])
    gc = points.sample_rasters(gc, {"mag_rtp": mag, "grav": grav})  # spatial join to pixels
    deps = points.load_deposits(P["deposits"], inp["deposit_spec"], grid)

    # ---- 2a. Model A: structure / CV ----
    st = structure.run(mag_f, grav_f, grid.res, cfg)
    layers.update({k: v for k, v in st.items() if not k.startswith("lineament_") and not k.startswith("_")})

    # mapped faults (independent geological evidence, distinct from CV lineaments)
    if cfg["source"] == "synthetic":
        mf = vector.synthetic_mapped_faults(resolve(cfg, "raw") / "synthetic" / "truth.json", grid)
    else:
        fsrc = cfg["sources"]["real"].get("faults")
        mf = vector.real_mapped_faults(Path(cfg["_root"]) / fsrc["path"], fsrc, grid) if fsrc else None
    if mf is not None:
        layers["dist_mapped_fault_m"] = mf

    # ---- 2b. Model B: geochemistry ----
    scored, gl = geochem.run(gc, grid, cfg)
    layers.update(gl)

    # ---- 3. Fusion + labels ----
    for k in list(layers):
        if not k.startswith("_") and layers[k].dtype.kind == "f":
            layers[k] = np.where(valid, layers[k], np.nan)
    ds = fusion.build_stack(grid, layers)
    ds["valid"] = (("y", "x"), valid)
    m = cfg["model"]
    ds = fusion.add_labels(ds, grid, deps["row"].values, deps["col"].values, m["label_buffer_m"],
                           m["negative_exclusion_m"])
    ds["label"] = ds["label"].where(ds["valid"], -1).astype("int8")

    # ---- 4. Train, validate, predict ----
    scv = train.cross_validate(ds, cfg, spatial=True, return_map=True)
    oof_map = np.where(valid, scv.pop("_oof_map"), np.nan)
    std_map = np.where(valid, scv.pop("_std_map"), np.nan).astype("float32")
    metrics = {"spatial_cv": scv,
               "spatial_cv_nobuffer": train.cross_validate(ds, cfg, spatial=True, buffer_m=0),
               "random_cv": train.cross_validate(ds, cfg, spatial=False),
               "ablation": train.ablation(ds, cfg)}
    model, feats, prob, pct = train.fit_predict(ds, cfg)
    prob = np.where(valid, prob, np.nan)
    pct = np.where(valid, pct, np.nan)
    ds["prospectivity"] = (("y", "x"), prob)
    ds["prospectivity_pct"] = (("y", "x"), pct)
    # held-out capture efficiency (from spatial-CV predictions, not the final in-sample model)
    metrics["capture"] = train.capture_efficiency(oof_map, deps["row"].values, deps["col"].values)
    ds["prospectivity_oof"] = (("y", "x"), oof_map.astype("float32"))
    ds["prospectivity_std"] = (("y", "x"), std_map)
    metrics["uncertainty"] = {"mean_cv_std": float(np.nanmean(std_map)),
                              "p95_cv_std": float(np.nanpercentile(std_map, 95)),
                              "note": "per-pixel std of area-percentile rank across the spatial-CV fold models, "
                                      "in percentile points (0-100)"}
    metrics["features"] = feats
    log.info("spatial CV AUC %.3f | random CV AUC %.3f | capture %s", metrics["spatial_cv"]["roc_auc"],
             metrics["random_cv"]["roc_auc"], metrics["capture"])

    # ---- 5. Explain ----
    sv, base = explain.shap_values(model, ds, feats)
    tg = explain.targets(ds, sv, feats, grid, m["top_area_frac"])
    explain.plots(model, ds, feats, out)
    figs = ["shap_beeswarm.png", "shap_importance.png"]
    if len(tg):
        explain.waterfall(model, ds, feats, int(tg.iloc[0]["row"]), int(tg.iloc[0]["col"]), out / "shap_target1.png")
        figs.append("shap_target1.png")

    # ---- 6. Export ----
    write_geotiff(out / "prospectivity.tif", prob, grid)
    write_geotiff(out / "prospectivity_percentile.tif", pct, grid)
    write_geotiff(out / "prospectivity_uncertainty.tif", std_map, grid)
    ds.to_netcdf(proc / "feature_stack.nc")
    import xarray as xr

    xr.Dataset({"shap": (("feature", "y", "x"), sv)}, coords={"feature": feats, "y": ds.y, "x": ds.x},
               attrs={"base_value": base}).to_netcdf(proc / "shap.nc")
    tg.drop(columns=["row", "col"]).to_file(out / "targets.gpkg", driver="GPKG")
    tg.to_crs("EPSG:4326").to_file(out / "targets.geojson", driver="GeoJSON")
    keep = ["x", "y", "anomaly_score", "target_anomaly", "signature_index", "iforest_score", "lof_score"] + \
           [c for c in scored.columns if c.endswith("_pivot")] + cfg["geochem"]["elements"]
    scored[[c for c in keep if c in scored] + ["geometry"]].to_crs("EPSG:4326").to_file(out / "geochem_scored.geojson")
    deps[["geometry"] + [c for c in deps.columns if c in ("NAME", "COMMODITY")]].to_crs("EPSG:4326") \
        .to_file(out / "deposits.geojson")
    web_layers(ds, grid, out / "web", {"lineament_deep": st["lineament_deep"],
                                       "lineament_shallow": st["lineament_shallow"]})
    joblib.dump({"model": model, "features": feats}, out / "model.joblib")
    if cfg["source"] == "synthetic":
        metrics["structure_recovery"] = structure_recovery(cfg, grid, st)
    metrics["data_source"] = cfg["source"]
    metrics["runtime_s"] = round(time.time() - t0, 1)
    if "_cnn_loss" in st:
        metrics["cnn_loss"] = [round(float(x), 6) for x in st["_cnn_loss"]]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report(out / "REPORT.md", cfg, metrics, tg, figs)
    log.info("done in %.0f s -> %s", metrics["runtime_s"], out)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="GeoAI-Prospector pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", choices=["synthetic", "real"], default=None)
    ap.add_argument("--if-changed", action="store_true", help="skip unless upstream data changed (scheduled runs)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    if a.source:
        cfg["source"] = a.source
    if a.if_changed and cfg["source"] == "real":
        changed = download.check_for_updates(cfg)
        log.info("upstream changes: %s", changed)
        if not any(changed.values()):
            log.info("nothing changed upstream - skipping run")
            return
        download.download_all(cfg, force=True)
    run(cfg)


if __name__ == "__main__":
    main()
