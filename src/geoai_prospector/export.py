"""Write GIS deliverables (GeoTIFF, GeoPackage, NetCDF) plus lightweight web layers
(PNG overlays in EPSG:4326 + GeoJSON) and an auto-generated markdown run report."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from rasterio.warp import Resampling, calculate_default_transform, reproject  # noqa: E402

from .config import Grid  # noqa: E402

log = logging.getLogger(__name__)

WEB_LAYERS = {  # layer: (colormap, robust stretch?, label)
    "prospectivity_pct": ("inferno", False, "AI prospectivity (percentile)"),
    "prospectivity_std": ("magma", True, "Model uncertainty (rank spread, pct points)"),
    "mag_rtp": ("turbo", True, "Magnetic RTP"),
    "mag_tilt": ("RdBu_r", True, "Magnetic tilt derivative"),
    "grav": ("viridis", True, "Bouguer gravity"),
    "grav_hgm": ("magma", True, "Gravity horizontal gradient"),
    "structural_complexity": ("cividis", True, "Lineament intersection index"),
    "cnn_recon_error": ("plasma", True, "CNN reconstruction error"),
    "geochem_target_anomaly": ("YlOrRd", True, "Cu-Au-Bi geochem anomaly"),
}


def write_geotiff(path: Path, arr: np.ndarray, grid: Grid):
    """Cloud-Optimized GeoTIFF (GDAL COG driver), falling back to tiled GTiff."""
    kw = dict(height=grid.height, width=grid.width, count=1, dtype="float32", crs=grid.crs,
              transform=grid.transform, nodata=np.nan, compress="deflate")
    try:
        with rasterio.open(path, "w", driver="COG", **kw) as dst:
            dst.write(arr.astype("float32"), 1)
    except rasterio.errors.RasterioIOError:
        with rasterio.open(path, "w", driver="GTiff", tiled=True, **kw) as dst:
            dst.write(arr.astype("float32"), 1)


def _to_wgs84(arr, grid: Grid, resampling=Resampling.bilinear):
    tr, w, h = calculate_default_transform(grid.crs, "EPSG:4326", grid.width, grid.height,
                                           grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    out = np.full((h, w), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=grid.transform, src_crs=grid.crs, src_nodata=np.nan,
              dst_transform=tr, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=resampling)
    west, north = tr.c, tr.f
    east, south = west + tr.a * w, north + tr.e * h
    return out, [[south, west], [north, east]]


def web_layers(ds, grid: Grid, out_dir: Path, lineament_masks: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (cmap, robust, label) in WEB_LAYERS.items():
        if name not in ds:
            continue
        a, bounds = _to_wgs84(ds[name].values, grid)
        lo, hi = (np.nanpercentile(a, [2, 98]) if robust else (np.nanmin(a), np.nanmax(a)))
        rgba = colormaps[cmap](np.clip((a - lo) / (hi - lo + 1e-12), 0, 1))
        rgba[..., 3] = np.where(np.isfinite(a), 1.0, 0.0)
        plt.imsave(out_dir / f"{name}.png", rgba)
        meta[name] = {"png": f"{name}.png", "bounds": bounds, "label": label, "cmap": cmap,
                      "vmin": float(lo), "vmax": float(hi)}
    for name, mask in lineament_masks.items():
        from scipy.ndimage import binary_dilation

        a, bounds = _to_wgs84(binary_dilation(mask).astype("float32"), grid, Resampling.nearest)
        rgba = np.zeros(a.shape + (4,))
        colour = (0.0, 0.9, 1.0) if "deep" in name else (1.0, 1.0, 1.0)
        rgba[..., :3] = colour
        rgba[..., 3] = np.nan_to_num(a) * 0.9
        plt.imsave(out_dir / f"{name}.png", rgba)
        meta[name] = {"png": f"{name}.png", "bounds": bounds,
                      "label": "Deep gravity lineaments (CV)" if "deep" in name else "Magnetic lineaments (CV)"}
    (out_dir / "layers.json").write_text(json.dumps(meta, indent=2))
    return meta


def report(path: Path, cfg: dict, metrics: dict, targets_gdf, figs: list[str]):
    cv, rnd = metrics["spatial_cv"], metrics["random_cv"]
    lines = [
        f"# {cfg['project']['name']} - run report",
        "",
        f"*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} | region: {cfg['project']['region']} | "
        f"data source: {cfg['source']} | grid {cfg['grid']['crs']} @ {cfg['grid']['resolution']} m*",
        "",
        "## Model performance",
        "",
        "| Validation | ROC-AUC | PR-AUC | base rate |",
        "|---|---|---|---|",
        f"| **Spatial block CV ({cfg['model']['block_size_m'] / 1000:.0f} km blocks, "
        f"{cv.get('buffer_m', 0) / 1000:.1f} km buffer)** | **{cv['roc_auc']:.3f}** | "
        f"**{cv['pr_auc']:.3f}** | {cv['base_rate']:.4f} |",
        *([f"| Spatial block CV, no buffer | {metrics['spatial_cv_nobuffer']['roc_auc']:.3f} | "
           f"{metrics['spatial_cv_nobuffer']['pr_auc']:.3f} | "
           f"{metrics['spatial_cv_nobuffer']['base_rate']:.4f} |"] if "spatial_cv_nobuffer" in metrics else []),
        f"| Random CV (leaky, for comparison) | {rnd['roc_auc']:.3f} | {rnd['pr_auc']:.3f} | {rnd['base_rate']:.4f} |",
        "",
        "Held-out capture efficiency (share of known deposits inside the top x % of the area, "
        "scored by models that never saw their block): "
        + ", ".join(f"top {k.split('_')[3].replace('pct', '%')}: {v:.0%}" for k, v in metrics["capture"].items()),
        "",
        *([f"Uncertainty: per-pixel spread of area-percentile rank across the {cfg['model']['n_folds']} "
           f"spatial-CV fold models - mean {metrics['uncertainty']['mean_cv_std']:.1f} pct points, 95th "
           f"percentile {metrics['uncertainty']['p95_cv_std']:.1f}. High-score, high-spread targets need "
           "data before trust."] if "uncertainty" in metrics else []),
        "",
        *([f"Synthetic ground-truth check (Model A): {metrics['structure_recovery']['deep_lineament_precision']:.0%} "
           f"of detected deep lineaments lie within {metrics['structure_recovery']['tolerance_m'] / 1000:.1f} km of a "
           f"planted fault; {metrics['structure_recovery']['deep_fault_recall']:.0%} of planted deep-fault length was "
           "recovered.", ""] if "structure_recovery" in metrics else []),
        "## Does fusion help? (spatial CV ROC-AUC)",
        "",
        "| Feature set | ROC-AUC | PR-AUC | features |",
        "|---|---|---|---|",
    ]
    for k, v in metrics["ablation"].items():
        lines.append(f"| {k} | {v['roc_auc']:.3f} | {v['pr_auc']:.3f} | {v['n_features']} |")
    lines += ["", "## Top targets", "", "| Rank | Status | Score | Rank spread (pct pts) | Area km2 | Why |",
              "|---|---|---|---|---|---|"]
    for _, t in targets_gdf.head(10).iterrows():
        lines.append(f"| {t['rank']} | {t['status']} | {t['peak_score']:.2f} | "
                     f"±{t.get('peak_std', float('nan')):.1f} | {t['area_km2']:.1f} | {t['explanation']} |")
    green = targets_gdf[targets_gdf["status"] == "greenfield"].head(10)
    lines += ["", "## Top greenfield targets (no known deposit within 1 km)", "",
              "| Rank | Score | Rank spread (pct pts) | Area km2 | Peak (MGA E, N) | Why |",
              "|---|---|---|---|---|---|"]
    for _, t in green.iterrows():
        lines.append(f"| {t['rank']} | {t['peak_score']:.2f} | ±{t.get('peak_std', float('nan')):.1f} | "
                     f"{t['area_km2']:.1f} | {t['peak_x']:.0f}, {t['peak_y']:.0f} | {t['explanation']} |")
    lines += ["", "## Figures", ""] + [f"![{f}]({f})" for f in figs]
    path.write_text("\n".join(lines))
