"""Synthetic but geologically plausible East Tennant style dataset.

Why it exists: the full pipeline (and CI) can run with no downloads, and because the
ground truth is known, you can check that each model recovers what was planted.

What it plants:
* fault-bounded basement blocks (two dominant trends, NW and NE, plus E-W), so faults
  appear as gradients in both grids; only the "deep" faults offset gravity strongly
* magnetite ironstone bullseyes at most deposits (Tennant Creek style)
* deposits clustered near fault intersections, plus a few "random" ones
* sedimentary cover thickening to the east, which damps the surface geochemistry
* biased sampling: regional sparse grid + dense prospect-scale grids
* below-detection values written as negative numbers, as in most legacy databases

Inputs are deliberately written in *different* CRSs and resolutions (mag in GDA94
geographic, gravity in GDA2020 geographic, points in GDA94 lon/lat) so the ETL has to
do real reprojection work.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter

from ..config import Grid

log = logging.getLogger(__name__)

ELEMENTS = {  # element: (unit, background median in that unit, log-sd, detection limit)
    "Cu": ("ppm", 25, 0.6, 2),
    "Au": ("ppb", 2, 0.9, 1),
    "Fe": ("pct", 4.0, 0.35, 0.01),
    "Bi": ("ppm", 0.3, 0.7, 0.1),
    "Co": ("ppm", 12, 0.5, 1),
    "Pb": ("ppm", 15, 0.6, 2),
    "Zn": ("ppm", 40, 0.6, 2),
    "As": ("ppm", 6, 0.7, 1),
    "Ni": ("ppm", 20, 0.5, 2),
}


def _faults(rng, grid: Grid, n_deep=6, n_shallow=9):
    """Faults as (point, unit direction, throw_mag, throw_grav, deep flag)."""
    faults = []
    trends = [135, 45, 90]
    for i in range(n_deep + n_shallow):
        deep = i < n_deep
        az = np.deg2rad(trends[i % 3] + rng.normal(0, 10))
        p = np.array([rng.uniform(grid.xmin + 10e3, grid.xmax - 10e3),
                      rng.uniform(grid.ymin + 10e3, grid.ymax - 10e3)])
        d = np.array([np.sin(az), np.cos(az)])
        faults.append(dict(p=p, d=d, deep=deep,
                           tm=rng.uniform(150, 400) * rng.choice([-1, 1]),
                           tg=(rng.uniform(4, 9) if deep else rng.uniform(0.3, 1.2)) * rng.choice([-1, 1])))
    return faults


def _intersections(faults, grid: Grid):
    pts = []
    for i in range(len(faults)):
        for j in range(i + 1, len(faults)):
            a, b = faults[i], faults[j]
            m = np.column_stack([a["d"], -b["d"]])
            if abs(np.linalg.det(m)) < 0.2:
                continue
            t = np.linalg.solve(m, b["p"] - a["p"])
            q = a["p"] + t[0] * a["d"]
            if grid.xmin + 3e3 < q[0] < grid.xmax - 3e3 and grid.ymin + 3e3 < q[1] < grid.ymax - 3e3:
                pts.append((q, a["deep"] + b["deep"]))
    return pts


def _write_geotiff(path, arr, transform, crs, nodata=np.nan):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(arr.astype("float32"), 1)


def _warp_to_geographic(arr, grid: Grid, dst_crs: str, res_deg: float):
    dst_crs = CRS.from_user_input(dst_crs)
    transform, w, h = calculate_default_transform(grid.crs, dst_crs, grid.width, grid.height,
                                                  grid.xmin, grid.ymin, grid.xmax, grid.ymax,
                                                  resolution=res_deg)
    out = np.full((h, w), np.nan, dtype="float32")
    reproject(arr.astype("float32"), out, src_transform=grid.transform, src_crs=grid.crs,
              dst_transform=transform, dst_crs=dst_crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return out, transform, dst_crs


def generate(grid: Grid, out_dir: Path, seed=42, n_samples=3500, n_deposits=30) -> dict:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Generate at 4x finer resolution than the model grid, so resampling is realistic
    fine = Grid(grid.crs, grid.res / 4, grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    X, Y = np.meshgrid(fine.x, fine.y)

    faults = _faults(rng, grid)
    mag = np.zeros(fine.shape)
    grav = np.zeros(fine.shape)
    for f in faults:
        # signed distance from fault line -> smooth step (fault offsets a layered basement)
        nvec = np.array([-f["d"][1], f["d"][0]])
        sd = (X - f["p"][0]) * nvec[0] + (Y - f["p"][1]) * nvec[1]
        step = np.tanh(sd / (1500 if f["deep"] else 600))
        mag += f["tm"] * step
        grav += f["tg"] * step
    # basement texture + regional trend + intrusions
    mag += gaussian_filter(rng.normal(0, 1, fine.shape), 12) * 3000
    grav += gaussian_filter(rng.normal(0, 1, fine.shape), 40) * 60 + (X - grid.xmin) / 1e4 * 1.5
    for _ in range(5):
        cx, cy = rng.uniform(grid.xmin, grid.xmax), rng.uniform(grid.ymin, grid.ymax)
        r = rng.uniform(3e3, 7e3)
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r ** 2))
        mag += rng.uniform(-300, 500) * blob
        grav += rng.uniform(-6, 6) * blob

    # deposits: mostly near fault intersections (deep ones favoured), some random
    inter = _intersections(faults, grid)
    rng.shuffle(inter)
    inter.sort(key=lambda t: -t[1])
    deposits = []
    n_struct = int(n_deposits * 0.75)
    for q, _ in inter:
        if len(deposits) >= n_struct:
            break
        deposits.append(q + rng.normal(0, 1200, 2))
    while len(deposits) < n_deposits:
        deposits.append(np.array([rng.uniform(grid.xmin + 2e3, grid.xmax - 2e3),
                                  rng.uniform(grid.ymin + 2e3, grid.ymax - 2e3)]))
    deposits = np.array(deposits)
    # magnetite ironstones: sharp magnetic + small gravity highs at ~75 % of deposits
    for d in deposits:
        if rng.random() < 0.5:
            blob = np.exp(-((X - d[0]) ** 2 + (Y - d[1]) ** 2) / (2 * 500 ** 2))
            mag += rng.uniform(150, 450) * blob
            grav += rng.uniform(0.2, 0.8) * blob
    # decoy ironstones that are barren
    for _ in range(30):
        cx, cy = rng.uniform(grid.xmin, grid.xmax), rng.uniform(grid.ymin, grid.ymax)
        mag += rng.uniform(150, 450) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 500 ** 2))

    mag += rng.normal(0, 8, fine.shape)            # nT noise
    grav += rng.normal(0, 0.05, fine.shape)        # mGal noise
    # survey gaps
    gap = ((X - grid.xmin) > 70e3) & ((X - grid.xmin) < 76e3) & ((grid.ymax - Y) > 10e3) & ((grid.ymax - Y) < 20e3)
    mag[gap] = np.nan
    grav[(X - grid.xmin) > 98.5e3] = np.nan

    mag_ll, mag_tr, mag_crs = _warp_to_geographic(mag, fine, "EPSG:4283", 0.0025)
    grav_ll, grav_tr, grav_crs = _warp_to_geographic(gaussian_filter(np.nan_to_num(grav, nan=np.nanmean(grav)), 2)
                                                     + np.where(np.isnan(grav), np.nan, 0), fine, "EPSG:7844", 0.008)
    _write_geotiff(out_dir / "mag_rtp.tif", mag_ll, mag_tr, mag_crs)
    _write_geotiff(out_dir / "grav_scba.tif", grav_ll, grav_tr, grav_crs)

    # ---- geochemistry ----
    xs, ys, kind = [], [], []
    # regional grid ~2.5 km with jitter
    gx, gy = np.meshgrid(np.arange(grid.xmin + 1e3, grid.xmax, 2500), np.arange(grid.ymin + 1e3, grid.ymax, 2500))
    sel = rng.random(gx.size) < 0.7
    xs += list(gx.ravel()[sel] + rng.normal(0, 300, sel.sum()))
    ys += list(gy.ravel()[sel] + rng.normal(0, 300, sel.sum()))
    kind += ["regional"] * int(sel.sum())
    # prospect grids: around 60 % of deposits and some random prospects (sampling bias)
    centres = [d for d in deposits if rng.random() < 0.4]
    centres += [np.array([rng.uniform(grid.xmin, grid.xmax), rng.uniform(grid.ymin, grid.ymax)]) for _ in range(25)]
    per = max(1, (n_samples - len(xs)) // max(1, len(centres)))
    for c in centres:
        xs += list(c[0] + rng.normal(0, 1500, per))
        ys += list(c[1] + rng.normal(0, 1500, per))
        kind += ["prospect"] * per
    xs, ys = np.array(xs), np.array(ys)
    keep = (xs > grid.xmin) & (xs < grid.xmax) & (ys > grid.ymin) & (ys < grid.ymax)
    xs, ys, kind = xs[keep], ys[keep], np.array(kind)[keep]
    n = len(xs)

    dmin = np.min(np.hypot(xs[:, None] - deposits[None, :, 0], ys[:, None] - deposits[None, :, 1]), axis=1)
    halo = np.exp(-(dmin ** 2) / (2 * 1000 ** 2))
    cover = np.clip((xs - grid.xmin) / (grid.xmax - grid.xmin), 0, 1)  # thicker cover to the east
    signal = halo * (1 - 0.95 * cover)
    shared = rng.normal(0, 1, n)  # mafic factor: Fe-Co-Ni co-vary
    rows = {"SAMPLE_ID": [f"SYN{i:06d}" for i in range(n)], "SAMPLE_TYPE": kind}
    lift = {"Cu": 6, "Au": 10, "Bi": 5, "Fe": 0.5, "Co": 0.5, "As": 1}
    for el, (unit, med, sd, dl) in ELEMENTS.items():
        z = rng.normal(0, sd, n) + (0.3 * shared if el in ("Fe", "Co", "Ni") else 0)
        v = med * np.exp(z) * (1 + lift.get(el, 0) * signal)
        # an unrelated Pb-Zn anomaly (false positive for a Cu-Au target)
        if el in ("Pb", "Zn"):
            v *= 1 + 12 * np.exp(-((xs - (grid.xmin + 25e3)) ** 2 + (ys - (grid.ymin + 70e3)) ** 2) / (2 * 3000 ** 2))
        v = np.where(v < dl, -dl, np.round(v, 3))         # censored values coded as -DL
        v[rng.random(n) < 0.005] = np.nan                  # random missing assays
        rows[f"{el}_{unit}"] = v
    df = pd.DataFrame(rows)
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4283", always_xy=True)
    df["LONGITUDE"], df["LATITUDE"] = to_ll.transform(xs, ys)
    df.to_csv(out_dir / "geochem.csv", index=False)

    lon, lat = to_ll.transform(deposits[:, 0], deposits[:, 1])
    dep = pd.DataFrame({"NAME": [f"Synthetic-{i:02d}" for i in range(len(deposits))],
                        "COMMODITY": rng.choice(["Cu-Au", "Au", "Cu", "Au-Cu-Bi"], len(deposits)),
                        "LONGITUDE": lon, "LATITUDE": lat})
    dep.to_csv(out_dir / "deposits.csv", index=False)

    truth = {"faults": [{"p": f["p"].tolist(), "d": f["d"].tolist(), "deep": f["deep"]} for f in faults],
             "deposits_xy": deposits.tolist()}
    (out_dir / "truth.json").write_text(json.dumps(truth))
    log.info("synthetic data: %d samples, %d deposits -> %s", n, len(deposits), out_dir)
    return {"magnetics": out_dir / "mag_rtp.tif", "gravity": out_dir / "grav_scba.tif",
            "geochem": out_dir / "geochem.csv", "deposits": out_dir / "deposits.csv"}
