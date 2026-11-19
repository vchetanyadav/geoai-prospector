"""Point ETL: geochemistry and deposit tables -> clean GeoDataFrames in the grid CRS,
joined to the exact pixel they fall in."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ..config import Grid

log = logging.getLogger(__name__)

UNIT_TO_PPM = {"ppm": 1.0, "ppb": 1e-3, "pct": 1e4, "%": 1e4, "g/t": 1.0}


def _to_gdf(df: pd.DataFrame, x: str, y: str, crs: str, grid: Grid) -> gpd.GeoDataFrame:
    df = df.dropna(subset=[x, y])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs).to_crs(grid.crs)
    gdf["x"], gdf["y"] = gdf.geometry.x, gdf.geometry.y
    rows, cols = grid.rowcol(gdf["x"].values, gdf["y"].values)
    inside = grid.inside(rows, cols)
    gdf = gdf.loc[inside].copy()
    gdf["row"], gdf["col"] = rows[inside], cols[inside]
    return gdf


def load_geochem(path: Path, src: dict, elements: list[str], grid: Grid, max_missing_frac=0.3) -> gpd.GeoDataFrame:
    """Standardise units to ppm, keep detection-limit flags, drop unusable rows.

    Censored values are coded as negative detection limits (NTGS/GA convention); the
    absolute value is kept in `<el>_dl` and `<el>` is set to NaN with `<el>_bdl = True`.
    """
    df = pd.read_csv(path, low_memory=False)
    gdf = _to_gdf(df, src["x"], src["y"], src["crs"], grid)
    out = gdf[["geometry", "x", "y", "row", "col"]].copy()
    for el in elements:
        spec = src["columns"][el]
        v = pd.to_numeric(gdf[spec["col"]], errors="coerce").values * UNIT_TO_PPM[spec["unit"]]
        bdl = v < 0
        out[f"{el}_bdl"] = bdl
        out[f"{el}_dl"] = np.where(bdl, -v, np.nan)
        out[el] = np.where(bdl | (v == 0), np.nan, v)
    missing = out[elements].isna() & ~out[[f"{e}_bdl" for e in elements]].values
    frac = missing.mean()
    drop_el = frac[frac > max_missing_frac].index.tolist()
    if drop_el:
        log.warning("dropping elements with >%.0f%% missing: %s", 100 * max_missing_frac, drop_el)
    keep_el = [e for e in elements if e not in drop_el]
    ok = ~missing[keep_el].any(axis=1)
    out = out.loc[ok].copy()
    # duplicates at one location (field repeats) -> keep the first
    out = out.loc[~out[["x", "y"]].round(0).duplicated()].copy()
    out.attrs["elements"] = keep_el
    log.info("geochem: %d samples kept (of %d), elements=%s", len(out), len(df), keep_el)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs)


def load_deposits(path: Path, src: dict, grid: Grid) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    filt = src.get("commodity_filter")
    if filt and src.get("commodity_col") in df:
        pat = "|".join(filt)
        df = df[df[src["commodity_col"]].astype(str).str.contains(pat, case=False)]
    gdf = _to_gdf(df, src["x"], src["y"], src["crs"], grid)
    log.info("deposits: %d in area", len(gdf))
    return gdf


def sample_rasters(gdf: gpd.GeoDataFrame, layers: dict[str, np.ndarray]) -> gpd.GeoDataFrame:
    """Spatial join: attach the value of each raster at the pixel holding each point."""
    gdf = gdf.copy()
    for name, arr in layers.items():
        gdf[name] = arr[gdf["row"].values, gdf["col"].values]
    return gdf
