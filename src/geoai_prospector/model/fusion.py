"""Spatial information fusion: one xarray Dataset where every pixel carries every feature,
plus the training labels."""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..config import Grid
from ..features.structure import distance_to

# Human-readable names and units, used by the SHAP explanations and the app
FEATURES = {
    "mag_rtp": ("Magnetic intensity (RTP)", "nT"),
    "mag_tilt": ("Magnetic tilt derivative", "rad"),
    "mag_hgm": ("Magnetic horizontal gradient", "nT/km"),
    "grav": ("Bouguer gravity", "mGal"),
    "grav_hgm": ("Gravity horizontal gradient", "mGal/km"),
    "grav_1vd": ("Gravity 1st vertical derivative", "mGal/km"),
    "dist_deep_fault_m": ("Distance to deep gravity lineament (CV-detected)", "m"),
    "dist_mapped_fault_m": ("Distance to mapped fault (geology map)", "m"),
    "dist_shallow_lineament_m": ("Distance to magnetic lineament", "m"),
    "lineament_density": ("Lineament density", "km/km2"),
    "structural_complexity": ("Lineament intersection / complexity index", ""),
    "cnn_pc1": ("CNN texture feature 1", ""),
    "cnn_pc2": ("CNN texture feature 2", ""),
    "cnn_pc3": ("CNN texture feature 3", ""),
    "cnn_recon_error": ("CNN reconstruction error (unusual texture)", ""),
    "geochem_anomaly_score": ("Geochem multivariate anomaly score", ""),
    "geochem_target_anomaly": ("Geochem Cu-Au-Bi signature anomaly", ""),
    "geochem_Cu_pivot": ("Cu enrichment (ILR pivot z-score)", ""),
    "geochem_Au_pivot": ("Au enrichment (ILR pivot z-score)", ""),
    "geochem_Bi_pivot": ("Bi enrichment (ILR pivot z-score)", ""),
}

GROUPS = {
    "geophysics": ["mag_rtp", "mag_tilt", "mag_hgm", "grav", "grav_hgm", "grav_1vd"],
    "structure": ["dist_deep_fault_m", "dist_mapped_fault_m", "dist_shallow_lineament_m", "lineament_density",
                  "structural_complexity",
                  "cnn_pc1", "cnn_pc2", "cnn_pc3", "cnn_recon_error"],
    "geochem": ["geochem_anomaly_score", "geochem_target_anomaly", "geochem_Cu_pivot", "geochem_Au_pivot",
                "geochem_Bi_pivot"],
}


def build_stack(grid: Grid, layers: dict[str, np.ndarray]) -> xr.Dataset:
    ds = xr.Dataset(coords={"y": grid.y, "x": grid.x})
    for k, v in layers.items():
        if k.startswith("_") or np.asarray(v).shape != grid.shape:
            continue
        ds[k] = (("y", "x"), np.asarray(v))
    ds.attrs.update(crs=grid.crs.to_string(), res=grid.res, transform=list(grid.transform)[:6])
    return ds


def add_labels(ds: xr.Dataset, grid: Grid, dep_rows, dep_cols, buffer_m: float, exclusion_m: float) -> xr.Dataset:
    """1 = within buffer of a known deposit, 0 = farther than exclusion, -1 = unused.

    This is a positive-unlabelled problem: 'negative' just means 'no known deposit',
    so we keep a ring around deposits out of training and treat scores as relative
    prospectivity, not absolute probability."""
    m = np.zeros(grid.shape, bool)
    m[dep_rows, dep_cols] = True
    d = distance_to(m, grid.res)
    lab = np.full(grid.shape, -1, dtype="int8")
    lab[d <= buffer_m] = 1
    lab[d > exclusion_m] = 0
    ds["label"] = (("y", "x"), lab)
    ds["dist_known_deposit_m"] = (("y", "x"), d)
    return ds


def feature_names(ds: xr.Dataset) -> list[str]:
    return [f for f in FEATURES if f in ds]


def to_matrix(ds: xr.Dataset, feats: list[str]) -> np.ndarray:
    return np.stack([ds[f].values.ravel() for f in feats], axis=1).astype("float32")
