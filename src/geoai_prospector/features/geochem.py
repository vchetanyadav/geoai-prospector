"""Model B - compositional geochemistry + unsupervised anomaly detection.

Assays are compositional: each sample is parts of a whole, so raw ppm values live on
a simplex and ordinary statistics (correlation, distance) are biased. We
1. replace censored (<DL) values with multiplicative replacement,
2. close the composition with a residual part ("Rest" = 1e6 ppm - sum),
3. map to real space with the isometric log-ratio (ILR) transform,
4. score anomalies with Isolation Forest + Local Outlier Factor (rank-averaged),
5. compute interpretable pivot coordinates for the pathfinder elements, and
6. grid the point results with inverse-distance weighting.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from ..config import Grid

log = logging.getLogger(__name__)


# ---------- compositional data analysis ----------

def multiplicative_replacement(X: np.ndarray, censored: np.ndarray, dl: np.ndarray, factor=0.65, total=1e6):
    """Martin-Fernandez et al. (2003): censored -> factor*DL, others rescaled so the
    total is preserved. X, censored, dl are (n, D); X is NaN where censored."""
    X = X.copy()
    delta = np.where(censored, factor * dl, 0.0)
    s = delta.sum(1, keepdims=True)
    X = np.where(censored, delta, X * (1 - s / total))
    return X


def close(X: np.ndarray, total=1e6) -> np.ndarray:
    """Append a residual part so each row sums to `total`, then scale to 1."""
    rest = total - X.sum(1, keepdims=True)
    rest = np.clip(rest, 1e-9 * total, None)
    C = np.hstack([X, rest])
    return C / C.sum(1, keepdims=True)


def clr(C: np.ndarray) -> np.ndarray:
    L = np.log(C)
    return L - L.mean(1, keepdims=True)


def helmert_basis(D: int) -> np.ndarray:
    """Orthonormal D x (D-1) basis of the clr hyperplane (pivot / Helmert contrasts)."""
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        k = D - j - 1  # number of remaining parts
        V[j, j] = np.sqrt(k / (k + 1))
        V[j + 1:, j] = -1 / np.sqrt(k * (k + 1))
    return V


def ilr(C: np.ndarray) -> np.ndarray:
    return clr(C) @ helmert_basis(C.shape[1])


def ilr_inv(Z: np.ndarray) -> np.ndarray:
    E = np.exp(Z @ helmert_basis(Z.shape[1] + 1).T)
    return E / E.sum(1, keepdims=True)


def pivot(C: np.ndarray, j: int) -> np.ndarray:
    """First pivot coordinate for part j: sqrt((D-1)/D) * ln(x_j / gmean(others)).
    Positive = the element is enriched relative to the rest of the sample."""
    D = C.shape[1]
    L = np.log(C)
    others = (L.sum(1) - L[:, j]) / (D - 1)
    return np.sqrt((D - 1) / D) * (L[:, j] - others)


# ---------- anomaly detection ----------

def _pct_rank(v):
    return (rankdata(v) - 1) / (len(v) - 1)


def score(gdf: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = cfg["geochem"]
    els = gdf.attrs.get("elements", g["elements"])
    X = gdf[els].to_numpy(float)
    cens = gdf[[f"{e}_bdl" for e in els]].to_numpy(bool)
    dl = gdf[[f"{e}_dl" for e in els]].fillna(0).to_numpy(float)
    X = multiplicative_replacement(X, cens, dl, g["bdl_factor"])
    C = close(X)
    Z = ilr(C)
    Zs = RobustScaler().fit_transform(Z)

    iso = IsolationForest(n_estimators=g["iforest"]["n_estimators"], contamination=g["iforest"]["contamination"],
                          random_state=g["iforest"]["seed"]).fit(Zs)
    s_iso = -iso.score_samples(Zs)
    lof = LocalOutlierFactor(n_neighbors=g["lof"]["n_neighbors"]).fit(Zs)
    s_lof = -lof.negative_outlier_factor_

    out = gdf.copy()
    out["iforest_score"] = _pct_rank(s_iso)
    out["lof_score"] = _pct_rank(s_lof)
    out["anomaly_score"] = (out["iforest_score"] + out["lof_score"]) / 2
    for el in g["pathfinders"]:
        if el in els:
            z = pivot(C, els.index(el))
            # robust z-score vs the survey: 0 = typical sample, +3 = strongly enriched relative to the rest
            mad = np.median(np.abs(z - np.median(z))) * 1.4826 + 1e-9
            out[f"{el}_pivot"] = (z - np.median(z)) / mad
    pf = [f"{e}_pivot" for e in g["pathfinders"] if e in els]
    # target-signature index: mean robust z of the pathfinders (Cu+Au+Bi co-enrichment)
    out["signature_index"] = out[pf].to_numpy().mean(1)
    # anomalous AND on-signature: separates Cu-Au anomalies from e.g. Pb-Zn ones
    out["target_anomaly"] = out["anomaly_score"] * _pct_rank(out["signature_index"].values)
    log.info("geochem: ILR dim %d, top-5%% anomalies %d", Z.shape[1], int((out["anomaly_score"] > 0.95).sum()))
    return out


def idw_grid(x, y, v, grid: Grid, k=8, power=2, max_dist_m=4000):
    """Inverse-distance weighting onto the grid. Cells farther than max_dist from any
    sample are NaN - XGBoost treats NaN as 'no data', so we never invent chemistry."""
    tree = cKDTree(np.column_stack([x, y]))
    X, Y = np.meshgrid(grid.x, grid.y)
    q = np.column_stack([X.ravel(), Y.ravel()])
    d, i = tree.query(q, k=min(k, len(x)))
    d = np.maximum(d, 1.0)
    w = 1 / d ** power
    w[d > max_dist_m] = 0
    ws = w.sum(1)
    est = np.where(ws > 0, (w * np.asarray(v)[i]).sum(1) / np.where(ws > 0, ws, 1), np.nan)
    return est.reshape(grid.shape).astype("float32"), d[:, 0].reshape(grid.shape).astype("float32")


def run(gdf, grid: Grid, cfg: dict):
    scored = score(gdf, cfg)
    gp = cfg["geochem"]["gridding"]
    layers = {}
    cols = ["anomaly_score", "target_anomaly", "signature_index"] + \
           [c for c in scored.columns if c.endswith("_pivot")]
    dist = None
    for c in cols:
        layers[f"geochem_{c}"], dist = idw_grid(scored["x"].values, scored["y"].values, scored[c].values, grid,
                                                gp["k"], gp["power"], gp["max_dist_m"])
    layers["_geochem_dist_m"] = dist  # coverage only, NOT a model feature (sampling bias!)
    return scored, layers
