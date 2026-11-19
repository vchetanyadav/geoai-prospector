"""Model C - spatially aware gradient boosting.

The trap in prospectivity modelling: neighbouring pixels are nearly identical, so a
random train/test split leaks information and reports inflated scores. We evaluate
with *spatial block* cross-validation (10 km blocks, StratifiedGroupKFold) and report
the random-split score next to it so the gap is visible.

Metrics a geologist cares about besides AUC:
* PR-AUC (positives are <1 % of pixels)
* capture efficiency: share of known deposits that fall in the top 5/10/20 % of the area
"""
from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from .fusion import GROUPS, feature_names, to_matrix

log = logging.getLogger(__name__)


def _blocks(ds: xr.Dataset, block_m: float) -> np.ndarray:
    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    bx = ((X - X.min()) // block_m).astype(int)
    by = ((Y.max() - Y) // block_m).astype(int)
    return (by * (bx.max() + 1) + bx).ravel()


def _model(cfg, pos_weight):
    p = cfg["model"]["xgb"]
    return XGBClassifier(**p, objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                         scale_pos_weight=pos_weight, random_state=cfg["model"]["seed"], n_jobs=-1)


def capture_efficiency(score_map: np.ndarray, dep_rows, dep_cols, fracs=(0.05, 0.10, 0.20)) -> dict:
    s = score_map[np.isfinite(score_map)]
    out = {}
    for f in fracs:
        thr = np.quantile(s, 1 - f)
        out[f"deposits_in_top_{int(f * 100)}pct_area"] = float(np.mean(score_map[dep_rows, dep_cols] >= thr))
    return out


def cross_validate(ds: xr.Dataset, cfg: dict, feats: list[str] | None = None, spatial=True,
                   return_map=False, buffer_m: float | None = None) -> dict:
    """Out-of-fold evaluation. With return_map=True (spatial only) every pixel of the
    area gets a score from the model that did NOT see its block, so capture efficiency
    is an honest held-out number.

    buffer_m (default from config model.cv_buffer_m): training samples closer than this
    to any test sample are dropped from that fold, so spatial autocorrelation across
    block edges cannot leak. Pass 0 to disable."""
    feats = feats or feature_names(ds)
    X = to_matrix(ds, feats)
    y = ds["label"].values.ravel()
    groups = _blocks(ds, cfg["model"]["block_size_m"])
    use = y >= 0
    Xu, yu, gu = X[use], y[use], groups[use]
    GX, GY = np.meshgrid(ds.x.values, ds.y.values)
    xy = np.column_stack([GX.ravel(), GY.ravel()])[use]
    if buffer_m is None:
        buffer_m = float(cfg["model"].get("cv_buffer_m", 0))
    k = cfg["model"]["n_folds"]
    seed = cfg["model"]["seed"]
    if spatial:
        splits = StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(Xu, yu, gu)
    else:
        splits = StratifiedKFold(k, shuffle=True, random_state=seed).split(Xu, yu)
    oof = np.full(len(yu), np.nan)
    oof_map = np.full(len(y), np.nan)
    full_preds = []
    for tr, te in splits:
        if spatial and buffer_m > 0:
            d, _ = cKDTree(xy[te]).query(xy[tr])
            tr = tr[d > buffer_m]
        w = (yu[tr] == 0).sum() / max(1, (yu[tr] == 1).sum())
        m = _model(cfg, w).fit(Xu[tr], yu[tr])
        oof[te] = m.predict_proba(Xu[te])[:, 1]
        if return_map and spatial:
            pred_all = m.predict_proba(X)[:, 1]
            full_preds.append(pred_all)
            pix = np.isin(groups, np.unique(gu[te]))
            oof_map[pix] = pred_all[pix]
    res = {"roc_auc": float(roc_auc_score(yu, oof)), "pr_auc": float(average_precision_score(yu, oof)),
           "base_rate": float(yu.mean()), "n_pos": int(yu.sum()), "n_neg": int((yu == 0).sum()),
           "n_features": len(feats), "buffer_m": float(buffer_m) if spatial else 0.0}
    if return_map:
        res["_oof_map"] = oof_map.reshape(ds.sizes["y"], ds.sizes["x"])
        # epistemic uncertainty in RANK space: how many percentile points a pixel's rank
        # moves between fold models (probability-space std collapses at the extremes)
        ranks = [100 * np.argsort(np.argsort(p)) / (len(p) - 1) for p in full_preds]
        res["_std_map"] = np.std(np.stack(ranks), axis=0).reshape(ds.sizes["y"], ds.sizes["x"])
    return res


def ablation(ds: xr.Dataset, cfg: dict) -> dict:
    """Does fusion beat any single modality? (spatial CV for all)."""
    all_f = feature_names(ds)
    sets = {
        "geophysics_only": [f for f in all_f if f in GROUPS["geophysics"]],
        "structure_only": [f for f in all_f if f in GROUPS["structure"]],
        "geochem_only": [f for f in all_f if f in GROUPS["geochem"]],
        "fused_all": all_f,
    }
    return {name: cross_validate(ds, cfg, f) for name, f in sets.items() if f}


def fit_predict(ds: xr.Dataset, cfg: dict):
    feats = feature_names(ds)
    X = to_matrix(ds, feats)
    y = ds["label"].values.ravel()
    use = y >= 0
    w = (y[use] == 0).sum() / max(1, (y[use] == 1).sum())
    model = _model(cfg, w).fit(X[use], y[use])
    prob = model.predict_proba(X)[:, 1].reshape(ds.sizes["y"], ds.sizes["x"])
    # percentile rank is the honest product for a PU problem: "top x % of the area"
    flat = prob.ravel()
    pct = (np.argsort(np.argsort(flat)) / (flat.size - 1)).reshape(prob.shape)
    return model, feats, prob.astype("float32"), pct.astype("float32")
