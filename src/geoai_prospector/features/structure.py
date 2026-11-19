"""Model A - computer vision for structural geology.

Two complementary detectors work on the geophysical grids treated as images:

1. Classical edge detection (Canny on the normalised field). Large smoothing on gravity
   picks long-wavelength, deep-sourced boundaries ("deep crustal faults"); small
   smoothing on magnetics picks shallow contacts and dykes. Edges are skeletonised and
   short fragments removed, then turned into continuous metrics:
   distance-to-fault, lineament density and structural complexity (where differently
   oriented lineaments meet, i.e. fault intersections).

2. A small convolutional autoencoder (PyTorch) trained self-supervised on
   magnetic+gravity patches. It needs no labels. Its latent features (reduced by PCA)
   describe geophysical texture, and its reconstruction error highlights unusual
   texture that the network could not compress, often complex structure or
   intrusions.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from skimage.feature import canny
from skimage.morphology import remove_small_objects, skeletonize

log = logging.getLogger(__name__)


def normalise(a: np.ndarray) -> np.ndarray:
    """Robust scaling to roughly [0, 1] (2nd-98th percentile)."""
    lo, hi = np.nanpercentile(a, [2, 98])
    return np.clip((np.nan_to_num(a, nan=np.nanmedian(a)) - lo) / (hi - lo + 1e-12), 0, 1)


def lineaments(field: np.ndarray, sigma: float, min_px: int) -> np.ndarray:
    """Canny -> skeleton -> drop short fragments. Returns a boolean edge mask."""
    edges = canny(normalise(field), sigma=sigma, low_threshold=0.05, high_threshold=0.15, use_quantiles=False)
    edges = skeletonize(edges)
    return remove_small_objects(edges, max_size=min_px, connectivity=2) if _has_max_size() else \
        remove_small_objects(edges, min_size=min_px, connectivity=2)


def _has_max_size() -> bool:
    import inspect

    return "max_size" in inspect.signature(remove_small_objects).parameters


def distance_to(mask: np.ndarray, res: float) -> np.ndarray:
    """Euclidean distance (m) from every pixel to the nearest True pixel."""
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype="float32")
    return (distance_transform_edt(~mask) * res).astype("float32")


def structural_complexity(field: np.ndarray, mask: np.ndarray, res: float, radius_m: float):
    """Lineament density and orientation diversity in a Gaussian window.

    Orientation is taken from the gradient direction (theta, mod pi). Using doubled
    angles, R = |sum w exp(2i theta)| / sum w is ~1 where all edges are parallel and ~0
    where several trends cross. complexity = density * (1 - R) peaks at intersections.
    """
    sm = gaussian_filter(normalise(field), 1.0)
    gx, gy = sobel(sm, axis=1), sobel(sm, axis=0)
    theta = np.arctan2(gy, gx)
    w = mask.astype(float)
    s = radius_m / res
    dens = gaussian_filter(w, s)
    c = gaussian_filter(w * np.cos(2 * theta), s)
    sn = gaussian_filter(w * np.sin(2 * theta), s)
    R = np.hypot(c, sn) / (dens + 1e-9)
    density_per_km = dens * (1000.0 / res)  # ~km of lineament per km2
    return density_per_km.astype("float32"), (dens * (1 - np.clip(R, 0, 1)) * 1000.0 / res).astype("float32")


# ---------------- CNN autoencoder ----------------

def _build_autoencoder(in_ch: int, latent: int):
    import torch.nn as nn

    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, latent, 3, stride=2, padding=1), nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(latent, 32, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, in_ch, 3, padding=1),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    return ConvAE()


def cnn_features(stack: np.ndarray, patch=32, stride=8, epochs=25, latent=32, n_pcs=3, seed=0):
    """Train the autoencoder on patches of `stack` (C, H, W) and return
    (pcs (n_pcs, H, W), recon_error (H, W), training loss curve)."""
    import torch
    import torch.nn.functional as F
    from sklearn.decomposition import PCA

    torch.manual_seed(seed)
    np.random.seed(seed)
    C, H, W = stack.shape
    x = np.stack([normalise(c) for c in stack]).astype("float32")
    patches = [x[:, i:i + patch, j:j + patch]
               for i in range(0, H - patch + 1, stride) for j in range(0, W - patch + 1, stride)]
    data = torch.from_numpy(np.stack(patches))
    # simple augmentation: 90 degree rotations + flips (structure has no preferred orientation a priori)
    model = _build_autoencoder(C, latent)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(data))
        tot = 0.0
        for b in range(0, len(data), 64):
            xb = data[perm[b:b + 64]]
            k = np.random.randint(4)
            xb = torch.rot90(xb, k, dims=(2, 3))
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=(3,))
            loss = F.mse_loss(model(xb), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        losses.append(tot / len(data))
    log.info("CNN autoencoder: %d patches, final MSE %.5f", len(data), losses[-1])

    # fully convolutional inference on the whole grid (pad to a multiple of 4)
    model.eval()
    ph, pw = (-H) % 4, (-W) % 4
    full = torch.from_numpy(np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect"))[None]
    with torch.no_grad():
        z = model.enc(full)
        rec = model.dec(z)
        err = ((rec - full) ** 2).mean(1)[0, :H, :W].numpy()
        zup = F.interpolate(z, size=full.shape[-2:], mode="bilinear", align_corners=False)[0, :, :H, :W].numpy()
    feats = zup.reshape(latent, -1).T
    pcs = PCA(n_components=n_pcs, random_state=seed).fit_transform(feats).T.reshape(n_pcs, H, W)
    return pcs.astype("float32"), gaussian_filter(err, 1.0).astype("float32"), losses


def run(mag: np.ndarray, grav: np.ndarray, res: float, cfg: dict) -> dict[str, np.ndarray]:
    s = cfg["structure"]
    src = {"mag": mag, "grav": grav}
    deep = lineaments(src[s["deep_source"]], s["deep_sigma"], s["min_lineament_px"])
    shallow = lineaments(src[s["shallow_source"]], s["shallow_sigma"], s["min_lineament_px"])
    dens, cplx = structural_complexity(src[s["deep_source"]], deep | shallow, res, s["density_radius_m"])
    out = {
        "lineament_deep": deep,
        "lineament_shallow": shallow,
        "dist_deep_fault_m": distance_to(deep, res),
        "dist_shallow_lineament_m": distance_to(shallow, res),
        "lineament_density": dens,
        "structural_complexity": cplx,
    }
    log.info("lineaments: deep %d px, shallow %d px", deep.sum(), shallow.sum())
    if s["cnn"]["enabled"]:
        c = s["cnn"]
        pcs, err, losses = cnn_features(np.stack([mag, grav]), c["patch"], c["stride"], c["epochs"],
                                        c["latent_channels"], c["n_pcs"], c["seed"])
        for i, pc in enumerate(pcs):
            out[f"cnn_pc{i + 1}"] = pc
        out["cnn_recon_error"] = err
        out["_cnn_loss"] = np.array(losses)
    return out
