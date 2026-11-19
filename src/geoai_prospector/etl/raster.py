"""Raster ETL: reproject any grid (GeoTIFF or NetCDF, any CRS) onto the model grid,
fill small gaps, and compute standard potential-field derivatives."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.fill import fillnodata
from rasterio.warp import Resampling, reproject

from ..config import Grid

log = logging.getLogger(__name__)


def _read_any(path: Path):
    """Return (array, transform, crs). NetCDF goes through xarray + rioxarray."""
    if path.suffix.lower() in (".nc", ".nc4", ".cdf"):
        import rioxarray  # noqa: F401  (registers .rio accessor)

        ds = xr.open_dataset(path)
        var = [v for v in ds.data_vars if ds[v].ndim == 2][0]
        da = ds[var]
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4283")  # GA grids without CRS metadata are GDA94 geographic
        da = da.rio.set_spatial_dims(x_dim=da.dims[-1], y_dim=da.dims[-2])
        return da.values.astype("float32"), da.rio.transform(), da.rio.crs
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        return arr, src.transform, src.crs


def to_grid(path: Path, grid: Grid, resampling="bilinear", fill_max_search_px=6) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a raster to the model grid. Returns (values, valid_mask_before_fill)."""
    arr, transform, crs = _read_any(Path(path))
    out = np.full(grid.shape, np.nan, dtype="float32")
    reproject(arr, out, src_transform=transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
              resampling=getattr(Resampling, resampling))
    valid = np.isfinite(out)
    log.info("%s: %s -> %s, %.1f%% valid", Path(path).name, crs.to_string(), grid.crs.to_string(), 100 * valid.mean())
    if fill_max_search_px and (~valid).any():
        sentinel = np.float32(-1e30)
        out = fillnodata(np.where(valid, out, sentinel), mask=valid.astype("uint8"),
                         max_search_distance=fill_max_search_px, smoothing_iterations=0)
        out[out == sentinel] = np.nan  # holes wider than the search distance stay NaN
    return out.astype("float32"), valid


# ---------- potential-field derivatives ----------

def _pad(a, n):
    return np.pad(np.where(np.isfinite(a), a, np.nanmean(a)), n, mode="reflect")


def hgm(a: np.ndarray, res: float) -> np.ndarray:
    """Horizontal gradient magnitude (units per metre)."""
    gy, gx = np.gradient(np.where(np.isfinite(a), a, np.nanmean(a)), res)
    return np.hypot(gx, gy)


def vertical_derivative(a: np.ndarray, res: float, order=1) -> np.ndarray:
    """nth vertical derivative via FFT: F^-1(|k|^n F(a)). Reflect-padded to limit edge ringing."""
    n = max(a.shape) // 4
    p = _pad(a, n)
    ky = np.fft.fftfreq(p.shape[0], d=res) * 2 * np.pi
    kx = np.fft.fftfreq(p.shape[1], d=res) * 2 * np.pi
    k = np.hypot(*np.meshgrid(kx, ky))
    out = np.real(np.fft.ifft2(np.fft.fft2(p - p.mean()) * k ** order))
    return out[n:-n, n:-n]


def tilt(a: np.ndarray, res: float) -> np.ndarray:
    """Tilt derivative arctan(VDR / HGM): amplitude-balanced edge enhancer, in radians."""
    return np.arctan2(vertical_derivative(a, res), hgm(a, res) + 1e-12)
