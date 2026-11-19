"""Config loading and the target grid definition shared by every stage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from affine import Affine
from rasterio.crs import CRS

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(path.resolve().parents[1])
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Grid:
    """A north-up regular grid. Every raster and point dataset is snapped to this."""

    crs: CRS
    res: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_config(cls, cfg: dict) -> Grid:
        g = cfg["grid"]
        xmin, ymin, xmax, ymax = g["bounds"]
        return cls(CRS.from_user_input(g["crs"]), float(g["resolution"]), xmin, ymin, xmax, ymax)

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.xmin, 0, -self.res, self.ymax)

    @property
    def x(self) -> np.ndarray:
        return self.xmin + self.res * (np.arange(self.width) + 0.5)

    @property
    def y(self) -> np.ndarray:
        return self.ymax - self.res * (np.arange(self.height) + 0.5)

    def rowcol(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        cols = np.floor((np.asarray(xs) - self.xmin) / self.res).astype(int)
        rows = np.floor((self.ymax - np.asarray(ys)) / self.res).astype(int)
        return rows, cols

    def inside(self, rows, cols) -> np.ndarray:
        return (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
