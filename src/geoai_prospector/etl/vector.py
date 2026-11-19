"""Vector ETL: independently mapped faults -> distance raster.

Scientific point: a magnetic or gravity lineament detected by computer vision is a
geophysical edge, not automatically a geological fault. The model therefore carries
BOTH kinds of evidence as separate features:

* ``dist_deep_lineament_m`` etc. - CV-detected edges (Model A, no interpretation)
* ``dist_mapped_fault_m`` - faults mapped by geologists (NTGS / GA surface geology),
  an independent line of evidence

In synthetic mode the planted deep faults stand in for the mapped-fault layer (in
reality only a subset of structures is mapped, so treat the synthetic case as the
optimistic bound).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import LineString, box

from ..config import Grid
from ..features.structure import distance_to

log = logging.getLogger(__name__)


def _rasterize_lines(geoms, grid: Grid) -> np.ndarray:
    shapes_ = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes_:
        return np.zeros(grid.shape, bool)
    mask = rasterize(shapes_, out_shape=grid.shape, transform=grid.transform, all_touched=True, fill=0)
    return mask.astype(bool)


def synthetic_mapped_faults(truth_path: Path, grid: Grid) -> np.ndarray | None:
    """Deep planted faults from truth.json as 'mapped' faults (distance in metres)."""
    if not truth_path.exists():
        return None
    truth = json.loads(truth_path.read_text())
    bbox = box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    L = 2 * max(grid.xmax - grid.xmin, grid.ymax - grid.ymin)
    geoms = []
    for f in truth["faults"]:
        if not f["deep"]:
            continue  # geological maps usually carry the major structures only
        p, d = np.array(f["p"]), np.array(f["d"])
        geoms.append(LineString([p - L * d, p + L * d]).intersection(bbox))
    mask = _rasterize_lines(geoms, grid)
    log.info("mapped faults (synthetic truth): %d px", int(mask.sum()))
    return distance_to(mask, grid.res)


def real_mapped_faults(path: Path, src: dict, grid: Grid) -> np.ndarray | None:
    """Mapped fault lines from a GeoPackage/shapefile/GeoJSON (e.g. NTGS 1:250k geology)."""
    if not path.exists():
        log.warning("no mapped-fault layer at %s - the model runs without it (see docs/DATA.md)", path)
        return None
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None and src.get("crs"):
        gdf = gdf.set_crs(src["crs"])
    gdf = gdf.to_crs(grid.crs)
    mask = _rasterize_lines(list(gdf.geometry), grid)
    log.info("mapped faults (%s): %d features, %d px", path.name, len(gdf), int(mask.sum()))
    return distance_to(mask, grid.res)
