import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from geoai_prospector.config import Grid
from geoai_prospector.etl import points, raster
from geoai_prospector.features import structure

GRID = Grid(rasterio.crs.CRS.from_epsg(7853), 1000, 520000, 7780000, 540000, 7800000)


def test_grid_geometry():
    assert GRID.shape == (20, 20)
    r, c = GRID.rowcol([520500], [7799500])
    assert (r[0], c[0]) == (0, 0)


def test_reprojects_geographic_raster_onto_grid(tmp_path):
    # a GDA94 lon/lat raster covering the grid, value = longitude
    lon0, lat0, res = 135.0, -19.8, 0.005
    w, h = 120, 120
    lons = lon0 + res * (np.arange(w) + 0.5)
    arr = np.tile(lons, (h, 1)).astype("float32")
    p = tmp_path / "lon.tif"
    with rasterio.open(p, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs="EPSG:4283",
                       transform=from_origin(lon0, lat0, res, res)) as dst:
        dst.write(arr, 1)
    out, valid = raster.to_grid(p, GRID)
    assert out.shape == GRID.shape and valid.all()
    assert np.all(np.diff(out, axis=1) > 0)  # longitude increases eastwards after reprojection


def test_geochem_loader_handles_bdl_and_units(tmp_path):
    df = pd.DataFrame({"LONGITUDE": [135.25, 135.26], "LATITUDE": [-19.75, -19.76],
                       "Cu_ppm": [50, -2], "Au_ppb": [1000, 5]})
    p = tmp_path / "g.csv"
    df.to_csv(p, index=False)
    grid = Grid(rasterio.crs.CRS.from_epsg(7853), 500, 500000, 7790000, 560000, 7830000)
    spec = {"x": "LONGITUDE", "y": "LATITUDE", "crs": "EPSG:4283",
            "columns": {"Cu": {"col": "Cu_ppm", "unit": "ppm"}, "Au": {"col": "Au_ppb", "unit": "ppb"}}}
    g = points.load_geochem(p, spec, ["Cu", "Au"], grid)
    assert len(g) == 2
    assert g["Au"].iloc[0] == 1.0  # 1000 ppb -> 1 ppm
    assert g["Cu_bdl"].iloc[1] and g["Cu_dl"].iloc[1] == 2


def test_lineaments_find_a_planted_fault():
    a = np.zeros((80, 80))
    a[:, 40:] = 10.0  # vertical step = fault
    edges = structure.lineaments(a, sigma=2, min_px=5)
    cols = np.where(edges)[1]
    assert len(cols) > 40 and abs(np.median(cols) - 39.5) <= 1.5
    d = structure.distance_to(edges, 500)
    assert d[:, 0].min() > 15000


def test_complexity_peaks_at_intersection():
    a = np.zeros((80, 80))
    a[:, 40:] += 10
    a[40:, :] += 10
    mask = structure.lineaments(a, 2, 5)
    _, cplx = structure.structural_complexity(a, mask, 500, 2000)
    r, c = np.unravel_index(np.argmax(cplx), cplx.shape)
    assert abs(r - 40) < 8 and abs(c - 40) < 8
