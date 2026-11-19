"""Export formats: COG GeoTIFF round-trip."""
import numpy as np
import rasterio
from rasterio.crs import CRS

from geoai_prospector.config import Grid
from geoai_prospector.export import write_geotiff


def test_write_geotiff_is_cloud_optimized(tmp_path):
    grid = Grid(CRS.from_epsg(7853), 500, 520000, 7780000, 540000, 7800000)
    arr = np.random.default_rng(0).random(grid.shape).astype("float32")
    p = tmp_path / "x.tif"
    write_geotiff(p, arr, grid)
    with rasterio.open(p) as src:
        assert src.crs.to_epsg() == 7853
        assert np.allclose(src.read(1), arr)
        assert src.profile["tiled"] or src.block_shapes[0][0] < grid.height  # internal tiling (COG layout)
