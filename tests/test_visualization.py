import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from loading.visualization import write_scene_vrt


class SceneVrtTests(unittest.TestCase):
    def test_stack_pixels_metadata_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / 'scene'
            scene.mkdir()
            transform = from_origin(100, 200, 20, 20)
            for token, value in [('B10', 10), ('B2', 2), ('B3', 3)]:
                with rasterio.open(scene / f'scene_{token}_toa.tif', 'w',
                                   driver='GTiff', width=3, height=2, count=1,
                                   dtype='float32', crs='EPSG:32632',
                                   transform=transform, nodata=-9999) as dst:
                    dst.write(np.full((2, 3), value, dtype='float32'), 1)
            vrt = write_scene_vrt(root, 'scene')
            self.assertNotIn(str(root), vrt.read_text())
            moved = root / 'relocated'
            shutil.move(str(scene), moved)
            with rasterio.open(moved / vrt.name) as source:
                self.assertEqual(source.count, 3)
                self.assertEqual(source.descriptions, ('B2', 'B3', 'B10'))
                self.assertEqual(source.transform, transform)
                self.assertEqual(source.crs.to_epsg(), 32632)
                self.assertEqual(source.nodatavals, (-9999,) * 3)
                np.testing.assert_array_equal(source.read()[:, 0, 0], [2, 3, 10])

    def test_no_saved_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(write_scene_vrt(directory, 'missing'))


if __name__ == '__main__':
    unittest.main()
