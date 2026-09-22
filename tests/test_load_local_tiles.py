import numpy as np

from loading.load_local_tiles import _calibrate_landsat, _fill_gaps


def test_fill_gaps_preserves_first_tile_and_uses_second_for_missing_pixels():
    first = np.array([[1.0, np.nan], [3.0, np.nan]], dtype="float32")
    second = np.array([[9.0, 2.0], [8.0, np.nan]], dtype="float32")

    result = _fill_gaps(first, second)

    np.testing.assert_allclose(
        result,
        np.array([[1.0, 2.0], [3.0, np.nan]], dtype="float32"),
        equal_nan=True,
    )


def test_landsat_8_band_6_is_reflective_not_thermal():
    metadata = {
        "REFLECTANCE_MULT_BAND_6": "0.00002",
        "REFLECTANCE_ADD_BAND_6": "-0.1",
        "SUN_ELEVATION": "60",
    }
    values = np.array([[10000.0]], dtype="float32")

    result = _calibrate_landsat(values, 6, "LC08", metadata)

    expected = (10000 * 0.00002 - 0.1) / np.cos(np.radians(30))
    np.testing.assert_allclose(result, [[expected]], rtol=1e-6)
