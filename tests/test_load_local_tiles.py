import numpy as np

from loading.load_local_tiles import (
    _calibrate_landsat,
    _fill_gaps,
    _sentinel2_band_path,
)


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


def test_sentinel_band_matcher_accepts_plain_band_names(tmp_path):
    band_path = tmp_path / "nested" / "B02.jp2"
    band_path.parent.mkdir()
    band_path.write_bytes(b"jp2")

    assert _sentinel2_band_path(tmp_path, "B02") == band_path


def test_sentinel_scene_directory_accepts_safe_suffix(tmp_path):
    from loading.load_local_tiles import _sentinel2_scene_dir

    product_id = "S2B_MSIL1C_20230403T143729_N0509_R096_T19HCD_20230403T175909"
    safe_dir = tmp_path / "T19HCD" / f"{product_id}.SAFE"
    safe_dir.mkdir(parents=True)

    assert _sentinel2_scene_dir(tmp_path, "T19HCD", product_id) == safe_dir
