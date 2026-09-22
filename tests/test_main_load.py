import json

import pandas as pd
import pytest

from loading import main_load


def write_config(tmp_path, **updates):
    config = {
        "study_area": "Area",
        "working_directory": str(tmp_path),
        "shapefile": str(tmp_path / "area.geojson"),
        "satellite": "Sentinel-2",
        "DOWNLOAD_MODE": "STAC-API",
        "CROP": True,
        "SAVE": False,
        "run_snowflakes": False,
    }
    config.update(updates)
    config["shapefile"] = str(tmp_path / "area.geojson")
    (tmp_path / "area.geojson").write_text("{}")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_crop_requires_resampling_parameters(tmp_path):
    config_path = write_config(tmp_path)

    with pytest.raises(ValueError, match="resampling_params"):
        main_load.run(config_path)


def test_raw_full_tile_mode_does_not_require_crop_parameters(tmp_path, monkeypatch):
    config_path = write_config(
        tmp_path,
        DOWNLOAD_MODE="RAW",
        CROP=False,
    )
    query_dir = tmp_path / "Area" / "QUERY"
    query_dir.mkdir(parents=True)
    pd.DataFrame({"Name": []}).to_csv(
        query_dir / "Sentinel2_2024-01-01_2024-02-01.csv", index=False
    )
    monkeypatch.setattr(main_load, "_download_raw", lambda *args: None)

    assert main_load.run(config_path) == []


def test_processed_dates_are_removed_before_download(tmp_path):
    study_dir = tmp_path / "Area"
    merged = study_dir / "MERGED" / "S2A_MSIL1C_20240115T100000_N0510_R001_merged_20240115T120000"
    merged.mkdir(parents=True)
    products = pd.DataFrame(
        {
            "Name": [
                "S2A_MSIL1C_20240115T100000_N0510_R001_T32TPS_20240115T120000.SAFE",
                "S2A_MSIL1C_20240116T100000_N0510_R001_T32TPS_20240116T120000.SAFE",
            ]
        }
    )

    remaining = main_load._remove_processed(products, study_dir, "Sentinel-2")

    assert remaining["Name"].tolist() == [products.iloc[1]["Name"]]
