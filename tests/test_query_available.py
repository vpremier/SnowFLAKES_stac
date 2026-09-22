from pathlib import Path

import pandas as pd

from data_download import query_available


def test_month_ranges_are_at_most_one_month():
    assert query_available.month_ranges("2024-01-15", "2024-04-10") == [
        ("2024-01-15", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-03-01", "2024-04-01"),
        ("2024-04-01", "2024-04-10"),
    ]


def test_query_period_filters_tiles_and_skips_existing_csv(tmp_path, monkeypatch):
    aoi = tmp_path / "area.geojson"
    aoi.write_text("{}")
    query_dir = tmp_path / "QUERY"
    query_dir.mkdir()
    existing = query_dir / "Sentinel2_2024-01-01_2024-02-01.csv"
    existing.write_text("Name,tile\nold,T00AAA\n")

    calls = []

    def fake_query(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame(
            {
                "Name": ["keep", "skip"],
                "tile": ["T19HDE", "T32TPS"],
            }
        )

    monkeypatch.setattr(query_available, "query_google_sentinel2", fake_query)
    outputs = query_available.query_period(
        aoi,
        query_dir,
        "2024-02-01",
        "2024-03-01",
        80,
        "sentinel2",
        skip_sentinel2_tiles={"T32TPS"},
    )

    assert len(calls) == 1
    assert outputs == [query_dir / "Sentinel2_2024-02-01_2024-03-01.csv"]
    saved = pd.read_csv(outputs[0])
    assert saved["Name"].tolist() == ["keep"]

    query_available.query_period(
        aoi,
        query_dir,
        "2024-01-01",
        "2024-02-01",
        80,
        "sentinel2",
    )
    assert len(calls) == 1


def test_run_queries_creates_study_area_query_directory(tmp_path, monkeypatch):
    aoi = tmp_path / "area.shp"
    aoi.write_text("placeholder")
    monkeypatch.setattr(
        query_available,
        "query_period",
        lambda *args: [Path(args[1]) / "Sentinel2_test.csv"],
    )

    outputs = query_available.run_queries(
        "Alps",
        tmp_path,
        aoi,
        "2024-01-01",
        "2024-01-02",
        satellite="sentinel2",
    )

    assert (tmp_path / "Alps" / "QUERY").is_dir()
    assert outputs == [tmp_path / "Alps" / "QUERY" / "Sentinel2_test.csv"]


def test_config_mode_maps_existing_config_keys(tmp_path, monkeypatch):
    aoi = tmp_path / "area.geojson"
    aoi.write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"study_area": "Alps", "working_directory": "work", '
        f'"shapefile": "{aoi}", "date_start": "2024-01-01", '
        '"date_end": "2024-02-01", "satellite": "Sentinel-2", '
        '"max_cloudcover": 70, "s2_tile_skip": ["19HDE"], '
        '"sentinel2_source": "odata"}'
    )
    received = {}

    def fake_run_queries(**kwargs):
        received.update(kwargs)
        return []

    monkeypatch.setattr(query_available, "run_queries", fake_run_queries)
    query_available._run_from_config(config_path)

    assert received["study_area"] == "Alps"
    assert received["satellite"] == "sentinel2"
    assert received["max_cloudcover"] == 70
    assert received["skip_sentinel2_tiles"] == ["19HDE"]
    assert received["sentinel2_source"] == "odata"


def test_odata_query_keeps_id_column_and_filters_tiles(tmp_path, monkeypatch):
    aoi = tmp_path / "area.geojson"
    aoi.write_text("{}")
    query_dir = tmp_path / "QUERY"
    query_dir.mkdir()
    monkeypatch.setenv("CDSE_USERNAME", "user")
    monkeypatch.setenv("CDSE_PASSWORD", "password")

    def fake_odata(*args, **kwargs):
        return pd.DataFrame(
            {
                "Id": ["uuid-keep", "uuid-skip"],
                "Name": ["keep", "skip"],
                "tile": ["T19HDE", "T32TPS"],
            }
        )

    monkeypatch.setattr(query_available, "query_cdse", fake_odata)
    output = query_available.query_period(
        aoi,
        query_dir,
        "2024-01-01",
        "2024-02-01",
        80,
        "sentinel2",
        skip_sentinel2_tiles={"T32TPS"},
        sentinel2_source="odata",
    )[0]

    saved = pd.read_csv(output)
    assert saved["Id"].tolist() == ["uuid-keep"]


def test_odata_does_not_skip_google_csv_without_id(tmp_path):
    output = tmp_path / "google.csv"
    output.write_text("Name,tile\nscene,T19HDE\n")

    assert not query_available._existing_sentinel2_query_is_compatible(
        output, "odata"
    )
    assert query_available._existing_sentinel2_query_is_compatible(
        output, "google"
    )
