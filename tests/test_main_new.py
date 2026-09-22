import main_new


def base_config(full_tiles):
    return {
        "download_full_tiles": full_tiles,
        "satellite": "Sentinel-2",
        "date_start": "2024-01-01",
        "date_end": "2024-01-02",
        "resampling_params": {
            "extent_target": [0, 0, 20, 20],
            "resolution": 10,
            "epsg_target": 32632,
        },
    }


def test_aoi_strategy_uses_existing_workflow(monkeypatch):
    config = base_config(False)
    monkeypatch.setattr(main_new, "load_config", lambda path: config)
    monkeypatch.setattr(main_new, "load_dotenv", lambda: None)

    received = {}

    def fake_workflow(start, end, path):
        received.update(start=start, end=end, path=path)
        return "aoi"

    monkeypatch.setattr(main_new, "run_workflow", fake_workflow)

    assert main_new.run("config.json") == "aoi"
    assert received == {
        "start": "2024-01-01",
        "end": "2024-01-02",
        "path": "config.json",
    }


def test_tile_strategy_downloads_then_processes(monkeypatch):
    config = base_config(True)
    products = object()
    calls = []
    monkeypatch.setattr(main_new, "load_config", lambda path: config)
    monkeypatch.setattr(main_new, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        main_new,
        "query_and_download_tiles",
        lambda received: calls.append(("download", received)) or products,
    )
    monkeypatch.setattr(
        main_new,
        "process_downloaded_tiles",
        lambda received_products, received_config: calls.append(
            ("process", received_products, received_config)
        ),
    )

    assert main_new.run("config.json") is products
    assert calls == [
        ("download", config),
        ("process", products, config),
    ]


def test_download_full_tiles_must_be_boolean():
    config = base_config("yes")
    try:
        main_new.check_new_config(config)
    except ValueError as error:
        assert "true or false" in str(error)
    else:
        raise AssertionError("String strategy flag must be rejected")
