#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create SnowFLAKES input arrays from query CSV files.

``DOWNLOAD_MODE``/``download_mode`` selects ``STAC-API`` or ``RAW``.
``CROP``/``crop`` must be true for STAC-API. With raw archives, false means
that the full union of the downloaded tiles is mosaicked. ``SAVE``/``save``
controls whether merged GeoTIFF bands are written.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, *args, **kwargs):
        path = Path(dotenv_path or ".env")
        if not path.is_file():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
        return True

from data_download.landsat_query_download import download_landsat
from data_download.sentinel2_query_download import download_cdse
from loading.download_sentinel2_s2dl import download_s2dl
from loading.load_local_tiles import load_landsat_tiles, load_sentinel2_tiles


def _value(config, upper, lower, default=None):
    if upper in config:
        return config[upper]
    return config.get(lower, default)


def _study_directory(config):
    working = config.get("working_directory")
    if working is None:
        output = config.get("output_directory")
        if output:
            working = str(Path(output).parent)
    if working is None:
        raise ValueError("Config requires 'working_directory'")
    study = config.get("study_area") or config.get("study_area_name")
    if study is None:
        study = Path(config["shapefile"]).stem
    return Path(working) / study


def _read_query_files(study_dir, prefix):
    paths = sorted((study_dir / "QUERY").glob(f"{prefix}_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No {prefix} query CSV files found in {study_dir / 'QUERY'}"
        )
    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if "Name" not in result.columns:
        raise ValueError(f"{prefix} query files must contain a 'Name' column")
    return result.drop_duplicates(subset=["Name"]).reset_index(drop=True)


def _dates(products, sensor):
    names = products["Name"].astype(str)
    if sensor == "Sentinel-2":
        tokens = names.str.split("_").str[2].str[:8]
    else:
        tokens = names.str.split("_").str[3]
    return sorted(
        f"{token[:4]}-{token[4:6]}-{token[6:8]}"
        for token in tokens.dropna().unique()
    )


def _processed_dates(study_dir, sensor):
    """Read dates represented by already-created merged scene folders."""
    merged_dir = study_dir / "MERGED"
    if not merged_dir.is_dir():
        return set()
    dates = set()
    for scene_dir in merged_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        parts = scene_dir.name.split("_")
        index = 2 if sensor == "Sentinel-2" else 3
        if len(parts) <= index:
            continue
        token = parts[index][:8]
        if len(token) == 8 and token.isdigit():
            dates.add(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
    return dates


def _remove_processed(products, study_dir, sensor):
    processed = _processed_dates(study_dir, sensor)
    if not processed:
        return products
    dates = set(_dates(products, sensor)) - processed
    names = products["Name"].astype(str)
    if sensor == "Sentinel-2":
        product_dates = names.str.split("_").str[2].str[:8]
    else:
        product_dates = names.str.split("_").str[3].str[:8]
    keep = product_dates.map(
        lambda token: f"{token[:4]}-{token[4:6]}-{token[6:8]}" in dates
    )
    skipped = len(products) - int(keep.sum())
    if skipped:
        print(f"Skipping {skipped} {sensor} products for processed dates")
    return products[keep].reset_index(drop=True)


def _save_bands(data, output_dir, scene_id):
    import rasterio

    output_dir = Path(output_dir) / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = data.rio.transform()
    crs = data.rio.crs
    for band in data.coords["band"].values:
        path = output_dir / f"{scene_id}_{band}_toa.tif"
        if path.exists():
            continue
        values = data.sel(band=band).squeeze().values.astype("float32")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=float("nan"),
        ) as destination:
            destination.write(values, 1)


def _download_raw(config, study_dir, sentinel2, landsat):
    raw_dir = study_dir / "RAW"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if sentinel2 is not None and not sentinel2.empty:
        source = str(config.get("sentinel2_source", "google")).lower()
        if source == "odata":
            username = os.getenv("CDSE_USERNAME")
            password = os.getenv("CDSE_PASSWORD")
            if not username or not password:
                raise ValueError("CDSE_USERNAME and CDSE_PASSWORD are required")
            download_cdse(sentinel2, raw_dir, username, password)
            _extract_sentinel2_archives(raw_dir / "Sentinel2")
        else:
            download_s2dl(sentinel2, raw_dir / "Sentinel2")

    if landsat is not None and not landsat.empty:
        username = os.getenv("ERS_USERNAME")
        token = os.getenv("ERS_TOKEN")
        if not username or not token:
            raise ValueError("ERS_USERNAME and ERS_TOKEN are required")
        results = landsat.rename(columns={"Name": "displayId"}).copy()
        download_landsat(
            results,
            raw_dir,
            username,
            token,
            pathrowList=config.get("landsat_tile_list"),
            tierList=config.get("landsat_tiers", ["T1"]),
            folder_style="mission",
        )


def _extract_sentinel2_archives(sentinel_dir):
    """Extract OData ZIP archives while retaining the downloaded archives."""
    for archive in sentinel_dir.glob("*/S2*.zip"):
        product_id = archive.stem
        product_dir = archive.parent / product_id
        if product_dir.is_dir() and any(product_dir.iterdir()):
            continue
        product_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(product_dir)


def _load_stac_date(config, sensor, date, save, platform=None):
    params = config["resampling_params"]
    common = {
        "outdir": str(_study_directory(config) / "MERGED"),
        "date": date,
        "resolution": params["resolution"],
        "extent_target": params["extent_target"],
        "epsg_target": params["epsg_target"],
        "save": save,
        "shp": config["shapefile"],
        "exclude_tiles": config.get("exclude_tiles"),
    }
    if sensor == "Sentinel-2":
        from loading import load_stac

        load_stac.setup_cdse_credentials()
        return load_stac.convert_sentinel2_bands(**common)

    from loading import load_stac_usgs

    load_stac_usgs.setup_usgs_credentials()
    platform = platform or config["satellite"].upper().replace("-", "_")
    platform = {
        "LT05": "LANDSAT_5",
        "LE07": "LANDSAT_7",
        "LC08": "LANDSAT_8",
        "LC09": "LANDSAT_9",
    }.get(platform, platform)
    return load_stac_usgs.convert_landsat_bands(
        platform=platform,
        **common,
    )


def _load_raw_date(config, sensor, products, date, crop, save):
    params = config.get("resampling_params", {})
    extent = params.get("extent_target") if crop else None
    resolution = params.get(
        "resolution", 10 if sensor == "Sentinel-2" else 30
    )
    epsg = params.get("epsg_target")
    if sensor == "Sentinel-2":
        raw_dir = _study_directory(config) / "RAW" / "Sentinel2"
        data, scene_id = load_sentinel2_tiles(
            products,
            raw_dir,
            date,
            extent,
            resolution,
            epsg,
            config.get("exclude_tiles"),
        )
    else:
        raw_dir = _study_directory(config) / "RAW"
        data, scene_id = load_landsat_tiles(
            products,
            raw_dir,
            date,
            extent,
            resolution,
            epsg,
            config.get("exclude_tiles"),
        )
    if save and data is not None:
        _save_bands(data, _study_directory(config) / "MERGED", scene_id)
    return data, scene_id


def run(config_path):
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path(config_path).resolve().parent / ".env")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    mode = str(_value(config, "DOWNLOAD_MODE", "download_mode", "STAC-API"))
    mode = mode.lower().replace("_", "-").replace(" ", "-")
    if mode in {"stac", "stac-api", "cdse-stac-api"}:
        mode = "stac-api"
    elif mode in {"raw", "archives", "download"}:
        mode = "raw"
    else:
        raise ValueError("DOWNLOAD_MODE must be STAC-API or RAW")

    crop = bool(_value(config, "CROP", "crop", False))
    save = bool(_value(config, "SAVE", "save", False))
    if mode == "stac-api" and not crop:
        raise ValueError("CROP must be true when DOWNLOAD_MODE is STAC-API")

    params = config.get("resampling_params", {})
    if crop:
        required = ("extent_target", "resolution", "epsg_target")
        missing = [key for key in required if params.get(key) is None]
        if missing:
            raise ValueError(
                "resampling_params is required when CROP=true; missing: "
                + ", ".join(missing)
            )

    study_dir = _study_directory(config)
    satellite = str(config.get("satellite", "both")).lower()
    sentinel2 = (
        _read_query_files(study_dir, "Sentinel2")
        if satellite.startswith("sentinel") or satellite == "both"
        else pd.DataFrame(columns=["Name"])
    )
    landsat = (
        _read_query_files(study_dir, "Landsat")
        if satellite.startswith("landsat") or satellite == "both"
        else pd.DataFrame(columns=["Name"])
    )
    sentinel2 = _remove_processed(sentinel2, study_dir, "Sentinel-2")
    landsat = _remove_processed(landsat, study_dir, "Landsat")
    _download_raw(config, study_dir, sentinel2, landsat) if mode == "raw" else None

    arrays = []
    if mode == "stac-api":
        for date in _dates(sentinel2, "Sentinel-2"):
            data, scene_id = _load_stac_date(config, "Sentinel-2", date, save)
            if data is not None:
                arrays.append(("Sentinel-2", date, scene_id, data))
        for date in _dates(landsat, "Landsat"):
            date_products = landsat[
                landsat["Name"].astype(str).str.split("_").str[3].str[:8]
                == date.replace("-", "")
            ]
            for sensor_code in sorted(
                date_products["Name"].astype(str).str.split("_").str[0].unique()
            ):
                data, scene_id = _load_stac_date(
                    config,
                    "Landsat",
                    date,
                    save,
                    platform=sensor_code,
                )
                if data is not None:
                    arrays.append(("Landsat", date, scene_id, data))
    else:
        for date in _dates(sentinel2, "Sentinel-2"):
            data, scene_id = _load_raw_date(config, "Sentinel-2", sentinel2, date, crop, save)
            if data is not None:
                arrays.append(("Sentinel-2", date, scene_id, data))
        for date in _dates(landsat, "Landsat"):
            date_products = landsat[
                landsat["Name"].astype(str).str.split("_").str[3].str[:8]
                == date.replace("-", "")
            ]
            for sensor_code in sorted(
                date_products["Name"].astype(str).str.split("_").str[0].unique()
            ):
                sensor_products = date_products[
                    date_products["Name"].astype(str).str.startswith(sensor_code + "_")
                ]
                data, scene_id = _load_raw_date(
                    config, "Landsat", sensor_products, date, crop, save
                )
                if data is not None:
                    arrays.append(("Landsat", date, scene_id, data))

    config["output_directory"] = str(study_dir)
    if config.get("run_snowflakes", True):
        from SnowFLAKES.main_SnowFLAKES import run_snowflakes

        for _, _, scene_id, data in arrays:
            run_snowflakes(config, data, scene_id)
    return arrays


def main():
    parser = argparse.ArgumentParser(description="Load query results for SnowFLAKES")
    parser.add_argument("config")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path(args.config).resolve().parent / ".env")
    run(args.config)


if __name__ == "__main__":
    main()
