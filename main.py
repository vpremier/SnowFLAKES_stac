#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the complete SnowFLAKES query, preparation, and processing workflow.

``DOWNLOAD_SENTINEL`` selects ``OData``, ``S3``, ``Google``, ``STAC-API``, or
false. ``DOWNLOAD_LANDSAT`` selects ``STAC-API``, ``USGS-M2M``, or false.
``CROP``/``crop`` must be true for STAC-API. With raw archives, false means
that each complete tile is processed separately while retaining its own
reprojected extent. ``SAVE``/``save`` controls whether prepared GeoTIFF bands
are written.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from data_download.landsat_query_download import download_landsat
from data_download.sentinel2_query_download import download_cdse
from data_download.sentinel2_s3_download import download_sentinel2_s3
from loading.download_sentinel2_s2dl import download_s2dl
from loading.load_local_tiles import (
    load_landsat_tiles,
    load_prepared_bands,
    load_sentinel2_tiles,
    prepared_bands_are_complete,
    prepared_grid_matches,
)


def _value(config, upper, lower, default=None):
    if upper in config:
        return config[upper]
    return config.get(lower, default)


def _download_flag(value, allowed, name):
    """Normalize a sensor download flag, accepting JSON false."""
    if value is False or value is None:
        return False
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if normalized in {"false", "none", "off", "0", ""}:
        return False
    aliases = {
        "stac": "stac-api",
        "cdse-stac": "stac-api",
        "usgs": "usgs-m2m",
        "m2m": "usgs-m2m",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed | {"false"}))
        raise ValueError(f"{name} must be one of: {choices}")
    return normalized


def _working_directory(config):
    working = config.get("working_directory")
    if working is None:
        output = config.get("output_directory")
        if output:
            working = str(Path(output).parent)
    if working is None:
        raise ValueError("Config requires 'working_directory'")
    return Path(working)


def _study_directory(config):
    working = _working_directory(config)
    study = config.get("study_area") or config.get("study_area_name")
    if study is None:
        study = Path(config["shapefile"]).stem
    return working / study


def _landsat_mission(sensor_code):
    return {
        "LT05": "Landsat-5",
        "LE07": "Landsat-7",
        "LC08": "Landsat-8",
        "LC09": "Landsat-9",
    }.get(str(sensor_code).upper(), str(sensor_code))


def _sensor_directory(root, sensor, platform=None):
    """Return the requested sensor/mission directory below a storage root."""
    root = Path(root)
    if sensor == "Sentinel-2":
        return root / "Sentinel-2"
    mission = _landsat_mission(platform) if platform else "Landsat"
    return root / "Landsat" / mission


def _read_query_files(study_dir, prefix, date_start=None, date_end=None):
    paths = sorted((study_dir / "QUERY").glob(f"{prefix}_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No {prefix} query CSV files found in {study_dir / 'QUERY'}"
        )
    frames = [pd.read_csv(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if "Name" not in result.columns:
        raise ValueError(f"{prefix} query files must contain a 'Name' column")
    result = result.drop_duplicates(subset=["Name"]).reset_index(drop=True)
    if date_start is None and date_end is None:
        return result

    sensor = "Sentinel-2" if prefix.lower().startswith("sentinel") else "Landsat"
    date_index = 2 if sensor == "Sentinel-2" else 3
    tokens = result["Name"].astype(str).str.split("_").str[date_index]
    parsed = pd.to_datetime(tokens.str[:8], format="%Y%m%d", errors="coerce")
    start = pd.Timestamp(date_start) if date_start is not None else parsed.min()
    end = pd.Timestamp(date_end) if date_end is not None else parsed.max() + pd.Timedelta(days=1)
    selected = result.loc[parsed.ge(start) & parsed.lt(end)].reset_index(drop=True)
    skipped = len(result) - len(selected)
    if skipped:
        print(
            f"Ignoring {skipped} {sensor} query scene(s) outside "
            f"the configured period [{start.date()}, {end.date()})"
        )
    return selected


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


def _remove_processed(
    products,
    study_dir,
    sensor,
    exclude_tiles=None,
    resolution=None,
    epsg=None,
    extent=None,
):
    """Exclude product groups having a complete cropped prepared cache."""
    if products.empty:
        return products
    names = products["Name"].astype(str)
    if sensor == "Sentinel-2":
        group_keys = names.str.split("_").str[2].str[:8]
    else:
        group_keys = (
            names.str.split("_").str[0]
            + "|"
            + names.str.split("_").str[3].str[:8]
        )

    keep = pd.Series(True, index=products.index)
    for _, indexes in products.groupby(group_keys).groups.items():
        group = products.loc[indexes]
        platform = (
            str(group.iloc[0]["Name"]).split("_")[0]
            if sensor != "Sentinel-2"
            else None
        )
        merged_dir = _sensor_directory(
            Path(study_dir) / "MERGED", sensor, platform
        )
        token = str(group.iloc[0]["Name"]).split("_")[
            2 if sensor == "Sentinel-2" else 3
        ][:8]
        date = f"{token[:4]}-{token[4:6]}-{token[6:8]}"
        scene_id = _prepared_scene_id(
            group,
            sensor,
            date,
            merge=True,
            exclude_tiles=exclude_tiles,
        )
        cache_matches = scene_id and prepared_bands_are_complete(
            merged_dir, scene_id
        )
        if cache_matches and resolution is not None and epsg is not None:
            cache_matches = prepared_grid_matches(
                merged_dir, scene_id, resolution, epsg, extent
            )
        if cache_matches:
            keep.loc[indexes] = False

    skipped = len(products) - int(keep.sum())
    if skipped:
        print(f"Skipping download of {skipped} cached {sensor} products")
    return products[keep].reset_index(drop=True)


def _selected_raw_products(products, sensor, date, exclude_tiles=None):
    """Select one sensor/date using the same rules as the local loaders."""
    names = products["Name"].astype(str)
    date_token = pd.Timestamp(date).strftime("%Y%m%d")
    date_index = 2 if sensor == "Sentinel-2" else 3
    tile_index = 5 if sensor == "Sentinel-2" else 2
    selected = products[
        names.str.split("_").str[date_index].str[:8] == date_token
    ]
    excluded = set(exclude_tiles or [])
    if excluded:
        selected = selected[
            ~selected["Name"].astype(str).str.split("_").str[tile_index].isin(
                excluded
            )
        ]
    return selected


def _prepared_scene_id(products, sensor, date, merge, exclude_tiles=None):
    """Derive the scene identifier used by the prepared GeoTIFF cache."""
    selected = _selected_raw_products(
        products, sensor, date, exclude_tiles=exclude_tiles
    )
    if selected.empty:
        return None
    scene_id = str(selected.iloc[0]["Name"]).removesuffix(".SAFE")
    parts = scene_id.split("_")
    if merge:
        parts[5 if sensor == "Sentinel-2" else 2] = "merged"
    return "_".join(parts)


def _remove_cached_tiles(
    products, study_dir, sensor, resolution=None, epsg=None
):
    """Exclude uncropped products having a complete per-tile cache."""
    if products.empty:
        return products
    keep = []
    for _, row in products.iterrows():
        scene_id = str(row["Name"]).removesuffix(".SAFE")
        tile = _tile_name(sensor, scene_id)
        platform = (
            scene_id.split("_", 1)[0] if sensor != "Sentinel-2" else None
        )
        output_dir = _sensor_directory(
            Path(study_dir) / "TILES", sensor, platform
        ) / tile
        cached = prepared_bands_are_complete(output_dir, scene_id)
        if cached and resolution is not None and epsg is not None:
            cached = prepared_grid_matches(
                output_dir, scene_id, resolution, epsg
            )
        keep.append(not cached)
    skipped = len(products) - sum(keep)
    if skipped:
        print(f"Skipping download of {skipped} cached {sensor} products")
    return products[pd.Series(keep, index=products.index)].reset_index(drop=True)


def _save_bands(data, output_dir, scene_id):
    import rasterio

    output_dir = Path(output_dir) / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = data.rio.transform()
    crs = data.rio.crs
    for band in data.coords["band"].values:
        path = output_dir / f"{scene_id}_{band}_toa.tif"
        if path.exists():
            print(f"  Prepared band already exists, skipping: {path.name}")
            continue
        print(f"  Saving prepared band {band}: {path}")
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


def _download_raw(
    config,
    study_dir,
    sentinel2,
    landsat,
    sentinel_source="google",
    landsat_source="usgs-m2m",
):
    raw_dir = _working_directory(config) / "RAW"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def log_error(sensor, scene, reason):
        log_path = raw_dir / "download_errors.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"{sensor}\t{scene}\t{reason}\n")
        print(f"Download validation failed for {scene}; logged to {log_path}")

    def validate_sentinel(products):
        invalid = []
        for _, row in products.iterrows():
            name = str(row["Name"])
            product = name.removesuffix(".SAFE")
            parts = product.split("_")
            if len(parts) < 6:
                log_error("Sentinel-2", name, "invalid product name")
                invalid.append(name)
                continue
            scene_dir = raw_dir / "Sentinel-2" / parts[5]
            candidates = [scene_dir / product, scene_dir / f"{product}.SAFE", scene_dir / f"{product}.zip"]
            existing = next((path for path in candidates if path.exists()), None)
            try:
                if existing is None or existing.stat().st_size == 0:
                    raise ValueError("missing or empty product")
                if existing.suffix.lower() == ".zip":
                    with zipfile.ZipFile(existing) as archive:
                        if archive.testzip() is not None:
                            raise ValueError("invalid ZIP member")
                elif not any(
                    path.is_file() and path.stat().st_size > 0
                    for path in existing.rglob("*")
                ):
                    raise ValueError("empty SAFE directory")
            except Exception as error:
                for candidate in candidates:
                    if candidate.is_dir():
                        shutil.rmtree(candidate, ignore_errors=True)
                    elif candidate.exists():
                        try:
                            candidate.unlink()
                        except OSError:
                            pass
                log_error("Sentinel-2", name, str(error))
                invalid.append(name)
        return invalid

    def validate_landsat(products):
        mission = {"LT05": "Landsat-5", "LE07": "Landsat-7", "LC08": "Landsat-8", "LC09": "Landsat-9"}
        invalid = []
        for _, row in products.iterrows():
            name = str(row["Name"])
            parts = name.split("_")
            sensor = parts[0] if parts else "unknown"
            pathrow = parts[2] if len(parts) > 2 else "unknown"
            archive = raw_dir / "Landsat" / mission.get(sensor, sensor) / pathrow / f"{name}.tar"
            try:
                if not archive.is_file() or archive.stat().st_size == 0:
                    raise ValueError("missing or empty archive")
                with tarfile.open(archive, mode="r") as tar:
                    if not tar.getmembers():
                        raise ValueError("empty TAR archive")
            except Exception as error:
                if archive.exists():
                    try:
                        archive.unlink()
                    except OSError:
                        pass
                log_error("Landsat", name, str(error))
                invalid.append(name)
        return invalid

    invalid_sentinel = []
    invalid_landsat = []
    if sentinel_source not in {False, "false", "none", ""} and sentinel2 is not None and not sentinel2.empty:
        source = str(sentinel_source).lower()
        products_to_validate = sentinel2
        if source == "s3":
            try:
                result = download_sentinel2_s3(
                    sentinel2, raw_dir / "Sentinel-2", config, return_status=True
                )
                failed_names = set(result.get("failed", []))
                if failed_names:
                    products_to_validate = sentinel2[
                        ~sentinel2["Name"].astype(str).isin(failed_names)
                    ]
            except Exception as error:
                print(f"Sentinel-2 S3 download failed: {error}; validating scenes and continuing")
        elif source == "odata":
            username = os.getenv("CDSE_USERNAME")
            password = os.getenv("CDSE_PASSWORD")
            if not username or not password:
                raise ValueError("CDSE_USERNAME and CDSE_PASSWORD are required")
            try:
                download_cdse(sentinel2, raw_dir, username, password)
                _extract_sentinel2_archives(raw_dir / "Sentinel-2")
            except Exception as error:
                print(f"Sentinel-2 OData download failed: {error}; validating scenes and continuing")
        else:
            try:
                # S2DL handles scenes independently.  Keep track of scenes it
                # actually attempted so a single failed scene does not make
                # every unconfirmed query result appear as another failure.
                result = download_s2dl(
                    sentinel2, raw_dir / "Sentinel-2", return_status=True
                )
                failed_names = set(result.get("failed", []))
                if failed_names:
                    products_to_validate = sentinel2[
                        ~sentinel2["Name"].astype(str).isin(failed_names)
                    ]
            except Exception as error:
                print(f"Sentinel-2 Google download failed: {error}; validating scenes and continuing")
        invalid_sentinel = validate_sentinel(products_to_validate)

    if landsat_source not in {False, "false", "none", ""} and landsat is not None and not landsat.empty:
        username = os.getenv("ERS_USERNAME")
        token = os.getenv("ERS_TOKEN")
        if not username or not token:
            raise ValueError("ERS_USERNAME and ERS_TOKEN are required")
        results = landsat.rename(columns={"Name": "displayId"}).copy()
        try:
            download_landsat(
                results,
                raw_dir,
                username,
                token,
                pathrowList=config.get("landsat_tile_list"),
                tierList=config.get("landsat_tiers", ["T1"]),
                folder_style="mission",
            )
        except Exception as error:
            print(f"Landsat download failed: {error}; validating scenes and continuing")
        invalid_landsat = validate_landsat(landsat)

    return invalid_sentinel, invalid_landsat


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


def _load_stac_date(
    config, sensor, date, save, platform=None, products=None
):
    params = config["resampling_params"]
    platform_code = platform
    if sensor != "Sentinel-2" and platform_code is None and products is not None:
        platform_code = str(products.iloc[0]["Name"]).split("_")[0]
    prepared_root = _sensor_directory(
        Path(_study_directory(config)) / "MERGED", sensor, platform_code
    )
    if products is not None:
        scene_id = _prepared_scene_id(
            products,
            sensor,
            date,
            merge=True,
            exclude_tiles=config.get("exclude_tiles"),
        )
        if scene_id is not None:
            if prepared_grid_matches(
                prepared_root,
                scene_id,
                params["resolution"],
                params["epsg_target"],
                params["extent_target"],
            ):
                cached = load_prepared_bands(prepared_root, scene_id, date)
                print(f"Loading prepared bands from cache: {scene_id}")
                return cached, scene_id

    common = {
        "outdir": str(prepared_root),
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


def _load_raw_date(
    config,
    sensor,
    products,
    date,
    crop,
    save,
    save_dir=None,
    merge=True,
):
    params = config.get("resampling_params", {})
    platform = (
        str(products.iloc[0]["Name"]).split("_")[0]
        if sensor != "Sentinel-2" and not products.empty
        else None
    )
    prepared_dir = Path(save_dir) if save_dir else _sensor_directory(
        Path(_study_directory(config)) / "MERGED", sensor, platform
    )
    prepared_scene_id = _prepared_scene_id(
        products,
        sensor,
        date,
        merge,
        exclude_tiles=config.get("exclude_tiles"),
    )
    target_extent = params.get("extent_target") if crop else None
    if prepared_scene_id is not None and prepared_grid_matches(
        prepared_dir,
        prepared_scene_id,
        params.get("resolution"),
        params.get("epsg_target"),
        target_extent,
    ):
        cached = load_prepared_bands(
            prepared_dir, prepared_scene_id, date
        )
        print(f"Loading prepared bands from cache: {prepared_scene_id}")
        return cached, prepared_scene_id

    extent = target_extent
    resolution = params.get(
        "resolution", 10 if sensor == "Sentinel-2" else 30
    )
    epsg = params.get("epsg_target")
    if sensor == "Sentinel-2":
        raw_dir = _working_directory(config) / "RAW" / "Sentinel-2"
        if not raw_dir.exists():
            legacy_dir = _working_directory(config) / "RAW" / "Sentinel2"
            if legacy_dir.exists():
                raw_dir = legacy_dir
        data, scene_id = load_sentinel2_tiles(
            products,
            raw_dir,
            date,
            extent,
            resolution,
            epsg,
            config.get("exclude_tiles"),
            merge=merge,
        )
    else:
        raw_dir = _working_directory(config) / "RAW"
        data, scene_id = load_landsat_tiles(
            products,
            raw_dir,
            date,
            extent,
            resolution,
            epsg,
            config.get("exclude_tiles"),
            merge=merge,
        )
    if save and data is not None:
        _save_bands(data, prepared_dir, scene_id)
    return data, scene_id


def _tile_name(sensor, product_name):
    """Return the MGRS tile or WRS path/row encoded in a product name."""
    parts = str(product_name).removesuffix(".SAFE").split("_")
    return parts[5] if sensor == "Sentinel-2" else parts[2]


def _tile_output_directory(study_dir, sensor, tile, platform=None):
    path = _sensor_directory(Path(study_dir) / "TILES", sensor, platform) / tile
    path.mkdir(parents=True, exist_ok=True)
    return path


def _snowflakes_output_directory(
    study_dir, sensor, tile=None, platform=None
):
    """Return the dedicated SnowFLAKES output root for a scene."""
    root = Path(study_dir) / "SnowFLAKES"
    sensor_root = _sensor_directory(root, sensor, platform)
    path = sensor_root if tile is None else sensor_root / tile
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_composites(
    config, sensor, data, scene_id, output_dir, rgb_data=None
):
    """Save RGB and false-color composites from the prepared DataArray."""
    from utils import save_false_color

    scene_dir = Path(output_dir) / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    overwrite = bool(config.get("overwrite", False))
    if sensor == "Sentinel-2":
        rgb_bands = ["B04", "B03", "B02"]
        false_color_bands = ["B11", "B8A", "B03"]
    else:
        rgb_bands = ["red", "green", "blue"]
        false_color_bands = ["swir16", "nir08", "green"]

    save_rgb = _save_rgb_enabled(config)
    save_fcc = bool(
        _value(
            config,
            "SAVE_FALSE_COLOR",
            "save_false_color",
            True,
        )
    )
    composites = []
    if save_rgb:
        composites.append(("RGB", rgb_bands, "rgb_10m"))
    if save_fcc:
        composites.append(("false-color", false_color_bands, "fcc"))

    available = set(str(band) for band in data.coords["band"].values)
    for label, bands, filename in composites:
        missing = [band for band in bands if band not in available]
        if missing:
            raise ValueError(
                f"Cannot create {label} composite for {scene_id}; "
                f"missing bands: {', '.join(missing)}"
            )
        path = scene_dir / f"{filename}.tif"
        if path.exists() and not overwrite:
            print(f"  {label.capitalize()} composite exists, skipping: {path}")
            continue
        print(
            f"  Saving {label} composite {path} from bands "
            f"{', '.join(bands)}"
        )
        composite_data = (
            rgb_data if filename == "rgb_10m" and rgb_data is not None else data
        )
        save_false_color(str(scene_dir), bands, composite_data, filename)


def _save_rgb_enabled(config):
    return bool(
        _value(
            config,
            "SAVE_RGB",
            "save_rgb",
            config.get("save_rgb_10m", False),
        )
    )


def _rgb_cache_exists(config, output_dir, scene_id):
    return (
        _save_rgb_enabled(config)
        and not bool(config.get("overwrite", False))
        and (Path(output_dir) / scene_id / "rgb_10m.tif").is_file()
    )


def _rgb_uses_analysis_data(config, sensor):
    """Return true when the analysis DataArray already has native RGB scale."""
    return (
        sensor == "Sentinel-2"
        and config.get("resampling_params", {}).get("resolution") == 10
    )


def _load_raw_rgb10(config, products, date, crop, merge=True, save_dir=None):
    """Load only Sentinel-2 RGB bands at native 10 m for visualization."""
    if not _save_rgb_enabled(config) or products.empty:
        return None, None
    params = config.get("resampling_params", {})
    extent = params.get("extent_target") if crop else None
    raw_dir = _working_directory(config) / "RAW" / "Sentinel-2"
    if not raw_dir.exists():
        legacy_dir = _working_directory(config) / "RAW" / "Sentinel2"
        if legacy_dir.exists():
            raw_dir = legacy_dir
    print(f"Loading Sentinel-2 RGB bands at 10 m for {date}")
    return load_sentinel2_tiles(
        products,
        raw_dir,
        date,
        extent,
        10,
        params["epsg_target"],
        config.get("exclude_tiles"),
        merge=merge,
        bands=["B04", "B03", "B02"],
    )


def _load_stac_rgb10(config, date):
    """Load only Sentinel-2 RGB assets at native 10 m from STAC."""
    if not _save_rgb_enabled(config):
        return None, None
    from loading import load_stac

    params = config["resampling_params"]
    load_stac.setup_cdse_credentials()
    print(f"Loading Sentinel-2 RGB bands at 10 m from STAC for {date}")
    return load_stac.convert_sentinel2_bands(
        outdir=str(_sensor_directory(Path(_study_directory(config)) / "MERGED", "Sentinel-2")),
        date=date,
        resolution=10,
        extent_target=params["extent_target"],
        epsg_target=params["epsg_target"],
        save=False,
        shp=config["shapefile"],
        exclude_tiles=config.get("exclude_tiles"),
        bands=["B04", "B03", "B02"],
    )


def _run_or_collect(
    config, sensor, date, data, scene_id, output_dir, collected, rgb_data=None
):
    """Consume one array immediately, or retain it for a library caller."""
    _save_composites(
        config, sensor, data, scene_id, output_dir, rgb_data=rgb_data
    )
    if config.get("run_snowflakes", True):
        from SnowFLAKES.main_SnowFLAKES import run_snowflakes

        scene_config = config.copy()
        scene_config["output_directory"] = str(output_dir)
        run_snowflakes(scene_config, data, scene_id)
    else:
        collected.append((sensor, date, scene_id, data))


def _process_cropped_raw(config, study_dir, sentinel2, landsat, save, collected):
    """Load cropped date mosaics and consume one DataArray at a time."""
    snowflakes_dir = _snowflakes_output_directory(study_dir, "Sentinel-2")
    for date in _dates(sentinel2, "Sentinel-2"):
        data, scene_id = _load_raw_date(
            config, "Sentinel-2", sentinel2, date, True, save
        )
        if data is not None:
            rgb_data = None
            if _rgb_uses_analysis_data(config, "Sentinel-2"):
                print("Reusing 10 m analysis DataArray for RGB; no second load")
                rgb_data = data
            elif not _rgb_cache_exists(config, snowflakes_dir, scene_id):
                rgb_data, _ = _load_raw_rgb10(
                    config, sentinel2, date, True, merge=True
                )
            _run_or_collect(
                config,
                "Sentinel-2",
                date,
                data,
                scene_id,
                snowflakes_dir,
                collected,
                rgb_data=rgb_data,
            )
            if config.get("run_snowflakes", True):
                del data
                if rgb_data is not None:
                    del rgb_data
                gc.collect()

    for date in _dates(landsat, "Landsat"):
        date_products = landsat[
            landsat["Name"].astype(str).str.split("_").str[3].str[:8]
            == date.replace("-", "")
        ]
        for sensor_code in sorted(
            date_products["Name"].astype(str).str.split("_").str[0].unique()
        ):
            snowflakes_dir = _snowflakes_output_directory(
                study_dir, "Landsat", platform=sensor_code
            )
            sensor_products = date_products[
                date_products["Name"].astype(str).str.startswith(sensor_code + "_")
            ]
            data, scene_id = _load_raw_date(
                config, "Landsat", sensor_products, date, True, save
            )
            if data is not None:
                _run_or_collect(
                    config,
                    "Landsat",
                    date,
                    data,
                    scene_id,
                    snowflakes_dir,
                    collected,
                )
                if config.get("run_snowflakes", True):
                    del data
                    gc.collect()


def _process_raw_tiles(config, study_dir, sentinel2, landsat, save, collected):
    """Reproject and process every raw tile on its own original footprint."""
    for sensor, products in (("Sentinel-2", sentinel2), ("Landsat", landsat)):
        for date in _dates(products, sensor):
            names = products["Name"].astype(str)
            date_index = 2 if sensor == "Sentinel-2" else 3
            date_products = products[
                names.str.split("_").str[date_index].str[:8]
                == date.replace("-", "")
            ]
            tile_names = date_products["Name"].map(
                lambda name: _tile_name(sensor, name)
            )
            for tile in sorted(tile_names.unique()):
                tile_products = date_products[tile_names == tile]
                if sensor == "Landsat":
                    sensor_codes = sorted(
                        tile_products["Name"]
                        .astype(str)
                        .str.split("_")
                        .str[0]
                        .unique()
                    )
                else:
                    sensor_codes = [None]

                for sensor_code in sensor_codes:
                    selected = tile_products
                    if sensor_code is not None:
                        selected = selected[
                            selected["Name"]
                            .astype(str)
                            .str.startswith(sensor_code + "_")
                        ]
                    platform = sensor_code if sensor != "Sentinel-2" else None
                    tile_dir = _tile_output_directory(
                        study_dir, sensor, tile, platform=platform
                    )
                    snowflakes_dir = _snowflakes_output_directory(
                        study_dir, sensor, tile, platform=platform
                    )
                    image_names = ", ".join(
                        selected["Name"].astype(str).tolist()
                    )
                    print("\n" + "=" * 60)
                    print(
                        f"Processing {sensor} tile {tile} for {date}"
                    )
                    print(f"Image(s): {image_names}")
                    print(f"Prepared-data directory: {tile_dir}")
                    print(f"SnowFLAKES directory: {snowflakes_dir}")
                    print("=" * 60)
                    data, scene_id = _load_raw_date(
                        config,
                        sensor,
                        selected,
                        date,
                        False,
                        save,
                        save_dir=tile_dir,
                        merge=False,
                    )
                    if data is not None:
                        rgb_data = None
                        if _rgb_uses_analysis_data(config, sensor):
                            print(
                                "Reusing 10 m analysis DataArray for RGB; "
                                "no second load"
                            )
                            rgb_data = data
                        elif sensor == "Sentinel-2" and not _rgb_cache_exists(
                            config, snowflakes_dir, scene_id
                        ):
                            rgb_data, _ = _load_raw_rgb10(
                                config,
                                selected,
                                date,
                                False,
                                merge=False,
                            )
                        _run_or_collect(
                            config,
                            sensor,
                            date,
                            data,
                            scene_id,
                            snowflakes_dir,
                            collected,
                            rgb_data=rgb_data,
                        )
                        if config.get("run_snowflakes", True):
                            del data
                            if rgb_data is not None:
                                del rgb_data
                            gc.collect()


def run(config_path):
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path(config_path).resolve().parent / ".env")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    legacy_mode = str(
        _value(config, "DOWNLOAD_MODE", "download_mode", "STAC-API")
    )
    legacy_mode = legacy_mode.lower().replace("_", "-").replace(" ", "-")
    if "DOWNLOAD_SENTINEL" in config or "download_sentinel" in config:
        sentinel_download = _download_flag(
            _value(config, "DOWNLOAD_SENTINEL", "download_sentinel", False),
            {"odata", "s3", "google", "stac-api"},
            "DOWNLOAD_SENTINEL",
        )
    elif legacy_mode in {"stac", "stac-api", "cdse-stac-api"}:
        sentinel_download = "stac-api"
    elif legacy_mode in {"s3", "sentinel2-s3", "sentinel-s3"}:
        sentinel_download = "s3"
    elif legacy_mode in {"raw", "archives", "download"}:
        sentinel_download = _download_flag(
            "google",
            {"odata", "s3", "google"},
            "DOWNLOAD_SENTINEL",
        )
    elif legacy_mode not in {"", "stac-api"}:
        raise ValueError(
            "Unsupported DOWNLOAD_MODE value. Use DOWNLOAD_SENTINEL and "
            "DOWNLOAD_LANDSAT instead."
        )
    else:
        sentinel_download = _download_flag(
            "google",
            {"odata", "s3", "google"},
            "DOWNLOAD_SENTINEL",
        )

    if "DOWNLOAD_LANDSAT" in config or "download_landsat_mode" in config:
        landsat_download = _download_flag(
            config.get("DOWNLOAD_LANDSAT", config.get("download_landsat_mode")),
            {"stac-api", "usgs-m2m"},
            "DOWNLOAD_LANDSAT",
        )
    elif legacy_mode in {"stac", "stac-api", "cdse-stac-api"}:
        landsat_download = "stac-api"
    else:
        landsat_download = "usgs-m2m"

    satellite = str(config.get("satellite", "both")).lower()
    if satellite.startswith("sentinel"):
        landsat_download = False
    elif satellite.startswith("landsat"):
        sentinel_download = False

    sentinel_stac = sentinel_download == "stac-api"
    landsat_stac = landsat_download == "stac-api"
    # A false flag disables downloading but still permits processing products
    # already present under RAW. STAC-API selects the remote processing path.
    raw_sentinel = (
        not satellite.startswith("landsat") and sentinel_download != "stac-api"
    )
    raw_landsat = (
        not satellite.startswith("sentinel") and landsat_download != "stac-api"
    )

    crop = bool(_value(config, "CROP", "crop", False))
    save = bool(_value(config, "SAVE", "save", False))
    run_snowflakes_enabled = bool(config.get("run_snowflakes", True))
    # RAW + uncropped + no-save/no-SnowFLAKES is an intentional download-only
    # mode: archives are downloaded and no in-memory processing is attempted.
    download_only = (
        (raw_sentinel or raw_landsat)
        and not crop
        and not save
        and not run_snowflakes_enabled
    )
    processing_requested = raw_sentinel or raw_landsat or sentinel_stac or landsat_stac
    if (
        not save
        and not run_snowflakes_enabled
        and processing_requested
        and not download_only
    ):
        raise ValueError(
            "SAVE=false requires run_snowflakes=true because the prepared "
            "DataArray exists only in memory and would otherwise be lost "
            "when the loading process exits"
        )
    if (sentinel_stac or landsat_stac) and not crop:
        raise ValueError("CROP must be true when using STAC-API downloads")

    params = config.get("resampling_params", {})
    if crop:
        required = ("extent_target", "resolution", "epsg_target")
        missing = [key for key in required if params.get(key) is None]
        if missing:
            raise ValueError(
                "resampling_params is required when CROP=true; missing: "
                + ", ".join(missing)
            )
    elif raw_sentinel or raw_landsat:
        required = ("resolution", "epsg_target")
        missing = [key for key in required if params.get(key) is None]
        if missing:
            raise ValueError(
                "resampling_params is required for RAW tile processing; "
                "missing: " + ", ".join(missing)
            )

    study_dir = _study_directory(config)
    sentinel2 = (
        _read_query_files(
            study_dir,
            "Sentinel2",
            config.get("date_start"),
            config.get("date_end"),
        )
        if satellite.startswith("sentinel") or satellite == "both"
        else pd.DataFrame(columns=["Name"])
    )
    landsat = (
        _read_query_files(
            study_dir,
            "Landsat",
            config.get("date_start"),
            config.get("date_end"),
        )
        if satellite.startswith("landsat") or satellite == "both"
        else pd.DataFrame(columns=["Name"])
    )
    if raw_sentinel or raw_landsat:
        sentinel_download_products = sentinel2 if raw_sentinel else sentinel2.iloc[0:0]
        landsat_download_products = landsat if raw_landsat else landsat.iloc[0:0]
        if crop:
            if _save_rgb_enabled(config) and not sentinel_download_products.empty:
                sentinel2_downloads = sentinel_download_products
            else:
                sentinel2_downloads = _remove_processed(
                    sentinel_download_products,
                    study_dir,
                    "Sentinel-2",
                    exclude_tiles=config.get("exclude_tiles"),
                    resolution=params["resolution"],
                    epsg=params["epsg_target"],
                    extent=params["extent_target"],
                )
            landsat_downloads = _remove_processed(
                landsat_download_products,
                study_dir,
                "Landsat",
                exclude_tiles=config.get("exclude_tiles"),
                resolution=params["resolution"],
                epsg=params["epsg_target"],
                extent=params["extent_target"],
            )
        else:
            if _save_rgb_enabled(config) and not sentinel_download_products.empty:
                sentinel2_downloads = sentinel_download_products
            else:
                sentinel2_downloads = _remove_cached_tiles(
                    sentinel_download_products,
                    study_dir,
                    "Sentinel-2",
                    resolution=params["resolution"],
                    epsg=params["epsg_target"],
                )
            landsat_downloads = _remove_cached_tiles(
                landsat_download_products,
                study_dir,
                "Landsat",
                resolution=params["resolution"],
                epsg=params["epsg_target"],
            )
        invalid_sentinel, invalid_landsat = _download_raw(
            config,
            study_dir,
            sentinel2_downloads,
            landsat_downloads,
            sentinel_source=sentinel_download,
            landsat_source=landsat_download,
        )
        if invalid_sentinel:
            sentinel2 = sentinel2[~sentinel2["Name"].isin(invalid_sentinel)].copy()
        if invalid_landsat:
            landsat = landsat[~landsat["Name"].isin(invalid_landsat)].copy()

        if download_only:
            print(
                "Download step complete; CROP=false, SAVE=false, and "
                "run_snowflakes=false, so processing stops here."
            )
            return []

    # When SnowFLAKES is enabled each DataArray is consumed immediately and
    # released. If it is disabled, retain the arrays for programmatic callers.
    arrays = []
    stac_sentinel_products = sentinel2 if sentinel_stac else sentinel2.iloc[0:0]
    stac_landsat_products = landsat if landsat_stac else landsat.iloc[0:0]
    if sentinel_stac or landsat_stac:
        for date in _dates(stac_sentinel_products, "Sentinel-2"):
            snowflakes_dir = _snowflakes_output_directory(
                study_dir, "Sentinel-2"
            )
            data, scene_id = _load_stac_date(
                config,
                "Sentinel-2",
                date,
                save,
                products=stac_sentinel_products,
            )
            if data is not None:
                rgb_data = None
                if _rgb_uses_analysis_data(config, "Sentinel-2"):
                    print("Reusing 10 m analysis DataArray for RGB; no second STAC load")
                    rgb_data = data
                elif not _rgb_cache_exists(config, snowflakes_dir, scene_id):
                    rgb_data, _ = _load_stac_rgb10(config, date)
                _run_or_collect(
                    config,
                    "Sentinel-2",
                    date,
                    data,
                    scene_id,
                    snowflakes_dir,
                    arrays,
                    rgb_data=rgb_data,
                )
                if config.get("run_snowflakes", True):
                    del data
                    if rgb_data is not None:
                        del rgb_data
                    gc.collect()
        for date in _dates(stac_landsat_products, "Landsat"):
            date_products = stac_landsat_products[
                stac_landsat_products["Name"].astype(str).str.split("_").str[3].str[:8]
                == date.replace("-", "")
            ]
            for sensor_code in sorted(
                date_products["Name"].astype(str).str.split("_").str[0].unique()
            ):
                snowflakes_dir = _snowflakes_output_directory(
                    study_dir, "Landsat", platform=sensor_code
                )
                sensor_products = date_products[
                    date_products["Name"]
                    .astype(str)
                    .str.startswith(sensor_code + "_")
                ]
                data, scene_id = _load_stac_date(
                    config,
                    "Landsat",
                    date,
                    save,
                    platform=sensor_code,
                    products=sensor_products,
                )
                if data is not None:
                    _run_or_collect(
                        config,
                        "Landsat",
                        date,
                        data,
                        scene_id,
                        snowflakes_dir,
                        arrays,
                    )
                    if config.get("run_snowflakes", True):
                        del data
                        gc.collect()
    if raw_sentinel or raw_landsat:
        raw_sentinel_products = sentinel2 if raw_sentinel else sentinel2.iloc[0:0]
        raw_landsat_products = landsat if raw_landsat else landsat.iloc[0:0]
        if crop:
            _process_cropped_raw(
                config, study_dir, raw_sentinel_products, raw_landsat_products, save, arrays
            )
        else:
            _process_raw_tiles(
                config, study_dir, raw_sentinel_products, raw_landsat_products, save, arrays
            )

    config["output_directory"] = str(study_dir)
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
