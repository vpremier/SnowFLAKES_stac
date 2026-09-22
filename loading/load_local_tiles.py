#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load and mosaic locally downloaded Sentinel-2 or Landsat Level-1 tiles."""

import os
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 - registers the xarray ``rio`` accessor
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds


SENTINEL2_BANDS = [
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B11", "B12", "B8A",
]

LANDSAT_BANDS = {
    "LT05": {
        1: "blue", 2: "green", 3: "red", 4: "nir08",
        5: "swir16", 6: "lwir", 7: "swir22",
    },
    "LE07": {
        1: "blue", 2: "green", 3: "red", 4: "nir08",
        5: "swir16", 6: "lwir", 7: "swir22",
    },
    "LC08": {
        1: "coastal", 2: "blue", 3: "green", 4: "red",
        5: "nir08", 6: "swir16", 7: "swir22", 10: "lwir11",
    },
    "LC09": {
        1: "coastal", 2: "blue", 3: "green", 4: "red",
        5: "nir08", 6: "swir16", 7: "swir22", 10: "lwir11",
    },
}


def _target_grid(extent, resolution):
    """Return the transform and shape for the configured output grid."""
    xmin, ymin, xmax, ymax = extent
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("The configured target extent is not valid")
    transform = from_origin(xmin, ymax, resolution, resolution)
    return transform, height, width


def _resolve_epsg(paths, epsg):
    if epsg is not None:
        return epsg
    with rasterio.open(paths[0]) as source:
        resolved = source.crs.to_epsg() if source.crs else None
    if resolved is None:
        raise ValueError("The source raster has no EPSG and none was configured")
    return resolved


def _union_extent(paths, epsg, resolution):
    """Return a target-grid-aligned union extent for complete tiles."""
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for path in paths:
        with rasterio.open(path) as source:
            bounds = transform_bounds(
                source.crs,
                f"EPSG:{epsg}",
                *source.bounds,
                densify_pts=21,
            )
        xmin = min(xmin, bounds[0])
        ymin = min(ymin, bounds[1])
        xmax = max(xmax, bounds[2])
        ymax = max(ymax, bounds[3])
    return [
        np.floor(xmin / resolution) * resolution,
        np.floor(ymin / resolution) * resolution,
        np.ceil(xmax / resolution) * resolution,
        np.ceil(ymax / resolution) * resolution,
    ]


def _read_on_grid(path, epsg, transform, height, width,
                  resampling=Resampling.bilinear):
    """Read one raster and reproject it using bilinear resampling by default."""
    destination = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata or 0,
            dst_transform=transform,
            dst_crs=f"EPSG:{epsg}",
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination


def _fill_gaps(mosaic, tile):
    """Apply the gap-filling rule used by the old ``merge_tiles.py``."""
    if mosaic is None:
        return tile
    missing = np.isnan(mosaic)
    mosaic[missing] = tile[missing]
    return mosaic


def _as_data_array(arrays, bands, date, epsg, transform):
    data = np.stack(arrays)[None, ...]
    height, width = arrays[0].shape
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    result = xr.DataArray(
        data,
        dims=("day", "band", "y", "x"),
        coords={
            "day": [pd.Timestamp(date).day],
            "band": bands,
            "y": y,
            "x": x,
        },
        attrs={"transform": transform, "epsg": int(epsg)},
    )
    return result.rio.write_crs(int(epsg)).rio.write_transform(transform)


def _sentinel2_band_path(scene_dir, band):
    """Find a band regardless of whether S2DL includes a filename prefix."""
    if not Path(scene_dir).is_dir():
        raise FileNotFoundError(f"Scene directory is missing: {scene_dir}")
    band_pattern = re.compile(rf"(?:^|[_-]){re.escape(band)}(?:$|[_-])")
    matches = []
    for path in Path(scene_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jp2", ".tif"}:
            continue
        if band_pattern.search(path.stem + "_"):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Band {band} is missing in {scene_dir}")
    return sorted(set(matches))[0]


def _sentinel2_scene_dir(outdir, tile, product_id):
    """Resolve S2DL/SAFE directory naming variants."""
    base = Path(outdir) / tile
    candidates = (base / product_id, base / f"{product_id}.SAFE")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Return the canonical path so the resulting error remains informative.
    return candidates[0]


def load_sentinel2_tiles(products, outdir, date, extent, resolution, epsg,
                         exclude_tiles=None):
    """Load S2DL products for one date and merge their MGRS tiles."""
    date_token = pd.Timestamp(date).strftime("%Y%m%d")
    selected = products[
        products["Name"].str.split("_").str[2].str.startswith(date_token)
    ]
    exclude_tiles = set(exclude_tiles or [])
    selected = selected[
        ~selected["Name"].str.split("_").str[5].isin(exclude_tiles)
    ]
    if selected.empty:
        return None, None

    source_paths = []
    for name in selected["Name"]:
        product_id = str(name).removesuffix(".SAFE")
        tile = product_id.split("_")[5]
        scene_dir = _sentinel2_scene_dir(outdir, tile, product_id)
        source_paths.append(_sentinel2_band_path(scene_dir, "B02"))
    epsg = _resolve_epsg(source_paths, epsg)
    if extent is None:
        extent = _union_extent(source_paths, epsg, resolution)
    transform, height, width = _target_grid(extent, resolution)

    mosaics = []
    for band in SENTINEL2_BANDS:
        mosaic = None
        for name in selected["Name"]:
            product_id = str(name).removesuffix(".SAFE")
            parts = product_id.split("_")
            tile = parts[5]
            scene_dir = _sentinel2_scene_dir(outdir, tile, product_id)
            band_path = _sentinel2_band_path(scene_dir, band)
            values = _read_on_grid(
                band_path, epsg, transform, height, width
            )
            baseline = int(parts[3].removeprefix("N"))
            offset = -1000 if baseline >= 400 else 0
            values = (values + offset) * 0.0001
            values[values <= 0] = np.nan
            mosaic = _fill_gaps(mosaic, values)
        mosaics.append(mosaic)

    scene_id = selected.iloc[0]["Name"].removesuffix(".SAFE")
    parts = scene_id.split("_")
    parts[5] = "merged"
    return (
        _as_data_array(mosaics, SENTINEL2_BANDS, date, epsg, transform),
        "_".join(parts),
    )


def _read_mtl(archive):
    with tarfile.open(archive) as tar:
        member = next(
            item for item in tar.getmembers()
            if item.name.upper().endswith("_MTL.TXT")
        )
        text = tar.extractfile(member).read().decode("utf-8")
    return {
        key.strip(): value.strip().strip('"')
        for key, value in (
            line.split("=", 1) for line in text.splitlines() if "=" in line
        )
    }


def _landsat_member(archive, band):
    suffix = f"_B{band}.TIF"
    with tarfile.open(archive) as tar:
        member = next(
            item.name for item in tar.getmembers()
            if item.name.upper().endswith(suffix)
        )
    return f"/vsitar/{os.path.abspath(archive)}/{member}"


def _landsat_archive(outdir, sensor, tile, name):
    mission = {
        "LT05": "Landsat-5",
        "LE07": "Landsat-7",
        "LC08": "Landsat-8",
        "LC09": "Landsat-9",
    }.get(sensor, sensor)
    mission_path = Path(outdir) / "Landsat" / mission / tile / f"{name}.tar"
    sensor_path = Path(outdir) / "Landsat" / sensor / tile / f"{name}.tar"
    return mission_path if mission_path.exists() else sensor_path


def _calibrate_landsat(values, band, sensor, metadata):
    if band == 6 and sensor == "LE07":
        suffix = next(
            suffix for suffix in ("_VCID_1", "_VCID_2")
            if f"RADIANCE_MULT_BAND_6{suffix}" in metadata
        )
    else:
        suffix = ""

    is_thermal = (
        (sensor in ("LT05", "LE07") and band == 6)
        or (sensor in ("LC08", "LC09") and band == 10)
    )
    if is_thermal:
        radiance = (
            values * float(metadata[f"RADIANCE_MULT_BAND_{band}{suffix}"])
            + float(metadata[f"RADIANCE_ADD_BAND_{band}{suffix}"])
        )
        k1 = float(metadata[f"K1_CONSTANT_BAND_{band}{suffix}"])
        k2 = float(metadata[f"K2_CONSTANT_BAND_{band}{suffix}"])
        result = k2 / np.log(k1 / radiance + 1)
    else:
        result = (
            values * float(metadata[f"REFLECTANCE_MULT_BAND_{band}"])
            + float(metadata[f"REFLECTANCE_ADD_BAND_{band}"])
        )
        zenith = np.radians(90 - float(metadata["SUN_ELEVATION"]))
        result = result / np.cos(zenith)
    result[result <= 0] = np.nan
    return result


def load_landsat_tiles(products, outdir, date, extent, resolution, epsg,
                       exclude_tiles=None):
    """Load downloaded USGS tar archives for one date and merge WRS tiles."""
    date_token = pd.Timestamp(date).strftime("%Y%m%d")
    names = products["Name"].astype(str)
    selected = products[names.str.split("_").str[3] == date_token]
    exclude_tiles = set(exclude_tiles or [])
    selected = selected[
        ~selected["Name"].str.split("_").str[2].isin(exclude_tiles)
    ]
    if selected.empty:
        return None, None

    source_paths = []
    for name in selected["Name"]:
        sensor, _, tile = name.split("_")[:3]
        source_paths.append(
            _landsat_member(_landsat_archive(outdir, sensor, tile, name), 2)
        )
    epsg = _resolve_epsg(source_paths, epsg)
    if extent is None:
        extent = _union_extent(source_paths, epsg, resolution)

    first_sensor = selected.iloc[0]["Name"].split("_")[0]
    band_map = LANDSAT_BANDS[first_sensor]
    mosaics = []
    for band in band_map:
        mosaic = None
        for name in selected["Name"]:
            sensor, _, tile = name.split("_")[:3]
            if sensor != first_sensor:
                continue
            archive = _landsat_archive(outdir, sensor, tile, name)
            if not archive.exists():
                raise FileNotFoundError(f"Downloaded scene is missing: {archive}")
            metadata = _read_mtl(archive)
            values = _read_on_grid(
                _landsat_member(archive, band),
                epsg,
                transform,
                height,
                width,
            )
            values = _calibrate_landsat(values, band, sensor, metadata)
            mosaic = _fill_gaps(mosaic, values)
        mosaics.append(mosaic)

    scene_id = selected.iloc[0]["Name"]
    parts = scene_id.split("_")
    parts[2] = "merged"
    return (
        _as_data_array(
            mosaics, list(band_map.values()), date, epsg, transform
        ),
        "_".join(parts),
    )
