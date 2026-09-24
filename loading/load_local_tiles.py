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
    # Match stackstac's default ``snap_bounds=True`` behavior: bounds are
    # expanded to whole target pixels before the raster shape is calculated.
    xmin = np.floor(xmin / resolution) * resolution
    ymin = np.floor(ymin / resolution) * resolution
    xmax = np.ceil(xmax / resolution) * resolution
    ymax = np.ceil(ymax / resolution) * resolution
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
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


def _combine_arrays(mosaic, tile, method):
    """Combine overlapping scenes using the same reducers as stackstac."""
    if mosaic is None:
        return tile
    stack = np.stack([mosaic, tile])
    if method == "max":
        valid = ~np.isnan(stack)
        return np.where(valid.any(axis=0), np.nanmax(stack, axis=0), np.nan)
    if method == "mean":
        valid = ~np.isnan(stack)
        count = valid.sum(axis=0)
        total = np.nansum(stack, axis=0)
        return np.divide(
            total,
            count,
            out=np.full_like(total, np.nan, dtype="float32"),
            where=count > 0,
        )
    raise ValueError(f"Unsupported scene combination method: {method}")


def _fill_gaps(mosaic, tile):
    """Backward-compatible first-valid gap filling helper."""
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


def _prepared_band_spec(scene_id):
    """Return DataArray band names and accepted cached filename tokens."""
    sensor = str(scene_id).split("_", 1)[0]
    if sensor.startswith("S2"):
        return [(band, (band,)) for band in SENTINEL2_BANDS]
    if sensor not in LANDSAT_BANDS:
        return []
    return [
        (name, (name, f"B{number}"))
        for number, name in LANDSAT_BANDS[sensor].items()
    ]


def prepared_bands_are_complete(output_dir, scene_id):
    """Return true when every prepared band exists for a saved scene."""
    scene_dir = Path(output_dir) / scene_id
    if not scene_dir.is_dir():
        return False
    for _, filename_tokens in _prepared_band_spec(scene_id):
        if not any(
            (scene_dir / f"{scene_id}_{token}_toa.tif").is_file()
            for token in filename_tokens
        ):
            return False
    return bool(_prepared_band_spec(scene_id))


def prepared_grid_matches(
    output_dir, scene_id, resolution, epsg, extent=None
):
    """Check that cached bands match the currently requested target grid."""
    if resolution is None or epsg is None:
        return False
    if not prepared_bands_are_complete(output_dir, scene_id):
        return False
    scene_dir = Path(output_dir) / scene_id
    _, filename_tokens = _prepared_band_spec(scene_id)[0]
    path = next(
        scene_dir / f"{scene_id}_{token}_toa.tif"
        for token in filename_tokens
        if (scene_dir / f"{scene_id}_{token}_toa.tif").is_file()
    )
    with rasterio.open(path) as source:
        source_epsg = source.crs.to_epsg() if source.crs else None
        source_resolution = (abs(source.transform.a), abs(source.transform.e))
        if source_epsg != int(epsg):
            return False
        if not np.allclose(source_resolution, (resolution, resolution)):
            return False
        if extent is not None:
            xmin, ymin, xmax, ymax = extent
            snapped = (
                np.floor(xmin / resolution) * resolution,
                np.floor(ymin / resolution) * resolution,
                np.ceil(xmax / resolution) * resolution,
                np.ceil(ymax / resolution) * resolution,
            )
            if not np.allclose(source.bounds, snapped):
                return False
    return True


def load_prepared_bands(output_dir, scene_id, date):
    """Reconstruct a SnowFLAKES DataArray from prepared GeoTIFF bands.

    ``None`` is returned for an absent or incomplete cache so callers can
    transparently fall back to the original STAC or raw-archive loader.
    """
    if not prepared_bands_are_complete(output_dir, scene_id):
        return None

    scene_dir = Path(output_dir) / scene_id
    arrays = []
    bands = []
    reference = None
    epsg = None
    transform = None
    print(f"Loading prepared scene from GeoTIFF cache: {scene_id}")
    for band, filename_tokens in _prepared_band_spec(scene_id):
        path = next(
            scene_dir / f"{scene_id}_{token}_toa.tif"
            for token in filename_tokens
            if (scene_dir / f"{scene_id}_{token}_toa.tif").is_file()
        )
        print(f"  Loading cached band {band}: {path.name}")
        with rasterio.open(path) as source:
            grid = (source.height, source.width, source.transform, source.crs)
            if reference is None:
                reference = grid
                transform = source.transform
                epsg = source.crs.to_epsg() if source.crs else None
                if epsg is None:
                    raise ValueError(f"Cached band has no EPSG CRS: {path}")
            elif grid != reference:
                raise ValueError(
                    f"Cached bands do not share the same grid: {path}"
                )
            arrays.append(source.read(1).astype("float32"))
        bands.append(band)

    return _as_data_array(arrays, bands, date, epsg, transform)


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
                         exclude_tiles=None, merge=True, bands=None):
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
    bands = list(bands or SENTINEL2_BANDS)

    print(
        f"Preparing Sentinel-2 date {date}: "
        f"{len(selected)} product(s)"
    )

    source_paths = []
    for name in selected["Name"]:
        product_id = str(name).removesuffix(".SAFE")
        tile = product_id.split("_")[5]
        scene_dir = _sentinel2_scene_dir(outdir, tile, product_id)
        source_paths.append(_sentinel2_band_path(scene_dir, bands[0]))
    epsg = _resolve_epsg(source_paths, epsg)
    if extent is None:
        extent = _union_extent(source_paths, epsg, resolution)
    transform, height, width = _target_grid(extent, resolution)
    print(
        f"  Target grid: EPSG:{epsg}, {resolution} m, "
        f"{width} x {height} pixels, extent={extent}"
    )

    mosaics = []
    for band_index, band in enumerate(bands, start=1):
        print(
            f"  Resampling/calibrating Sentinel-2 band {band} "
            f"({band_index}/{len(bands)})"
        )
        mosaic = None
        for name in selected["Name"]:
            product_id = str(name).removesuffix(".SAFE")
            parts = product_id.split("_")
            tile = parts[5]
            scene_dir = _sentinel2_scene_dir(outdir, tile, product_id)
            band_path = _sentinel2_band_path(scene_dir, band)
            print(f"    Tile {tile}, image {product_id}")
            values = _read_on_grid(
                band_path, epsg, transform, height, width
            )
            baseline = int(parts[3].removeprefix("N"))
            offset = -1000 if baseline >= 400 else 0
            values = (values + offset) * 0.0001
            values[values <= 0] = np.nan
            mosaic = _combine_arrays(mosaic, values, "max")
        mosaics.append(mosaic)

    scene_id = selected.iloc[0]["Name"].removesuffix(".SAFE")
    parts = scene_id.split("_")
    if merge:
        parts[5] = "merged"
    return (
        _as_data_array(mosaics, bands, date, epsg, transform),
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
                       exclude_tiles=None, merge=True):
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

    print(
        f"Preparing Landsat date {date}: {len(selected)} product(s)"
    )

    source_paths = []
    for name in selected["Name"]:
        sensor, _, tile = name.split("_")[:3]
        source_paths.append(
            _landsat_member(_landsat_archive(outdir, sensor, tile, name), 2)
        )
    epsg = _resolve_epsg(source_paths, epsg)
    if extent is None:
        extent = _union_extent(source_paths, epsg, resolution)
    transform, height, width = _target_grid(extent, resolution)
    print(
        f"  Target grid: EPSG:{epsg}, {resolution} m, "
        f"{width} x {height} pixels, extent={extent}"
    )

    first_sensor = selected.iloc[0]["Name"].split("_")[0]
    band_map = LANDSAT_BANDS[first_sensor]
    mosaics = []
    for band_index, band in enumerate(band_map, start=1):
        band_name = band_map[band]
        print(
            f"  Resampling/calibrating Landsat band B{band} "
            f"({band_name}, {band_index}/{len(band_map)})"
        )
        mosaic = None
        for name in selected["Name"]:
            sensor, _, tile = name.split("_")[:3]
            if sensor != first_sensor:
                continue
            archive = _landsat_archive(outdir, sensor, tile, name)
            print(f"    Tile {tile}, image {name}")
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
            mosaic = _combine_arrays(mosaic, values, "mean")
        # Match the STAC Landsat path: reduce overlapping raw values first,
        # then apply the radiometric conversion once to the reduced raster.
        first_archive = _landsat_archive(
            outdir,
            first_sensor,
            selected.iloc[0]["Name"].split("_")[2],
            selected.iloc[0]["Name"],
        )
        metadata = _read_mtl(first_archive)
        mosaics.append(_calibrate_landsat(mosaic, band, first_sensor, metadata))

    scene_id = selected.iloc[0]["Name"]
    parts = scene_id.split("_")
    if merge:
        parts[2] = "merged"
    return (
        _as_data_array(
            mosaics, list(band_map.values()), date, epsg, transform
        ),
        "_".join(parts),
    )
