#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:05:41 2026

@author: cmarin
"""

import os
import subprocess
from datetime import datetime, timezone
import rasterio


def get_raster_epsg(raster_path):
    """
    Read EPSG code from a raster.
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs

    if crs is None:
        raise ValueError(f"No CRS found in {raster_path}")

    epsg = crs.to_epsg()

    if epsg is None:
        raise ValueError(
            "Could not determine EPSG code from raster. "
            "Use a projected DEM with a clear EPSG code."
        )

    return epsg


def run_grass_sunmask_from_datetime(
    dem_path,
    output_shadow_path,
    date_time_utc,
    epsg=None,
    grass_bin="grass",
    overwrite=True,
):
    """
    Generate a terrain cast-shadow mask using GRASS r.sunmask.

    Parameters
    ----------
    dem_path : str
        Path to DEM GeoTIFF.

    output_shadow_path : str
        Output shadow mask GeoTIFF.

    date_time_utc : datetime
        Acquisition datetime in UTC.

    epsg : int, optional
        EPSG code of the DEM/projected location.
        If None, it is read from the DEM.

    grass_bin : str
        GRASS executable name or path.

    overwrite : bool
        If True, overwrite existing files.

    Returns
    -------
    output_shadow_path : str
        Path to generated shadow mask.

    Notes
    -----
    GRASS r.sunmask output convention should be checked visually.
    Depending on GRASS version/settings, output categories may need inversion.
    """

    dem_path = os.path.abspath(dem_path)
    output_shadow_path = os.path.abspath(output_shadow_path)

    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    if epsg is None:
        epsg = get_raster_epsg(dem_path)

    if date_time_utc.tzinfo is None:
        date_time_utc = date_time_utc.replace(tzinfo=timezone.utc)
    else:
        date_time_utc = date_time_utc.astimezone(timezone.utc)

    year = date_time_utc.year
    month = date_time_utc.month
    day = date_time_utc.day 
    hour = date_time_utc.hour
    minute = date_time_utc.minute
    second = date_time_utc.second

    overwrite_flag = "--overwrite" if overwrite else ""

    grass_script = f"""
set -e

r.import input="{dem_path}" output=dem {overwrite_flag}

g.region raster=dem

r.sunmask \\
    elevation=dem \\
    output=sunmask \\
    year={year} \\
    month={month} \\
    day={day} \\
    hour={hour} \\
    minute={minute} \\
    second={second} \\
    timezone=0 \\
    {overwrite_flag}

r.out.gdal \\
    input=sunmask \\
    output="{output_shadow_path}" \\
    format=GTiff \\
    type=Byte \\
    {overwrite_flag}
"""

    cmd = [
        grass_bin,
        "--tmp-location",
        f"EPSG:{epsg}",
        "--exec",
        "bash",
        "-c",
        grass_script,
    ]

    print("Running GRASS r.sunmask...")
    print(f"DEM: {dem_path}")
    print(f"Output: {output_shadow_path}")
    print(f"EPSG: {epsg}")
    print(f"UTC datetime: {date_time_utc.isoformat()}")

    subprocess.run(cmd, check=True)

    return output_shadow_path


dem_path = "/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Azufre/Sentinel2_xgboost/01_TEST_auxiliary_folder/DEM.tif"
output_shadow_path = "/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Azufre/Sentinel2_xgboost/S2B_MSIL1C_20220227T143729_N0510_R096_merged_20240523T181541/auxiliary/S2B_MSIL1C_20220227T143729_N0510_R096_merged_20240523T181541_shadow_mask_grass.tif"

date_time_utc = datetime(2022, 2, 27, 14, 37, 29, tzinfo=timezone.utc)

run_grass_sunmask_from_datetime(
    dem_path=dem_path,
    output_shadow_path=output_shadow_path,
    date_time_utc=date_time_utc,
)