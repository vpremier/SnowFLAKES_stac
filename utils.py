#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:22:44 2025

@author: vpremier
"""

import geopandas as gpd
import numpy as np
from pyproj import CRS
import glob
import os
import rasterio


def save_false_color(wd, bands, ds):

    # Extract bands from xarray dataset
    b1 = ds.sel(band = bands[0]).values
    b2 = ds.sel(band = bands[1]).values
    b3 = ds.sel(band = bands[2]).values

    output_path = os.path.join(wd, "false_color_composite.tif")

    # Stack as RGB
    rgb = np.stack([b1, b2, b3])

    # Get raster metadata from dataset
    height, width = b1.shape

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": rgb.dtype,
        "crs": ds.rio.crs,
        "transform": ds.rio.transform()
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(rgb)
            
            

def get_shape_extent(shape_name, epsg=3035, outres=500, merge=True, row=None):
    # Read shapefile
    gdf = gpd.read_file(shape_name)

    # Get current CRS
    crs_shp = gdf.crs
    target_crs = CRS.from_epsg(epsg)

    # Reproject if needed
    if crs_shp != target_crs:
        print('The input shapefile is in another reference system. Reprojecting...')
        gdf = gdf.to_crs(target_crs)

    if merge:
        if row is not None:
            raise ValueError("`row` must be None when `merge=True`.")
        xmin, ymin, xmax, ymax = gdf.total_bounds
    else:
        if row is None:
            raise ValueError("You must provide a `row` index when `merge=False`.")
        geom = gdf.iloc[row].geometry  # safer than gdf[row]
        xmin, ymin, xmax, ymax = geom.bounds

    # Round to outres
    xMin = round(int(xmin / outres) * outres, 5)
    yMin = round(int(ymin / outres) * outres, 5)
    xMax = round(int(np.ceil(xmax / outres)) * outres, 5)
    yMax = round(int(np.ceil(ymax / outres)) * outres, 5)

    return xMin, yMin, xMax, yMax





