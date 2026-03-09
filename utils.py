#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:22:44 2025

@author: vpremier
"""

import geopandas as gpd
import numpy as np
from pyproj import CRS



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





