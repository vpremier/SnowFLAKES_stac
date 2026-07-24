#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:22:44 2025

@author: vpremier
"""

import geopandas as gpd
import numpy as np
from pyproj import CRS
import os
import rasterio
import rioxarray
import glob
import time

from SnowFLAKES.utilities import *








def remove_glaciers(outdir):
    for scf_path in glob.glob(os.path.join(outdir, "*", "SCF")):
        
        files = os.listdir(scf_path)
        
        has_glacier = any(f.endswith("SnowFLAKES_GLACIERS.tif") for f in files)
        has_snowflakes = any(f.endswith("SnowFLAKES.tif") for f in files)
        
        # Condition: glacier exists BUT SnowFLAKES.tif does NOT
        if has_glacier and not has_snowflakes:
            for f in files:
                if "SnowFLAKES_GLACIERS" in f:
                    file_path = os.path.join(scf_path, f)
                    
                    print(f"🗑 Removing {file_path}")
                    os.remove(file_path)
    return
                
                
                
                
def load_with_retry(data, max_retries=5, wait_seconds=30):
    for attempt in range(max_retries):
        try:
            data.load()
            return True  # success

        except Exception as e:
            print(f"⚠️ Load failed (attempt {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                sleep_time = wait_seconds * (attempt + 1)  # linear backoff
                print(f"⏳ Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)
            else:
                print("❌ Max retries reached.")
                raise e
                
                
                

def get_processed_dates(config):
    working_folder = config['output_directory']

    files = glob.glob(os.path.join(working_folder, "*/*SnowFLAKES*.tif"))
    

    dates = sorted({
        (
            f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"
            if fname.startswith("S2")
            else f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]}"
        )
        for f in files
        for fname in [os.path.basename(f)]
        for parts in [fname.split("_")]
        if fname.startswith(("S2", "L"))
    })
    
    return dates
    

def get_dates_to_skip(config):
    working_folder = config['output_directory']

    scenes_to_skip_clouds = read_log(working_folder, '00_skip_cloud_masks')
    dates_to_skip_emptyitems = read_log(working_folder, '00_dates_no_items')

    
    # Combine and remove duplicates
    all_scenes = set(scenes_to_skip_clouds) #set(scenes_to_skip) 
    
    # Extract dates
    dates = sorted({
        (
            f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"
            if scene.startswith("S2")
            else f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]}"
        )
        for scene in all_scenes
        for parts in [scene.split("_")]
        if scene.startswith(("S2", "L"))
    })
    
    dates = sorted(set(dates) | set(dates_to_skip_emptyitems))
    return dates
    


def get_dates_to_process(files, config):
    
    # extract dates 
    dates_to_download = sorted({
        (
            # Sentinel-2 (starts with S2...)
            f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"
            if f.startswith("S2")
            # Landsat (starts with L...)
            else f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]}"
        )
        for f in files
        for parts in [f.split("_")]
    })

    # already processed
    processed_dates = set(get_processed_dates(config))
    
    # to skip (clouds / failed)
    dates_to_skip = set(get_dates_to_skip(config))
    
    
    # final dates to process
    dates_to_process = sorted(
        set(dates_to_download) - processed_dates - dates_to_skip 
    )     

    return dates_to_process       
    
    
    

    
    
    
def save_false_color(wd, bands, ds):

    # Extract bands from xarray dataset
    b1 = np.squeeze(ds.sel(band = bands[0]).values)
    b2 = np.squeeze(ds.sel(band = bands[1]).values)
    b3 = np.squeeze(ds.sel(band = bands[2]).values)

    output_path = os.path.join(wd, "false_color_composite.tif")

    # Stack as RGB
    rgb = np.stack([b1, b2, b3])

    # Get raster metadata from dataset
    height, width = ds.sizes['y'], ds.sizes['x']

    try: 
        crs = ds.rio.crs
    except:
        crs = CRS.from_epsg(ds.epsg.item())
        
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": rgb.dtype,
        "crs": crs,
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


