#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:36:25 2026

@author: vpremier
"""
import os
import numpy as np
import logging
from rasterio.enums import Resampling
import rasterio as rio
import re
import boto3
import time
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import box
from pyproj import CRS

from sentinelhub import (
    BBox,
    DataCollection,
    SHConfig,
)
import pystac_client
from loading.sh_datacube import load
from shapely.geometry import mapping

from loading.utils_stac import (open_image, get_bbox_wgs84)


def convert_sentinel2_bands(outdir,
                            date,
                            resolution=None,
                            img4ext = None,
                            extent_target=None,
                            epsg_target=None,
                            reproj_type=Resampling.bilinear,
                            suffix='toa',
                            na_value = "NaN",
                            calibration=True,
                            ow=False,
                            max_cc = 90,
                            idList = [],
                            filter_by_geometry = True,
                            save = True,
                            shp=None,
                            exclude_tiles=None):
  
    
    
    # out directory
    os.makedirs(outdir, exist_ok=True)

    # logging file
    log_file = os.path.join(outdir, "sentinel2.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Started Sentinel-2 loading from CDSE")


    # credentials: set up the configuration
    session = boto3.Session(profile_name="sentinelhub")
    creds = session.get_credentials().get_frozen_credentials()
    
    # create credentials here https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings
    
    config = SHConfig()
    config.sh_client_id = creds.access_key
    config.sh_client_secret = creds.secret_key
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'


        
    URL = "https://stac.dataspace.copernicus.eu/v1"
    cat = pystac_client.Client.open(URL)
    cat.add_conforms_to("ITEM_SEARCH")


    # define target information (extent, resolution etc)
    if img4ext:
        print('Reading extent, resolution and epsg from an image..')
        img, info = open_image(img4ext)
        extent_target = info['extent']
        crs = rio.crs.CRS.from_wkt(info['projection'])
        epsg_target = crs.to_epsg()
        resolution = info['geotransform'][1]
    else:
        assert extent_target and resolution and epsg_target, \
            "Please specify the target extent, resolution and EPSG or enter the path to a target image"
              
    
    # determine AOI bbox in wgs84
    if filter_by_geometry:
        print('Filtering STAC by geometry')
        # determine AOI bbox in wgs84
        print('Filtering STAC by geometry')
        if shp:
            # Load shapefile
            gdf = gpd.read_file(shp)
            
            # Ensure it's in WGS84 (required by STAC APIs)
            gdf = gdf.to_crs(epsg=4326)
            
            # Merge all geometries into one (important if multiple features)
            # TODO: Check if we can use gdf.union_all() should be used instead of gdf.unary_union
            # geom = gdf.unary_union
            geom = gdf.union_all()
            geom = geom.simplify(0.05, preserve_topology=True)
            geometry = mapping(geom)
            
        else:
      
            bbox_of_interest = get_bbox_wgs84(img4ext=img4ext, 
                                              extent_target=extent_target, 
                                              epsg_target=epsg_target, 
                                              buffer_m=1000)
            
            geometry = mapping(box(*bbox_of_interest))
        
        params = {
            "collections": ["sentinel-2-l1c"],
            "intersects": geometry,
            "datetime": f"{date}",
            "query": {
                "eo:cloud_cover": {
                    "lte": max_cc
                    }
                }
            }
        
            
    # date format
    date_start = datetime.strptime(date, "%Y-%m-%d")
    date_end = date_start + timedelta(days=1)
    
    # bounding box
    bbox = BBox(extent_target, epsg_target) 
    
    items = list(cat.search(**params).items_as_dicts())
    
    if len(items) == 0:
        return None, None

    # id of the scene 
    image_id = items[0]['id']
    
    print(f"Loading {image_id}")
    
    # for Sentinel-2: needs to be changed in case of other sensors
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", 
             "B10", "B11", "B12", "B8A"]
    
    # Split by underscore
    parts = image_id.split("_")
    
    # Replace the tile (3rd element, index 2) with "merged"
    parts[5] = "merged"
    
    match = re.search(r'_N(\d{4})_', image_id)
    baseline = int(match.group(1)) if match else None
    
    # Reconstruct the new ID
    merged_image_id = "_".join(parts)
    
    logging.info(f"Processing {image_id}")                
    print("Processing %s " %image_id)
    
    
    start = time.time()

    # create folder
    if save:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, f"{merged_image_id}"), exist_ok=True)
        
    try:
        cube = load(
            DataCollection.SENTINEL2_L1C,
            bands=bands,
            bbox=bbox,  # BBox in EPSG:3035 from the params above
            time=(date_start, date_end),
            resolution=50,
            filter=f"eo:cloud_cover < {max_cc}",
            config=config,
        )
        # lazy: nothing downloaded yet, one request per timestamp
    
        data = cube.compute(scheduler="threads", num_workers=16)

        # Replace 0 with NaN
        data = data.where(data != 0, np.nan)
        
        # Group by day and compute mean
        data = data.groupby("time.day").max(dim="time", skipna=True)

    except Exception as e:
        msg = f"Failed to load data for {merged_image_id}: {str(e)}"
        logging.error(msg)
        print(msg)
        return    
    

    
    #=== Extract info_src from xarray ===
    transform = data.rio.transform()
  
    width = len(data.x)    
    height = len(data.y) 

    dst_crs = CRS.from_epsg(epsg_target)
    
    
    if calibration:
        data = data * 0.0001
        # offset = -1000
        # if baseline >= 400:
        #     data = (data + offset) * 0.0001
          
        # else:
        #     raise ValueError("Old Sentinel-2 processing baseline (<0400) not supported")
            
    data = data.where(data > 0)        
            
    # Iterate through bands
    if save:
        for band_name in data.band.values:
        
            out_path = os.path.join(outdir, f"{merged_image_id}", 
                                    f"{merged_image_id}_{band_name}_{suffix}.tif")
            
            if os.path.exists(out_path) and not ow:
                print(f"Skipping {out_path} (already exists)")
                continue
        
            band_data = np.squeeze(data.sel(band=str(band_name)).values.astype("float32"))

            # === Save GeoTIFF ===
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'float32',
                'crs': dst_crs,
                'transform': transform,
                'nodata': np.nan,
            }
            
            with rio.open(out_path, 'w', **profile) as dst:
                dst.write(band_data, 1)
    
            print(f"Saved {out_path}")
        
        
    end = time.time()
    print(f"Total runtime of the program is {end - start} seconds")
    
    return data, merged_image_id



