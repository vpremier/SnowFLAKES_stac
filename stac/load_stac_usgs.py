#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:04:04 2026

@author: vpremier
"""

import os
import time
import boto3
import numpy as np
import requests as rq
from rasterio.enums import Resampling
import logging
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import mapping
from shapely.geometry import box
import json 
from rasterio.session import AWSSession
import stackstac
from datetime import datetime
import rasterio as rio

from stac.utils_stac import *



def fetch_stac_server(params):
    """ 
    Queries the STAC server (STAC) backend.
    params is a Python dictionary to pass as JSON to the request.
    """
    
    search_url = f"https://landsatlook.usgs.gov/stac-server/search"
    query_return = rq.post(search_url, json=params).json()
    error = query_return.get("message", "")
    if error:
        raise Exception(f"STAC-Server failed and returned: {error}")
        
    print(f"Items Found: {len(query_return['features'])}")   
    
    for q in query_return['features']: print(f"Platform: {q['properties']['platform']}, Cloud Cover: {q['properties']['eo:cloud_cover']}, Collection: {q['description']}, ID: {q['id']}")
        
    return query_return['features']




def use_s3_assets(items):
    for item in items:
        for key, asset in item["assets"].items():
            if "alternate" in asset and "s3" in asset["alternate"]:
                asset["href"] = asset["alternate"]["s3"]["href"]
    return items




def get_MTL_file(query_item):
    session = boto3.Session(profile_name="default")
    s3_client = session.client('s3')
    object_key  = query_item['assets']['MTL.json']['alternate']['s3']['href']

    # Split the S3 path to extract the bucket name and object key
    bucket_name, key = object_key.replace("s3://", "").split('/', 1)

    # Retrieve the JSON file object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=key, RequestPayer='requester')

    # Read the content of the JSON file
    json_content = response['Body'].read().decode('utf-8')

    # Parse the JSON content 
    return json.loads(json_content)




def get_query_items(date, img4ext = None, extent_target=None, resolution=None,
                    epsg_target=None, max_cc = 90, filter_by_geometry = True, 
                    shp=None, platform = 'LANDSAT_8', idList = []):
    
    
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

    # convert the date to the required format
    day = datetime.strptime(date, "%Y-%m-%d")
    
    date_rfc3339 = f"{day.strftime('%Y-%m-%dT00:00:00Z')}/{day.strftime('%Y-%m-%dT23:59:59Z')}"
    

    # determine AOI bbox in wgs84
    if filter_by_geometry:
        print('Filtering STAC by geometry')
        # determine AOI bbox in wgs84
        if shp:
            # Load shapefile
            gdf = gpd.read_file(shp)
            
            # Ensure it's in WGS84 (required by STAC APIs)
            gdf = gdf.to_crs(epsg=4326)
            
            # Merge all geometries into one (important if multiple features) 
            geom = gdf.unary_union
            geom = geom.simplify(0.05, preserve_topology=True)
            geometry = mapping(geom)
            
        else:
      
      
            bbox_of_interest = get_bbox_wgs84(img4ext=img4ext, 
                                              extent_target=extent_target, 
                                              epsg_target=epsg_target, 
                                              buffer_m=1000)
            
            geometry = mapping(box(*bbox_of_interest))
            
            
        params = {
            "collections": ["landsat-c2l1"],
            "intersects": geometry,
            "datetime": date_rfc3339,
            "query": {
                "eo:cloud_cover": {
                    "lte": max_cc
                    },
                "platform": {'in': [platform]}
                }
            }
        
    elif idList:
        print('Filtering STAC by input ID list')
        raise NotImplementedError("Filtering by ID list not implemented yet")

    # query
    query_return = fetch_stac_server(params) 
    items = use_s3_assets(query_return)
    
    return items
   
    
   
    
def get_scene_center_time(date, img4ext = None, extent_target=None, 
                          resolution=None, epsg_target=None, max_cc = 90, 
                          filter_by_geometry = True, shp=None, 
                          platform = 'LANDSAT_8', idList = []):
    
    items = get_query_items(date, img4ext = img4ext, extent_target=extent_target, 
                            resolution=resolution, epsg_target=epsg_target, 
                            max_cc = max_cc, filter_by_geometry = filter_by_geometry, 
                            shp=shp, platform = platform, idList = idList)
    
    MTL_info = get_MTL_file(items[0]) 
    
    time = MTL_info['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SCENE_CENTER_TIME']
    time_clean = time.split('.')[0]
    
    dt = datetime.strptime(
        date + ' ' + time_clean,
        '%Y-%m-%d %H:%M:%S'
    )
    
    return dt




def convert_landsat_bands(outdir, date, resolution=None, img4ext = None, 
                            extent_target=None, epsg_target=None, 
                            reproj_type=Resampling.bilinear, suffix='toa',
                            na_value = "NaN", calibration=True, ow=False,
                            max_cc = 90, platform = 'LANDSAT_8', idList = [], 
                            filter_by_geometry = True,
                            save = True, shp=None, exclude_tiles=None):   
    
    # out directory
    os.makedirs(outdir, exist_ok=True)
    
    # logging file
    log_file = os.path.join(outdir, "landsat.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Started Landsat loading from USGS STAC catalogue")
    
    
    items = get_query_items(date, img4ext = img4ext, extent_target=extent_target, 
                            resolution=resolution, epsg_target=epsg_target, 
                            max_cc = max_cc, filter_by_geometry = filter_by_geometry, 
                            shp=shp, platform = platform, idList = idList)
    
    start = time.time()


    # no items
    if len(items) == 0:
        return None, None
    
    # id of the scene 
    image_id = items[0]['id']
    
    print(f"Loading {image_id}")
    
    # Split by underscore
    parts = image_id.split("_")
    
    # Replace the tile (3rd element, index 2) with "merged"
    parts[2] = "merged"
    
    # Reconstruct the new ID
    merged_image_id = "_".join(parts)
    
    logging.info(f"Processing {merged_image_id}")                
    print("Processing %s " %merged_image_id)
    

    if platform == "LANDSAT_5" or platform == "LANDSAT_7":
        # #Riccardo usava per L5 solo queste??
        # bands = ['blue', 'green', 'red', 'nir08', 'swir16']
        bands = {
                    1: 'blue',
                    2: 'green',
                    3: 'red',
                    4: 'nir08',
                    5: 'swir16',
                    6: 'lwir',
                    7: 'swir22'
                }
    elif platform == "LANDSAT_8" or platform == "LANDSAT_9":
        # bands 1, 2, 3, 4, 5 , 6, 7, 10
        bands = {
                    1: 'coastal',
                    2: 'blue',
                    3: 'green',
                    4: 'red',
                    5: 'nir08',
                    6: 'swir16',
                    7: 'swir22',
                    10: 'lwir11'
                }


    # create folder
    if save:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, f"{merged_image_id}"), exist_ok=True)
        
        
    # import stackstac
    
    try:
        session = boto3.Session(profile_name="default")
        aws_session = AWSSession(session)
    
        with rio.Env(aws_session):
            data = stackstac.stack(
               items=items,
               bounds=extent_target,
               epsg=epsg_target,
               resolution=resolution,
               resampling=reproj_type,
               assets=list(bands.values()),
               xy_coords="center"
            )
        
        # Replace 0 with NaN
        data = data.where(data != 0, np.nan)
        
        # Group by day and compute mean
        data = data.groupby("time.day").mean(dim="time", skipna=True)
    
    except Exception as e:
        msg = f"Failed to load data for {merged_image_id}: {str(e)}"
        logging.error(msg)
        print(msg)
        return    
        
    
    #=== Extract info_src from xarray ===
    transform = data.attrs['transform']
  
    width = len(data.x)    
    height = len(data.y) 

    dst_crs = CRS.from_epsg(epsg_target)
    

    
    if calibration:
        # see also from here     
        #https://code.usgs.gov/eros-user-services/processing_landsat_data/scaling-landsat-collection-2-level-1-data/-/blob/main/Scaling_Landsat_C2_L1_Data_v2.ipynb?ref_type=heads

        # conversion to Top of Atmosphere reflectance
        MTL_info = get_MTL_file(items[0]) 
        
        zenith = np.radians(90 - float(MTL_info['LANDSAT_METADATA_FILE']['IMAGE_ATTRIBUTES']['SUN_ELEVATION']))
        
        for band in bands:        
            band_name = bands[band]
        
            if band_name.startswith("lwir"):
                # thermal bands
                
                # Get parameters from the MTL file
                K1 = float(MTL_info['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS'][f'K1_CONSTANT_BAND_{band}'])
                K2 = float(MTL_info['LANDSAT_METADATA_FILE']['LEVEL1_THERMAL_CONSTANTS'][f'K2_CONSTANT_BAND_{band}'])
        
                thermal_rad_add = float(MTL_info['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f'RADIANCE_ADD_BAND_{band}'])
                thermal_rad_mult = float(MTL_info['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f'RADIANCE_MULT_BAND_{band}'])
        
                thermal_array = data.sel(band=band_name)
        
                radiance_thermal = thermal_array * thermal_rad_mult + thermal_rad_add
        
                result = K2 / (np.log(K1 / radiance_thermal + 1))
        
                data.loc[dict(band=band_name)] = result
        
            else:
                # convert band to reflectance
                add_val = float(MTL_info['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f'REFLECTANCE_ADD_BAND_{band}'])
                mult_val = float(MTL_info['LANDSAT_METADATA_FILE']['LEVEL1_RADIOMETRIC_RESCALING'][f'REFLECTANCE_MULT_BAND_{band}'])
        
                result = (data.sel(band=band_name) * mult_val + add_val) / np.cos(zenith)
        
                data.loc[dict(band=band_name)] = result
        
    data = data.where(data > 0)     

        
    # Iterate through bands
    if save:
        for band in bands:
        
            out_path = os.path.join(outdir, f"{merged_image_id}", 
                                    f"{merged_image_id}_B{band}_{suffix}.tif")
            
            if os.path.exists(out_path) and not ow:
                print(f"Skipping {out_path} (already exists)")
                continue
        
            band_data = np.squeeze(data.sel(band=bands[band]).values.astype("float32"))
   
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





if __name__ == "__main__":    
    
    
    resolution = 50
    epsg_target = 32719
    img4ext = r'/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/Landsat/L1-LC08/01_TEST_auxiliary_folder/DEM.tif'


    date = "2018-06-29"

    outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test'

    start = time.time()


    data, merged_image_id = convert_landsat_bands(outdir, date, resolution=resolution, img4ext=img4ext, 
                            epsg_target=None, reproj_type=Resampling.cubic, 
                            suffix='toa', na_value = "NaN", calibration=True, 
                            ow=False, platform = 'LANDSAT_8')
    
    
    # rename bands like sentinel-2?
    
    data = data.load()

    end = time.time()
    print(f"Total runtime of the program is {end - start} seconds")