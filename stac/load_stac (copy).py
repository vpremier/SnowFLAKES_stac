#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:03:56 2026

@author: vpremier
"""

from dotenv import load_dotenv
import os
import geopandas as gpd
import numpy as np
from pyproj import CRS
import stackstac
import time
import re
import logging
import rasterio as rio
import pystac_client
from shapely.geometry import box
from shapely.geometry import mapping
from rasterio.enums import Resampling
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO
from affine import Affine    
# import odc.stac
import requests as r 
from datetime import datetime, timedelta

from stac.utils_stac import *
import requests as rq
from rasterio.session import AWSSession
import boto3
from rioxarray import merge

            
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




def convert_landsat_bands(outdir, date, resolution=None, img4ext = None, 
                            extent_target=None, epsg_target=None, 
                            reproj_type=Resampling.bilinear, suffix='toa',
                            na_value = "NaN", calibration=True, ow=False,
                            max_cc = 90, idList = [], filter_by_geometry = True,
                            save = True, shp=None, exclude_tiles=None):        
    """
    Loads Sentinel-2 L1C data from the Copernicus Data Space STAC API,
    reprojects it to a user-defined grid, applies radiometric calibration, and
    optionally saves each band as a GeoTIFF.
    
    The function uses `stackstac` to lazily load Sentinel-2 assets and returns the
    result as an `xarray.DataArray` with dimensions:
    
        (time, band, y, x)
    
    
    Parameters
    ----------
        
    outdir : str, optional
        Output directory where GeoTIFF files will be written.
        
    date : str
        Acquisition date in format "YYYY-MM-DD". Only this day will be queried.
    
    resolution : float, optional
        Target pixel size in map units. If None, the native resolution is used.
    
    img4ext : str, optional
        Path to a reference raster used to extract target extent, projection,
        and resolution.
    
    extent_target : list, optional
        Output bounding box [xmin, ymin, xmax, ymax].
    
    epsg_target : int, optional
        Target coordinate reference system EPSG code.
    
    reproj_type : rasterio.enums.Resampling
        Resampling method used during reprojection.
    
    suffix : str
        Suffix appended to output filenames (e.g. "toa").
    
    na_value : float or str
        NoData value for output rasters.
    
    calibration : bool
        Apply Sentinel-2 reflectance calibration.
    
    ow : bool
        Overwrite existing files.
    
    max_cc : int
        Maximum allowed cloud cover percentage.
    
    idList : list
        Optional list of Sentinel-2 scene IDs.
    
    filter_by_geometry : bool
        If True, STAC search is constrained to the target geometry.
    
    save : bool
        If True, the output bands are saved as GeoTiff.
    
    Returns
    -------
    xarray.DataArray
        Sentinel-2 data cube with dimensions (time, band, y, x)
        containing calibrated reflectance values.
    """
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
    
    


    # convert the date to the required format
    day = datetime.strptime(date, "%Y-%m-%d")
    
    date_rfc3339 = f"{day.strftime('%Y-%m-%dT00:00:00Z')}/{day.strftime('%Y-%m-%dT23:59:59Z')}"
   
    
    
    
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
                    }
                }
            }
    
        query_return = fetch_stac_server(params) 
        
        [print(f['properties']['landsat:scene_id']) for f in query_return] 
        
        search = [l['href'] for l in catalog_links if l['rel'] == 'search'][0]   #retreive search endpoint from STAC Catalog
        query = r.post(search, json=params).json()   # send POST request to the stac-search endpoint with params passed in
        print(f"Items Found: {len(query['features'])}")

        query['features'][0] 
        # print the keys of the properties of a STAC item
        query['features'][0]['properties'].keys()


    elif idList:
        print('Filtering STAC by input ID list')
        # Looking for Sentinel-2 L1C
        params = {
            "collections": ["landsat-c2l1"],
            "datetime": f"{date}",
            "ids":idList,     
        }
        
    LP_items = [l['href'] for l in query['links'] if l['rel'] == 'items'][0]    # Set the items endpoint to variable


    start = time.time()

    items = list(cat.search(**params).items_as_dicts())
    print(f"Number of STAC items returned: {len(items)}")
    

    band_links = {}  # Initialize as an empty dictionary
    
    for query_item in query_return:
        for b in bands:
            band_link = query_item['assets'][b]['alternate']['s3']['href']
    
            # Create a list for the band if it doesn't exist yet
            band_links.setdefault(b, []).append(band_link)
    
    
    def use_s3_assets(items):
        for item in items:
            for key, asset in item["assets"].items():
                if "alternate" in asset and "s3" in asset["alternate"]:
                    asset["href"] = asset["alternate"]["s3"]["href"]
        return items
    
    
    items = use_s3_assets(query_return)
    
    
    # AWS credentials
    aws_session = AWSSession(
        boto3.Session(profile_name="default"),
        requester_pays=True
    )    


    import stackstac
    import rasterio as rio
    
    with rio.Env(aws_session):
        stack = stackstac.stack(
            items,                  # your STAC items (query["features"])
            assets=["swir16"],      # or "SR_B6" for Landsat C2 L2
            resolution=resolution,
            epsg=epsg_target,
            bounds=extent_target,
        )
        
    stack.load()
    
    
    with rio.Env(aws_session):
        with rio.open("s3://usgs-landsat/collection02/level-1/standard/oli-tirs/2025/232/084/LC08_L1TP_232084_20250217_20250226_02_T1/LC08_L1TP_232084_20250217_20250226_02_T1_B6.TIF") as src:
            print(src.shape)
    
    
    
    
    import boto3
    import os
    
    session = boto3.Session(profile_name="default")
    creds = session.get_credentials().get_frozen_credentials()
    
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
        
        
    os.environ["AWS_REQUEST_PAYER"] = "requester"
    
    
    import stackstac

    stack = stackstac.stack(
        items,              # patched to use s3://
        assets=["swir16"],
        resolution=30,
        chunksize=None
    )
        
    
    
    
    
    
    
    
    mergeList = []
    for link in band_links['swir16']:
        with rio.Env(aws_session):
            mergeList.append(rioxarray.open_rasterio(link))  # Open file and append to list
    mergeList
    
    
    swir_mosaic = merge.merge_arrays(mergeList, nodata=0)
    swir_mosaic
    
    
    
    if exclude_tiles:
        exclude_tiles = [f"MGRS-{t}" for t in exclude_tiles]
        
        filtered_items = [
            item for item in items
            if item['properties'].get('grid:code') not in exclude_tiles
        ]
        
        items = filtered_items
        print(f"Number of items after exclusion: {len(items)}")


    if len(items) == 0:

        return None, None
    
    # for Landsat-8: needs to be changed in case of other sensors
    bands = ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16',
             'swir22', 'lwir11']
    

    # id of the scene 
    image_id = items[0]['id']
    
    print(f"Loading {image_id}")

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
    
        
    # create folder
    if save:
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, f"{merged_image_id}"), exist_ok=True)

    try:
        data = stackstac.stack(
            items=items,
            bounds=extent_target,
            epsg=epsg_target,
            resolution=resolution,
            assets=bands,
            resampling=reproj_type,
            gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
                {
                    "GDAL_NUM_THREADS": -1,
                    "GDAL_HTTP_UNSAFESSL": "YES",
                    "GDAL_HTTP_TCP_KEEPALIVE": "YES",
                    "AWS_VIRTUAL_HOSTING": "FALSE",
                    "AWS_HTTPS": "YES",
                }
                ),
            )
        
        # dask_chunk_size = 1024
        # data = odc.stac.load(
        #     items,
        #     bands=bands,
        #     chunks={"y":dask_chunk_size, "x":dask_chunk_size, "time": 1},
        #     crs=epsg_target,
        #     resolution=resolution,
        #     bounds=extent_target,
        # )
        
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
        offset = -1000
        if baseline >= 400:
            data = (data + offset) * 0.0001
        else:
            raise ValueError("Old Sentinel-2 processing baseline (<0400) not supported")
            
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


    

        
    

if __name__ == "__main__":    
    
    
    resolution = 50
    epsg_target = 32719
    img4ext = r'/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/Landsat/LC08/01_TEST_auxiliary_folder/LC08_DEM.tif'

    shape_name = r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Echaurren/EsteroGlaciarEchaurren/polygon/polygon.shp'
    filter_by_geometry = True
    shp = None


    date = "2015-06-14"

    outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test'

    start = time.time()


    data = convert_sentinel2_bands(outdir, date, resolution=resolution, img4ext=img4ext, 
                            epsg_target=None, reproj_type=Resampling.cubic, 
                            suffix='toa', na_value = "NaN", calibration=True, 
                            ow=False)

    end = time.time()
    print(f"Total runtime of the program is {end - start} seconds")




# ds_odata = xr.open_dataset(r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test/tile/T19HCC/S2C_MSIL1C_20250206T143811_N0511_R096_T19HCC_20250206T161931/T19HCC_20250206T143811_B01_tmp.tif')


    
# opzione per salvare output oppure leggo array

# processare ghiacciai?
# check openeo





