#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:03:56 2026

@author: vpremier
"""
from dotenv import load_dotenv
import os

import rasterio
from shapely.geometry import shape
import pystac_client
from shapely.geometry import box, shape
from shapely.geometry import mapping
from rasterio.enums import Resampling

import stackstac
import geopandas as gpd

import numpy as np
from pyproj import CRS
import odc.stac

import time

import logging

import rasterio as rio

from utils import *




def convert_sentinel2_bands(outdir, date_start, date_end, resolution=None, 
                          img4ext = None, extent_target=None, epsg_target=None,
                          reproj_type=Resampling.bilinear, suffix='_boa',
                          na_value = "NaN", calibration=True, ow=False,
                          max_cc = 80, idList = [], filter_by_geometry = True):        
    """
    Converts and optionally reprojects Landsat bands from an xarray dataset to single-band GeoTIFFs.

    This function supports reprojection based on user-defined resolution, extent, EPSG code,
    or by matching the spatial configuration of a reference image (`img4ext`). Bands are 
    saved individually as GeoTIFF files.

    Parameters:
        data (xarray.Dataset): Dataset loaded using odc.stac.stac_load containing Landsat bands.
        outdir (str): Output directory where GeoTIFFs will be saved.
        image_id (str): Identifier for the Landsat scene, used to name output files.
        bands (dict): Dictionary mapping xarray band names to short names (e.g., {"SR_B2": "blue"}).
        resolution (float, optional): Target spatial resolution in meters. If None, native resolution is used.
        img4ext (str, optional): Path to a reference image whose extent and projection will be used.
        extent_target (list, optional): Custom output extent [xmin, ymin, xmax, ymax]. Ignored if `img4ext` is set.
        epsg_target (int or str, optional): Target coordinate reference system (EPSG code). Required if reprojection is desired.
        reproj_type (rasterio.enums.Resampling, optional): Resampling method for reprojection. Default is bilinear.
        suffix (str, optional): Suffix to add to the output filenames (e.g., "_toa" or "_boa").
        na_value (str or float, optional): Value used to represent NoData in the output. Default is "NaN".
        calibration (bool, optional): Whether to apply reflectance calibration to the bands. Default is True.
        ow (bool, optional): Overwrite existing output files. Default is False.

    Returns:
        None. Saves one GeoTIFF file per band in the specified output directory.
    """

        
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
    
    
    # credentials:  S3 Credentials from CDSE 
    # see https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials
    S3_ENDPOINT = "eodata.dataspace.copernicus.eu"

    load_dotenv()  # loads .env file
    os.environ["AWS_S3_ENDPOINT"] = S3_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # option 1 - use stackstac
    CDSE_URL = "https://stac.dataspace.copernicus.eu/v1"
    cat = pystac_client.Client.open(CDSE_URL)
    cat.add_conforms_to("ITEM_SEARCH")
    
    # define target information (extent, resolution etc)
    if img4ext:
        print('Reading extent, resolution and epsg from an image..')
        img, info = open_image(img4ext)
        extent_target = info['extent']
        crs = rasterio.crs.CRS.from_wkt(info['projection'])
        epsg_target = crs.to_epsg()
        resolution = info['geotransform'][1]
    else:
        assert extent_target and resolution and epsg_target, \
            "Please specify the target extent, resolution and EPSG or enter the path to a target image"
            
    # read from a shapefil?


        
    
    # determine AOI bbox in wgs84
    if filter_by_geometry:
        print('Filtering STAC by geometry')
        bbox_of_interest = get_bbox_wgs84(img4ext=img4ext, 
                                          extent_target=extent_target, 
                                          epsg_target=epsg_target, 
                                          buffer_m=1000)
        
        geometry = mapping(box(*bbox_of_interest))
        
        params = {
            "collections": ["sentinel-2-l1c"],
            "intersects": geometry,
            "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z"
            }
        
    elif idList:
        print('Filtering STAC by input ID list')
        # Looking for Sentinel-2 L1C
        params = {
            "collections": ["sentinel-2-l1c"],
            "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
            "ids":idList
        }

    items = list(cat.search(**params).items_as_dicts())
    print(f"Number of STAC items returned: {len(items)}")
    
    # for Sentinel-2: needs to be changed in case of other sensors
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", 
             "B11", "B12", "B8A"]
    

    # id of the scene 
    image_id = items[0]['id']
    
    print(f"Loading {image_id}")

    # Split by underscore
    parts = image_id.split("_")
    
    # Replace the tile (3rd element, index 2) with "merged"
    parts[5] = "merged"
    
    # Reconstruct the new ID
    merged_image_id = "_".join(parts)
    
    logging.info(f"Processing {image_id}")                
    print("Processing %s " %image_id)
    
    
    sensor = image_id.split('_')[0]
    
    # create folder
    os.makedirs(os.path.join(outdir, sensor), exist_ok=True)
    os.makedirs(os.path.join(outdir, sensor, f"{merged_image_id}"), exist_ok=True)



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
        # Replace 0 with NaN
        # data = data.where(data != 0, np.nan)
        
        # Group by day and compute mean
        data = data.groupby("time.day").mean(dim="time", skipna=True)
        


    except Exception as e:
        msg = f"Failed to load data for {merged_image_id}: {str(e)}"
        logging.error(msg)
        print(msg)
        return    
    
    # # === Extract info_src from xarray ===
    transform = data.attrs['transform']
  
    width = len(data.x)    
    height = len(data.y) 

    dst_crs = CRS.from_epsg(epsg_target)
    
    
    # Iterate through bands
    for band_name in data.band.values:
        out_path = os.path.join(outdir, sensor, f"{merged_image_id}", 
                                f"{merged_image_id}_{band_name}_{suffix}.tif")
        if os.path.exists(out_path) and not ow:
            print(f"Skipping {out_path} (already exists)")
            continue
        
        
        band_data = np.squeeze(data.sel(band=str(band_name)).values.astype("float32"))
        # nodata_val = data[band_name].attrs.get('nodata', -9999)
        
        # band_data[band_data == nodata_val] = np.nan
        
 
        # === Apply constants  ===
        if calibration:
            offset = -1000
            band_data = (band_data + offset)* 0.0001

        band_data[band_data <=0] = np.nan
        

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

    # # option 2
    # start = time.time()

    # catalog = pystac_client.Client.open(
    #             "https://stac.dataspace.copernicus.eu/v1/"
    #         )

    # items = catalog.search(
    #             collections=["sentinel-2-l1c"],
    #             bbox=bbox_of_interest,
    #             datetime=f"{date_start}/{date_end}",
    #         ).item_collection()

    # odc.stac.configure_rio(
    #     aws={
    #         "endpoint_url": S3_ENDPOINT,
    #         "aws_access_key_id": ACCESS_KEY,
    #         "aws_secret_access_key": SECRET_KEY,
    #         "region_name": "default"
    #     },
    #     AWS_VIRTUAL_HOSTING=False
    # )

    # dataset = odc.stac.load(
    #     items,
    #     bbox=bbox_of_interest,
    #     crs="EPSG:4326"
    # )

    # dataset[0,3].plot.imshow()
    
    # end = time.time()
    # print(f"Total runtime of the program is {end - start} seconds")

    

        
    

    
    


                



                

        
    
           
            

              
            

            
        
          



resolution = 20
epsg_target = 32719
img4ext = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test/tile/T19HCC/S2C_MSIL1C_20250206T143811_N0511_R096_T19HCC_20250206T161931/T19HCC_20250206T143811_B01_toa.tif'

shape_name = r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Echaurren/EsteroGlaciarEchaurren/polygon/polygon.shp'
extent_target = get_shape_extent(shape_name, epsg=32719, outres =500)

date_start = "2025-02-06"
date_end = "2025-02-07"

outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test'

start = time.time()

idList = ['S2C_MSIL1C_20250206T143811_N0511_R096_T19HCC_20250206T161931']

convert_sentinel2_bands(outdir, date_start, date_end, resolution=resolution, 
                          img4ext=img4ext, epsg_target=None,
                          reproj_type=Resampling.cubic, suffix='toa',
                          na_value = "NaN", calibration=True, ow=False,
                          max_cc = 80)

end = time.time()
print(f"Total runtime of the program is {end - start} seconds")

# dubbi: posso leggere direttamente in epsg 32719?


# ds_odata = xr.open_dataset(r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test/tile/T19HCC/S2C_MSIL1C_20250206T143811_N0511_R096_T19HCC_20250206T161931/T19HCC_20250206T143811_B01_tmp.tif')


    
    # read the extent, fix issue offset (PB), problema se ho più item
    # vuoi leggere id oppure direttamente fare un merge? 
    # opzione per salvare output oppure leggo array
    
    # check offset
    # processare ghiacciai?
    # check openeo





