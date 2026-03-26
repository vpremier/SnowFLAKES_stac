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



from stac.utils_stac import *


# clms_urban-atlas_land-cover-use_europe_V025ha_vector_static_v01
# clms_urban-atlas_street-tree-layer_europe_V005ha_vector_static_v01
# cop-dem-eea-10-laea-tif
# cop-dem-glo-30-dged-cog
        
# collection = "cop-dem-glo-30-dged-cog"


def load_cdse_collection(collection, outdir, resolution=None, img4ext = None, 
                            extent_target=None, epsg_target=None, 
                            reproj_type=Resampling.bilinear, save=True, 
                            ow=False, shp=None):
    
    start = time.time()

    # out directory
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "DEM.tif")
    

    # credentials:  S3 Credentials from CDSE 
    # see https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials
    S3_ENDPOINT = "eodata.dataspace.copernicus.eu"

    load_dotenv()  # loads .env file
    os.environ["AWS_S3_ENDPOINT"] = S3_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_HTTPS"] = "YES"
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
    
    # option 1 - use stackstac
    CDSE_URL = "https://stac.dataspace.copernicus.eu/v1"
    # cat = pystac_client.Client.open(CDSE_URL)
    
    
    retry = Retry(
        total=5,
        backoff_factor=8,  # waits 0, 16s, 32s, 64s, 128s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
        raise_on_status=False,  # prevents urllib3 raising before pystac sees it
        respect_retry_after_header=True,  # Not certain that this header is ever set
        retry_after_max=300,  # cap retry to 5 minutes
    )
    
    cat = pystac_client.Client.open(CDSE_URL, stac_io=StacApiIO(max_retries=retry))
    
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
    print('Filtering STAC by geometry')
    if shp:
        # Load shapefile
        gdf = gpd.read_file(shp)
        
        # Ensure it's in WGS84 (required by STAC APIs)
        gdf = gdf.to_crs(epsg=4326)
        
        # Merge all geometries into one (important if multiple features)        
        geometry = mapping(gdf.unary_union)
        
    else:


        bbox_of_interest = get_bbox_wgs84(img4ext=img4ext, 
                                          extent_target=extent_target, 
                                          epsg_target=epsg_target, 
                                          buffer_m=1000)
        
        geometry = mapping(box(*bbox_of_interest))
    
    params = {"collections": [collection],
              "intersects": geometry}
    
    items = list(cat.search(**params).items_as_dicts())
    print(f"Number of STAC items returned: {len(items)}")
    
    data = stackstac.stack(
        items=items,
        bounds=extent_target,
        epsg=epsg_target,
        resolution=resolution,
        resampling=reproj_type,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(
            {
                "GDAL_NUM_THREADS": -1,
                "GDAL_HTTP_UNSAFESSL": "YES",
                "GDAL_HTTP_TCP_KEEPALIVE": "YES",
                "AWS_VIRTUAL_HOSTING": "FALSE",
                "AWS_HTTPS": "YES"
            }
            ),
        )
    
    data = data.mean(dim="time", skipna=True)

    
            # con questa configurazione non dava più problemi
            # "GDAL_HTTP_MAX_RETRY":"5",
            #     "GDAL_HTTP_RETRY_DELAY":"2",
            #     "GDAL_HTTP_TIMEOUT":"30",
            #     "GDAL_NUM_THREADS":"1",
            #     "CPL_VSIL_CURL_USE_HEAD":"NO",
            #     "GDAL_DISABLE_READDIR_ON_OPEN":"EMPTY_DIR",
            #     "CPL_VSIL_CURL_USE_HEAD": "NO",
            #     "GDAL_HTTP_TCP_KEEPALIVE": "NO"
            
            # questa era commentata
            # "GDAL_NUM_THREADS": "1",
            # "GDAL_HTTP_UNSAFESSL": "YES",
            # "GDAL_HTTP_TCP_KEEPALIVE": "YES",
            # "AWS_VIRTUAL_HOSTING": "FALSE",
            # "AWS_HTTPS": "YES",
            # "GDAL_HTTP_MAX_RETRY": "5",
            # "GDAL_HTTP_RETRY_DELAY": "2",
            # "GDAL_HTTP_TIMEOUT": "30",
            
    
    #  === Extract info_src from xarray ===
    transform = Affine(
        resolution, 0, extent_target[0],
        0, -resolution, extent_target[3]
    )
      

    width = len(data.x)    
    height = len(data.y) 

    dst_crs = CRS.from_epsg(epsg_target)
    

    if save:
        band_data = np.squeeze(data).values.astype("float32")

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
    
    return data
            
    
def convert_sentinel2_bands(outdir, date, resolution=None, img4ext = None, 
                            extent_target=None, epsg_target=None, 
                            reproj_type=Resampling.bilinear, suffix='toa',
                            na_value = "NaN", calibration=True, ow=False,
                            max_cc = 90, idList = [], filter_by_geometry = True,
                            save = True, shp=None):        
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
    # cat = pystac_client.Client.open(CDSE_URL)
    

    
    retry = Retry(
        total=5,
        backoff_factor=8,  # waits 0, 16s, 32s, 64s, 128s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
        raise_on_status=False,  # prevents urllib3 raising before pystac sees it
        respect_retry_after_header=True,  # Not certain that this header is ever set
        retry_after_max=300,  # cap retry to 5 minutes
    )
    
    cat = pystac_client.Client.open(CDSE_URL, stac_io=StacApiIO(max_retries=retry))


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
            "collections": ["sentinel-2-l1c"],
            "intersects": geometry,
            "datetime": f"{date}",
            "query": {
                "eo:cloud_cover": {
                    "lte": max_cc
                    }
                }
            }
        
        
    elif idList:
        print('Filtering STAC by input ID list')
        # Looking for Sentinel-2 L1C
        params = {
            "collections": ["sentinel-2-l1c"],
            "datetime": f"{date}",
            "ids":idList,     
        }

    start = time.time()

    items = list(cat.search(**params).items_as_dicts())
    print(f"Number of STAC items returned: {len(items)}")
    
    # for Sentinel-2: needs to be changed in case of other sensors
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", 
             "B10", "B11", "B12", "B8A"]
    

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
        # Replace 0 with NaN
        # data = data.where(data != 0, np.nan)
        
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

    

        
    

if __name__ == "__main__":    

    resolution = 20
    epsg_target = 32719
    img4ext = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test/tile/T19HCC/S2C_MSIL1C_20250206T143811_N0511_R096_T19HCC_20250206T161931/T19HCC_20250206T143811_B01_toa.tif'

    shape_name = r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Echaurren/EsteroGlaciarEchaurren/polygon/polygon.shp'
    extent_target = get_shape_extent(shape_name, epsg=32719, outres =500)

    date = "2025-02-06"

    outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/stac_test'

    start = time.time()

    idList = ['S2C_MSIL1C_20250206T143811_N0511_R096_T19HCC_20250206T161931']

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





