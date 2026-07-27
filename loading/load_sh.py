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
from rasterio.transform import from_origin
import rasterio as rio
import re
import boto3
import time
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import box
from pyproj import CRS, Transformer
from math import floor

from sentinelhub import (
    BBox,
    CRS as SHCRS,
    DataCollection,
    SHConfig,
)
import pystac_client
from loading.sh_datacube import load
from shapely.geometry import mapping

from loading.utils_stac import (open_image, get_bbox_wgs84)


SH_SUPPORTED_EXTRA_EPSG = {
    2154,
    2180,
    2193,
    3003,
    3004,
    3005,
    3006,
    3031,
    3035,
    3161,
    3346,
    3413,
    3416,
    3578,
    3580,
    3765,
    3794,
    3844,
    3912,
    3995,
    4026,
    5514,
    28992,
    32184,
}


def is_sentinelhub_crs_supported(epsg: int) -> bool:
    """ Check whether an EPSG code is documented as supported by Process API.
        See: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process/Crs.html
    """
    epsg = int(epsg)

    return (
        epsg in {4326, 3857}
        or 32601 <= epsg <= 32660
        or 32701 <= epsg <= 32760
        or epsg in SH_SUPPORTED_EXTRA_EPSG
    )


def choose_sentinelhub_request_epsg(
    extent: list[float] | tuple[float, float, float, float],
    source_epsg: int,
) -> int:
    """
    Return the original CRS when supported.

    Otherwise, choose the WGS84 UTM CRS containing the center of the AOI.
    For large AOIs spanning multiple UTM zones, use EPSG:4326.
    """
    source_epsg = int(source_epsg)

    if is_sentinelhub_crs_supported(source_epsg):
        return source_epsg

    minx, miny, maxx, maxy = map(float, extent)

    transformer = Transformer.from_crs(
        f"EPSG:{source_epsg}",
        "EPSG:4326",
        always_xy=True,
    )

    corners = [
        transformer.transform(minx, miny),
        transformer.transform(minx, maxy),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
    ]

    longitudes = [point[0] for point in corners]
    latitudes = [point[1] for point in corners]

    min_lon = min(longitudes)
    max_lon = max(longitudes)
    min_lat = min(latitudes)
    max_lat = max(latitudes)

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    longitude_span = max_lon - min_lon

    # A single UTM zone is 6 degrees wide. For broad AOIs, WGS84 is safer.
    if longitude_span > 6:
        return 3857

    if center_lat > 84:
        return 3413

    if center_lat < -80:
        return 3031

    zone = floor((center_lon + 180) / 6) + 1
    zone = max(1, min(zone, 60))

    if center_lat >= 0:
        return 32600 + zone

    return 32700 + zone


def make_sentinelhub_bbox(
    extent: list[float] | tuple[float, float, float, float],
    source_epsg: int,
) -> tuple[BBox, BBox, int]:
    """
    Create both the target bbox and the Process API request bbox.
    """
    source_epsg = int(source_epsg)

    bbox_target = BBox(
        bbox=extent,
        crs=SHCRS(source_epsg),
    )

    request_epsg = choose_sentinelhub_request_epsg(
        extent=extent,
        source_epsg=source_epsg,
    )

    if request_epsg == source_epsg:
        bbox_request = bbox_target
    else:
        bbox_request = bbox_target.transform_bounds(
            SHCRS(request_epsg)
        )

    return bbox_target, bbox_request, request_epsg


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
    # bbox = BBox(extent_target, epsg_target)
    bbox_target, bbox_request, request_epsg = make_sentinelhub_bbox(
        extent=extent_target,
        source_epsg=epsg_target,
    )

    if request_epsg != int(epsg_target):
        # Give the local reprojection enough neighbouring source pixels for
        # bilinear interpolation. The halo affects only the download; the
        # result is cropped onto the exact configured target grid below.
        halo = 2 * float(resolution)
        bbox_request = BBox(
            [
                bbox_request.min_x - halo,
                bbox_request.min_y - halo,
                bbox_request.max_x + halo,
                bbox_request.max_y + halo,
            ],
            bbox_request.crs,
        )

    print(f"Target CRS: EPSG:{epsg_target}")
    print(f"Sentinel Hub request CRS: EPSG:{request_epsg}")
    print(f"Target bbox: {bbox_target}")
    print(f"Request bbox: {bbox_request}")

    logging.info(
        "Target CRS EPSG:%s; Sentinel Hub request CRS EPSG:%s",
        epsg_target,
        request_epsg,
    )
    
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
            bbox=bbox_request,  # BBox in EPSG:3035 from the params above
            time=(date_start, date_end),
            resolution=resolution,
            filter=f"eo:cloud_cover < {max_cc}",
            config=config,
        )
        # lazy: nothing downloaded yet, one request per timestamp
    
        data = cube.compute(scheduler="threads", num_workers=16)

        # Replace 0 with NaN
        data = data.where(data != 0, np.nan)
        
        # Group by day and compute mean
        data = data.groupby("time.day").max(dim="time", skipna=True)

        if data.sizes.get("day") != 1:
            raise ValueError(
                f"Expected one day of data, found {data.sizes.get('day')}"
            )

        day_value = data.coords["day"].values
        data = data.squeeze("day", drop=True)

        if request_epsg != int(epsg_target):
            xmin, ymin, xmax, ymax = map(float, extent_target)
            target_resolution = float(resolution)

            width_exact = (xmax - xmin) / target_resolution
            height_exact = (ymax - ymin) / target_resolution

            if not np.isclose(width_exact, round(width_exact)):
                raise ValueError(
                    "Target extent width is not divisible by the target "
                    f"resolution: {xmax - xmin} / {target_resolution}"
                )

            if not np.isclose(height_exact, round(height_exact)):
                raise ValueError(
                    "Target extent height is not divisible by the target "
                    f"resolution: {ymax - ymin} / {target_resolution}"
                )

            target_width = round(width_exact)
            target_height = round(height_exact)

            target_transform = from_origin(
                xmin,
                ymax,
                target_resolution,
                target_resolution,
            )

            # Make the source CRS explicit before reprojection.
            data = data.rio.set_spatial_dims(
                x_dim="x",
                y_dim="y",
                inplace=False,
            )
            data = data.rio.write_crs(
                f"EPSG:{request_epsg}",
                inplace=False,
            )

            data = data.rio.reproject(
                f"EPSG:{epsg_target}",
                shape=(target_height, target_width),
                transform=target_transform,
                resampling=reproj_type,
                nodata=np.nan,
            )

            expected_bounds = (xmin, ymin, xmax, ymax)
            actual_bounds = data.rio.bounds()
            if not np.allclose(actual_bounds, expected_bounds, atol=1e-6):
                raise ValueError(
                    f"Unexpected output bounds: {actual_bounds}; "
                    f"expected {expected_bounds}"
                )

        data = data.expand_dims(day=day_value)
        data = data.transpose("day", "band", "y", "x")


    except Exception as e:
        msg = f"Failed to load data for {merged_image_id}: {str(e)}"
        logging.error(msg)
        print(msg)
        raise
    

    
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

