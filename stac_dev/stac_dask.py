#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:21:38 2026

@author: vpremier
"""

import datetime, os

import boto3
from rasterio.session import AWSSession
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import rioxarray
from pystac_client import Client as psClient
from odc.stac import stac_load, configure_rio
from odc.geo.geobox import GeoBox
from dask.distributed import Client, LocalCluster


# Retrieve AWS credentials using boto3
session = boto3.Session(profile_name='cdse')
credentials = session.get_credentials()

# Set GDAL env variables for S3 buccket access on CDSE
os.environ['AWS_S3_ENDPOINT'] = "eodata.dataspace.copernicus.eu"
os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
os.environ['AWS_VIRTUAL_HOSTING'] = "FALSE"

# Create small localCluster
cluster = LocalCluster(n_workers=6, threads_per_worker=1)
client = Client(cluster)


# Small bounding box in EPSG:4326 (Germany)
extent_target = [573030.3488, 5048649.9999, 813030.3488, 5308649.9999]
epsg_target=25832

# Convert bounding box to EPSG:3035
bbox_of_interest = get_bbox_wgs84(img4ext=None, 
                                  extent_target=extent_target, 
                                  epsg_target=epsg_target, 
                                  buffer_m=1000)

# Extra rasterio config
configure_rio(
    cloud_defaults=True,
    verbose=True
)

# Create GeoBox from projected bbox
gbox = GeoBox.from_bbox(extent_target, resolution=50, crs="EPSG:25832")
print(gbox)

# Query STAC catalog
catalog = psClient.open("https://stac.dataspace.copernicus.eu/v1/")
query = catalog.search(
    collections=["sentinel-2-l1c"],
    bbox=bbox_of_interest,
    datetime=[datetime.datetime(2017, 10, 6), datetime.datetime(2017, 10, 7)],
)

# Load a minimal set of bands
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", 
         "B10", "B11", "B12", "B8A"]
ds = stac_load(
    query.items(),
    bands=bands,
    groupby="solar_day",
    resampling="cubic",
    geobox=gbox,
    chunks={"x": -1, "y": -1},
)

# Compute
ds = ds.compute()
