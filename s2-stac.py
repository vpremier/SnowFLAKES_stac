#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:03:56 2026

@author: vpremier
"""
from dotenv import load_dotenv
import os

from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
import pystac_client
from shapely import to_geojson
import json

import rioxarray 
import stackstac

S3_ENDPOINT = "eodata.dataspace.copernicus.eu"


os.environ["AWS_S3_ENDPOINT"] = S3_ENDPOINT
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")



geometry = {'type': 'Polygon',
'coordinates': [[[-56.055536, -12.63809],
  [-56.055536, -12.523493],
  [-55.88178, -12.523493],
  [-55.88178, -12.63809],
  [-56.055536, -12.63809]]]}

bounds = shape(geometry).bounds


CDSE_URL = "https://stac.dataspace.copernicus.eu/v1"
cat = pystac_client.Client.open(CDSE_URL)
cat.add_conforms_to("ITEM_SEARCH")

start_dt = "2025-07-01"
end_dt = "2025-07-30"



params = {
    "collections": ["sentinel-2-l1c"],
    "intersects": geometry,
    "datetime": f"{start_dt}T00:00:00Z/{end_dt}T23:59:59Z"
}

items = list(cat.search(**params).items_as_dicts())
print(f"Number of STAC items returned: {len(items)}")



stack = stackstac.stack(
    items=items,
    resolution=(0.00025, 0.00025),
    bounds_latlon=bounds,
    epsg=4326,
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

stack[0,3].plot.imshow()