#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:36:25 2026

@author: vpremier
"""

import pickle
from datetime import datetime, timedelta
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
)
import pystac_client
import matplotlib.pyplot as plt



config = SHConfig("free-cdse")

URL = "https://stac.dataspace.copernicus.eu/v1"
cat = pystac_client.Client.open(URL)
cat.add_conforms_to("ITEM_SEARCH")

# date
date = "2015-08-18"
start = datetime.strptime(date, "%Y-%m-%d")
end = start + timedelta(days=1)


# bounding box
resolution = 50
epsg_target = 32719
extent_target = [366000, 6205000, 428500, 6342500]

bbox = BBox(extent_target, epsg_target)


    
params = {
    "limit": 100,
    "collections": "sentinel-2-l1c",
    "datetime": f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
    "bbox": tuple(extent_target),
    "query": {"eo:cloud_cover": {"lte": 90}},
    "sortby": "properties.eo:cloud_cover",
    "fields": {"exclude": ["geometry"]},
}
stac_items = list(cat.search(**params).item_collection())
