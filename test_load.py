#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:56:49 2026

@author: vpremier
"""

def main():
    from dotenv import load_dotenv
    load_dotenv()  # loads .env file
    
    import os
    
    os.environ["CDSE_S3_ACCESS_KEY"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["CDSE_S3_SECRET_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
    os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("CDSE_S3_ACCESS_KEY")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("CDSE_S3_SECRET_KEY")
    os.environ["AWS_HTTPS"] = "YES"
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
    
    from urllib3 import Retry
    import pystac_client
    from pystac_client.stac_api_io import StacApiIO
    import odc.stac
    from shapely.geometry import mapping
    from shapely.geometry import box
    from rasterio.enums import Resampling
    import stackstac
    import numpy as np
    from dask.diagnostics import ProgressBar
    
    
    from dask.distributed import Client
    
    client = Client()
    print(client)
    
    
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
        # retry_after_max=300,  # cap retry to 5 minutes
    )
    
    cat = pystac_client.Client.open(CDSE_URL, stac_io=StacApiIO(max_retries=retry))
    
    
    cat.add_conforms_to("ITEM_SEARCH")
    
    
    
    bbox_of_interest = [9.923401957204756, 45.512035122015476, 13.198352068831694, 47.93600367442509]
    geometry = mapping(box(*bbox_of_interest))
    
    from odc.geo.geobox import GeoBox
    
    
    extent_target = [573030.3488, 5048649.9999, 813030.3488, 5308649.9999]
    resolution = 50
    epsg_target = 25832
    reproj_type = Resampling.bilinear
    
    
    
    # define a geobox for my region
    bounds = (573030.3488, 5048649.9999, 813030.3488, 5308649.9999)
    geobox = GeoBox.from_bbox(bounds, crs=f"epsg:{epsg_target}", resolution=resolution)
    
    
    date = "2018-06-30"
    
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", 
             "B10", "B11", "B12", "B8A"]
    
    params = {
        "collections": ["sentinel-2-l1c"],
        "intersects": geometry,
        "datetime": f"{date}",
        "query": {
            "eo:cloud_cover": {
                "lte": 90
                }
            }
        }
    
    
    URL = "https://stac.dataspace.copernicus.eu/v1"
    cat = pystac_client.Client.open(URL)
    cat.add_conforms_to("ITEM_SEARCH")

    
    search = cat.search(**params)
    
    # option 1
    items = list(cat.search(**params).items_as_dicts())
    
    
    
    stack = stackstac.stack(
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
    stack = stack.where(stack != 0, np.nan)
    
    # Group by day and compute mean
    stack = stack.groupby("time.day").mean(dim="time", skipna=True)      
    
    stack = stack.astype("float32")
    stack = stack.reset_coords(drop=True)
    
    
    stack = stack.copy()
    stack.attrs = {}
    
    for coord in stack.coords:
        stack.coords[coord].attrs = {}
    
    # optional but recommended
    # stack = stack.chunk({"day": 1, "y": 2048, "x": 2048})

    # output_path = "my_band.zarr"
    
    # with ProgressBar():
    #     stack.to_dataset(name="B01").to_zarr(output_path, mode="w")
    
    stack.load()


    # stack = odc.stac.load(
    #     search.items(),
    #     geobox=geobox,
    #     bands = bands,
    #     resampling="bilinear",
    #     chunks =  {
    #     "time": 1,
    #     "band":1,
    #     "y": dask_chunk_size,
    #     "x": dask_chunk_size
    # })

    
    return stack



if __name__ == "__main__":
    import time

    start = time.time()
    
    ds = main()
    
    end = time.time()
    print(f"Total runtime of the program is {end - start} seconds")

    
    


    
"""
B01 loading
    with odc stac load
    dask_chunk_size = 1024 13.35 s 1 band
    dask_chunk_size = 2048 14.26 s 1 band
    
    stack stac 4 s

B01 and B02 loading
    stack stac 114.43715620040894
    
    
stackstac + compute (B02) 121 s

stackstac + load (B02) 76 s




"""