#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:02:12 2026

@author: vpremier
"""

import json
import os
import pandas as pd
import shutil
import time

from SnowFLAKES.main_SnowFLAKES import run_snowflakes
from data_download.main import run_query_download

from loading import (
    load_stac, 
    load_stac_usgs, 
    load_sh
)

from utils import (
    remove_glaciers,
    get_dates_to_process,
    load_with_retry,
    save_false_color
)





def run_workflow(date_start, date_end, config_path):
    
    
    # Read
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Modify dates
    config["date_start"] = date_start
    config["date_end"] = date_end
    
    # resampling parameters
    resolution = config["resampling_params"]["resolution"]
    extent_target = config["resampling_params"]["extent_target"]
    epsg_target = config["resampling_params"]["epsg_target"]
    # bbox = get_shape_extent(shp, epsg=32719, outres =500)
    
    
    # differentiate for Sentinel-2 and Landsat
    if config["satellite"] == "Sentinel-2":
        config["query_sentinel2"] = True
        config["download_sentinel2"] = False
        config["download_landsat"] = False
        config["query_landsat"] = False
        
    elif config["satellite"].startswith("Landsat"):
        config["query_sentinel2"] = False
        config["download_sentinel2"] = False
        config["download_landsat"] = False
        config["query_landsat"] = True
    

    # Write back
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("Config updated")
    
    
    # Run the data query ------------------------------------------------------
    run_query_download(config_path)
    
    # Look for the data in our folder
    outdir = config["output_directory"]
    data_df = pd.read_csv(os.path.join(outdir, 'query.csv'))
    
    if data_df.empty:
        return
    
    log_file = os.path.join(outdir, "failed_dates.txt")
    empty_items_dates = os.path.join(outdir, "00_dates_no_items.log")


    
    files = [f.split('.')[0] for f in data_df['Name'].to_list()]
    
    if not config['simple_class']:
        remove_glaciers(outdir)
    
    dates_to_process = get_dates_to_process(files, config)    
                                
    while len(dates_to_process) > 0:
        print("\n" + "="*60)
        print(f"📅 Period: {date_start} → {date_end}")
        print(f"⏳ Pending scenes: {len(dates_to_process)}")
        print("="*60 + "\n") 
        
        failed_dates = []
        # Run the STAC loading
        for i, date in enumerate(dates_to_process):
            print(date)
        
            try:
                
                if config["satellite"] == "Sentinel-2":
                    # Select the Sentinel-2 backend from the configuration.  Keep
                    # the CDSE STAC API as the fallback for older configurations
                    # that do not yet define ``download_mode``.
                    download_mode = (str(config.get("download_mode", "cdse stac api"))
                                     .strip().lower().replace("_", " "))
                    sentinel2_kwargs = {
                        "outdir": outdir,
                        "date": date,
                        "resolution": resolution,
                        "extent_target": extent_target,
                        "epsg_target": epsg_target,
                        "save": False,
                        "shp": config["shapefile"],
                        "exclude_tiles": config["exclude_tiles"],
                    }

                    if download_mode == "sentinelhub":
                        # Sentinel Hub backend (loading/load_sh.py).
                        data, scene_id = load_sh.convert_sentinel2_bands(
                            **sentinel2_kwargs
                        )
                    elif download_mode == "cdse stac api":
                        # Copernicus Data Space STAC backend (loading/load_stac.py).
                        load_stac.setup_cdse_credentials()
                        data, scene_id = load_stac.convert_sentinel2_bands(
                            **sentinel2_kwargs
                        )
                    else:
                        raise ValueError(
                            "Unsupported Sentinel-2 download_mode "
                            f"{config.get('download_mode')!r}. Expected "
                            "'sentinelhub' or 'cdse stac api'."
                        )
                    
                elif config["satellite"].startswith("Landsat"):
                    # Landsat: USGS STAC
                    load_stac_usgs.setup_usgs_credentials()

                    data, scene_id = load_stac_usgs.convert_landsat_bands(outdir, 
                                                                          date, 
                                                                          resolution=resolution, 
                                                                          extent_target=extent_target, 
                                                                          epsg_target=epsg_target,
                                                                          save = False,
                                                                          platform = config["satellite"].upper().replace("-", "_"),
                                                                          shp=config['shapefile'],
                                                                          exclude_tiles=config['exclude_tiles'])
                    
                
                if scene_id is None:
                    with open(empty_items_dates, "a") as f:
                        f.write(f"{date}\n")
                                        
       
                
                # data = data.chunk({
                #         "day": 1,     # or small number
                #         "band": len(data.band),
                #         "x": 1024,
                #         "y": 1024
                #     })
                # ds = data.to_dataset(name="sentinel2")
                # ds = ds.reset_coords(drop=True)
                # data.to_zarr("output.zarr", mode="w")
                
                # data = data.chunk({"y": 2048, "x": 2048})
                # ds = data.compute()
                
                
                print(list(data.coords["band"].values))               
                # create folder
                os.makedirs(os.path.join(outdir, scene_id), exist_ok=True)
            
                # loading in the memory the STAC
                load_with_retry(data, max_retries=20, wait_seconds=2)
                
                
                # save RGB for visualization
                if config["satellite"] == "Sentinel-2":
                    save_false_color(os.path.join(outdir, scene_id), ["B11", "B8A", "B03"], data)
                    
                elif config["satellite"].startswith("Landsat"):
                    save_false_color(os.path.join(outdir, scene_id), ["swir16", "nir08", "green"], data)

                
                time.sleep(2)
                
                try:
                    run_snowflakes(config, data, scene_id)
                finally:
                    # Auxiliary rasters are intermediate products.  Remove only
                    # the scene-specific auxiliary directory when requested;
                    # the scene folder and its final products are preserved.
                    if config.get("remove_auxiliary", False):
                        scene_aux_folder = os.path.join(
                            outdir, scene_id, "auxiliary"
                        )
                        if os.path.isdir(scene_aux_folder):
                            shutil.rmtree(scene_aux_folder)
                
            except Exception as e:
                print(f"Error processing date {date}: {e}")
                failed_dates.append(date)
                
                with open(log_file, "a") as f:
                    f.write(f"{date},{str(e)}\n")
                    
                    
          
        
        # Recompute dates to process (removes processed ones automatically)
        dates_to_process = get_dates_to_process(files, config)
    
        # Optional: stop if nothing changed (avoid infinite loop)
        if set(dates_to_process) == set(failed_dates):
            print("Only failing dates remain. Stopping to avoid infinite loop.")
            break     
        

if __name__ == "__main__":
    
    
    start = pd.Timestamp("2013-04-01")
    end = pd.Timestamp("2015-03-31")
    # start = pd.Timestamp("2024-03-05")
    # end = pd.Timestamp("2024-03-06")
    # shape of the AOI
    # config_path = './config_snowcop_landsat.json'
    # config_path = './config/config_snowcop.json'
    config_path = './config/config_snowcop_landsat.json'

    
    step = pd.Timedelta(days=60)
    
    date_pairs = []
    
    current = start
    while current < end:
        next_date = current + step
        if next_date > end:
            next_date = end
    
        date_pairs.append((
            current.strftime("%Y-%m-%d"),
            next_date.strftime("%Y-%m-%d")
        ))
    
        current = next_date

    for date_start, date_end in date_pairs:
        run_workflow(date_start, date_end, config_path)

        
            


    
# add layer uncertainty
# guarda land cover
