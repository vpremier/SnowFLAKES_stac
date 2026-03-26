#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:02:12 2026

@author: vpremier
"""

import subprocess
import json
import os
import pandas as pd
from stac import load_stac
import time
import shutil
import glob

from dotenv import load_dotenv
load_dotenv()

from utils import *
from SnowFLAKES.main_SnowFLAKES import run_snowflakes
from data_download.main import run_query_download


def run_workflow(date_start, date_end, config_path):
    
    # Read
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Modify dates
    config["date_start"] = date_start
    config["date_end"] = date_end
    config["query_sentinel2"] = True
    config["download_sentinel2"] = False
    
    
    # resampling parameters
    resolution = config["resampling_params"]["resolution"]
    extent_target = config["resampling_params"]["extent_target"]
    epsg_target = config["resampling_params"]["epsg_target"]
    # bbox = get_shape_extent(shp, epsg=32719, outres =500)


    # Write back
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("Config updated")
    
    
    # Run the data query ------------------------------------------------------
    run_query_download(config_path)
    
    # Look for the data in our folder
    outdir = config["output_directory"]
    data_df = pd.read_csv(os.path.join(outdir, 'query_sentinel2.csv'))
    
    if data_df.empty:
        return
    
    log_file = os.path.join(outdir, "failed_dates.txt")

    
    s2_files = [f.split('.')[0] for f in data_df['Name'].to_list()]
    
    dates_to_process = get_dates_to_process(s2_files, config)    
                                
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
                data, scene_id = load_stac.convert_sentinel2_bands(outdir, date, 
                                                        resolution=resolution, 
                                                         extent_target=extent_target, 
                                                         epsg_target=epsg_target,
                                                         save = False,
                                                         shp=config['shapefile'])
                
                print(list(data.coords["band"].values))               
                # create folder
                os.makedirs(os.path.join(outdir, scene_id), exist_ok=True)
            
                # loading in the memory the STAC
                load_with_retry(data, max_retries=20, wait_seconds=2)
                
                
                # save RGB for visualization
                save_false_color(os.path.join(outdir, scene_id), ["B11", "B8A", "B03"], data)
                
                time.sleep(2)
                
                run_snowflakes(config, data, scene_id)
                
            except Exception as e:
                print(f"Error processing {scene_id} on date {date}: {e}")
                failed_dates.append(date)
                
                with open(log_file, "a") as f:
                    f.write(f"{date},{scene_id},{str(e)}\n")
        
        # Recompute dates to process (removes processed ones automatically)
        dates_to_process = get_dates_to_process(s2_files, config)
    
        # Optional: stop if nothing changed (avoid infinite loop)
        if set(dates_to_process) == set(failed_dates):
            print("Only failing dates remain. Stopping to avoid infinite loop.")
            break     
        

if __name__ == "__main__":
    
    
    start = pd.Timestamp("2017-01-01")
    end = pd.Timestamp("2018-05-15")
    # start = pd.Timestamp("2020-04-25")
    # end = pd.Timestamp("2020-04-26")
    # shape of the AOI
    config_path = './config_fram3s.json'
    
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

        
            
# modifica scenes to skip! le classifica con thematic    


# write readme`
# remove auxiliary?
    
# add layer uncertainty
# check ghiacciai
# guarda land cover


# fare lo stesso per Landsat

# provare ASTER???


### to do

# add time duration
# write documentatio 
    







    