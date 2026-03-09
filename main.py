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
import glob
from stac import load_stac
# from SnowFLAKES import TS_to_PNG
import shutil
import time

from utils import get_shape_extent
import geopandas as gpd
from shapely.geometry import box
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime



def run_workflow(date_start, date_end, shp, resolution):
    config_download_path = "./data_download/config.json"
    
    # Read
    with open(config_download_path, "r") as f:
        config_download = json.load(f)
    
    # Modify dates
    config_download["date_start"] = date_start
    config_download["date_end"] = date_end
    config_download["shapefile"] = shp
    config_download["query_sentinel2"] = True
    config_download["download_sentinel2"] = False



    # Write back
    with open(config_download_path, "w") as f:
        json.dump(config_download, f, indent=2)
    
    print("Config updated")
    
    
    # Run the data query ------------------------------------------------------
    subprocess.run("./data_download.sh", shell=True)
    
    # Look for the data in our folder
    outdir = config_download["output_directory"]
    data_df = pd.read_csv(os.path.join(outdir, 'query_sentinel2.csv'))
    

    
    
    s2_files = [f.split('.')[0] for f in data_df['Name'].to_list()]
    
    # extract dates 
    dates = sorted({
        f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        for d in (i.split("_")[2][:8] for i in s2_files)
    })
    
    
    # bbox = get_shape_extent(shp, epsg=32719, outres =500)
    bbox = [391500, 6278500, 404500, 6289000]


    # Run the STAC loading
    for date in dates:
    
        data = load_stac.convert_sentinel2_bands(outdir, date, resolution=resolution, 
                                                 extent_target=bbox, 
                                                 epsg_target=32719)
        time.sleep(2)
    



        

    
    
    # # Run SnowFLAKES ----------------------------------------------------------
    # config_sf_path = "./SnowFLAKES/input_json/fram3s.json"
    
    # # Read
    # with open(config_sf_path, "r") as f:
    #     config_sf = json.load(f)
    
    # config_sf['resolution'] = config_prep['resampling_parameters']['resolution']
    # config_sf['satellite'] = config_prep['satellite']
    # config_sf['Start Date'] = config_download["date_start"]
    # config_sf['End Date'] = config_download["date_end"]
    # config_sf['working_folder'] = wd
    
    # # Write back
    # with open(config_sf_path, "w") as f:
    #     json.dump(config_sf, f, indent=2)
    
    # print("Config updated")
    
    # subprocess.run("./run_SnowFLAKES.sh", shell=True)
    
    
    # # Save PNGs to visualize the results of SnowFLAKES ------------------------
    # png_folder = os.path.join(config_sf['working_folder'], 'PNG')
    # os.makedirs(png_folder, exist_ok=True)
    
    # scf_subfolder_name = config_sf['SVM_folder_name']
        
    # start_date = config_download["date_start"].replace("-", "")
    # end_date = config_download["date_end"].replace("-", "")
        
    # TS_to_PNG.save_scene_png(config_sf['working_folder'], png_folder, start_date, end_date, scf_subfolder_name)
            
            
    # # Move relevant results and delete unuseful data --------------------------     
            
    # # Move information to keep in another folder
    # # classified map, NDSI, NDVI, shadow mask
    # data_list = glob.glob(config_sf['working_folder'] + '/S2*')
    
    # for d in data_list:
    #     # look for snowflakes output
    #     id_scene = os.path.basename(d)
    
    #     try:       
    #         for par in ["SnowFLAKES", "NDSI", "NDVI","shadow_mask"]:
    #             output = glob.glob(d + f'/*/*{par}.tif').pop()
                
    #             par_dir = os.path.join(config_sf['working_folder'], par)
    #             os.makedirs(par_dir, exist_ok=True)
    #             new_name = os.path.join(par_dir, id_scene + '_' + par + '.tif')
    #             shutil.move(output, new_name)
            
    #         # Log classified image
    #         with open(classified_log, "a") as f:
    #             f.write(f"{id_scene}\n")
                
    #         print(f"✔ Classified: {id_scene}")
                
        
    #     except:
    #         print('No classified data')
            
    #         # Log cloudy images
    #         with open(cloudy_log, "a") as f:
    #             f.write(f"{id_scene}\n")
    
    #         print(f"⚠ Cloudy or incomplete: {id_scene}")
            
    #     # remove the image
    #     shutil.rmtree(d)
        
    # # remove also the downloaded archives
    # try:
    #     for d in downloaded:
    #         os.remove(d)
    # except:
    #     return
    
    return
        

if __name__ == "__main__":
    
    

    start = pd.Timestamp("2017-10-01")
    end = pd.Timestamp("2017-10-15")
    
    resolution = 20
    
    step = pd.Timedelta(days=15)
    
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

    
    # shape of the AOI
    shp = r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Echaurren/EsteroGlaciarEchaurren/polygon/polygon.shp'



    
    
    for date_start, date_end in date_pairs:
            
        run_workflow(date_start, date_end, shp, resolution)
    
    


# provare a salvare come zarr

# salvare anche i nuvolosi?
    
# add layer uncertainty
# check ghiacciai
# guarda land cover


# aggiungere download


# fare lo stesso per Landsat

# provare ASTER???

