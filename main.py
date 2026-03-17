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
from SnowFLAKES.main_SnowFLAKES import run_snowflakes
import time

from dotenv import load_dotenv
load_dotenv()

from utils import *


def run_workflow(date_start, date_end, shp):
    config_path = "./config.json"
    
    # Read
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Modify dates
    config["date_start"] = date_start
    config["date_end"] = date_end
    config["shapefile"] = shp
    config["query_sentinel2"] = True
    config["download_sentinel2"] = False


    # Write back
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("Config updated")
    
    
    # Run the data query ------------------------------------------------------
    subprocess.run("./data_download.sh", shell=True)
    
    # Look for the data in our folder
    outdir = config["output_directory"]
    data_df = pd.read_csv(os.path.join(outdir, 'query_sentinel2.csv'))
    
    if data_df.empty:
        return

    
    s2_files = [f.split('.')[0] for f in data_df['Name'].to_list()]
    
    # extract dates 
    dates = sorted({
        f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        for d in (i.split("_")[2][:8] for i in s2_files)
    })
    
    # resampling parameters
    resolution = config["resampling_params"]["resolution"]
    extent_target = config["resampling_params"]["extent_target"]
    epsg_target = config["resampling_params"]["epsg_target"]
    # bbox = get_shape_extent(shp, epsg=32719, outres =500)

    # Run the STAC loading
    for date in dates:
        
    
        data, scene_id = load_stac.convert_sentinel2_bands(outdir, date, 
                                                resolution=resolution, 
                                                 extent_target=extent_target, 
                                                 epsg_target=epsg_target,
                                                 save = False)
        
        if os.path.exists(os.path.join(outdir,scene_id)) and not config["overwrite"]:
            print(f"Scene {scene_id} already processed. Set overwrite as True in the config file.")
            return
        
        os.makedirs(os.path.join(outdir,scene_id), exist_ok=True)


        
        # save RGB for visualization
        save_false_color(os.path.join(outdir, scene_id), ["B11", "B8A", "B03"], data)
        
        time.sleep(2)
        
        run_snowflakes(config, data, scene_id)
        



        

    
    
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
    
    

    start = pd.Timestamp("2015-04-01")
    end = pd.Timestamp("2025-03-31")
    
    
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
    shp = r'/mnt/CEPH_PROJECTS/SNOWCOP/ValidationDataset/SMB/glaciers/Azufre.geojson'



    
    
    for date_start, date_end in date_pairs:
            
        run_workflow(date_start, date_end, shp)
    
    


# write readme`
    
# add layer uncertainty
# check ghiacciai
# guarda land cover


# fare lo stesso per Landsat

# provare ASTER???


### to do

# add time duration
# write documentatio 
    
    # aggiungere buffer per laghi

    # aggiornare doc string

    # si vede bordo tra le tile

    # soglie shadow - non shadow alpi e ande
    