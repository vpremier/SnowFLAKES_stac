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
from data_download.query_available import run_queries

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


from SnowFLAKES.utilities import get_uncertainty



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
    
    
    satellite = str(config.get("satellite", "")).strip().lower()
    if satellite not in {"sentinel-2", "sentinel2", "landsat", "both"}:
        raise ValueError(
            "satellite must be Sentinel-2, Landsat, or both"
        )
    config["output_directory"] = os.path.join(
        config["working_directory"],
        config.get("study_area") or os.path.splitext(
            os.path.basename(config["shapefile"])
        )[0],
    )
    
    
    # Run the modern data query directly.  Keep the resulting frames in the
    # same single DataFrame used by the debugging loop below.
    satellite_key = satellite.replace("_", "-")
    if satellite_key in {"sentinel-2", "sentinel2"}:
        query_satellite = "sentinel2"
    elif satellite_key in {"landsat", "both"}:
        query_satellite = satellite_key
    else:
        raise ValueError("satellite must be Sentinel-2, Landsat, or both")

    sentinel_source = str(
        config.get("DOWNLOAD_SENTINEL", "Google")
    ).strip().lower().replace("_", "-")
    if sentinel_source in {"stac", "stac-api", "cdse-stac-api"}:
        sentinel_source = "odata"
    elif sentinel_source not in {"google", "odata", "s3"}:
        sentinel_source = "google"

    query_paths = run_queries(
        study_area=config.get("study_area") or os.path.splitext(
            os.path.basename(config["shapefile"])
        )[0],
        working_directory=config["working_directory"],
        aoi=config["shapefile"],
        date_start=config["date_start"],
        date_end=config["date_end"],
        max_cloudcover=float(config.get("max_cloudcover", 90)),
        satellite=query_satellite,
        skip_sentinel2_tiles=config.get("exclude_tiles") or [],
        download_sentinel=sentinel_source,
    )

    outdir = config["output_directory"]
    frames = []
    for query_path in query_paths:
        try:
            frame = pd.read_csv(query_path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    data_df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["Name"])
    )
    
    # sceneList = glob.glob(outdir + os.sep + 'L*')
    # for scene in sceneList:
    #     scene_id = os.path.basename(scene)
    #     get_uncertainty(scene_id, config)


    if data_df.empty:
        return
    
    log_file = os.path.join(outdir, "failed_dates.txt")
    empty_items_dates = os.path.join(outdir, "00_dates_no_items.log")


    
    files = [f.split('.')[0] for f in data_df['Name'].to_list()]
    
    if not config.get('simple_class', False):
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
            date_token = str(date).replace("-", "")
            date_names = [name for name in files if date_token in name]
            date_sensor = (
                "Sentinel-2"
                if any(name.startswith("S2") for name in date_names)
                else "Landsat"
            )
        
            try:
                
                if date_sensor == "Sentinel-2":
                    # Select the Sentinel-2 backend from the configuration.  Keep
                    # the CDSE STAC API as the fallback for older configurations
                    # that do not yet define ``download_mode``.
                    download_mode = (str(config.get(
                        "DOWNLOAD_SENTINEL",
                        config.get("download_mode", "cdse stac api")
                    ))
                                     .strip().lower().replace("_", " "))
                    if download_mode in {"stac-api", "stac api", "odata", "s3"}:
                        download_mode = "cdse stac api"
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
                    
                elif date_sensor == "Landsat":
                    # Landsat: USGS STAC
                    load_stac_usgs.setup_usgs_credentials()

                    platform_code = next(
                        (name.split("_", 1)[0] for name in date_names
                         if name.startswith(("LT05_", "LE07_", "LC08_", "LC09_"))),
                        "LC08",
                    )
                    platform_name = {
                        "LT05": "LANDSAT_5",
                        "LE07": "LANDSAT_7",
                        "LC08": "LANDSAT_8",
                        "LC09": "LANDSAT_9",
                    }[platform_code]
                    data, scene_id = load_stac_usgs.convert_landsat_bands(outdir, 
                                                                          date, 
                                                                          resolution=resolution, 
                                                                          extent_target=extent_target, 
                                                                          epsg_target=epsg_target,
                                                                          save = False,
                                                                          platform = platform_name,
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
                if date_sensor == "Sentinel-2":
                    save_false_color(os.path.join(outdir, scene_id), ["B11", "B8A", "B03"], data, "fcc")

                    # Optional visualization-only RGB load. Keep the analysis
                    # cube at its configured resolution and request only the
                    # three native 10 m RGB bands when explicitly enabled.
                    if config.get("save_rgb_10m", False):
                        rgb_path = os.path.join(outdir, scene_id, "rgb_10m.tif")
                        if not os.path.exists(rgb_path) or config.get("overwrite", False):
                            rgb_kwargs = dict(sentinel2_kwargs)
                            rgb_kwargs.update({
                                "resolution": 10,
                                "bands": ["B04", "B03", "B02"],
                                "save": False,
                            })
                            if download_mode == "sentinelhub":
                                rgb_data, _ = load_sh.convert_sentinel2_bands(**rgb_kwargs)
                            else:
                                rgb_data, _ = load_stac.convert_sentinel2_bands(**rgb_kwargs)
                            load_with_retry(rgb_data, max_retries=20, wait_seconds=2)
                            save_false_color(
                                os.path.join(outdir, scene_id),
                                ["B04", "B03", "B02"],
                                rgb_data,
                                "rgb_10m",
                            )
                            del rgb_data
                        else:
                            print(f"Skipping existing {rgb_path}")

                    
                elif date_sensor == "Landsat":
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

    # start = pd.Timestamp("2021-05-01")
    # end = pd.Timestamp("2023-03-31")
    
    start = pd.Timestamp("2018-01-21")
    end = pd.Timestamp("2018-01-22")


    # shape of the AOI
    config_path = './config/config_mendoza_new.json'


    
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
        print('ciao')
        # run_workflow(date_start, date_end, config_path)
        
