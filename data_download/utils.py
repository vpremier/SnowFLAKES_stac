#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:08:02 2023

@author: vpremier
"""

import json
import os
from datetime import datetime


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config



def check_config_consistency(config):
    """
    Validate required config fields for preprocessing.
    Raises ValueError if any check fails.
    """
    # Flags: check they exist and are bool
    for flag in ["query_landsat", "query_sentinel2", "download_landsat", "download_sentinel2"]:
        if flag not in config:
            raise ValueError(f"Missing required flag: '{flag}'")
        if not isinstance(config[flag], bool):
            raise ValueError(f"Flag '{flag}' must be a boolean.")

    # Output directory: must be string and not empty
    outdir = config.get("output_directory")
    if not isinstance(outdir, str) or not outdir.strip():
        raise ValueError("'output_directory' must be a non-empty string.")

    # Shapefile: must be string and not empty
    shp = config.get("shapefile")
    if not isinstance(shp, str) or not shp.strip():
        raise ValueError("'shapefile' must be a non-empty string.")
    if not os.path.isfile(shp):
        raise ValueError(f"Shapefile path does not exist: '{shp}'")

    # Dates: must be string and valid dates
    date_start = config.get("date_start")
    date_end = config.get("date_end")
    if not isinstance(date_start, str) or not isinstance(date_end, str):
        raise ValueError("'date_start' and 'date_end' must be strings in 'YYYY-MM-DD' format.")
    try:
        start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format.")
    if start_dt >= end_dt:
        raise ValueError("'date_start' must be before 'date_end'.")

    # Cloud cover: int in [0, 100]
    max_cc = config.get("max_cloudcover")
    if not isinstance(max_cc, int) or not (0 <= max_cc <= 100):
        raise ValueError("'max_cloudcover' must be an integer between 0 and 100.")

    # Landsat satellite list: must be list with valid options
    landsat_satellite = config.get("landsat_satellite")
    valid_satellites = {"LT05", "LE07", "LC08", "LC09"}
    if not isinstance(landsat_satellite, list):
        raise ValueError("'landsat_satellite' must be a list.")
    invalid = [s for s in landsat_satellite if s not in valid_satellites]
    if invalid:
        raise ValueError(f"Invalid Landsat satellites: {invalid}. Allowed: {valid_satellites}.")

    # s2_tile_list and landsat_tile_list: must be lists (can be empty)
    # for tile_key in ["s2_tile_list", "landsat_tile_list"]:
    #     if tile_key not in config:
    #         raise ValueError(f"Missing '{tile_key}' in config.")
    #     if not isinstance(config[tile_key], list):
    #         raise ValueError(f"'{tile_key}' must be a list.")



   

    










          
            

  
