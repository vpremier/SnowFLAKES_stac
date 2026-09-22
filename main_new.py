#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run SnowFLAKES with AOI streaming or complete tile downloads.

Add ``"download_full_tiles": false`` to the configuration. When it is false,
the existing STAC-based workflow is used. When it is true, complete Sentinel-2
tiles are downloaded from Google with S2DL, or Landsat Collection 2 Level-1
scenes are downloaded through USGS M2M. The local tiles are then calibrated,
merged on the configured target grid, and passed to SnowFLAKES.
"""

import argparse
import os
import shutil
import time

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

from data_download.landsat_query_download import download_landsat, query_landsat
from data_download.sentinel2_google_query import query_google_sentinel2
from data_download.utils import load_config
from loading.download_sentinel2_s2dl import download_s2dl
from loading.load_local_tiles import load_landsat_tiles, load_sentinel2_tiles


LANDSAT_CODES = {
    "Landsat-5": "LT05",
    "Landsat-7": "LE07",
    "Landsat-8": "LC08",
    "Landsat-9": "LC09",
}


def run_workflow(date_start, date_end, config_path):
    """Import the existing AOI workflow only when that strategy is selected."""
    from main import run_workflow as run_aoi_workflow
    return run_aoi_workflow(date_start, date_end, config_path)


def run_snowflakes(config, data, scene_id):
    """Import SnowFLAKES only when image processing starts."""
    from SnowFLAKES.main_SnowFLAKES import run_snowflakes as process_scene
    return process_scene(config, data, scene_id)


def check_new_config(config):
    """Validate the settings used to choose the download strategy."""
    full_tiles = config.get("download_full_tiles", False)
    if not isinstance(full_tiles, bool):
        raise ValueError("'download_full_tiles' must be true or false")

    satellite = config.get("satellite")
    if satellite != "Sentinel-2" and satellite not in LANDSAT_CODES:
        raise ValueError(
            "'satellite' must be Sentinel-2 or Landsat-5/7/8/9"
        )

    if full_tiles:
        params = config.get("resampling_params", {})
        for key in ("extent_target", "resolution", "epsg_target"):
            if params.get(key) is None:
                raise ValueError(
                    f"resampling_params.{key} is required for tile downloads"
                )


def query_and_download_tiles(config):
    """Query and download complete products for the configured sensor."""
    outdir = config["output_directory"]
    os.makedirs(outdir, exist_ok=True)
    query_path = os.path.join(outdir, "query.csv")

    if config["satellite"] == "Sentinel-2":
        products = query_google_sentinel2(
            config["date_start"],
            config["date_end"],
            data_collection=config.get("s2_data_collection", "S2MSI1C"),
            shp=config["shapefile"],
            max_cc=config["max_cloudcover"],
            tile=config.get("s2_tile"),
        )
        products.to_csv(query_path, index=False)
        if not products.empty:
            download_s2dl(products, outdir)
        return products

    username = os.getenv("ERS_USERNAME")
    token = os.getenv("ERS_TOKEN")
    if not username or not token:
        raise ValueError(
            "ERS_USERNAME and ERS_TOKEN are required for Landsat M2M"
        )

    results = query_landsat(
        config["date_start"],
        config["date_end"],
        username,
        token,
        shp=config["shapefile"],
        max_cc=config["max_cloudcover"],
        sat=[LANDSAT_CODES[config["satellite"]]],
        tierList=config.get("landsat_tiers", ["T1"]),
    )
    products = results.rename(columns={"displayId": "Name"})
    products.to_csv(query_path, index=False)
    if not results.empty:
        download_landsat(
            results,
            outdir,
            username,
            token,
            pathrowList=config.get("landsat_tile_list"),
            tierList=config.get("landsat_tiers", ["T1"]),
        )
    return products


def load_local_date(products, config, date):
    """Calibrate and merge all downloaded tiles belonging to one date."""
    params = config["resampling_params"]
    common = {
        "products": products,
        "outdir": config["output_directory"],
        "date": date,
        "extent": params["extent_target"],
        "resolution": params["resolution"],
        "epsg": params["epsg_target"],
        "exclude_tiles": config.get("exclude_tiles"),
    }
    if config["satellite"] == "Sentinel-2":
        return load_sentinel2_tiles(**common)
    return load_landsat_tiles(**common)


def process_downloaded_tiles(products, config):
    """Merge local tiles by date and run the normal SnowFLAKES processing."""
    from utils import get_dates_to_process, remove_glaciers, save_false_color

    if products.empty:
        print("The query did not return any products")
        return

    outdir = config["output_directory"]
    if not config["simple_class"]:
        remove_glaciers(outdir)

    names = products["Name"].astype(str).str.replace(
        r"\.(SAFE|tar)$", "", regex=True
    )
    dates = get_dates_to_process(names.tolist(), config)
    failed_log = os.path.join(outdir, "failed_dates.txt")
    empty_log = os.path.join(outdir, "00_dates_no_items.log")

    for date in dates:
        print("\n" + "=" * 60)
        print(f"Processing locally downloaded tiles for {date}")
        print("=" * 60 + "\n")
        try:
            data, scene_id = load_local_date(products, config, date)
            if scene_id is None:
                with open(empty_log, "a") as file:
                    file.write(f"{date}\n")
                continue

            scene_dir = os.path.join(outdir, scene_id)
            os.makedirs(scene_dir, exist_ok=True)

            if config["satellite"] == "Sentinel-2":
                save_false_color(
                    scene_dir, ["B11", "B8A", "B03"], data, "fcc"
                )
            else:
                save_false_color(
                    scene_dir, ["swir16", "nir08", "green"], data, "fcc"
                )

            try:
                run_snowflakes(config, data, scene_id)
            finally:
                if config.get("remove_auxiliary", False):
                    auxiliary = os.path.join(scene_dir, "auxiliary")
                    if os.path.isdir(auxiliary):
                        shutil.rmtree(auxiliary)

        except Exception as error:
            print(f"Error processing date {date}: {error}")
            with open(failed_log, "a") as file:
                file.write(f"{date},{error}\n")


def run(config_path):
    """Run the configured AOI or complete-tile strategy."""
    load_dotenv()
    config = load_config(config_path)
    check_new_config(config)

    if not config.get("download_full_tiles", False):
        return run_workflow(
            config["date_start"], config["date_end"], config_path
        )

    products = query_and_download_tiles(config)
    process_downloaded_tiles(products, config)
    return products


def main():
    parser = argparse.ArgumentParser(
        description="Run SnowFLAKES using AOI streaming or full tile downloads"
    )
    parser.add_argument("config", help="Path to the JSON configuration")
    args = parser.parse_args()

    start = time.time()
    run(args.config)
    elapsed = time.time() - start
    print("\nThe workflow completed.")
    print(
        f"Execution time: {int(elapsed // 60)} minutes and "
        f"{int(elapsed % 60)} seconds"
    )


if __name__ == "__main__":
    main()
