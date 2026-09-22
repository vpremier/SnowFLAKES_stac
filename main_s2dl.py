#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query and download Sentinel-2 scenes with S2DL."""

import os
import sys
import time
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

from data_download.sentinel2_query_download import query_cdse
from data_download.utils import load_config
from loading.download_sentinel2_s2dl import download_s2dl


def run_s2dl(config_path):
    """Read the config and download its Sentinel-2 query with S2DL."""
    load_dotenv()
    config = load_config(config_path)

    if config['satellite'] != 'Sentinel-2':
        raise ValueError("main_s2dl.py requires satellite='Sentinel-2'")

    outdir = config['output_directory']
    os.makedirs(outdir, exist_ok=True)

    if config['query_sentinel2']:
        username = os.getenv('CDSE_USERNAME')
        password = os.getenv('CDSE_PASSWORD')
        if not username or not password:
            raise ValueError(
                'Set CDSE_USERNAME and CDSE_PASSWORD before running the query'
            )

        s2List = query_cdse(
            config['date_start'],
            config['date_end'],
            username,
            password,
            data_collection=config.get('s2_data_collection', 'S2MSI1C'),
            shp=config['shapefile'],
            max_cc=config['max_cloudcover'],
            tile=config.get('s2_tile'),
            filter_date=True,
            filter_baseline=True,
        )

        query_path = os.path.join(outdir, 'query.csv')
        s2List.to_csv(query_path, index=False)
        print(f'Query saved at {query_path}')
    else:
        query_path = config['s2List_path']

    if not os.path.isfile(query_path):
        raise FileNotFoundError(f'Query file does not exist: {query_path}')

    s2List = pd.read_csv(query_path)
    if s2List.empty:
        print('The query did not return Sentinel-2 scenes')
        return []

    return download_s2dl(s2List, outdir)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python main_s2dl.py path_to_config.json')
        raise SystemExit(1)

    start_time = time.time()
    scenes = run_s2dl(sys.argv[1])
    elapsed = time.time() - start_time

    print(f'\nDownloaded/read {len(scenes)} Sentinel-2 scenes')
    print(f'Execution time: {int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds')
