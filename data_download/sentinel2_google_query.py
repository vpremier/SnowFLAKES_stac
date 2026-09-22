#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query Sentinel-2 products available in the Google public data bucket."""

import argparse
import os
from datetime import datetime
from xml.etree import ElementTree

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box
from tqdm import tqdm

from data_download.sentinel_filters import get_filtered_baseline, filter_RON


GOOGLE_BUCKET_API = (
    'https://storage.googleapis.com/storage/v1/'
    'b/gcp-public-data-sentinel-2/o'
)
GOOGLE_BUCKET_URL = (
    'https://storage.googleapis.com/gcp-public-data-sentinel-2'
)
CDSE_PRODUCTS_URL = (
    'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
)


def normalize_tile(tile):
    """Return an MGRS tile in the form ``T19HDE``."""
    tile = str(tile).strip().upper()
    if tile.startswith('MGRS-'):
        tile = tile[5:]
    if not tile.startswith('T'):
        tile = 'T' + tile
    if len(tile) != 6:
        raise ValueError(f'Invalid Sentinel-2 tile: {tile}')
    return tile


def get_bounds(shp):
    """Return the WGS84 bounding-box WKT used to discover MGRS tiles."""
    if isinstance(shp, (str, os.PathLike)):
        if not os.path.exists(shp):
            raise FileNotFoundError(f'Shapefile does not exist: {shp}')
        gdf = gpd.read_file(shp)
    elif isinstance(shp, gpd.GeoDataFrame):
        gdf = shp.copy()
    else:
        raise ValueError(f"Unsupported input type for 'shp': {type(shp)}")

    if gdf.empty:
        raise ValueError('The study-area file is empty')
    if gdf.crs is None:
        raise ValueError('The study-area CRS is missing')
    if gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')

    return box(*gdf.total_bounds).wkt


def get_intersecting_tiles(date_start, date_end, shp, data_collection):
    """Use the public CDSE catalogue to discover intersecting MGRS tiles."""
    bounds = get_bounds(shp)
    query_filter = (
        "Collection/Name eq 'SENTINEL-2'"
        " and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
        f"'productType' and att/OData.CSC.StringAttribute/Value eq '{data_collection}')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{bounds}')"
        f" and ContentDate/Start ge {date_start}T00:00:00.000Z"
        f" and ContentDate/Start lt {date_end}T00:00:00.000Z"
    )
    params = {
        '$filter': query_filter,
        '$select': 'Name',
        '$top': 1000,
    }

    names = []
    url = CDSE_PRODUCTS_URL
    while url:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        page = response.json()
        names.extend(product['Name'] for product in page.get('value', []))
        url = page.get('@odata.nextLink')
        params = None

    tiles = {
        name.split('_')[5]
        for name in names
        if len(name.split('_')) > 5
    }
    return sorted(normalize_tile(tile) for tile in tiles)


def list_tile_products(tile, data_collection, session):
    """List product IDs stored by Google for one MGRS tile."""
    tile = normalize_tile(tile)
    level = 'L1C' if data_collection == 'S2MSI1C' else 'L2A'
    level_prefix = '' if level == 'L1C' else 'L2/'
    prefix = (
        f'{level_prefix}tiles/{tile[1:3]}/{tile[3]}/{tile[4:6]}/'
    )
    params = {
        'prefix': prefix,
        'delimiter': '/',
        'maxResults': 1000,
    }

    product_ids = []
    while True:
        response = session.get(GOOGLE_BUCKET_API, params=params, timeout=60)
        response.raise_for_status()
        page = response.json()

        for product_prefix in page.get('prefixes', []):
            product_name = product_prefix.rstrip('/').split('/')[-1]
            if product_name.endswith('.SAFE'):
                product_ids.append(product_name[:-5])

        page_token = page.get('nextPageToken')
        if not page_token:
            break
        params['pageToken'] = page_token

    return product_ids


def get_google_product_url(product_id):
    """Return the public Google URL and metadata filename for a product."""
    parts = product_id.split('_')
    level = parts[1][-3:]
    tile = normalize_tile(parts[5])
    level_prefix = '' if level == 'L1C' else 'L2/'
    base_url = (
        f'{GOOGLE_BUCKET_URL}/{level_prefix}tiles/'
        f'{tile[1:3]}/{tile[3]}/{tile[4:6]}/{product_id}.SAFE'
    )
    metadata_name = f'MTD_MSI{level}.xml'
    return base_url, metadata_name


def get_cloud_cover(product_id, session):
    """Read cloud cover from the product metadata stored by Google."""
    base_url, metadata_name = get_google_product_url(product_id)
    response = session.get(
        f'{base_url}/{metadata_name}', timeout=60
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)

    cloud_tags = {
        'Cloud_Coverage_Assessment',
        'CLOUDY_PIXEL_PERCENTAGE',
    }
    for element in root.iter():
        tag = element.tag.rsplit('}', 1)[-1]
        if tag in cloud_tags and element.text is not None:
            return float(element.text)

    raise ValueError(f'Cloud cover is missing for {product_id}')


def query_google_sentinel2(
    date_start,
    date_end,
    data_collection='S2MSI1C',
    shp=None,
    max_cc=90,
    tile=None,
    filter_baseline=True,
    RON_list=None,
):
    """Return Sentinel-2 products that can be downloaded with S2DL.

    Provide either ``tile`` (a string or list of MGRS tiles) or ``shp``. When
    only an AOI is supplied, CDSE is queried without authentication to discover
    its intersecting MGRS tiles. Product IDs and cloud cover are then read from
    Google's public bucket.
    """
    if data_collection not in ('S2MSI1C', 'S2MSI2A'):
        raise ValueError("data_collection must be 'S2MSI1C' or 'S2MSI2A'")

    start = datetime.strptime(date_start, '%Y-%m-%d')
    end = datetime.strptime(date_end, '%Y-%m-%d')
    if start >= end:
        raise ValueError('date_start must be before date_end')

    if tile is None:
        if shp is None:
            raise ValueError("Provide either 'tile' or 'shp'")
        tiles = get_intersecting_tiles(
            date_start, date_end, shp, data_collection
        )
    elif isinstance(tile, (str, int)):
        tiles = [normalize_tile(tile)]
    else:
        tiles = [normalize_tile(item) for item in tile]

    print(f"Querying Google tiles: {', '.join(tiles)}")
    session = requests.Session()
    rows = []

    for current_tile in tiles:
        product_ids = list_tile_products(
            current_tile, data_collection, session
        )
        for product_id in product_ids:
            sensing_time = datetime.strptime(
                product_id.split('_')[2], '%Y%m%dT%H%M%S'
            )
            if start <= sensing_time < end:
                rows.append({
                    'Name': product_id + '.SAFE',
                    'tile': current_tile,
                    'SensingTime': sensing_time,
                })

    products = pd.DataFrame(rows)
    if products.empty:
        print('No Google Sentinel-2 products found')
        return products

    if filter_baseline:
        products = get_filtered_baseline(products)

    if RON_list:
        products = filter_RON(products, RON_list)

    cloud_cover = []
    for name in tqdm(products['Name'], desc='Reading cloud cover'):
        cloud_cover.append(get_cloud_cover(name[:-5], session))
    products['cloudCover'] = cloud_cover
    products = products[products['cloudCover'] < max_cc].copy()

    products['BaseURL'] = [
        get_google_product_url(name[:-5])[0]
        for name in products['Name']
    ]
    products = products.sort_values(['SensingTime', 'tile']).reset_index(drop=True)

    print('\n' + '=' * 60)
    print('Google Sentinel-2 Query Summary')
    print('=' * 60)
    print(
        f'Found {len(products)} scenes from {date_start} to {date_end} '
        f'with cloud cover below {max_cc}%'
    )
    print(f"Tiles: {', '.join(tiles)}")
    print('=' * 60 + '\n')

    return products


def main():
    parser = argparse.ArgumentParser(
        description='Query S2DL-compatible Sentinel-2 products from Google'
    )
    parser.add_argument('date_start', help='Start date: YYYY-MM-DD')
    parser.add_argument('date_end', help='End date: YYYY-MM-DD')
    parser.add_argument('output_csv', help='Path for the query CSV')
    parser.add_argument('--shp', help='Study-area vector file')
    parser.add_argument('--tile', nargs='+', help='One or more MGRS tiles')
    parser.add_argument('--max-cc', type=float, default=90)
    parser.add_argument(
        '--collection', choices=['S2MSI1C', 'S2MSI2A'], default='S2MSI1C'
    )
    args = parser.parse_args()

    products = query_google_sentinel2(
        args.date_start,
        args.date_end,
        data_collection=args.collection,
        shp=args.shp,
        max_cc=args.max_cc,
        tile=args.tile,
    )
    output_directory = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(output_directory, exist_ok=True)
    products.to_csv(args.output_csv, index=False)
    print(f'Query saved at {args.output_csv}')


if __name__ == '__main__':
    main()
