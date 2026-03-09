#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:51:20 2025

@author: vpremier
"""
import requests
import os
import geopandas as gpd
import pandas as pd
import cgi
import json
import sys
import time

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def sendRequest(url, data, apiKey = None, exitIfNoResponse = True):
    """
    Send a request to an M2M endpoint and returns the parsed JSON response.
    This function is taken from 
    https://m2m.cr.usgs.gov/api/docs/example/download_data-py
    https://code.usgs.gov/eros-user-services/machine_to_machine/m2m_querying_metadatafilters
    
    
    Parameters:
    endpoint_url (str): The URL of the M2M endpoint
    payload (dict): The payload to be sent with the request

    Returns:
    dict: Parsed JSON response
    """  
    
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers = headers)  
    
    try:
      httpStatusCode = response.status_code 
      if response == None:
          print("No output from service")
          if exitIfNoResponse: sys.exit()
          else: return False
      output = json.loads(response.text)
      if output['errorCode'] != None:
          print(output['errorCode'], "- ", output['errorMessage'])
          if exitIfNoResponse: sys.exit()
          else: return False
      if  httpStatusCode == 404:
          print("404 Not Found")
          if exitIfNoResponse: sys.exit()
          else: return False
      elif httpStatusCode == 401: 
          print("401 Unauthorized")
          if exitIfNoResponse: sys.exit()
          else: return False
      elif httpStatusCode == 400:
          print("Error Code", httpStatusCode)
          if exitIfNoResponse: sys.exit()
          else: return False
    except Exception as e: 
          response.close()
          print(e)
          if exitIfNoResponse: sys.exit()
          else: return False
    response.close()
    
    return output['data']



def prompt_ERS_login(serviceURL, username, token):
    """
    Log in to the EROS Registration Service (ERS)
    https://ers.cr.usgs.gov/
    
    This function is adapted from 
    https://code.usgs.gov/eros-user-services/machine_to_machine/m2m_landsat_9_search_download
    
    
    Parameters:
    serviceURL (str): The URL of the M2M endpoint
    username (str): Your username
    token (str): Your token. For generating it see: https://www.usgs.gov/media/files/m2m-application-token-documentation
    
    """  
    
    print("Logging in...\n")

    # Use requests.post() to make the login request
    response = requests.post(f"{serviceURL}login-token", json={'username': username, 'token': token})

    # Check for successful response
    if response.status_code == 200:  
        apiKey = response.json()['data']
        print('\nLogin Successful, API Key Received!')
        headers = {'X-Auth-Token': apiKey}
        return apiKey
    else:
        print("\nLogin was unsuccessful, please try again or create an account at: https://ers.cr.usgs.gov/register.")
       


def downloadfiles(download):
    # Set up retry logic
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    downloadId = download['downloadId']
    print("    DOWNLOADING: " + download['url'])

    try:
        # Request the file
        with session.get(download['url'], stream=True, timeout=60) as response:
            response.raise_for_status()  # Raise HTTPError for bad responses

            # Parse filename
            content_disposition = cgi.parse_header(response.headers.get('Content-Disposition', ''))[1]
            filename = os.path.basename(content_disposition.get('filename', f'download_{downloadId}'))
            filepath = os.path.join('data_dir', filename)  # Change 'data_dir' as needed

            # Write to file in chunks
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

        print(f"    SAVED: {filepath}")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"    FAILED TO DOWNLOAD: {download['url']}")
        print(f"    ERROR: {e}")
        return None





def query_landsat(date_start, date_end, username, token, shp = None,
                         max_cc = 90, sat = ['LT05','LE07','LC08','LC09']):
    
    """Returns list of matching Landsat scenes for a selected period and
        for a specific area (defined from a shapefile). The username and
        password of your USGS EROS account are required. Please see
        
        https://pypi.org/project/usgsm2m/
        https://m2m.cr.usgs.gov/
        
        It is necessary to request the use of the Machine-to-Machine API.
        The request is done from the USGS account.
    
    Parameters
    ----------
    date_start : str
        starting date
    date_end : str
        ending date
    shp : str 
        path to a shapefile with your area of interest. Any crs is accepted
    username : str
        username of your USCGS account
    token : str
        token from your USCGS account (see: https://www.usgs.gov/media/files/m2m-application-token-documentation)
    max_cc : int, optional
        maximum cloud coverage. Default is 90%
    sat : list, optional
        list with desired missions. If not specified, all matching missions are selected. 
        Possible options are LT05, LE07, LC08 and LC09
    
    Returns
    -------
    results : pd.DataFrame that contains:
            - displayId
            - entityId
    """
    
    
    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    
    # log in
    apiKey = prompt_ERS_login(serviceUrl, username, token)    
    

    # Collection 2 Level 1 (Landsat): Dataset Alias for each satellite
    satellite = {'LT05':'landsat_tm_c2_l1',
                 'LE07':'landsat_etm_c2_l1',
                 'LC08':'landsat_ot_c2_l1',
                 'LC09':'landsat_ot_c2_l1'}
     
    if shp is None:          
        # Bounding Box Coordinates
        lat_min, lon_min, lat_max, lon_max = -90, -180, 90, 180
        
    elif isinstance(shp, (str, os.PathLike)) and os.path.exists(shp):
        #read the shapefile
        gdf = gpd.read_file(shp)
        
        # convert crs (otherwise may result in an error)
        if not gdf.crs == 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')  
        
        # Bounding Box Coordinates
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = bounds
        
    # shp is already a geometry (shapely, GeoSeries, or GeoDataFrame)
    elif isinstance(shp, gpd.GeoDataFrame):
        gdf = shp
  
        # ensure WGS84 CRS
        if gdf.crs is not None and gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        bounds = gdf.total_bounds
        lon_min, lat_min, lon_max, lat_max = bounds
    else:
        raise ValueError(f"Unsupported input type for 'shp': {type(shp)}")
    
    if sat == []:
        sat = ['LT05','LE07','LC08','LC09']
        
    # Request
    results = []
    for key in satellite:
        
        if key not in sat:
            continue
        
        # filter by date
        acquisitionFilter = {'start' : date_start, 'end' : date_end}
        
        # filter spatially
        spatialFilter =  {'filterType' : 'mbr',
                           'lowerLeft' : {'latitude' : lat_min,\
                                          'longitude' : lon_min},
                          'upperRight' : { 'latitude' : lat_max,\
                                          'longitude' : lon_max}}
            
            
        scene_search  = {'datasetName': satellite[key],
                         'maxResults':10000, # default is 100
                            'sceneFilter' : {
                                'spatialFilter': spatialFilter,
                                'cloudCoverFilter' : {'min' : 0, 'max' : max_cc},
                                'acquisitionFilter' : acquisitionFilter,}
                            }

        # send request
        scenes  = sendRequest(serviceUrl + "scene-search", scene_search, apiKey)
        

        for result in scenes['results']:
            # Add this scene to the list I would like to download
            results.append({
                'displayId': result['displayId'],
                'entityId': result['entityId']
            })
        
    # After the loop:
    unique_results = {}
    for r in results:
        unique_results[r['displayId']] = r  # Keep only the last occurrence
    
    results = list(unique_results.values())
            
        
    landsat_sensor = [r['displayId'].split('_')[0] for r in results]
    landsat_sensor = list(dict.fromkeys(landsat_sensor))  
    
    tileList = [r['displayId'].split('_')[2] for r in results]
    tileList = list(dict.fromkeys(tileList))  
    
    print('\n' + '='*60)
    print('Landsat Query Summary')
    print('='*60)
    
    print('Found %i Landsat scenes from %s to %s with maximum cloud coverage %i%%\n'
          % (len(results), date_start, date_end, max_cc))
    
    print('Found %i sensors:\n  %s\n'
          % (len(landsat_sensor), ', '.join(map(str, landsat_sensor))))
    
    print('The shapefile intersects %i tiles (path/row):\n  %s\n'
          % (len(tileList), ', '.join(map(str, tileList))))
    
    print('='*60 + '\n')


    

    return pd.DataFrame(results)



def download_landsat(results, outdir, username, token,
                     pathrowList=None, tierList=None):
    
    """Downloads a list of Landsat given as input. Possibility to filter 
        by tile and tier. The same credentials as function 
        query_landsat() are required.
        
        Downloads Landsat scenes and saves them in a structured folder:
        outdir/Landsat/SENSOR/TILE/SCENE.tar
        
    
    Parameters
    ----------
    results : pd.DataFrame
        contains the list of landsat scenes to be downloaded. It is returned as
        output by the function query_landsat
    outdir : str
        path where you want to save the archives
    username : str
        username of your USCGS account
    token : str
        token from your USCGS account (see: https://www.usgs.gov/media/files/m2m-application-token-documentation)
    pathrow : list, optional
        path/row (str) to download    
    tierList : list, optional
        tiers (str) to download 
    
    """


    serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    apiKey = prompt_ERS_login(serviceUrl, username, token)

    # Map satellite ID to dataset name
    satellite = {
        'LT05': 'landsat_tm_c2_l1',
        'LE07': 'landsat_etm_c2_l1',
        'LC08': 'landsat_ot_c2_l1',
        'LC09': 'landsat_ot_c2_l1'
    }

    # Extract info
    results['satellite'] = results['displayId'].str.split('_').str[0]
    results['pathrow'] = results['displayId'].str.split('_').str[2]
    results['tier'] = results['displayId'].str.split('_').str[-1]

    # Mark already downloaded
    def is_downloaded(row):
        sensor = row['satellite']
        tile = row['pathrow']
        filename = row['displayId'] + '.tar'
        filepath = os.path.join(outdir, 'Landsat', sensor, tile, filename)
        return os.path.exists(filepath)

    results['already_downloaded'] = results.apply(is_downloaded, axis=1)
    filtered = results[~results['already_downloaded']].copy()

    # Apply optional filters
    if pathrowList:
        filtered = filtered[filtered['pathrow'].isin(pathrowList)]
    if tierList:
        filtered = filtered[filtered['tier'].isin(tierList)]

    print(f"Already downloaded: {results['already_downloaded'].sum()} scenes")
    print(f"To download: {len(filtered)} scenes")



    # Loop over satellite groups
    for sat_id, group_df in filtered.groupby('satellite'):
        dataset_name = satellite.get(sat_id)
        if not dataset_name:
            print(f"Unknown satellite ID: {sat_id}, skipping.")
            continue

        download_payload = {
            'datasetName': dataset_name,
            'entityIds': group_df['entityId'].tolist()
        }

        downloadOptions = sendRequest(serviceUrl + "download-options",
                                      download_payload, apiKey)

        availableproducts = []
        for product in downloadOptions:
            if product['available'] and product['downloadSystem'] == 'ls_zip':
                availableproducts.append({
                    'entityId': product['entityId'],
                    'productId': product['id']
                })

        if not availableproducts:
            print(f"No available products for satellite {sat_id}.")
            continue

        requestedDownloadsCount = len(availableproducts)
        label = f"download-{sat_id}"

        download_req_payload = {'downloads': availableproducts, 'label': label}
        requestResults = sendRequest(serviceUrl + "download-request",
                                     download_req_payload, apiKey)

        if requestResults['preparingDownloads']:
            downloadIds = []
            download_retrieve_payload = {'label': label}

            print("\nRequesting additional download URLs...")
            while len(downloadIds) < (requestedDownloadsCount - len(requestResults['failed'])):
                moreDownloadUrls = sendRequest(serviceUrl + "download-retrieve",
                                               download_retrieve_payload, apiKey)

                for download in moreDownloadUrls['available']:
                    if (str(download['downloadId']) in requestResults['newRecords'] or
                            str(download['downloadId']) in requestResults['duplicateProducts']):
                        download_scene(download, group_df, outdir)
                        downloadIds.append(download['downloadId'])

                remaining = requestedDownloadsCount - len(downloadIds) - len(requestResults['failed'])
                if remaining > 0:
                    print(f"  {remaining} downloads still preparing. Waiting 30s...")
                    time.sleep(30)

        else:
            print("\nAll downloads available immediately:\n")
            for download in requestResults['availableDownloads']:
                print(download)
                download_scene(download, group_df, outdir)
                  
         
                
def download_scene(download, group_df, outdir):
    """
    Download a single scene and save to Landsat/SENSOR/TILE/SCENE.tar
    """
    url = download['url']
    entityId = download['entityId']

    # Find displayId to get SENSOR and TILE
    row = group_df[group_df['entityId'] == entityId].iloc[0]
    sensor = row['satellite']
    tile = row['pathrow']
    scene = row['displayId'] + '.tar'

    # Build target folder and ensure it exists
    dest_dir = os.path.join(outdir, 'Landsat', sensor, tile)
    os.makedirs(dest_dir, exist_ok=True)

    filepath = os.path.join(dest_dir, scene)

    print(f"DOWNLOADING: {scene} -> {dest_dir}")

    downloadResponse = requests.get(url, stream=True)
    if 'Content-Disposition' in downloadResponse.headers:
        content_disposition = cgi.parse_header(downloadResponse.headers['Content-Disposition'])[1]
        filename = os.path.basename(content_disposition['filename'])
    else:
        filename = scene

    with open(filepath, 'wb') as f:
        for chunk in downloadResponse.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Saved: {filepath}\n")



if __name__ == "__main__":   
    
    """
    Landsat download
    """
    from dotenv import load_dotenv
    load_dotenv()

    # dates for the query/download
    date_start = '2015-06-01'
    date_end = '2015-06-30'
    
        
    # shapefile wth the AOI
    shp = r'/mnt/CEPH_PROJECTS/FRAM3S/AOI/UTM_32N/province_boundary_25832.gpkg'
    
    # directory where you want to download your data
    outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/' 
    
    results = query_landsat(date_start, 
                            date_end, 
                            os.getenv("ERS_USERNAME"), 
                            os.getenv("ERS_TOKEN"), 
                            shp = shp, 
                            max_cc=50)
    
    start = time.time()
    
    download_landsat(results, outdir, os.getenv("ERS_USERNAME"), 
                        os.getenv("ERS_TOKEN"), pathrowList = ['192027'], tierList = ['T1'])
    
    end = time.time()
    print(f"Download time: {end - start:.2f} seconds")
    
    
    
    
    
