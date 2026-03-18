#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:07:58 2025

@author: vpremier
"""
import subprocess
import requests
import os
import geopandas as gpd
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from shapely.geometry import box, shape


import matplotlib.pyplot as plt

from sentinel_filters import *


def query_cdse(date_start, date_end, username, psw, 
                         data_collection = "S2MSI1C", shp = None,
                         max_cc = 90, tile = None, filter_date = True,
                         filter_baseline = True, RON_list = None):
    
    """Returns list of matching Sentinel-2 scenes for a selected period and
        for a specific area (defined from a shapefile). The username and
        password of your Copernicus Data Space Ecosystem account are required. 
        Please see
        
        https://dataspace.copernicus.eu/
        https://documentation.dataspace.copernicus.eu/APIs/On-Demand%20Production%20API.html
            
        Parameters
        ----------
        date_start : str
            starting date
        date_end : str
            ending date
        shp : str 
            path to a shapefile with your area of interest. Any crs is accepted
        username : str
            username of your CDSE account
        psw : str
            password of your CDSE account
        data_collection : str
            default is "S2MSI1C" that refers to the Sentinel-2 L1C data
        max_cc : int, optional
            maximum cloud coverage. Default is 90%
        tile : str, optional
            specific tile to be downloaded
        filter_date : bool, optional
            whether to filter double dates, if their footprints overlap. 
            Keep the biggest footprint. Only for Sentinel-2
        filter_baseline : bool, optional
            whether to filter double baseline for the same date. 
            Only for Sentinel-2
        RON_list : list, optional
            whether to filter on a list of relative orbit numbers (RON).
            Only for Sentinel-2
        
        Returns
        -------
        products : list
            list of the matching scenes
    """   
    
    # Define supported data collections
    allowed_collections = ["S2MSI1C","S2MSI2A", "SY_2_SYN___", "LANDSAT-5", 
                           "LANDSAT-7", "LANDSAT-8-ESA"]
    
    if data_collection not in allowed_collections:
        print(f"Allowed data collections: {allowed_collections}")
        print(f"You provided: '{data_collection}'")
        raise ValueError(f"Invalid data_collection: '{data_collection}'. Please use one of the allowed options.")

    

    # access to the Copernicus Dataspce ecosystem
    data = {
           "client_id": "cdse-public",
           "username": username,
           "password": psw,
           "grant_type": "password",
       }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
        )
        
        
    if shp is None:  
        boundsdata=box(*[-180,-90,180, 90]).wkt
        
    elif isinstance(shp, (str, os.PathLike)) and os.path.exists(shp):
        # search by polygon, time, and CDSE query keywords
        gdf = gpd.read_file(shp)
        
        # convert crs (otherwise may result in an error)
        if not gdf.crs == 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326') 
            
        boundsdata=box(*gdf.total_bounds).wkt
        
    elif isinstance(shp, gpd.GeoDataFrame):
        gdf = shp
  
        # ensure WGS84 CRS
        if gdf.crs is not None and gdf.crs.to_string() != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
         
        boundsdata=box(*gdf.total_bounds).wkt

    else:
         raise ValueError(f"Unsupported input type for 'shp': {type(shp)}")
    



    # query for SENTINEL-2 or SENTINEL-3 data
    if data_collection in ['S2MSI1C',"S2MSI2A",'SY_2_SYN___']:
        query = ('').join([f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=",
                           "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'",
                           " and att/OData.CSC.StringAttribute/Value eq '",
                           data_collection,
                           "')",
                           " and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'",
                           " and att/OData.CSC.DoubleAttribute/Value lt ",
                            str(max_cc),
                            ") and OData.CSC.Intersects(area=geography'SRID=4326;",
                            boundsdata,
                            "') and ContentDate/Start gt ",
                            date_start,
                            "T00:00:00.000Z and ContentDate/Start lt ",
                            date_end,
                            "T00:00:00.000Z&$top=1000"])
        if tile:   
            query = ('').join([f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=",
                               "contains(Name,'",
                               tile,
                               "') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'",
                               " and att/OData.CSC.StringAttribute/Value eq '",
                               data_collection,
                               "')",
                               " and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'",
                               " and att/OData.CSC.DoubleAttribute/Value lt ",
                                str(max_cc),
                                ") and OData.CSC.Intersects(area=geography'SRID=4326;",
                                boundsdata,
                                "') and ContentDate/Start gt ",
                                date_start,
                                "T00:00:00.000Z and ContentDate/Start lt ",
                                date_end,
                                "T00:00:00.000Z&$top=1000"])
            
            
    elif data_collection in ["LANDSAT-5","LANDSAT-7","LANDSAT-8-ESA"]:
        query = ('').join([
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=",
            "Collection/Name eq '", data_collection, "'",
            " and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'",
            " and att/OData.CSC.DoubleAttribute/Value lt ", str(max_cc), ")",
            " and OData.CSC.Intersects(area=geography'SRID=4326;", boundsdata, "')",
            " and ContentDate/Start gt ", date_start, "T00:00:00.000Z",
            " and ContentDate/Start lt ", date_end, "T00:00:00.000Z",
            "&$top=1000"
        ])

        

    json = requests.get(query).json()
    
    products = pd.DataFrame.from_dict(json['value'])
    
    if data_collection in ["S2MSI1C", "S2MSI2A"]:
        # ---- Filter by processing baseline (keep newest) ----
        if filter_baseline and not products.empty:
            products = get_filtered_baseline(products)

        # ---- Filter by geometry overlap (same date, same scene, keep the biggest) ----
        if filter_date and not products.empty:
            products = get_filtered_date(products)

        # ---- Filter by RON ----
        if RON_list:
            products = filter_RON(products, RON_list)

            

    print('\n' + '='*60)
    print(f'{data_collection} Query Summary')
    print('='*60)
    
    print('Found %i %s scenes from %s to %s with maximum cloud coverage %i%%\n'
          % (len(products), data_collection, date_start, date_end, max_cc))
    
    if not products.empty:
        products['tile'] = products['Name'].str.split('_').str[5]
        tiles = products['tile'].unique().tolist()

        print('The shapefile intersects %i tiles:\n  %s\n'
          % (len(tiles), ', '.join(tiles)))
    
    print('='*60 + '\n')

    
    return products



def download_cdse(s2List, outdir, username, psw):
    """Downloads a list of Sentinel-2 given as input. Credentials 
        from CDSE: please check
        
        https://dataspace.copernicus.eu/
        https://documentation.dataspace.copernicus.eu/APIs/OData.html
        
    Parameters
    ----------
    s2List : pd.DataFrame
        dataframe with the Sentinel-2 scenes (organised as in the output of function
                                   get_matching_s2_cop() )
    outdir : str
        path where you want to save the archives
    username : str
        username of your CDSE account
    psw : str
        password of your CDSE account
    
    """
    
    
    def get_access_token(username, psw):
        # access token is valid for 10 minutes
        # refresh token is valid for 60 minutes and can be used for generating
        # a new access token without using your credentials
        
        # refresh token (expires in 600s)
        cmd_str = ('').join(['''
                curl --location \
                --location --request POST 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' \
                --data-urlencode 'grant_type=password' \
                --data-urlencode 'username=%s' ''' %username,
                '''--data-urlencode 'password=%s' ''' %psw,
                '''--data-urlencode 'client_id=cdse-public' '''])
        
        result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE)
        token = result.stdout.decode('utf-8')[:-1]
        access_token = token.split('"access_token":"')[1].split('","expires_in"')[0]
        time_token = datetime.now()
        
        return access_token, time_token
    


    def download_file(s2_id, access_token, outname):
        url = ('').join([f"https://zipper.dataspace.copernicus.eu/odata/v1/Products(",
                        s2_id,
                        ")/$value"])
  
        headers = {"Authorization": f"Bearer {access_token}"}
        
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, headers=headers, stream=True)
        
        with open(outname, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    
    # fake time token
    time_token = datetime.strptime("1991-06-29", "%Y-%m-%d")
    for i in tqdm(range(0, len(s2List))):
        time_refresh = (datetime.now() - time_token).total_seconds()
        
        if time_refresh >= 600:
            print("Refreshing token")
            access_token, time_token = get_access_token(username, psw)

        fileName = s2List.loc[i]['Name']
        s2_id = s2List.loc[i]['Id']

        # Extract tile: safe filename format: ..._TxxXYZ_...
        try:
            tile = fileName.split('_')[5]
        except IndexError:
            print(f"Error parsing tile for {fileName}")
            continue
        
        # Build new folder path: outdir/Sentinel2/TxxXYZ/
        scene_dir = os.path.join(outdir, 'Sentinel2', tile)
        os.makedirs(scene_dir, exist_ok=True)
        
        outname = os.path.join(scene_dir, fileName.replace('.SAFE', '.zip'))
                
        if os.path.exists(outname) and os.stat(outname).st_size>0:
            print('%s already downloaded' %fileName.replace('.SAFE','.zip'))
        
        else:
            print("Downloading %s" %fileName)
            try:
                download_file(s2_id, access_token, outname)
            except:
                print('Error')

     




 



if __name__ == "__main__":   
        
    """
    Sentinel-2 download
    """
    
    import time

    from dotenv import load_dotenv
    load_dotenv()
    
    
    # dates for the query/download
    date_start = '2024-03-28'
    date_end = '2024-03-29'
    
    
    tile = 'T19JCH'
    
    # shapefile wth the AOI
    shp = r'/mnt/CEPH_PROJECTS/SNOWCOP/AOI/DOMAIN/AOI-basins.shp'
    
    # directory where you want to download your data
    outdir = r'/mnt/CEPH_PROJECTS/SNOWCOP/Vale/test/' + tile
  
    
    # it is possible to query also other collection (default is S2MSI1C)
    s2List = query_cdse(date_start, 
                        date_end, 
                        os.getenv("CDSE_USERNAME"), 
                        os.getenv("CDSE_PASSWORD"), 
                        data_collection = "S2MSI2A",
                        shp=shp,
                        max_cc = 90, 
                        tile=tile, 
                        filter_baseline = True,
                        filter_date = True,
                        RON_list = []) 

    
    start = time.time()
    
    download_cdse(s2List, outdir, os.getenv("CDSE_USERNAME"), os.getenv("CDSE_PASSWORD"))  
    
    end = time.time()
    print(f"Download time: {end - start:.2f} seconds")

