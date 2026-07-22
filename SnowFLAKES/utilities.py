#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:03:27 2024

@author: rbarella
"""
import os
from osgeo import gdal, osr
from pathlib import Path
from scipy.ndimage import binary_dilation
import netCDF4
import numpy as np
import glob
import rasterio
from rasterio.warp import transform_bounds
import re
from datetime import datetime
import geopandas as gpd

from loading.load_stac_usgs import get_scene_center_time



def save_tif(array, reference_raster_path, output_path, nodata=255, dtype=rasterio.uint8):
    """
    Save an array as a GeoTIFF using metadata from a reference raster.
    """

    if dtype is None:
        dtype = array.dtype

    with rasterio.open(reference_raster_path) as src:
        meta = src.meta.copy()

    meta.update(
        dtype=dtype,
        nodata=nodata,
        count=1
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array.astype(dtype), 1)

    print(f"Saved to: {output_path}")
    
    
    
def find_path(folder, pattern):
    # look for file containing a specific pattern in a given folder
    matches = list(Path(folder).glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in '{folder}'"
        )

    if len(matches) > 1:
        print(f"Warning: multiple matches for {pattern}, using first")

    return matches[0]



def load_map(folder, pattern):
    # load the map
    path = find_path(folder, pattern)
    image = open_image(path)[0]
    return image



def build_valid_scene(no_data_mask, *invalid_masks, iterations=2):
    """
    Combine arbitrary boolean masks into a validity mask.
    """
    invalid = np.logical_or.reduce(invalid_masks + (no_data_mask,))
    
    if iterations>=1:
        invalid_dilated = binary_dilation(invalid, iterations=iterations)
        return ~invalid_dilated
    
    else:
        return ~invalid



def find_closest_valid_scf(working_folder, date):
    """
    base_path: root Sentinel2 directory
    target_date: datetime object
    check_func: function that returns True if SCF is valid
    """
    
    folders = os.listdir(working_folder)
    
    target_date = datetime.strptime(date, "%Y%m%d")
    
    date_pattern = re.compile(r'_(\d{8}T\d{6})_')
    
    candidates = []
    
    for folder in folders:
        match = date_pattern.search(folder)
        if match:
            date_str = match.group(1)
            folder_date = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
            
            diff = abs((folder_date - target_date).total_seconds())
            
            candidates.append((diff, folder_date, folder))
    
    # Sort by closest date
    candidates.sort(key=lambda x: x[0])
    

    for _, folder_date, folder in candidates:
        scf_path = os.path.join(working_folder, folder, "SCF")
        
        if not os.path.exists(scf_path):
            continue
        
        # -------------------------
        # 3. find shapefile
        # -------------------------
        shp_files = [f for f in os.listdir(scf_path) if f.endswith(".shp")]
        if not shp_files:
            continue
        
        # check SnowFLAKES.tif exists
        tif_files = [f for f in os.listdir(scf_path) if f.endswith("SnowFLAKES.tif")]
        if not tif_files:
            continue
        
        shp_path = os.path.join(scf_path, shp_files[0])

        try:
            gdf = gpd.read_file(shp_path)

            unique_values = set(gdf["value"].unique())
            print(f"{folder_date} → values: {unique_values}")

            # -------------------------
            # 4. validation condition
            # -------------------------
            if unique_values == {1, 2}:

                print(f"✅ Valid SCF found at {folder_date}")

                return os.path.join(scf_path, tif_files[0])

        except Exception as e:
            print(f"⚠️ Error reading {shp_path}: {e}")

    return None
            
            
            
def is_month_in_range(month, start_month, end_month):
    """
    Check if a given month (1-12) is in the range [start_month, end_month],
    supporting ranges that wrap around the year (e.g. October to April).
    """
    if start_month <= end_month:
        return start_month <= month <= end_month
    else:
        return month >= start_month or month <= end_month



def create_empty_files(working_folder):
    """
    Creates two empty text files in the specified folder if they don't already exist:
    '00_scenes_to_skip.txt' and '00_skip_cloud_masks.txt'.

    Parameters
    ----------
    working_folder : str
        The folder where the files will be created.
    """
    # Define file paths
    scenes_to_skip_path = os.path.join(working_folder, '00_scenes_to_skip.log')
    skip_cloud_masks_path = os.path.join(working_folder, '00_skip_cloud_masks.log')
    skip_empty_items_path = os.path.join(working_folder, '00_dates_no_items.log')


    # Create the empty text files only if they don't already exist
    if not os.path.exists(scenes_to_skip_path):
        with open(scenes_to_skip_path, 'w') as f:
            pass  # Just create an empty file
        print(f"Created file: {scenes_to_skip_path}")
    else:
        print(f"File already exists: {scenes_to_skip_path}")

    if not os.path.exists(skip_cloud_masks_path):
        with open(skip_cloud_masks_path, 'w') as f:
            pass  # Just create an empty file
        print(f"Created file: {skip_cloud_masks_path}")
    else:
        print(f"File already exists: {skip_cloud_masks_path}")
        
    if not os.path.exists(skip_empty_items_path):
        with open(skip_empty_items_path, 'w') as f:
            pass  # Just create an empty file
        print(f"Created file: {skip_empty_items_path}")
    else:
        print(f"File already exists: {skip_empty_items_path}")
        
    return scenes_to_skip_path, skip_cloud_masks_path, skip_empty_items_path



def scenes_skip(working_folder):
    txt_scenes_to_skip_path = glob.glob(os.path.join(working_folder, '00_scenes_to_skip.log'))[0]
    with open(txt_scenes_to_skip_path, "r") as file:
        content = file.read().strip()
        if content:
            date_list = content.split('\n')
        else:
            date_list = []  # Empty file case

    return date_list



def cloud_mask_to_skip(working_folder):
    txt_scenes_to_skip_path = glob.glob(os.path.join(working_folder, '00_skip_cloud_masks.log'))[0]
    with open(txt_scenes_to_skip_path, "r") as file:
        content = file.read().strip()
        if content:
            date_list = content.split('\n')
        else:
            date_list = []  # Empty file case

    return date_list



def empty_items_to_skip(working_folder):
    txt_scenes_to_skip_path = glob.glob(os.path.join(working_folder, '00_dates_no_items.log'))[0]
    with open(txt_scenes_to_skip_path, "r") as file:
        content = file.read().strip()
        if content:
            date_list = content.split('\n')
        else:
            date_list = []  # Empty file case

    return date_list



def get_sensor(acquisition_name):
    """Determines the satellite mission based on the acquisition name."""
    acquisition_name = os.path.basename(acquisition_name)

    if 'LT04' in acquisition_name:
        return 'L4'
    elif 'LT05' in acquisition_name or acquisition_name[:3] == 'LT5':
        return 'L5'
    elif 'LE07' in acquisition_name or acquisition_name[:3] == 'LE7':
        return 'L7'
    elif 'LC08' in acquisition_name or acquisition_name[:3] == 'LC8':
        return 'L8'
    elif 'LC09' in acquisition_name:
        return 'L8'
    elif 'S2' in acquisition_name:
        return 'S2'
    elif 'PRS' in acquisition_name:
        return 'PRISMA'
    else:
        raise ValueError(f"Invalid acquisition name: {acquisition_name}")



def define_bands(data, valid_mask, sensor):
    """
    Extracts significant bands and generates stretched versions for a given sensor.

    Parameters
    ----------
    L_image : numpy.ndarray
        3D matrix of shape (bands, height, width) representing the spectral image.

    valid_mask : numpy.ndarray
        2D boolean matrix indicating valid pixels.

    sensor : str
        Sensor type ("S2", "L8", "L5", "L7", "L4").

    Returns
    -------
    dict
        Dictionary containing significant bands and stretched bands for GREEN and SWIR.
    """
    # Define band indices for different sensors
    band_mapping = {
        'L4': {'GREEN': 1, 'SWIR': 4, 'NIR': 3, 'RED': 2, 'BLUE': 0},
        'L5': {'GREEN': 'green', 'SWIR': 'swir16', 'NIR': 'nir08', 'RED': 'red', 'BLUE': 'blue'},
        'L7': {'GREEN': 'green', 'SWIR': 'swir16', 'NIR': 'nir08', 'RED': 'red', 'BLUE': 'blue'},
        'L8': {'GREEN': 'green', 'SWIR': 'swir16', 'NIR': 'nir08', 'RED': 'red', 'BLUE': 'blue'},
        'S2': {'GREEN': 'B03', 'SWIR': 'B11', 'NIR': 'B8A', 'RED': 'B04', 'BLUE': 'B02'},
        'PRISMA': {'GREEN': 19, 'SWIR': 122, 'NIR': 46, 'RED': 36, 'BLUE': 9}
    }

    # Check if the sensor is supported
    if sensor not in band_mapping:
        raise ValueError(f"Sensor '{sensor}' is not supported.")

    # Get band indices for the current sensor
    indices = band_mapping[sensor]

    # Extract bands using the indices
    bands = {name: np.squeeze(data.sel(band=idx).values) for name, idx in indices.items()}

    return bands



def select_band_names(sensor, suffix):
    """
    Returns the list of band names based on the sensor and suffix.
    """
    
    # SCF
    if sensor in ['L4', 'L5', 'L7'] and suffix == 'scf':
        return ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
    
    elif sensor == 'L8' and suffix == 'scf':
        return ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
    
    elif sensor == 'S2' and suffix == 'scf':
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    
    # CLOUDS
    if sensor in ['L4', 'L5', 'L7'] and suffix == 'cloud':
        return ['blue', 'green', 'red', 'nir08', 'swir16', 'lwir', 'swir22']
    
    elif sensor == 'L8' and suffix == 'cloud':
        return ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11']
    
    elif sensor == 'S2' and suffix == 'cloud':
        return ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    # all bands: not used anymore
    elif sensor == 'L7' and suffix == 'scfT':
        # return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B7']
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    elif sensor == 'L8' and suffix == 'scfT':
        #return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B10', 'B11']
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    elif sensor == 'S2' and suffix == 'scfT':
        return ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    else:
        # Graceful fallback with logging instead of exception
        print(f"Warning: Unsupported sensor or suffix combination: sensor={sensor}, suffix={suffix}")
        return []



def open_image(image_path, ncdf_layer='fsc'):
    """Opens an image and reads its metadata.

    Parameters
    ----------
    image_path : str
        path to an image
    ncdf_layer: optional , string of the name of wich layer of ncdf to open
    Returns
    -------
    image : osgeo.gdal.Dataset
        the opened image
    information : dict
        dictionary containing image metadata
    """

    ext = os.path.basename(image_path).split('.')[-1]

    if ext == 'nc':
        nc_data = netCDF4.Dataset(image_path, 'r')
        vars_nc = list(nc_data.variables)
        # ncdf_layer="fsc_unc"
        scf_name = list(filter(lambda x: x.startswith(ncdf_layer), vars_nc))[0]
        dataset = gdal.Open("NETCDF:{0}:{1}".format(image_path, scf_name))
        proj = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + geotransform[1] * cols
        miny = maxy + geotransform[5] * rows
        extent = [minx, miny, maxx, maxy]
        X_Y_raster_size = [cols, rows]
        information = {}
        information['geotransform'] = geotransform
        information['extent'] = extent
        information['geotransform'] = tuple(map(lambda x: round(x, 4) or x, information['geotransform']))
        information['extent'] = tuple(map(lambda x: round(x, 4) or x, information['extent']))
        information['X_Y_raster_size'] = X_Y_raster_size
        information['projection'] = proj

        image_output = np.array(dataset.ReadAsArray(0, 0, cols, rows))

    else:
        image = gdal.Open(image_path)
        cols = image.RasterXSize
        rows = image.RasterYSize
        geotransform = image.GetGeoTransform()
        proj = image.GetProjection()
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + geotransform[1] * cols
        miny = maxy + geotransform[5] * rows
        X_Y_raster_size = [cols, rows]
        extent = [minx, miny, maxx, maxy]
        information = {}
        information['geotransform'] = geotransform
        information['extent'] = extent
        information['X_Y_raster_size'] = X_Y_raster_size
        information['projection'] = proj
        projection = osr.SpatialReference(wkt=image.GetProjection())
        with rasterio.open(image_path, 'r+') as rds:
            epsg_code = str(rds.crs).split(':')[1]
        information['EPSG'] = epsg_code
        # print(cols,rows )
        image_output = np.array(image.ReadAsArray(0, 0, cols, rows))

    if image is None:
        print('could not open ' + image_path)
        return

    return image_output, information



def define_datetime(scene_id, config):

    # get sensor 
    sensor = get_sensor(scene_id)

    # Sentinel-2
    if sensor == 'S2' and scene_id.split('_')[1] == 'MSIL1C':
        
        date = scene_id.split('_')[2].split('T')[0]
        date_time_str = os.path.basename(scene_id).split('_')[2].split('T')[0] + \
                        os.path.basename(scene_id).split('_')[2].split('T')[1]
        date_time = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')

    elif sensor == 'S2' and scene_id.split('_')[1] == 'OPER':

        date = scene_id.split('_')[7][1:].split('T')[0]
        
    elif sensor.sta


    else:
        try:
            date = os.path.basename(scene_id).split('_')[3]
        except:
            date = os.path.basename(glob.glob(scene_id + os.sep + "*B1_toa.tif")[0]).split('_')[3]

        # retrieve the MTL file from the STAC catalogue
        date_time = get_scene_center_time(datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'), 
                                          extent_target=config["resampling_params"]['extent_target'], 
                                          resolution=config["resampling_params"]['resolution'], 
                                          epsg_target=config["resampling_params"]['epsg_target'], 
                                          max_cc = config['max_cloudcover'], 
                                          filter_by_geometry = True, 
                                          shp=config['shapefile'], 
                                          platform = config["satellite"].upper().replace("-", "_"), 
                                          idList = [])

    return date_time, date



def get_hemisphere(raster_path):
    """
    Determines whether a raster is in the Northern or Southern Hemisphere
    for any reference coordinate system (CRS).

    Parameters:
        raster_path (str): Path to the raster file.

    Returns:
        str: 'Northern Hemisphere', 'Southern Hemisphere', or 'Equator' 
             if the raster spans the equator.
    """
    try:
        with rasterio.open(raster_path) as raster:
            # Transform the bounds to WGS84 (EPSG:4326)
            bounds_wgs84 = transform_bounds(
                raster.crs,  # Source CRS
                "EPSG:4326",  # Target CRS
                raster.bounds.left,
                raster.bounds.bottom,
                raster.bounds.right,
                raster.bounds.top
            )

            # Extract the geographic bounds in WGS84
            _, bottom, _, top = bounds_wgs84

            # Determine the hemisphere
            if top > 0 and bottom > 0:
                return "N"
            elif top < 0 and bottom < 0:
                return "S"
            else:
                return "E"
    except Exception as e:
        return f"Error processing the raster: {e}"



def snow_around_glacier(FSC_SVM_map_path, curr_aux_folder, auxiliary_folder_path):

    # subdirectory SCF
    scf_folder = curr_aux_folder.replace('auxiliary', 'SCF')
    
    scf_map = load_map(scf_folder, '*SnowFLAKES.tif')
    glacier_mask = load_map(auxiliary_folder_path, '*glacier*.tif')
    
    
    # check snow presence around the glacier (buffer of 100 pixels)
    
    # Convert glacier mask to boolean
    glacier = glacier_mask > 0
    
    # Create 100-pixel buffer around glacier
    glacier_buffer = binary_dilation(glacier, iterations=100)
    
    # Keep only the ring outside the glacier
    buffer_ring = glacier_buffer & (~glacier)
    
    n_buffer_pixels = np.count_nonzero(buffer_ring)
    
    if n_buffer_pixels == 0:
        return True
    
    snow_pixels = np.count_nonzero(
        (scf_map > 0) & buffer_ring
    )
    
    snow_fraction = snow_pixels / n_buffer_pixels
    
    # Return True if less than 5% snow covered
    return snow_fraction < 0.5



def remove_low_scf(FSC_SVM_map_path, bands, curr_aux_folder, dem_path):
    
    # Load the SCF map 
    with rasterio.open(FSC_SVM_map_path) as scf_src:
        scf_data = scf_src.read(1)  # Reading first band
     
    shadow_mask = load_map(curr_aux_folder, '*shadow_mask.tif')
    distance_idx = load_map(curr_aux_folder, '*distance.tif')

    
    swir = bands["SWIR"]
    green = bands["GREEN"]
    
    
    # remove SCF lower than 10%
    scf_data[scf_data < 10] = 0

    condition1 = np.logical_and.reduce((swir > 0.4,
                                scf_data < 50,
                                shadow_mask == 0))
    
    condition2 = np.logical_and.reduce((green < 0.15,
                                scf_data < 50,
                                shadow_mask == 0))
    
    scf_data[condition1 | condition2] = 0
    
    save_tif(scf_data, dem_path, FSC_SVM_map_path, dtype=rasterio.uint8)

    











