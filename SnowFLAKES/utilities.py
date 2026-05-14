#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:03:27 2024

@author: rbarella
"""
import os
from osgeo import gdal, ogr, osr
import netCDF4
import numpy as np
import glob
import rasterio
from rasterio.warp import transform_bounds
import time
import re
from datetime import datetime
import geopandas as gpd

from stac.load_stac_usgs import get_scene_center_time


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


def data_filter(start_date, end_date, working_folder, sensor, scenes_to_skip):
    """
    Filters the scenes in the working folder by date, sensor, and skips specified scenes.

    Parameters
    ----------
    start_date : str
        Start date in the format "yyyymmdd".
    end_date : str
        End date in the format "yyyymmdd".
    working_folder : str
        The folder containing the acquisitions.
    sensor : str
        The sensor identifier (e.g., "S2", "L8").
    scenes_to_skip : list
        A list of dates (in "yyyymmdd" format) to exclude from the results.

    Returns
    -------
    list
        A list of filtered scene directories between the provided dates, excluding scenes in `scenes_to_skip`.
    """
    skipped_dates = []

    # Filter for Sentinel-2
    if sensor == 'S2':
        acquisitions = sorted(glob.glob(os.path.join(working_folder, 'S2*')))
        acquisitions_filtered = []
        for f in acquisitions:
            # Extract date for filtering
            
            if os.path.basename(f).split('_')[1] == 'MSIL1C':
                date = os.path.basename(f).split('_')[2].split('T')[0]
            elif os.path.basename(f).split('_')[1] == 'OPER':
                date = os.path.basename(f).split('_')[7][1:].split('T')[0]
            else:
                continue

            # Check date conditions
            if start_date <= date <= end_date:
                if date not in scenes_to_skip:
                    acquisitions_filtered.append(f)
                else:
                    skipped_dates.append(date)

    elif sensor == 'PRISMA':
        acquisitions = sorted(glob.glob(os.path.join(working_folder, 'PRS*')))
        acquisitions_filtered = []
        for f in acquisitions:
            date = os.path.basename(f).split('_')[4][:-6]
            if start_date <= date <= end_date:
                if date not in scenes_to_skip:
                    acquisitions_filtered.append(f)
                else:
                    skipped_dates.append(date)

    # Filter for Landsat (L8, L7, etc.)
    else:
        acquisitions = sorted(glob.glob(os.path.join(working_folder, 'L*T[12]')))
        acquisitions_filtered = []
        for f in acquisitions:
            date = os.path.basename(f).split('_')[3]
            if start_date <= date <= end_date:
               
                if date not in scenes_to_skip:
                    acquisitions_filtered.append(f)
                else:
                    skipped_dates.append(date)

    # Print skipped dates
    if skipped_dates:
        print(f"The following dates are being skipped: {', '.join(sorted(set(skipped_dates)))}")
    else:
        print("No dates were skipped.")

    print(f"Filtered acquisitions...")
    return acquisitions_filtered


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


def get_input_param(input_data, name):
    """Retrieves the value associated with a given name from a DataFrame."""
    filtered_data = input_data[input_data['Name'] == name]
    value = input_data[input_data['Name'] == name]['Value'].values[0]

    if len(filtered_data) > 1:
        raise ValueError(f"Multiple matching rows found for name: {name}")

    if isinstance(value, str):

        print(f'the parameter {name} is set to {value}')

    elif np.isnan(value):
        return None

    return filtered_data['Value'].iloc[0]


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
        'L5': {'GREEN': 1, 'SWIR': 4, 'NIR': 3, 'RED': 2, 'BLUE': 0},
        'L7': {'GREEN': 1, 'SWIR': 4, 'NIR': 3, 'RED': 2, 'BLUE': 0},
        'L8': {'GREEN': 2, 'SWIR': 5, 'NIR': 4, 'RED': 3, 'BLUE': 1},
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


def create_vrt_files(L_acquisitions_filt, sensor, resolution):
    """
    Creates VRT files for each acquisition based on the sensor type.
    """
    for elem in L_acquisitions_filt:
        if sensor == 'S2':
            create_vrt(elem, elem, 'cloud', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'scf', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'scfT', resolution=resolution, overwrite=False)

        elif sensor == 'L8' or sensor == 'L9':

            create_vrt(elem, elem, 'scf', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'scfT', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'cloud', resolution=resolution, overwrite=False)

        elif sensor == 'L7':

            create_vrt(elem, elem, 'scf', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'scfT', resolution=resolution, overwrite=False)
            create_vrt(elem, elem, 'cloud', resolution=resolution, overwrite=False)


        else:
            create_vrt(elem, elem, 'scf', resolution=resolution, overwrite=False)


def create_vrt(folder, outfolder, suffix="scf", resolution=30, overwrite=False):
    """
    Creates a VRT using selected bands for the specified sensor.
    """
    vrtname = os.path.join(outfolder, os.path.basename(folder) + f'_{suffix}.vrt')

    if os.path.exists(vrtname) and not overwrite:
        print(f"{vrtname} already exists.")
        return

    # Determine the sensor type and get the bands
    sensor = get_sensor(folder)
    band_name_list = select_band_names(sensor, suffix)

    # Find the band files in the folder
    file_list = find_band_files(folder, band_name_list, sensor)
    if not file_list:
        print(f"No band files found for sensor: {sensor}, suffix: {suffix}.")
        return

    # Create the VRT using GDAL
    create_vrt_with_gdal(file_list, vrtname, resolution, band_name_list)


def select_band_names(sensor, suffix):
    """
    Returns the list of band names based on the sensor and suffix.
    """
    if sensor in ['L4', 'L5', 'L7'] and suffix == 'scf':
        return ['B1', 'B2', 'B3', 'B4', 'B5']
    elif sensor == 'L7' and suffix == 'scfT':
        # return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B7']
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    elif sensor == 'L7' and suffix == 'cloud':
        # return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B7']
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    elif sensor == 'L8' and suffix == 'scf':
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    elif sensor == 'L8' and suffix == 'scfT':
        #return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B10', 'B11']
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    elif sensor == 'L8' and suffix == 'cloud':
        return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']
    elif sensor == 'S2' and suffix == 'cloud':
        return ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    elif sensor == 'S2' and suffix == 'scf':
        return ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    elif sensor == 'S2' and suffix == 'scfT':
        return ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    else:
        # Graceful fallback with logging instead of exception
        print(f"Warning: Unsupported sensor or suffix combination: sensor={sensor}, suffix={suffix}")
        return []


def find_band_files(folder, band_name_list, sensor):
    """
    Finds the file paths for the selected bands in the folder.
    """
    file_list = []
    for band in band_name_list:
        pattern = os.path.join(folder, f"*{band}_*.tif")
        file_list.extend(glob.glob(pattern))

    # Handle Landsat (non-S2) cases with uppercase .TIF extension
    if not file_list and sensor != 'S2':
        for band in band_name_list:
            pattern = os.path.join(folder, f"*{band}.TIF")
            file_list.extend(glob.glob(pattern))

    # Handle Sentinel-2 .tif extension cases
    if sensor == 'S2' and not file_list:
        for band in band_name_list:
            pattern = os.path.join(folder, f"*{band}.tif")
            file_list.extend(glob.glob(pattern))

    return file_list


def create_vrt_with_gdal(file_list, vrtname, resolution, band_name_list):
    """no idea yet
    Creates a VRT file using GDAL.
    """
    file_string = " ".join(file_list)
    cmd = f"gdalbuildvrt -separate -r bilinear -tr {resolution} {resolution} {vrtname} {file_string}"
    print(f"Running command: {cmd}")
    os.system(cmd)

    # Set the band descriptions in the VRT
    VRT_dataset = gdal.Open(vrtname, gdal.GA_Update)
    for idx, band_name in enumerate(band_name_list, 1):
        VRT_dataset.GetRasterBand(idx).SetDescription(band_name)
    VRT_dataset = None


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


def save_image(image_to_save, path_to_save, driver_name, datatype, geotransform, proj, NoDataValue=None):
    '''
    adfGeoTransform[0] / * top left x * /
    adfGeoTransform[1] / * w - e pixel resolution * /
    adfGeoTransform[2] / * rotation, 0 if image is "north up" * /
    adfGeoTransform[3] / * top left y * /
    adfGeoTransform[4] / * rotation, 0 if image is "north up" * /
    adfGeoTransform[5] / * n - s pixel resolution * /


    enum  	GDALDataType {
    GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3,
    GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7,
    GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11,
    GDT_TypeCount = 12}
    '''

    driver = gdal.GetDriverByName(driver_name)

    if len(np.shape(image_to_save)) == 2:
        bands = 1
        cols = np.shape(image_to_save)[1]
        rows = np.shape(image_to_save)[0]

    if len(np.shape(image_to_save)) > 2:
        bands = np.shape(image_to_save)[0]
        cols = np.shape(image_to_save)[2]
        rows = np.shape(image_to_save)[1]

    outDataset = driver.Create(path_to_save, cols, rows, bands, datatype)

    outDataset.SetGeoTransform(geotransform)

    if proj != None:
        outDataset.SetProjection(proj)

    if bands > 1:

        for i in range(1, bands + 1):
            outDataset.GetRasterBand(i).WriteArray(image_to_save[(i - 1), :, :], 0, 0)
            if NoDataValue != None:
                outDataset.GetRasterBand(i).SetNoDataValue(NoDataValue)

    else:
        outDataset.GetRasterBand(1).WriteArray(image_to_save, 0, 0)
        if NoDataValue != None:
            outDataset.GetRasterBand(1).SetNoDataValue(NoDataValue)

    outDataset = None

    print('Image Saved')

    return;


def reproj_point(x, y, srIn, srOut):
    '''
    trasform a point from one crt to another

    Parameters
    ----------
    x : TYPE
        x coord.
    y : TYPE
        y coord.
    srIn : str
        old crs.
    srOut : str
        target crs.

    Returns
    -------
    x : int
        output x coord.
    y : int
        output y coord.

    '''
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)

    coordTransform = osr.CoordinateTransformation(srIn, srOut)

    point.Transform(coordTransform)

    (x, y) = point.GetX(), point.GetY()

    return (x, y)


def subimages_definer(dim_img, max_dim=9000):
    """
    Determines the optimal number of subdivisions (nrow, ncol) required to ensure
    that the image can be processed without exceeding memory limitations.

    Parameters
    ----------
    dim_img : list of int
        Dimensions of the image as [x_pixels, y_pixels].
    max_dim : int, optional
        Maximum dimension that can be processed at once (default is 9000).

    Returns
    -------
    x : int
        Adjusted x dimension after subdivision.
    y : int
        Adjusted y dimension after subdivision.
    nrow : int
        Number of subdivisions along the x-axis.
    ncol : int
        Number of subdivisions along the y-axis.
    del_row : int
        Number of extra pixels to handle for odd-sized rows.
    del_col : int
        Number of extra pixels to handle for odd-sized columns.
    """
    x, y = dim_img  # Extract image dimensions (x and y)

    # Initialize subdivision counts and deltas for odd dimensions
    nrow, ncol = 1, 1
    del_row, del_col = 0, 0

    # Continue splitting the image until both dimensions fit within max_dim
    while x > max_dim or y > max_dim:
        if x > y:
            # Split along the x-axis
            if x % 2 != 0:  # Handle odd number of pixels
                x = np.floor(x / 2)
                del_row += 1
            else:
                x = np.floor(x / 2)
            nrow *= 2  # Double the number of rows
        else:
            # Split along the y-axis
            if y % 2 != 0:  # Handle odd number of pixels
                y = np.floor(y / 2)
                del_col += 1
            else:
                y = np.floor(y / 2)
            ncol *= 2  # Double the number of columns

    return int(x), int(y), nrow, ncol, del_row, del_col


def extent_cutter(img_info, nrow, ncol, resolution, x, y):
    """
    Calculates the extents of the image subdivisions based on the number of rows
    and columns, as well as the image resolution.

    Parameters
    ----------
    img_info : dict
        Dictionary containing image metadata, including its extent.
    nrow : int
        Number of subdivisions along the x-axis.
    ncol : int
        Number of subdivisions along the y-axis.
    resolution : float
        Image resolution (pixel size).
    x : int
        Number of pixels in each subdivision along the x-axis.
    y : int
        Number of pixels in each subdivision along the y-axis.

    Returns
    -------
    extent_outputs : list of tuples
        A list of tuples where each tuple contains the extent (min_x, min_y, max_x, max_y)
        for a subdivision.
    """
    extent_input = img_info['extent']
    minx, miny = extent_input[0], extent_input[1]

    extent_outputs = []

    # Iterate through each row and column to define extents for each subimage
    for i in range(nrow):
        x_min_i = minx + i * resolution * x
        x_max_i = minx + (i + 1) * resolution * x

        for j in range(ncol):
            y_min_i = miny + j * resolution * y
            y_max_i = miny + (j + 1) * resolution * y
            extent_outputs.append((x_min_i, y_min_i, x_max_i, y_max_i))

    return extent_outputs


def process_image(img_info, max_dim=9000):
    """
    Integrates the subimage definition and extent calculation for processing large images.

    Parameters
    ----------
    img_info : dict
        Dictionary containing image metadata, including dimensions and extent.
    max_dim : int, optional
        Maximum dimension that can be processed at once (default is 9000).

    Returns
    -------
    subimage_extents : list of tuples
        List of extents for each subimage defined for processing.
    """
    # Get the image dimensions from img_info
    dim_img = [img_info['X_Y_raster_size'][0], img_info['X_Y_raster_size'][1]]

    # Step 1: Determine the subdivisions (nrow, ncol) and adjusted dimensions
    x, y, nrow, ncol, del_row, del_col = subimages_definer(dim_img, max_dim)

    print(f"Subdivided image into {nrow} rows and {ncol} columns")
    print(f"Adjusted dimensions: {x} x {y} (extra rows: {del_row}, extra cols: {del_col})")

    # Step 2: Calculate the extents for each subimage
    resolution = img_info['geotransform'][1]
    subimage_extents = extent_cutter(img_info, nrow, ncol, resolution, x, y)

    print("Subimage extents calculated:")
    for extent in subimage_extents:
        print(extent)

    return subimage_extents


def define_datetime(sensor, acquisition_name, config):
    from datetime import datetime
    '''
    Parameters
    ----------
    sensor : str
        "S2","L8"....
    acquisition_name : str
        working folder name .

    Returns
    -------
    date_time : datetime
        example datetime.datetime(2022, 8, 2, 10, 26, 11).
    date : str
        yyyymmdd.

    '''
    if sensor == 'S2' and os.path.basename(acquisition_name).split('_')[1] == 'MSIL1C':

        date = os.path.basename(acquisition_name).split('_')[2].split('T')[0]
        date_time_str = os.path.basename(acquisition_name).split('_')[2].split('T')[0] + \
                        os.path.basename(acquisition_name).split('_')[2].split('T')[1]
        date_time = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')

    elif sensor == 'S2' and os.path.basename(acquisition_name).split('_')[1] == 'OPER':

        date = os.path.basename(acquisition_name).split('_')[7][1:].split('T')[0]

    elif sensor == 'PRISMA':

        date = os.path.basename(acquisition_name).split('_')[4][:-6]
        date_time_str = os.path.basename(acquisition_name).split('_')[4]
        date_time = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')


    else:
        try:
            date = os.path.basename(acquisition_name).split('_')[3]
        except:
            date = os.path.basename(glob.glob(acquisition_name + os.sep + "*B1_toa.tif")[0]).split('_')[3]

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


def perform_pca_on_geotiff(input_tiff, valid_mask, variance_threshold=0.95, no_data_value=-999):
    """
    Reads a hyperspectral image, performs PCA to reduce dimensionality to the optimal number of components,
    and saves the resulting PCA bands in a new GeoTIFF. PCA is applied only to valid areas defined by `valid_mask`.

    Parameters:
    - input_tiff: str, path to the input GeoTIFF file.
    - valid_mask: numpy array, boolean mask where True indicates valid data.
    - variance_threshold: float, the cumulative variance ratio to retain (default is 0.95).
    - no_data_value: float, the value to assign for invalid pixels in the output image (default is -9999).
    """
    import os
    import numpy as np
    import rasterio
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    output_tiff = input_tiff[:-4] + '_PCA.tif'

    if not os.path.exists(output_tiff):
        # Load hyperspectral image
        with rasterio.open(input_tiff) as src:
            img_data = src.read().astype(np.float32)
            n_bands, height, width = img_data.shape

            # Check valid_mask dimensions
            if valid_mask.shape != (height, width):
                raise ValueError("valid_mask dimensions do not match the image dimensions.")

            # Flatten the spatial dimensions
            img_data_reshaped = img_data.reshape(n_bands, -1).T
            valid_mask_flat = valid_mask.flatten()

            # Filter data based on valid_mask
            valid_data = img_data_reshaped[valid_mask_flat]

            # Standardize valid data
            scaler = StandardScaler()
            valid_data_scaled = scaler.fit_transform(valid_data)

            # Determine optimal number of components
            pca = PCA()
            pca.fit(valid_data_scaled)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1

            print(f"Optimal number of components: {optimal_components}")

            # Perform PCA with the optimal number of components
            pca = PCA(n_components=optimal_components)
            pca_data = pca.fit_transform(valid_data_scaled)

            # Prepare the output array with no_data values
            pca_data_full = np.full((optimal_components, height * width), no_data_value, dtype=np.float32)
            pca_data_full[:, valid_mask_flat] = pca_data.T

            # Reshape to (optimal_components, height, width)
            pca_data_reshaped = pca_data_full.reshape(optimal_components, height, width)

            # Write PCA bands to a new GeoTIFF file
            profile = src.profile
            profile.update(count=optimal_components, dtype=rasterio.float32, nodata=no_data_value)

            with rasterio.open(output_tiff, 'w', **profile) as dst:
                dst.write(pca_data_reshaped)

        print("PCA transformation complete. Output saved as:", output_tiff)


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


def give_time(start):
    """
    Prints the elapsed time since the given start time in hours, minutes, and seconds.

    Parameters:
        start (float): The start time (e.g., from time.time()).
    """
    elapsed = time.time() - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Elapsed time: {int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s")


















