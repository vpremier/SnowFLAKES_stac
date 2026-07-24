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
import rasterio
from rasterio.warp import transform_bounds
import re
from datetime import datetime
import geopandas as gpd

from loading.load_stac_usgs import get_scene_center_time



def create_folder(working_folder, folder_name):
    """Create a folder if it does not already exist.

    Parameters
    ----------
    working_folder : str or os.PathLike
        Parent directory.
    folder_name : str or os.PathLike
        Name of the folder to create.

    Returns
    -------
    str
        Path to the created or existing folder.
    """
    folder_path = os.path.join(working_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    return folder_path



def create_log(working_folder, log_name):
    """Ensure that a log file exists and return its path.

    Parameters
    ----------
    working_folder : str or os.PathLike
        Directory containing the log file.
    log_name : str
        Log filename without the ``.log`` extension.

    Returns
    -------
    str
        Path to the log file.
    """
    os.makedirs(working_folder, exist_ok=True)
    path = os.path.join(working_folder, f'{log_name}.log')

    if not os.path.exists(path):
        with open(path, 'w') as f:
            pass
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")

    return path



def read_log(working_folder, log_name):
    """Return the non-empty entries from a SnowFLAKES log.

    Parameters
    ----------
    working_folder : str or os.PathLike
        Directory containing the log file.
    log_name : str
        Log filename without the ``.log`` extension.

    Returns
    -------
    list of str
        Stripped, non-empty log entries in file order.
    """
    path = create_log(working_folder, log_name)

    with open(path, 'r') as file:
        return [line.strip() for line in file if line.strip()]



def save_tif(array, reference_raster_path, output_path, nodata=255, dtype=rasterio.uint8):
    """Save a two-dimensional array as a single-band GeoTIFF.

    Parameters
    ----------
    array : numpy.ndarray
        Two-dimensional array to write.
    reference_raster_path : str or os.PathLike
        Raster whose spatial metadata and profile are copied.
    output_path : str or os.PathLike
        Destination GeoTIFF path.
    nodata : int or float, optional
        Output no-data value. The default is 255.
    dtype : str or numpy.dtype, optional
        Output data type. The default is ``rasterio.uint8``; if ``None``, the
        input array's data type is used.
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
    
    
    
def load_map(folder, pattern, return_path=False):
    """Load the first raster in a folder matching a glob pattern.

    Parameters
    ----------
    folder : str or os.PathLike
        Directory containing the raster.
    pattern : str
        Glob pattern used to select the raster.
    return_path : bool, optional
        If ``True``, also return the matching path. The default is ``False``.

    Returns
    -------
    numpy.ndarray or tuple
        Raster values, or ``(array, path)`` when ``return_path=True``.

    Raises
    ------
    FileNotFoundError
        If no raster matches the pattern.

    Notes
    -----
    If multiple rasters match, the function prints a warning and uses the first
    result.
    """
    matches = list(Path(folder).glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in '{folder}'"
        )

    if len(matches) > 1:
        print(f"Warning: multiple matches for {pattern}, using first")

    path = matches[0]
    image = open_image(path)[0]

    if return_path:
        return image, path

    return image



def valid_mask(data, no_data_value=np.nan):
    """
    Generate a valid-data mask from a multiband DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray with dimensions ('band', 'y', 'x').
    no_data_value : float or int, optional
        Value representing no-data. Default is NaN.

    Returns
    -------
    xarray.DataArray
        Boolean mask where True indicates that all bands are valid.
    """
    if no_data_value is None or np.isnan(no_data_value):
        return np.squeeze(data.notnull().all(dim="band").values)

    return np.squeeze((data != no_data_value).all(dim="band").values)



def build_valid_scene(no_data_mask, *invalid_masks, iterations=2):
    """Combine invalid-pixel masks into a scene-validity mask.

    Parameters
    ----------
    no_data_mask : numpy.ndarray
        Boolean mask where ``True`` identifies no-data pixels.
    *invalid_masks : numpy.ndarray
        Additional boolean masks where ``True`` identifies invalid pixels.
    iterations : int, optional
        Number of binary-dilation iterations applied to the combined invalid
        mask. Values below 1 disable dilation. The default is 2.

    Returns
    -------
    numpy.ndarray
        Boolean mask where ``True`` identifies valid pixels.
    """
    invalid = np.logical_or.reduce(invalid_masks + (no_data_mask,))
    
    if iterations>=1:
        invalid_dilated = binary_dilation(invalid, iterations=iterations)
        return ~invalid_dilated
    
    else:
        return ~invalid



def get_sensor(acquisition_name):
    """Identify the sensor family from a satellite product name.

    Parameters
    ----------
    acquisition_name : str or os.PathLike
        Product identifier or path to a product.

    Returns
    -------
    str
        SnowFLAKES sensor label: ``L4``, ``L5``, ``L7``, ``L8``, or ``S2``.
        Landsat 9 returns ``L8`` because both missions use the same band mapping.

    Raises
    ------
    ValueError
        If the product name does not have a supported satellite prefix.
    """
    product_name = os.path.basename(os.fspath(acquisition_name).rstrip(os.sep)).upper()

    sensor_prefixes = (
        (("LT04", "LT4"), "L4"),
        (("LT05", "LT5"), "L5"),
        (("LE07", "LE7"), "L7"),
        (("LC08", "LC8", "LC09", "LC9"), "L8"),
        (("S2A", "S2B", "S2C"), "S2"),
    )

    for prefixes, sensor in sensor_prefixes:
        if product_name.startswith(prefixes):
            return sensor

    raise ValueError(f"Invalid acquisition name: {acquisition_name}")



def define_bands(data, sensor):
    """Extract the spectral bands used by SnowFLAKES.

    Parameters
    ----------
    data : xarray.DataArray
        Scene data containing a ``band`` coordinate.
    sensor : str
        SnowFLAKES sensor label returned by :func:`get_sensor`.

    Returns
    -------
    dict
        Arrays keyed by ``GREEN``, ``SWIR``, ``NIR``, ``RED``, and ``BLUE``.

    Raises
    ------
    ValueError
        If the sensor is not supported.
    """
    band_mapping = {
        'L4': {'GREEN': 1, 'SWIR': 4, 'NIR': 3, 'RED': 2, 'BLUE': 0},
        'L5': {'GREEN': 'green', 'SWIR': 'swir16', 'NIR': 'nir08', 'RED': 'red', 'BLUE': 'blue'},
        'L7': {'GREEN': 'green', 'SWIR': 'swir16', 'NIR': 'nir08', 'RED': 'red', 'BLUE': 'blue'},
        'L8': {'GREEN': 'green', 'SWIR': 'swir16', 'NIR': 'nir08', 'RED': 'red', 'BLUE': 'blue'},
        'S2': {'GREEN': 'B03', 'SWIR': 'B11', 'NIR': 'B8A', 'RED': 'B04', 'BLUE': 'B02'}
    }

    if sensor not in band_mapping:
        raise ValueError(f"Sensor '{sensor}' is not supported.")

    return {
        name: np.squeeze(data.sel(band=band_name).values)
        for name, band_name in band_mapping[sensor].items()
    }



def select_band_names(sensor, suffix):
    """Return the band names required for a processing stage.

    Parameters
    ----------
    sensor : str
        SnowFLAKES sensor label returned by :func:`get_sensor`.
    suffix : {"scf", "cloud"}
        Processing stage: snow-cover fraction or cloud detection.

    Returns
    -------
    list of str
        Band names in the order expected by the processing stage.

    Raises
    ------
    ValueError
        If the sensor or processing stage is unsupported.
    """
    sensor_group = 'L457' if sensor in ('L4', 'L5', 'L7') else sensor
    band_mapping = {
        ('L457', 'scf'): ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22'],
        ('L8', 'scf'): ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22'],
        ('S2', 'scf'): ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        ('L457', 'cloud'): ['blue', 'green', 'red', 'nir08', 'swir16', 'lwir', 'swir22'],
        ('L8', 'cloud'): ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11'],
        ('S2', 'cloud'): ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
    }

    try:
        return band_mapping[(sensor_group, suffix)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported sensor/suffix combination: {sensor!r}, {suffix!r}"
        ) from exc



def define_datetime(scene_id, config):
    """Return the acquisition date and time encoded by a scene identifier.

    Sentinel-2 timestamps are read directly from their product names.
    A Landsat identifier contains only the acquisition date, so its scene-centre
    time is obtained from the USGS STAC item's MTL metadata.

    Parameters
    ----------
    scene_id : str or os.PathLike
        Product identifier or path. Supported names are compact Sentinel-2
        Level-1C products, legacy Sentinel-2 ``OPER`` products, and Landsat
        4--9 products.
    config : dict
        SnowFLAKES configuration, used to query Landsat metadata.

    Returns
    -------
    date_time : datetime.datetime
        Timezone-naive acquisition datetime.
    date : str
        Acquisition date in ``YYYYMMDD`` format.
    """

    scene_name = os.path.basename(os.fspath(scene_id).rstrip(os.sep))
    sensor = get_sensor(scene_name)
    parts = scene_name.split('_')

    try:
        if sensor == 'S2' and parts[1] == 'MSIL1C':
            date_time = datetime.strptime(parts[2], '%Y%m%dT%H%M%S')
            date = date_time.strftime('%Y%m%d')

        elif sensor == 'S2' and parts[1] == 'OPER':
            match = re.search(r'V?(\d{8}T\d{6})', parts[7])
            if match is None:
                raise ValueError("missing legacy Sentinel-2 sensing time")
            date_time = datetime.strptime(match.group(1), '%Y%m%dT%H%M%S')
            date = date_time.strftime('%Y%m%d')

        elif sensor in ("L4", "L5", "L7", "L8"):
            date = parts[3]
            query_date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
            date_time = get_scene_center_time(
                query_date,
                extent_target=config["resampling_params"]['extent_target'],
                resolution=config["resampling_params"]['resolution'],
                epsg_target=config["resampling_params"]['epsg_target'],
                max_cc=config['max_cloudcover'],
                filter_by_geometry=True,
                shp=config['shapefile'],
                platform=config["satellite"].upper().replace("-", "_"),
                idList=[]
            )
        else:
            raise ValueError(f"Unsupported scene identifier: {scene_id!r}")
    except IndexError as exc:
        raise ValueError(f"Invalid scene identifier: {scene_id!r}") from exc

    return date_time, date



def get_hemisphere(raster_path):
    """Determine the hemisphere covered by a raster.

    Parameters
    ----------
    raster_path : str or os.PathLike
        Path to a georeferenced raster.

    Returns
    -------
    {"N", "S", "E"}
        ``N`` for the Northern Hemisphere, ``S`` for the Southern Hemisphere,
        or ``E`` if the raster crosses the equator.

    Raises
    ------
    ValueError
        If the raster has no coordinate reference system.
    """
    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        _, bottom, _, top = transform_bounds(
            raster.crs,
            "EPSG:4326",
            raster.bounds.left,
            raster.bounds.bottom,
            raster.bounds.right,
            raster.bounds.top
        )

    if bottom >= 0 and top > 0:
        return "N"
    if top <= 0 and bottom < 0:
        return "S"
    return "E"






# To be updated


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









def snow_around_glacier(wd, scene_id):

    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")
    
    
    scf_map = load_map(scene_folder, '*SnowFLAKES.tif')
    glacier_mask = load_map(auxiliary_folder, '*glacier*.tif')
    
    
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



def remove_low_scf(scene_id, data, FSC_SVM_map_path, curr_aux_folder):
    
    # load information for current scene
    sensor = get_sensor(scene_id)
    bands = define_bands(data, sensor)
    
    # Load the SCF map 
    with rasterio.open(FSC_SVM_map_path) as scf_src:
        scf_data = scf_src.read(1)  # Reading first band
     
    shadow_mask, shadow_path = load_map(curr_aux_folder, '*shadow_mask.tif', return_path=True)
    # distance_idx = load_map(curr_aux_folder, '*distance.tif')
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')

    
    swir = bands["SWIR"]
    green = bands["GREEN"]
    
    # scf correction based on diff B NIR and shadow mask
    pixels_to_correct = np.logical_and.reduce((diff_B_NIR > 0, 
                                               diff_B_NIR < 0.06, 
                                               shadow_mask == 1, 
                                               scf_data > 0, 
                                               scf_data < 50))

    # SCF_map[np.logical_and.reduce((SCF_map > 0, SCF_map <= 100, distance_idx == 255))] = 0

    
    # remove SCF lower than 10%
    scf_data[scf_data < 10] = 0

    condition1 = np.logical_and.reduce((swir > 0.2,
                                scf_data < 50,
                                shadow_mask == 0))
    
    condition2 = np.logical_and.reduce((green < 0.15,
                                scf_data < 50,
                                shadow_mask == 0))
    
    scf_data[pixels_to_correct] = 0
    scf_data[condition1 | condition2] = 0
    
    save_tif(scf_data, shadow_path, FSC_SVM_map_path, dtype=rasterio.uint8)

    
