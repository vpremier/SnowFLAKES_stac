#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:39:26 2024

@author: rbarella
"""

import os
import logging
from osgeo import gdal, osr
import geopandas as gpd
import numpy as np
import cv2
import rasterio
from shapely.geometry import Polygon
import glob
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer
from datetime import timezone
from pathlib import Path
import rioxarray
from rasterio.crs import CRS
from skimage.filters import threshold_otsu


from rasterio.transform import from_origin
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge

from SnowFLAKES.utilities import *

from pysolar.solar import *



def create_auxiliary_folder(working_folder, folder_name='01_TEST_auxiliary_folder'):
    """
    Creates an auxiliary folder in the working directory for storing permanent layers (e.g., DEM, masks).

    Parameters
    ----------
    working_folder : str
        The main working directory where the auxiliary folder will be created.

    Returns
    -------
    str
        The path of the auxiliary folder.
    """
    # Define path for the ancillary folder
    auxiliary_folder_path = os.path.join(working_folder, folder_name)

    # Check if the folder exists, create if not
    if not os.path.exists(auxiliary_folder_path):
        os.makedirs(auxiliary_folder_path)
        logging.info(f"Auxiliary folder created at {auxiliary_folder_path}.")
    else:
        logging.info(f"Auxiliary folder already exists at {auxiliary_folder_path}.")

    return auxiliary_folder_path



def water_identifier(data, auxiliary_folder_path):
    '''
    This function cut the water mask from the copernicus on the extent given by a ref image
    https://global-surface-water.appspot.com/download


    Parameters
    ----------
    ref_img_path : str
        path from a refernce image to cut water mask on .
    ancillary : bool, optional
        Presnce of ancillry folder, to store or not there the water mask. The default is False.

    Returns
    -------
    target_wb_mask_path : str
        water mask path.

    '''
    
    # path to the water mask file
    water_mask_file = glob.glob("/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/*")[0]

    target_wb_mask_path = os.path.join(auxiliary_folder_path, "Water_Mask.tif")

    if not os.path.exists(target_wb_mask_path):
        
        # ---- get CRS and bounds from xarray ----
        epsg_code = data.epsg.item()
        
        resolution = float(abs(data.x[1] - data.x[0]))

        E_min_old = float(data.x.min())
        E_max_old = float(data.x.max() + resolution)
        N_min_old = float(data.y.min() - resolution)
        N_max_old = float(data.y.max())

        
        # ---- water mask CRS ----
        with rasterio.open(water_mask_file) as d_target:
            srOut = d_target.crs
        
        # ---- transform extent to water-mask CRS ----
        if epsg_code != 4326:
        
            transformer = Transformer.from_crs(
                f"EPSG:{epsg_code}",
                srOut,
                always_xy=True
            )
        
            E_min, N_min = transformer.transform(E_min_old, N_min_old)
            E_max, N_max = transformer.transform(E_max_old, N_max_old)
        
            resolution /= 100000
        
        else:
        
            E_min = E_min_old
            E_max = E_max_old
            N_min = N_min_old
            N_max = N_max_old
        
        
        # ---- compute tile corners (unchanged logic) ----
        V1 = (int(np.floor(E_min / 10) * 10), int(np.ceil(N_min / 10) * 10))
        V2 = (int(np.floor(E_min / 10) * 10), int(np.ceil(N_max / 10) * 10))
        V3 = (int(np.floor(E_max / 10) * 10), int(np.ceil(N_min / 10) * 10))
        V4 = (int(np.floor(E_max / 10) * 10), int(np.ceil(N_max / 10) * 10))
        
        V_LIST = [V1, V2, V3, V4]
        
        nome_tile = []
        
        for v in V_LIST:
        
            if v[0] >= 0:
                E = str(int(np.floor(v[0] / 10) * 10))
                lat = "E"
                W = None
            else:
                W = str(int(abs(np.floor(v[0] / 10) * 10)))
                lat = "W"
                E = None
        
            if v[1] >= 0:
                N = str(int(np.ceil(v[1] / 10) * 10))
                lon = "N"
                S = None
            else:
                S = str(int(abs(np.floor(v[1] / 10) * 10)))
                lon = "S"
                N = None
        
            if W is None and N is None:
                nome = f"extent_{E}{lat}_{S}{lon}v1_4_2021.tif"
            elif W is None and S is None:
                nome = f"extent_{E}{lat}_{N}{lon}v1_4_2021.tif"
            elif E is None and N is None:
                nome = f"extent_{W}{lat}_{S}{lon}v1_4_2021.tif"
            else:
                nome = f"extent_{W}{lat}_{N}{lon}v1_4_2021.tif"
        
            file = f"/mnt/CEPH_BASEDATA/GIS/WORLD/WATER/Global_water_mask/{nome}"
        
            if file not in nome_tile:
                nome_tile.append(file)
        
        
    
        # ---- open and mosaic tiles ----
        src_files = [rasterio.open(f) for f in nome_tile]
        mosaic, mosaic_transform = merge(src_files)
        
        
        # ---- target grid from xarray ----
        width = data.sizes["x"]
        height = data.sizes["y"]
        
        dst_transform = from_bounds(E_min_old, N_min_old, E_max_old, N_max_old, width, height)
        dst_crs = f"EPSG:{epsg_code}"
        
        dst = np.empty((height, width), dtype=np.float32)
        
        
        # ---- reproject to Sentinel grid ----
        reproject(
            source=mosaic[0],
            destination=dst,
            src_transform=mosaic_transform,
            src_crs=src_files[0].crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        
        
        # Postprocess mask
        
        if np.sum(dst == 255) > 0:
        
            K = np.ones((30, 30)).astype(np.uint8)
        
            Water_dilated = cv2.dilate((dst == 255).astype(np.uint8), K, iterations=1)
        
            dst[Water_dilated == 1] = 255
            dst[dst == 210] = 1
            dst[dst == 255] = 1
        
        
        # Save result
        
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": f"EPSG:{epsg_code}",
            "transform": dst_transform
        }
        
        with rasterio.open(target_wb_mask_path, "w", **profile) as dst_file:
            dst_file.write(dst.astype("uint8"), 1)
        
        
        for src in src_files:
            src.close()


    return target_wb_mask_path



def glacier_mask_cutting(external_glacier_mask_path, water_mask_path):
    """
    Generates a glacier mask raster file from a shapefile and water mask.

    Parameters
    ----------
    external_glacier_mask_path : str
        Path to the shapefile containing glacier outlines.
    water_mask_path : str
        Path to the water mask raster.

    Returns
    -------
    str
        Path to the generated glacier mask raster file.
    """
    # Define output paths
    base_path = os.path.dirname(water_mask_path)
    glacier_shp_path = os.path.join(base_path, 'glacier_mask.shp')
    glacier_mask_path = os.path.join(base_path, 'glacier_mask.tif')

    # Check if the raster mask already exists
    if os.path.exists(glacier_mask_path):
        return glacier_mask_path

    # Open the water mask and extract extent and CRS
    img, img_info = open_image(water_mask_path)
    raster = gdal.Open(water_mask_path)
    proj = osr.SpatialReference(wkt=raster.GetProjection())
    epsg_code = proj.GetAttrValue("AUTHORITY", 1)
    crs_epsg = f"EPSG:{epsg_code}"

    # Get the extent and resolution from the water mask
    extent = img_info['extent']
    geotransform = img_info['geotransform']
    resolution = geotransform[1]
    E_min, N_min, E_max, N_max = extent

    # Create a bounding box polygon from the extent
    polygon = Polygon([
        (E_min, N_min), (E_min, N_max),
        (E_max, N_max), (E_max, N_min),
        (E_min, N_min)
    ])
    bounding_box = gpd.GeoDataFrame(geometry=[polygon], crs=crs_epsg)

    # Load and reproject the glacier shapefile
    glacier_gdf = gpd.read_file(external_glacier_mask_path)
    glacier_gdf = glacier_gdf.to_crs(crs_epsg)

    # Validate and fix geometries
    glacier_gdf['geometry'] = glacier_gdf['geometry'].apply(
        lambda geom: geom.buffer(0) if geom.is_valid else geom
    )

    # Clip glacier shapefile to the bounding box
    clipped_glaciers = gpd.clip(glacier_gdf, bounding_box)

    if not clipped_glaciers.empty:
        # Save the clipped shapefile
        clipped_glaciers.to_file(glacier_shp_path)

        # Rasterize the shapefile
        cmd = (
            f"gdal_rasterize -burn 1 -a_nodata 0 "
            f"-te {E_min} {N_min} {E_max} {N_max} "
            f"-tr {resolution} {resolution} "
            f"{glacier_shp_path} {glacier_mask_path}"
        )
        os.system(cmd)
    else:
        # Create an empty raster if no glaciers are found
        empty_glacier_mask = np.zeros_like(img)
        save_image(
            empty_glacier_mask,
            glacier_mask_path,
            'GTiff',
            1,
            geotransform,
            raster.GetProjection()
        )

    return glacier_mask_path



def calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False):
    '''
    Calculate slope and aspect from an input DEM.

    Parameters
    ----------
    dem_path : str
        Path to the existing DEM file.
    outdir : str
        Output directory where the slope and aspect files will be saved.
    resolution : float
        Output resolution.
    reproj_type : str, optional
        GDAL resampling method for reprojection (e.g., 'bilinear', 'cubic'). The default is 'cubic'.
    overwrite : bool, optional
        If True, overwrite existing slope and aspect files. The default is False.
    Ancillary_folder : bool, optional
        If True, save files in an ancillary folder within the output directory. The default is False.

    Returns
    -------
    slopePath : str
        Path to the saved slope file.
    aspectPath : str
        Path to the saved aspect file.
    '''

    slopePath = os.path.join(auxiliary_folder_path, os.path.basename(dem_path).replace('DEM.tif', 'slope.tif'))
    aspectPath = os.path.join(auxiliary_folder_path, os.path.basename(dem_path).replace('DEM.tif', 'aspect.tif'))

    print(slopePath)

    ################### Calculate slope
    if os.path.exists(slopePath) and not overwrite:
        print('Slope file was already created and saved')
    else:
        cmd = f"gdaldem slope {dem_path} {slopePath} -of GTiff -compute_edges" 
        os.system(cmd)
        print(f"Slope saved at {slopePath}")

    ################### Calculate aspect
    if os.path.exists(aspectPath) and not overwrite:
        print('Aspect file was already created and saved')
    else:
        cmd = f"gdaldem aspect {dem_path} {aspectPath} -of GTiff -compute_edges -zero_for_flat"
        os.system(cmd)
        print(f"Aspect saved at {aspectPath}")

    return slopePath, aspectPath



def create_default_cloud_mask(data, path_cloud_mask):
    """
    Create a default cloud mask (all ones) matching the grid of an xarray dataset.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Reference dataset with rioxarray metadata.
    path_cloud_mask : str or Path
        Output cloud mask path.
    """

    path_cloud_mask = Path(path_cloud_mask)

    # Raster dimensions
    height = data.sizes["y"]
    width = data.sizes["x"]

    # Create default mask (1 = clear)
    cloud_mask = np.ones((height, width), dtype=np.uint8)

    # Save raster using metadata from xarray
    with rasterio.open(
        path_cloud_mask,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(data.epsg.item()),
        transform=data.rio.transform(),
    ) as dst:
        dst.write(cloud_mask, 1)
        
        

def S2_clouds_classifier(data, cloud_bands, no_data_value, path_cloud_mask, 
                         cloud_prob, overwrite_cloud=0,
                         average_over=2, dilation_size=3):
    from s2cloudless import S2PixelCloudDetector
    """
    Classifies clouds in a Sentinel-2 image.

    Args:
        stack_clouds_path: Path to the stack of cloud bands.
        ref_img_path: Path to the stack of SCF bands.
        cloud_prob: Cloud probability threshold.
        overwrite_cloud: Whether to overwrite the existing cloud mask.
        average_over: Size of the averaging window.
        dilation_size: Size of the dilation operation.

    Returns:
        path_cloud_mask: Path to the generated cloud mask.
        cloud_cover_percentage: Cloud cover percentage.
    """

    # load the bands
    available_bands = list(data.coords["band"].values) 
    
    if "B10" not in available_bands:
    
        bands_data = []
    
        for b in cloud_bands:
            if b in available_bands:
                bands_data.append(np.squeeze(data.sel(band=b).values))
            else:
                # Fill missing band (e.g. B10) with zeros
                shape = data.sel(band=available_bands[0]).shape
                bands_data.append(np.squeeze(np.zeros(shape)))
        
        cloud_bands_image = np.stack(bands_data, axis=0)
        cloud_bands_image[cloud_bands_image == no_data_value] = np.nan

    
    else:
            
        cloud_bands_image = np.squeeze(data.sel(band=cloud_bands).values)
        cloud_bands_image[cloud_bands_image == no_data_value] = np.nan
        
 
    
    
    temporary_cloud_mask_path = path_cloud_mask.replace('.tif', '60m.tif')

    if not os.path.exists(temporary_cloud_mask_path):
        
        n_bands, width, height = cloud_bands_image.shape
        
        # Convert to (H, W, C)
        cloud_bands_image = np.transpose(cloud_bands_image, (1, 2, 0))
        
        # Create stack (batch, W, H, bands)
        Stack_to_classify = np.zeros((1, width, height, n_bands))
        
        Stack_to_classify[0] = cloud_bands_image
        Stack_to_classify[0, :, :, :][cloud_bands_image[:, :, 0] == 255] = 0
        
        print("Bands for cloud classification ready...")

        try:
            cloud_detector = S2PixelCloudDetector(threshold=cloud_prob, average_over=average_over,
                                                  dilation_size=dilation_size)
            cloud_mask = cloud_detector.get_cloud_masks(np.array(Stack_to_classify)) + 1

            cloud_cover_percentage = np.sum(cloud_mask[0, :, :] == 2) / \
                                     (np.shape(cloud_mask[0, :, :])[0] * np.shape(cloud_mask[0, :, :])[1])

            save_image(cloud_mask[0, :, :], temporary_cloud_mask_path, 'GTiff', 1, 
                       data.rio.transform().to_gdal(), CRS.from_epsg(data.epsg.item()).to_wkt())
        except Exception as e:
            print(f"Error during cloud classification: {e}")
            # Handle the error appropriately, e.g., log it or raise an exception

    if not os.path.exists(path_cloud_mask) or overwrite_cloud == 1:
        
        resolution = (data.x[1]-data.x[0]).item()

        E_min = float(data.x.min() - resolution/2)
        E_max = float(data.x.max() + resolution/2)
        N_min = float(data.y.min() - resolution/2)
        N_max = float(data.y.max() + resolution/2)


        cmd = 'gdalwarp -te ' + ' '.join((str(E_min), str(N_min), str(E_max), str(N_max))) + \
              ' -r nearest -tr ' + ' '.join((str(resolution), '-' + str(resolution))) + ' ' + ' '.join(
            (temporary_cloud_mask_path, path_cloud_mask))
        os.system(cmd)
        clud_tot = open_image(path_cloud_mask)[0]

        cloud_cover_percentage = np.sum(clud_tot[:, :] == 2) / \
                                 (np.shape(clud_tot[:, :])[0] * np.shape(clud_tot[:, :])[1])

        os.remove(temporary_cloud_mask_path)
    else:
        cloud_mask = open_image(path_cloud_mask)[0]

        cloud_cover_percentage = np.sum(cloud_mask == 2) / \
                                 (np.shape(cloud_mask)[0] * np.shape(cloud_mask)[1])

    return path_cloud_mask, cloud_cover_percentage;



def generate_no_data_mask(L_image, sensor, no_data_value=np.nan):
    """
    Generates a no-data mask for a given image based on the sensor.

    Args:
        L_image: The input image as a NumPy array.
        sensor: The sensor type (e.g., "L5", "L7").
        no_data_value: The value representing no data in the image.

    Returns:
        The generated no-data mask as a NumPy boolean array.
    """

    if np.isnan(no_data_value):
        if sensor == "L5":
            no_data_mask = (np.isnan(L_image[5, :, :]) | np.isnan(L_image[0, :, :])).astype(bool)  #
            no_data_mask = np.any(np.isnan(L_image), axis=0)
        elif sensor == "L7":
            # no_data_mask = (np.isnan(L_image[0, :, :]) | np.isnan(L_image[np.max(np.shape(L_image)[0] - 1), :, :]) | np.isnan(L_image[6, :, :])).astype(bool)
            no_data_mask = np.any(np.isnan(L_image), axis=0)
        else:
            # no_data_mask = (np.isnan(L_image[0, :, :]) | np.isnan(L_image[np.max(np.shape(L_image)[0] - 1), :, :])).astype(bool)
            no_data_mask = np.any(np.isnan(L_image), axis=0)
    else:
        # Handle other no-data values if needed
        raise NotImplementedError("Handling of non-NaN no-data values is not implemented yet.")

    valid_mask = np.logical_not(no_data_mask)

    return no_data_mask, valid_mask



def spectral_idx_computer(B1, B2, idx_name, no_data_mask, curr_aux_folder, 
                          sensor, output_filename, data, B3=None, B4=None):
    """
    Computes a spectral index and saves the result in the specified folder.

    Parameters
    ----------
    B1, B2 : numpy.ndarray
        Input bands used to calculate the spectral index.

    idx_name : str
        Name of the spectral index (e.g., 'NDSI', 'NDVI', 'shad_idx').

    no_data_mask : numpy.ndarray
        Mask indicating no-data values.

    curr_aux_folder : str
        Path to the folder where the output will be saved.

    sensor : str
        Type of sensor.

    output_filename : str
        Name of the output file.

    ref_img_path : str
        Path to the reference image to obtain metadata.

    Returns
    -------
    numpy.ndarray
        Computed spectral index.
    """

    # Define the calculations for each index
    calculations = {
        'normDiff': lambda B1, B2, B3, B4: (B1 - B2) / (B1 + B2),
        'shad_idx': lambda B1, B2, B3, B4: (B1 - B2) / (B1 + B2) / B1,
        'band_diff': lambda B1, B2, B3, B4: B1 - B2,
        'EVI': lambda B1, B2, B3, B4: 2.5 * (B1 - B2) / (B1 + 2.4 * B2 + 1),
        'NDSIplus': lambda B1, B2, B3, B4: 2 * (B1 + B2 - B3 - B4) / (B1 + B2 + B3 + B4),
        'idx6': lambda B1, B2, B3, B4: 2 * (2 * B1 - B2 - B3) / (2 * B1 + B2 + B3),
        'bandRatioGlaciers': lambda B1, B2, B3, B4: B1 / B2

    }

    # Check if the index name is in the dictionary
    if idx_name not in calculations:
        raise ValueError(f"Index '{idx_name}' is not supported.")

    # Perform the calculation
    idx_out = calculations[idx_name](B1, B2, B3, B4)

    # Set the pixels corresponding to no_data_mask to invalid (e.g., np.nan)
    idx_out[no_data_mask] = np.nan

    # Save the computed band to a file in curr_aux_folder
    output_path = os.path.join(curr_aux_folder, output_filename)
    
    # Raster dimensions
    height = data.sizes["y"]
    width = data.sizes["x"]
    
    # Save raster using metadata from xarray
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(data.epsg.item()),
        transform=data.rio.transform(),
    ) as dst:
        dst.write(idx_out, 1)
        
    print(f"Spectral index {idx_name} saved at {output_path}")
    return;
    
    

def solar_incidence_angle_calculator(data, scene_id, date_time, slopePath, aspectPath, curr_aux_folder, date):
    """
    Calculates the solar incidence angle based on slope, aspect, sun altitude, and azimuth.

    Parameters
    ----------
    img_info : dict
        Dictionary containing image metadata such as extent, geotransform, and EPSG code.

    date_time : datetime
        Date and time for which the solar position is calculated.

    slope_path : str
        Path to the slope GeoTIFF.

    aspect_path : str
        Path to the aspect GeoTIFF.

    curr_aux_folder : str
        Path to the folder where the solar incidence angle result will be saved.

    Returns
    -------
    numpy.ndarray
        Array representing the solar incidence angle.
    """
    # Extract image metadata
    resolution = float(abs(data.x[1] - data.x[0]))

    E_min = float(data.x.min())
    E_max = float(data.x.max() + resolution)
    N_min = float(data.y.min() - resolution)
    N_max = float(data.y.max())
    
    epsg_code = data.epsg.item()

    # Transform the coordinates to WGS84
    transformer = Transformer.from_crs(f"epsg:{epsg_code}", "epsg:4326", always_xy=True)
    central_E = E_min + (E_max - E_min) / 2
    central_N = N_min + (N_max - N_min) / 2
    Central_WGS84 = transformer.transform(central_E, central_N)

    # Convert date_time to UTC timezone
    datetime_object = date_time.replace(tzinfo=timezone.utc)

    # Get sun altitude and azimuth
    sun_altitude = get_altitude(Central_WGS84[1], Central_WGS84[0], datetime_object)
    sun_azimuth = get_azimuth(Central_WGS84[1], Central_WGS84[0], datetime_object)

    # Convert angles from degrees to radians
    sun_zenith_rad = np.radians(90 - sun_altitude)
    sun_azimuth_rad = np.radians(sun_azimuth)

    # Read slope and aspect from the files
    with rasterio.open(slopePath) as slope_ds, rasterio.open(aspectPath) as aspect_ds:
        slope = slope_ds.read(1)
        aspect = aspect_ds.read(1)
        profile = slope_ds.profile

    # Convert slope and aspect from degrees to radians
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)

    # Calculate the solar incidence angle
    cos_i = (
    np.cos(sun_zenith_rad) * np.cos(slope_rad)
    + np.sin(sun_zenith_rad) * np.sin(slope_rad)
    * np.cos(aspect_rad - sun_azimuth_rad)
    )

    cos_i = np.clip(cos_i, -1.0, 1.0) # this take into account the numerical approximations
    
    solar_incidence_angle = np.degrees(np.arccos(cos_i))

    # Set no-data areas (where slope is invalid) to NaN
    solar_incidence_angle[np.isnan(slope)] = np.nan

    # Save the solar incidence angle to a GeoTIFF in the curr_aux_folder
    output_path = os.path.join(curr_aux_folder, f'{scene_id}_solar_incidence_angle.tif')
    profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(solar_incidence_angle.astype(np.float32), 1)

    print(f"Solar incidence angle saved at {output_path}")
    return solar_incidence_angle, sun_altitude, sun_azimuth



def generate_shadow_mask(scene_id, curr_aux_folder, auxiliary_folder_path, no_data_mask, NIR):
    """
    Generate a shadow mask dynamically without setting thresholds and save as GeoTIFF.

    Parameters:
    - curr_aux_folder: Path to the auxiliary folder containing GeoTIFF files for indices.
    """
    # Find the paths to the necessary GeoTIFF files
    ndvi_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    idx6_path = glob.glob(os.path.join(curr_aux_folder, '*idx6.tif'))[0]
    evi_path = glob.glob(os.path.join(curr_aux_folder, '*EVI.tif'))[0]
    shad_idx_path = glob.glob(os.path.join(curr_aux_folder, '*shad_idx.tif'))[0]

    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]

    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]

    # Read input GeoTIFFs
    with rasterio.open(idx6_path) as src1, \
            rasterio.open(shad_idx_path) as src2, \
            rasterio.open(ndvi_path) as src_ndvi, \
            rasterio.open(evi_path) as src_evi, \
            rasterio.open(path_cloud_mask) as src_clouds, \
            rasterio.open(path_water_mask) as src_water, \
            rasterio.open(solar_incidence_angle_path) as src_angle:
        # Read data arrays
        index1 = src1.read(1).astype(float)
        index2 = src2.read(1).astype(float)
        ndvi = src_ndvi.read(1).astype(float)
        evi = src_evi.read(1).astype(float)
        cloud_mask = src_clouds.read(1).astype(int)
        water_mask = src_water.read(1).astype(int)
        solar_incidence_angle = src_angle.read(1).astype(float)

        # Read metadata for output
        meta = src1.meta.copy()

    # Normalize indices to range [0, 1]
    def normalize(arr):
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        return (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)

    # curr_range = (90, 180)
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))
    
    # curr_range = (min(np.nanmax(solar_incidence_angle[curr_scene_valid])-1, 90), 180)

    # curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0],
    #                                                                    solar_incidence_angle < curr_range[1]))

    index1_norm = normalize(index1)
    index2_norm = normalize(index2)
    ndvi_norm = normalize(ndvi)
    evi_norm = normalize(evi)
    nir_norm = normalize(NIR)


    # Combine indices to create a composite shadow score
    # Shadow pixels maximize index1 and index2, minimize ndvi and evi
    # shadow_score = (index1_norm + index2_norm) - (ndvi_norm + evi_norm + normalize(NIR))
    
    
    
    
    curr_range = (70, 180)
    curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0],
                                                                       solar_incidence_angle < curr_range[1]))
    
    
    self_shadow = np.logical_and(curr_scene_valid, solar_incidence_angle >= 90)

    # shadow_score = (
    #     index1_norm *
    #     index2_norm *
    #     (1 - ndvi_norm) *
    #     (1 - evi_norm) *
    #     (1 - nir_norm)
    # )
    
    shadow_score = (
        (index1_norm + index2_norm) /
        (ndvi_norm + evi_norm + nir_norm + 1e-6)
    )
    
    shadow_score[~curr_scene_valid] = np.nan
    
    threshold = np.nanpercentile(shadow_score[curr_scene_valid], 85)
    # threshold = threshold_otsu(shadow_score[curr_scene_valid])

    
    spectral_shadow = shadow_score > threshold
    casted_shadow = np.logical_and(spectral_shadow, curr_angle_valid)
    
    shadow_mask = np.logical_or(casted_shadow, self_shadow)

    # shadow_mask = cv2.medianBlur(shadow_mask.astype(np.uint8)*255, 5)


    # plt.hist(valid.flatten(), bins=500)


    
    # try:
    #     threshold = np.percentile(shadow_score[curr_angle_valid], [10, 95])[0]
    #     # plt.hist(shadow_score[curr_angle_valid].flatten(), bins=100)
    #     # Create shadow mask: positive values indicate shadow
    #     shadow_mask = (shadow_score > threshold).astype(np.uint8)
    # except:
    #     shadow_mask = np.zeros_like(shadow_score, dtype=bool).astype(np.uint8)

    # Update metadata for output
    meta.update({
        "dtype": "uint8",
        "count": 1,
        "nodata": 255,  # Use 255 as the nodata value for uint8
        "compress": "lzw"  # Compression to reduce file size
    })
    
    # Save shadow mask to GeoTIFF
    # shadow_score_path = os.path.join(curr_aux_folder, f'{scene_id}_shadow_score.tif')
    # with rasterio.open(shadow_score_path, "w", **meta) as dst:
    #     dst.write(shadow_score, 1)

    # Save shadow mask to GeoTIFF
    shadow_mask_path = os.path.join(curr_aux_folder, f'{scene_id}_shadow_mask.tif')
    with rasterio.open(shadow_mask_path, "w", **meta) as dst:
        dst.write(shadow_mask, 1)

    print(f"Shadow mask saved to {shadow_mask_path}")

    return shadow_mask_path



def adiacency_indexes(scene_id, curr_aux_folder, auxiliary_folder_path, no_data_mask, bands):
    """
    Generate a snow-proximity / altitude-constrained distance index.

    This function creates an auxiliary raster that describes how far each valid pixel is from
    high-confidence snow pixels, while also masking out areas that are considered too low in
    elevation to be relevant for snow training or snow classification.

    The output raster is later used as an additional validity/proximity constraint during
    training pixel selection. In the current SnowFLAKES workflow, pixels with value 255 are
    treated as invalid or excluded.

    The output file is saved as:

        <scene_id>_index_of_distance.tif

    inside `curr_aux_folder`.

    Parameters
    ----------
    scene_id : str
        Scene identifier. Used to infer the sensor type and to name the output file.

    curr_aux_folder : str
        Path to the current scene auxiliary folder.
        This folder must contain:
            *cloud_Mask.tif
            *NDSI.tif

    auxiliary_folder_path : str
        Path to the general auxiliary folder.
        This folder must contain:
            *Water_Mask.tif
            *DEM.tif

    no_data_mask : numpy.ndarray
        Boolean mask where True indicates no-data / invalid pixels.

    bands : dict
        Dictionary of spectral bands, usually returned by `define_bands(...)`.
        This function requires:
            bands['NIR']

    Returns
    -------
    None
        The function does not return an object.
        It writes a GeoTIFF distance index to disk.

    Output
    ------
    GeoTIFF
        Path:
            curr_aux_folder/<scene_id>_index_of_distance.tif

        Data type:
            uint8

        Values:
            0-254 : normalized distance/proximity index
            255   : no-data / excluded pixel

    Processing Steps
    ----------------
    1. Load auxiliary data:
        - cloud mask
        - water mask
        - NDSI raster
        - DEM raster
        - NIR band from the input band dictionary

    2. Build a valid-scene mask:
        A pixel is valid only if it is:
            - not cloud
            - not water
            - not no-data

        Current assumptions:
            cloud_mask == 2 means cloud
            water_mask == 1 means water
            no_data_mask == True means invalid

    3. Identify high-confidence snow and snow-free pixels:
        Snow-free pixels:
            NDSI < 0

        Snow pixels:
            NDSI > 0.6
            NIR > 0.45

        These are stored in an internal `snow_map`:
            0 = unclassified
            1 = sure no-snow
            2 = sure snow

    4. Compute distance from sure-snow pixels:
        The Euclidean distance transform is computed from pixels where:

            snow_map == 2

        The distance is then normalized to the range 0-1.

    5. Estimate an elevation threshold:
        The DEM values of sure-snow pixels are extracted.
        If sure-snow pixels exist, the minimum snow-relevant elevation is estimated as:

            altitude_min_threshold = 1st percentile of snow elevation - 500 m

        Pixels below this threshold are excluded.

    6. Combine distance and altitude:
        The normalized distance index is kept only where:
            - the pixel is valid
            - the pixel is above the altitude threshold

        Pixels outside this area are assigned 255.

    7. Save the result as a GeoTIFF.

    Notes
    -----
    The output is called an "index_of_distance", but larger values actually indicate
    pixels farther away from high-confidence snow pixels after normalization.

    This raster is used later in training selection with a rule such as:

        curr_distance_idx != 255

    meaning that pixels with value 255 are excluded.

    Important Assumptions
    ---------------------
    - The DEM, NDSI, cloud mask, water mask, and spectral bands are already aligned
      on the same grid.
    - NIR reflectance is scaled such that a threshold of 0.45 is meaningful.
    - NDSI is in the expected range, usually approximately -1 to 1.
    - DEM units are meters.
    - The CRS and geotransform are copied from the cloud mask metadata.

    Potential Issues
    ----------------
    1. The function name has a typo:
           adiacency_indexes
       should probably be:
           adjacency_indexes

    2. `sensor = get_sensor(scene_id)` is currently unused.

    3. `valid_mask = np.logical_not(no_data_mask)` is computed but not used.

    4. The altitude mask is assigned twice:

           altitude_mask = ...
           altitude_mask = (dem >= altitude_min_threshold)

       The second assignment can be problematic if `altitude_min_threshold` is NaN.

    5. If no sure-snow pixels exist, `altitude_min_threshold` becomes NaN.
       Because of the second altitude-mask assignment, the result may exclude all pixels.

    6. If all distance values are equal, this normalization can divide by zero:

           distance_from_snow_normalized =
               (distance - min) / (max - min)

    7. `np.nanmax(distance_from_snow)` can fail if all values are NaN.

    8. The output name is:

           <scene_id>_index_of_distance.tif

       but other parts of the code search for:

           *distance.tif

       This works because the filename contains "distance", but the naming should be
       kept consistent.

    Example
    -------
    >>> adiacency_indexes(
    ...     scene_id="S2A_MSIL2A_20240315T104031",
    ...     curr_aux_folder="/path/to/scene/auxiliary",
    ...     auxiliary_folder_path="/path/to/auxiliary",
    ...     no_data_mask=no_data_mask,
    ...     bands=bands
    ... )

    This creates:

        /path/to/scene/auxiliary/S2A_MSIL2A_20240315T104031_index_of_distance.tif
    """
    
    sensor = get_sensor(scene_id)

    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    dem_path = glob.glob(os.path.join(auxiliary_folder_path, '*DEM.tif'))[0]

    valid_mask = np.logical_not(no_data_mask)

    # Load masks and other necessary data
    cloud_mask, curr_image_info = open_image(path_cloud_mask)
    water_mask = open_image(path_water_mask)[0]
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))
    dem = open_image(dem_path)[0]
    NDSI = open_image(NDSI_path)[0]
    NIR = bands['NIR']

    # Create the snow map
    snow_map = np.zeros_like(NDSI, dtype=np.uint8)
    no_snow_sure = (NDSI < 0) & curr_scene_valid
    snow_sure = (NDSI > 0.6) & (NIR > 0.45) & curr_scene_valid
    snow_map[no_snow_sure] = 1
    snow_map[snow_sure] = 2

    # Calculate distance from snow_sure
    distance_from_snow = np.full_like(snow_map, np.nan, dtype=np.float32)
    snow_sure_pixels = (snow_map == 2)
    distance_from_snow[curr_scene_valid] = distance_transform_edt(~snow_sure_pixels)[curr_scene_valid]
    distance_from_snow = np.nan_to_num(distance_from_snow, nan=np.nanmax(distance_from_snow))
    distance_from_snow_normalized = (distance_from_snow - np.nanmin(distance_from_snow)) / (
            np.nanmax(distance_from_snow) - np.nanmin(distance_from_snow)
    )

    # Set altitude threshold
    valid_dem = dem[np.logical_and(curr_scene_valid, snow_map == 2)]

    if valid_dem.size > 0:
        altitude_min_threshold = np.percentile(valid_dem, 1) - 500
    else:
        altitude_min_threshold = np.nan  # Oppure scegli un valore predefinito sensato

    altitude_mask = (dem >= altitude_min_threshold) if not np.isnan(altitude_min_threshold) else np.zeros_like(dem,
                                                                                                               dtype=bool)

    altitude_mask = (dem >= altitude_min_threshold)

    # Combine distance and altitude into index_of_distance
    index_of_distance = np.zeros_like(snow_map, dtype=np.float32)
    index_of_distance[curr_scene_valid] = (
            distance_from_snow_normalized[curr_scene_valid] * altitude_mask[curr_scene_valid]
    )

    # Convert to uint8 for saving
    index_of_distance_uint8 = (index_of_distance * 254).astype(np.uint8)  # Scale if needed

    # Set no-data value for areas outside altitude_mask
    no_data_value = 255  # Choose the no-data value, e.g., 0 or 255
    index_of_distance_uint8[np.logical_or(~altitude_mask, ~curr_scene_valid)] = no_data_value

    # Save the result as a GeoTIFF
    output_path = os.path.join(curr_aux_folder, f"{scene_id}_index_of_distance.tif")
    transform = from_origin(curr_image_info['geotransform'][0], curr_image_info['geotransform'][3],
                            curr_image_info['geotransform'][1], -curr_image_info['geotransform'][5])
    with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=index_of_distance_uint8.shape[0],
            width=index_of_distance_uint8.shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=curr_image_info['projection'],
            transform=transform,
            nodata=no_data_value,
    ) as dst:
        dst.write(index_of_distance_uint8, 1)


# to be updated











def water_mask_cutting(water_mask_path, ref_img_path, auxiliary_folder_path):
    '''
    Parameters
    ----------
    water_mask_path : str
        path of water mask to cut .
    ref_img_path : str
        path of a reference image.
    Ancillary_folder : bool


    Returns
    -------
    target_wb_mask_path : str
        water mask path.


    '''
    if auxiliary_folder_path != None:
        target_wb_mask_path = auxiliary_folder_path + os.sep + os.path.basename(
            os.path.dirname(os.path.dirname(ref_img_path))) + "_Water_Mask.tif"
    else:
        target_wb_mask_path = ref_img_path[:-8] + "_Water_Mask.tif"

    if not os.path.exists(target_wb_mask_path):
        # clip the wbm with FSC extent

        img_info = open_image(ref_img_path)[1]

        d = gdal.Open(ref_img_path)
        with rasterio.open(ref_img_path, 'r+') as rds:
            epsg_code_ref = str(rds.crs).split(':')[1]

        E_min = (img_info['extent'][0])
        N_min = (img_info['extent'][1])
        E_max = (img_info['extent'][2])
        N_max = (img_info['extent'][3])
        img_res = str(img_info['geotransform'][1])

        extent_string = ' '.join([str(E_min), str(N_min), str(E_max), str(N_max)])
        cmd = 'gdalwarp -t_srs EPSG:' + epsg_code_ref + ' -te ' + extent_string + ' -tr ' + ' '.join(
            [img_res, img_res]) + \
              ' -of GTiff ' + ' '.join([water_mask_path, target_wb_mask_path])

        os.system(cmd)

        water_mask = open_image(target_wb_mask_path)[0]

        # dialte the nan value (255) of the water mask into the 1 value of water mask
        if np.sum(water_mask == 255) > 0:
            K = np.ones((30, 30)).astype(np.uint8)
            Water_dilated = cv2.dilate((water_mask == 255).astype(np.uint8), K, iterations=1)
            # create a single water mask with 0-1
            water_mask[Water_dilated == 1] = 255
            water_mask[water_mask == 210] = 1
            water_mask[water_mask == 255] = 1
            os.remove(target_wb_mask_path)
            save_image(water_mask.astype('uint8'), target_wb_mask_path, 'GTiff', 1, img_info['geotransform'],
                       img_info['projection'])

    return target_wb_mask_path








def landsat_cloud_classifier(data, cloud_bands, no_data_value,
                             path_cloud_mask, sensor, valid_mask, Nprocesses=8,
                             dilate_iterations=5):
    from xgboost import XGBClassifier
    import pickle
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.features import shapes
    from skimage.morphology import binary_erosion, binary_dilation, disk
    from joblib import Parallel, delayed
    import glob
    import os
    
    cloud_bands_image = np.squeeze(data.sel(band=cloud_bands).values)
    cloud_bands_image[cloud_bands_image == no_data_value] = np.nan

    # Select model based on sensor
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if sensor == 'L7':
        model_filepath = os.path.join(script_dir, 'Aux_files', 'Landsat-7_cloud_model_xgboost.p')
    elif sensor == 'L8':
        model_filepath = os.path.join(script_dir, 'Aux_files', 'Landsat-8_9_cloud_model_xgboost8.p')

    
    # Load the XGBoost model and associated data
    with open(model_filepath, 'rb') as model_file:
        svm_dict = pickle.load(model_file)
    xgboost_model = svm_dict['xgboostModel']
    normalizer = svm_dict['normalizer']
    
    if sensor == 'L7':
        new_names = ['blue', 'green', 'red', 'nir08', 'swir16', 'lwir', 'swir22']
    elif sensor == 'L8':
        new_names = ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'lwir11']
    
    svm_dict['feature_names'] = new_names
    feature_names = svm_dict['feature_names']

    # Create mapping from feature name -> band index
    band_map = {name: i for i, name in enumerate(cloud_bands)}
    
    # Get indices for requested feature names
    band_indices = [band_map[name] for name in feature_names]
    
    # Extract features
    features = np.column_stack([
        cloud_bands_image[i][valid_mask]
        for i in band_indices
    ])

    # Normalize features
    features = np.nan_to_num(features)
    features = normalizer.transform(features)

    # Split features for parallel processing
    feature_blocks = np.array_split(features, Nprocesses)

    # Classify in parallel using XGBoost
    def classify_block(block):
        return xgboost_model.predict(block)

    predictions_blocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(classify_block)(block) for block in feature_blocks
    )
    predictions = np.concatenate(predictions_blocks) + 1  # Adjust class indices

    # Create the output raster
    class_map = np.zeros((data.sizes['y'], data.sizes['x']), dtype='uint8')
    class_map[valid_mask] = predictions

    # Invert class_map values (1 ↔ 2)
    class_map[class_map == 1] = 3  # Temporary placeholder
    class_map[class_map == 2] = 1
    class_map[class_map == 3] = 2

    # Apply erosion and dilation
    # Apply erosion
    print("Applying morphological operations...")
    struct_element = disk(2)  # Structuring element for erosion and dilation
    eroded_map = binary_erosion(class_map == 2, footprint=struct_element).astype(np.uint8)

    # Apply dilation iteratively
    dilated_map = eroded_map
    for _ in range(dilate_iterations):
        dilated_map = binary_dilation(dilated_map, footprint=struct_element).astype(np.uint8)

    # Update class_map with morphological operations
    class_map[class_map > 0] = dilated_map[class_map > 0] * 2
    class_map = np.nan_to_num(class_map, nan=0).astype(np.uint8)
    class_map[class_map == 0] = 1
    
    # Save raster using metadata from xarray
    with rasterio.open(
        path_cloud_mask,
        "w",
        driver="GTiff",
        height=data.sizes['y'],
        width=data.sizes['x'],
        count=1,
        dtype="uint8",
        crs=CRS.from_epsg(data.epsg.item()),
        transform=data.rio.transform(),
    ) as dst:
        dst.write(class_map, 1)

        
        

    print(f"Classified and processed raster saved to {path_cloud_mask}.")

    clud_tot = open_image(path_cloud_mask)[0]

    cloud_cover_percentage = np.sum(clud_tot[:, :] == 2) / \
                             (np.shape(clud_tot[:, :])[0] * np.shape(clud_tot[:, :])[1])

    return path_cloud_mask, cloud_cover_percentage


































