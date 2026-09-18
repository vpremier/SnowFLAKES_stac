#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:39:26 2024

@author: rbarella
"""

import os
import shutil
from osgeo import gdal, osr
import geopandas as gpd
import numpy as np
import cv2
import rasterio
from shapely.geometry import Polygon
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer
from datetime import timezone
from rasterio.crs import CRS
from omnicloudmask import predict_from_array
from scipy.ndimage import binary_dilation

from pysolar.solar import get_altitude, get_azimuth

from rasterio.enums import Resampling

from SnowFLAKES.utilities import (
    load_map,
    build_valid_scene,
    save_tif, 
    open_image,
    get_sensor,
    valid_mask,
    create_folder,
    define_bands,
    define_datetime,
    create_log
)


from loading.load_stac import (
    load_cdse_collection,
    load_latest_cdse_asset,
    setup_cdse_credentials,
)




def create_omnicloudmask(data, scene_id, auxiliary_folder, curr_aux_folder,
                         no_data_value = np.nan, dilation_iterations=3):
    """Generate and save an OmniCloudMask cloud-classification raster."""
    if dilation_iterations < 0:
        raise ValueError("dilation_iterations must be non-negative")

    sensor = get_sensor(scene_id)
    dem_path = os.path.join(auxiliary_folder, "DEM.tif")
    path_cloud_mask = os.path.join(curr_aux_folder, f'{scene_id}_cloud_Mask.tif')
    os.makedirs(curr_aux_folder, exist_ok=True)

    if sensor == 'S2':
        input_array = np.squeeze(
            data.sel(band=["B04", "B03", "B8A"]).values
        ).astype(np.float32)

    elif sensor in ['L5', 'L7', 'L8']:
        input_array = np.squeeze(
            data.sel(band=["red", "green", "nir08"]).values
        ).astype(np.float32)
    else:
        raise ValueError(f"Cloud masking is not supported for sensor '{sensor}'")

    mask = predict_from_array(input_array, no_data_value=no_data_value)
    mask = np.squeeze(mask)

    # -------------------------
    # Dilate only thick clouds
    # -------------------------
    thick_cloud = (mask == 1)

    if dilation_iterations == 0:
        dilated = thick_cloud
    else:
        dilated = binary_dilation(
            thick_cloud,
            iterations=dilation_iterations
        )

    # Assign only newly dilated pixels to class 1
    mask[dilated] = 1

    # cloud cover given by thick and thin clouds
    cloud_cover_percentage = (np.sum(mask == 1) + np.sum(mask == 2))/ mask.size

    save_tif(mask, dem_path, path_cloud_mask, dtype=rasterio.uint8)

    return path_cloud_mask, cloud_cover_percentage



def spectral_idx_computer(data, B1, B2, idx_name, curr_aux_folder,
                          output_filename, no_data_value=np.nan, B3=None, B4=None):
    """Compute, save, and return a spectral index raster."""
    validMask = valid_mask(data, no_data_value=no_data_value)

    calculations = {
        'normDiff': lambda B1, B2, B3, B4: (B1 - B2) / (B1 + B2),
        'shad_idx': lambda B1, B2, B3, B4: (B1 - B2) / (B1 + B2) / B1,
        'band_diff': lambda B1, B2, B3, B4: B1 - B2,
        'EVI': lambda B1, B2, B3, B4: 2.5 * (B1 - B2) / (B1 + 2.4 * B2 + 1),
        'NDSIplus': lambda B1, B2, B3, B4: 2 * (B1 + B2 - B3 - B4) / (B1 + B2 + B3 + B4),
        'idx6': lambda B1, B2, B3, B4: 2 * (2 * B1 - B2 - B3) / (2 * B1 + B2 + B3),
        'bandRatioGlaciers': lambda B1, B2, B3, B4: B1 / B2

    }

    if idx_name not in calculations:
        raise ValueError(f"Index '{idx_name}' is not supported.")
    if idx_name == 'idx6' and B3 is None:
        raise ValueError("Index 'idx6' requires B3.")
    if idx_name == 'NDSIplus' and (B3 is None or B4 is None):
        raise ValueError("Index 'NDSIplus' requires B3 and B4.")

    with np.errstate(divide='ignore', invalid='ignore'):
        idx_out = np.asarray(calculations[idx_name](B1, B2, B3, B4), dtype=np.float32)

    if idx_out.shape != validMask.shape:
        raise ValueError(
            f"Index shape {idx_out.shape} does not match valid-mask shape {validMask.shape}."
        )

    idx_out[~validMask | ~np.isfinite(idx_out)] = no_data_value

    os.makedirs(curr_aux_folder, exist_ok=True)
    output_path = os.path.join(curr_aux_folder, output_filename)
    height = data.sizes["y"]
    width = data.sizes["x"]

    try:
        crs = CRS.from_epsg(int(data.epsg.item()))
    except (AttributeError, TypeError, ValueError):
        crs = data.rio.crs

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        nodata=no_data_value,
        crs=crs,
        transform=data.rio.transform(),
    ) as dst:
        dst.write(idx_out, 1)

    print(f"Spectral index {idx_name} saved at {output_path}")
    return idx_out



def water_identifier(
    auxiliary_folder_path,
    reference_raster_path=None,
    collection="clms_lcm_global_10m_yearly_v1_cog",
    asset="map",
    water_value=100,
    dilation_kernel_size=2,
    overwrite=False,
):
    """Download and save a CDSE water mask aligned with a reference raster.

    The newest annual CLMS 10 m Land Cover Map is read from the CDSE STAC
    catalogue. Its permanent-water class is converted to the binary convention
    used by SnowFLAKES (water=1, all other classes=0).
    """
    print("Generating water mask...")

    target_wb_mask_path = os.path.join(auxiliary_folder_path, "Water_Mask.tif")

    if os.path.exists(target_wb_mask_path) and not overwrite:
        with rasterio.open(target_wb_mask_path) as existing_mask:
            tags = existing_mask.tags()
        if (
            tags.get("source_collection") == collection
            and tags.get("source_asset") == asset
        ):
            return target_wb_mask_path
        print("Replacing water mask created from the legacy local dataset...")

    if reference_raster_path is None:
        reference_raster_path = os.path.join(auxiliary_folder_path, "DEM.tif")
    if not os.path.exists(reference_raster_path):
        raise FileNotFoundError(
            f"Reference raster for the water mask does not exist: {reference_raster_path}"
        )
    if dilation_kernel_size < 0:
        raise ValueError("dilation_kernel_size must be non-negative")

    water_classes, source_item = load_latest_cdse_asset(
        collection,
        asset,
        reference_raster_path,
        reproj_type=Resampling.nearest,
    )

    water_classes = np.asarray(water_classes)
    with rasterio.open(reference_raster_path) as reference:
        expected_shape = (reference.height, reference.width)
    if water_classes.shape != expected_shape:
        raise ValueError(
            f"Water mask shape {water_classes.shape} does not match "
            f"reference grid {expected_shape}"
        )

    water_mask = (water_classes == water_value).astype(np.uint8)
    if dilation_kernel_size > 0 and np.any(water_mask):
        kernel = np.ones(
            (dilation_kernel_size, dilation_kernel_size),
            dtype=np.uint8,
        )
        water_mask = cv2.dilate(water_mask, kernel, iterations=1)

    save_tif(
        water_mask,
        reference_raster_path,
        target_wb_mask_path,
        nodata=255,
        dtype=rasterio.uint8,
    )
    with rasterio.open(target_wb_mask_path, "r+") as output:
        output.update_tags(
            source_collection=collection,
            source_asset=asset,
            source_item=source_item,
            water_value=water_value,
        )
    print(f"Water mask saved at {target_wb_mask_path}")

    return target_wb_mask_path



def glacier_mask_cutting(external_glacier_mask_path, water_mask_path):
    """Generate a glacier mask clipped to the water-mask extent."""
    print("Generating glacier mask...")

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
        
        # save tif
        save_tif(empty_glacier_mask, water_mask_path, glacier_mask_path, dtype=rasterio.uint8)
        print(f"Glacier mask saved at {glacier_mask_path}")

    return glacier_mask_path



def calc_slope_aspect(dem_path, auxiliary_folder_path, overwrite=False):
    """Generate slope and aspect rasters from a DEM."""

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



def get_altitude_azimuth(data, date_time):
    
    # Extract image metadata
    resolution = float(abs(data.x[1] - data.x[0]))

    E_min = float(data.x.min())
    E_max = float(data.x.max() + resolution)
    N_min = float(data.y.min() - resolution)
    N_max = float(data.y.max())
    
    try:
        epsg_code = data.epsg.item()
    except:
        epsg_code = data.rio.crs.to_epsg()
        
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
    
    return sun_altitude, sun_azimuth
    

    
def solar_incidence_angle_calculator(data, scene_id, date_time, slopePath, aspectPath, curr_aux_folder, date):
    """Compute and save solar-incidence angles from terrain and acquisition time."""

    sun_altitude, sun_azimuth = get_altitude_azimuth(data, date_time)
    
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
    return solar_incidence_angle



def generate_shadow_mask(scene_id, curr_aux_folder, auxiliary_folder, no_data_mask, NIR):
    """Generate and save a composite terrain and cloud-shadow mask."""
    
    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    ndvi = load_map(curr_aux_folder, '*NDVI.tif')
    idx6 = load_map(curr_aux_folder, '*idx6.tif')
    shad_idx = load_map(curr_aux_folder, '*shad_idx.tif')
    evi = load_map(curr_aux_folder, '*EVI.tif')
    solar_incidence_angle = load_map(curr_aux_folder, '*solar_incidence_angle.tif')
    water_mask = load_map(auxiliary_folder, '*Water_Mask.tif')

    _, NDSI_path = load_map(curr_aux_folder, '*NDSI.tif', return_path=True)


    # Normalize indices to range [0, 1]
    def normalize(arr):
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        return (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)

    
    # validity mask
    curr_scene_valid = build_valid_scene(no_data_mask,
                                         cloud_mask == 1,
                                         cloud_mask == 2,
                                         water_mask == 1,
                                         iterations = 0)
    
    # SIA between 70 and 180
    curr_angle_valid = np.logical_and.reduce((curr_scene_valid, 
                                              solar_incidence_angle >= 70,
                                              solar_incidence_angle < 180))
    


    idx6_norm = normalize(idx6)
    shad_idx_norm = normalize(shad_idx)
    ndvi_norm = normalize(ndvi)
    evi_norm = normalize(evi)
    nir_norm = normalize(NIR)


    # Combine indices to create a composite shadow score
    shadow_score = (
        (idx6_norm + shad_idx_norm) /
        (ndvi_norm + evi_norm + nir_norm + 1e-6)
    )
    
    shadow_score[~curr_scene_valid] = np.nan
    
    threshold = np.nanpercentile(shadow_score[curr_scene_valid], 85)
    
    # DIFFERENT SHADOWS
    self_shadow = np.logical_and(curr_scene_valid, solar_incidence_angle >= 90)
    cloud_shadow = cloud_mask == 3

    spectral_shadow = shadow_score > threshold
    casted_shadow = np.logical_and(spectral_shadow, curr_angle_valid)
    
    shadow_mask = np.logical_or.reduce((casted_shadow, self_shadow, cloud_shadow))


    # Save shadow mask to GeoTIFF
    shadow_mask_path = os.path.join(curr_aux_folder, f'{scene_id}_shadow_mask.tif')

    save_tif(shadow_mask, NDSI_path, shadow_mask_path, dtype=rasterio.uint8)
    
    print(f"Shadow mask saved to {shadow_mask_path}")

    return shadow_mask_path



def thematic_map_classifier(scene_id, data, curr_aux_folder, auxiliary_folder,
                            validMask):
    """Classify and save a threshold-based thematic snow map."""


    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder, '*Water_Mask.tif')
    NDSI, NDSI_path = load_map(curr_aux_folder, '*NDSI.tif', return_path=True)
    NDVI = load_map(curr_aux_folder, '*NDVI.tif')
    
    # Set a fixed NDSI threshold (candidate pixels) and a NDVI threshold to avoid vegetation
    ndsi_threshold = 0.4
    ndvi_threshold = 0.5
    

    # Mark invalid pixels as 255 (no-data, clouds, or water)
    thematic_map = np.zeros_like(NDSI, dtype=np.uint8)

    
    # Build candidate mask: valid pixels with sufficient NDSI and low NDVI.
    snow_mask = validMask & (NDSI > ndsi_threshold) & (NDVI < ndvi_threshold)
    thematic_map[snow_mask] = 100

    # non-valid pixels
    thematic_map[np.logical_not(validMask)] = 255
    
    # Optionally mark cloud and water areas with distinct codes:
    thematic_map[cloud_mask == 1] = 205  # thick clouds
    thematic_map[cloud_mask == 2] = 205  # thin clouds
    thematic_map[water_mask == 1] = 210
    

    # Define output path
    output_path = os.path.join(curr_aux_folder, f'{scene_id}_simple_class.tif')

    # save output tif file
    save_tif(thematic_map, NDSI_path, output_path, dtype=rasterio.uint8)

    return output_path


    
def adjacency_index(scene_id, curr_aux_folder, auxiliary_folder, no_data_mask):
    """Compute and save an altitude-constrained distance-from-snow index."""

    # Load masks and other necessary data
    snow_map = load_map(curr_aux_folder, '*simple_class.tif')
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder, '*Water_Mask.tif')
    dem, dem_path = load_map(auxiliary_folder, '*DEM.tif', return_path=True)

    # valid mask
    curr_scene_valid = build_valid_scene(no_data_mask,
                                         cloud_mask == 1,
                                         cloud_mask == 2,
                                         water_mask == 1,
                                         iterations=0)
    
    # Calculate distance from snow_sure
    distance_from_snow = np.full_like(snow_map, np.nan, dtype=np.float32)
    snow_sure_pixels = (snow_map == 100)
    distance_from_snow[curr_scene_valid] = distance_transform_edt(~snow_sure_pixels)[curr_scene_valid]
    distance_from_snow = np.nan_to_num(distance_from_snow, nan=np.nanmax(distance_from_snow))
    distance_from_snow_normalized = (distance_from_snow - np.nanmin(distance_from_snow)) / (
            np.nanmax(distance_from_snow) - np.nanmin(distance_from_snow)
    )
    
    
    # Set altitude threshold, kind of snowline altitude
    valid_dem = dem[np.logical_and(curr_scene_valid, snow_map == 100)]

    if valid_dem.size == 0:
        altitude_mask = np.zeros_like(dem, dtype=bool)
    else:
        altitude_min_threshold = np.percentile(valid_dem, 1) - 500
        altitude_mask = dem >= altitude_min_threshold
    

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

    save_tif(index_of_distance_uint8, dem_path, output_path, 
             nodata=no_data_value, dtype=rasterio.uint8)
    
    return output_path
    
    

def create_auxiliary_information(scene_id, data, config):
    """Generate the auxiliary rasters required for scene classification."""
    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   
    
    sensor = get_sensor(scene_id)
    
    # Extract date and time from the folder name
    date_time, date = define_datetime(scene_id, config)

    # No data value
    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")

    # Scene's auxiliary folder
    curr_aux_folder = create_folder(scene_folder, "auxiliary")
    
    # valid mask
    validMask = valid_mask(data, no_data_value=no_data_value)
    no_data_perc = np.sum(~validMask) / (data.sizes["y"] * data.sizes["x"])
    
    
    
    # Load DEM, and compute slope and aspect ----------------------------------
    dem_path = os.path.join(auxiliary_folder, "DEM.tif")

    if not os.path.exists(dem_path):

        setup_cdse_credentials()
        dem = load_cdse_collection("cop-dem-glo-30-dged-cog",
                                   auxiliary_folder,
                                   resolution=config['resampling_params']['resolution'],
                                   extent_target=config['resampling_params']['extent_target'],
                                   epsg_target=config['resampling_params']['epsg_target'])

    slopePath, aspectPath = calc_slope_aspect(dem_path, 
                                              auxiliary_folder,
                                              overwrite=config['overwrite'])
    
    
    # Extract permanent water from the latest CLMS 10 m land-cover map.
    water_mask_path = water_identifier(auxiliary_folder)


    # Generate glacier mask
    external_glacier_mask_path = config['external_glacier_mask_path']
    glacier_mask_path = glacier_mask_cutting(external_glacier_mask_path, water_mask_path)
    
    
    # Generate cloud mask -----------------------------------------------------
    cloud_scenes_file = create_log(wd, '00_skip_cloud_masks')

    path_cloud_mask, cc_perc = create_omnicloudmask(data,
                                                    scene_id,
                                                    auxiliary_folder,
                                                    curr_aux_folder,
                                                    no_data_value = no_data_value)

    cloud_perc_corr = cc_perc / (1 - no_data_perc)

    if no_data_perc > 0.8 or cloud_perc_corr > 0.6:
        print(f'TOO MANY INVALID PIXELS for image {scene_id}')

        # Save the scene in the log file
        with open(cloud_scenes_file, "a") as f:
            f.write(f"{scene_id}\n")

        # delete folder
        shutil.rmtree(scene_folder)
        
        return False
    
    
    
    # Compute spectral indices: NDVI, NDSI, band difference, and shadow index
    bands = define_bands(data, sensor)
    
    spectral_idx_computer(data, bands['GREEN'], bands['NIR'], 'normDiff',
                          curr_aux_folder, f"{scene_id}_NDWI.tif", no_data_value)
    spectral_idx_computer(data, bands['NIR'], bands['RED'], 'normDiff',
                          curr_aux_folder, f"{scene_id}_NDVI.tif", no_data_value)
    spectral_idx_computer(data, bands['GREEN'], bands['SWIR'], 'normDiff',
                          curr_aux_folder, f"{scene_id}_NDSI.tif", no_data_value)
    spectral_idx_computer(data, bands['BLUE'], bands['NIR'], 'band_diff',
                          curr_aux_folder, f"{scene_id}_diffBNIR.tif", no_data_value)
    spectral_idx_computer(data, bands['GREEN'], bands['SWIR'], 'shad_idx',
                          curr_aux_folder, f"{scene_id}_shad_idx.tif", no_data_value)
    spectral_idx_computer(data, bands['NIR'], bands['RED'], 'EVI',
                          curr_aux_folder, f"{scene_id}_EVI.tif", no_data_value)
    spectral_idx_computer(data, bands['GREEN'], bands['RED'], 'idx6',
                          curr_aux_folder, f"{scene_id}_idx6.tif", no_data_value,
                          B3=bands['NIR'])

    
    
    
    # Calculate solar incidence angle
    solar_incidence_angle = solar_incidence_angle_calculator(data,
                                                             scene_id,
                                                             date_time,
                                                             slopePath,
                                                             aspectPath,
                                                             curr_aux_folder,
                                                             date
                                                         )
    
    
    # shadow mask
    shadow_mask_path = generate_shadow_mask(scene_id, 
                                            curr_aux_folder, 
                                            auxiliary_folder, 
                                            ~validMask, 
                                            bands['NIR'])
    
    # SCF with threshold methods
    SCF_thematic_path = thematic_map_classifier(scene_id, data, curr_aux_folder, 
                                                auxiliary_folder, validMask)

    

    # adiecency map
    adjacency_index_path = adjacency_index(scene_id, curr_aux_folder, auxiliary_folder, ~validMask)

    return True













