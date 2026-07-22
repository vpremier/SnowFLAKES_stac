#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:59:20 2024

@author: rbarella
"""
import os
import numpy as np
import shutil
from datetime import datetime as dt
import time
import geopandas as gpd
import rasterio

from SnowFLAKES.auxiliary_folder_population import (
    calc_slope_aspect,
    glacier_mask_cutting,
    create_omnicloudmask,
    create_folder,
)

from SnowFLAKES.utilities import (
    get_sensor,
    create_log,
    define_datetime)

from SnowFLAKES.training_collection import *
from SnowFLAKES.SCF_functions import *
from SnowFLAKES.ice import run_snow_ice_classification

from loading.load_stac import (
    load_cdse_collection,
    setup_cdse_credentials,
)

from utils import load_with_retry


def run_snowflakes(config, data, scene_id):
    
    print(f"Running SnowFLAKES for {scene_id}")
    start = time.time()

    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")    
    
    # Snow Cover Fraction folder
    SCF_folder = create_folder(scene_folder, config['SVM_folder_name'])
    
    # log files: create log files
    skipped_scenes_file = create_log(wd, '00_scenes_to_skip')
    cloud_scenes_file = create_log(wd, '00_skip_cloud_masks')
    
    # overwrite
    ow = config['overwrite']

    # sensor    
    sensor = get_sensor(scene_id)
    
    # Extract date and time from the folder name
    date_time, date = define_datetime(scene_id, config)

    


    
    


    
    # No data value
    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)

    

    # Load DEM, and compute slope and aspect
    dem_path = os.path.join(auxiliary_folder_path, "DEM.tif")

    if not os.path.exists(dem_path):

        setup_cdse_credentials()
        dem = load_cdse_collection("cop-dem-glo-30-dged-cog",
                                   auxiliary_folder_path,
                                   resolution=config['resampling_params']['resolution'],
                                   extent_target=config['resampling_params']['extent_target'],
                                   epsg_target=config['resampling_params']['epsg_target'])

    slopePath, aspectPath = calc_slope_aspect(dem_path, 
                                              auxiliary_folder, 
                                              reproj_type='bilinear', 
                                              overwrite=ow)



    # Generate water mask
    print("Generating water mask...")
    external_water_mask_path = config['External_water_mask']
    if not external_water_mask_path:
        water_mask_path = water_identifier(data, auxiliary_folder_path)
    else:
        #TODO: to be updated
        water_mask_path = water_mask_cutting(external_water_mask_path, ref_img_path, auxiliary_folder_path)

    print(f"Water mask saved at {water_mask_path}")
    
    
    # Generate glacier mask
    external_glacier_mask_path = config['external_glacier_mask_path']
    classify_glaciers = config['classify_glaciers']
    glacier_mask_path = glacier_mask_cutting(external_glacier_mask_path, water_mask_path)


    if classify_glaciers == 'yes':
        glaciers_model_svm = config['glaciers_model_name']
        start_glaciers_month = int(config['start_glaciers_month'])
        end_glaciers_month = int(config['end_glaciers_month'])

        dt_start_glaciers_month = dt(1900, start_glaciers_month, 1)
        dt_end_glaciers_month = dt(1900, end_glaciers_month, 1)

    else:
        print("No glacier mask created.")





    # load bands: bands used for SCF
    all_bands = select_band_names(sensor, 'scf') # curr_band_stack_path
    all_bands_image = np.squeeze(data.sel(band=all_bands).values)
    all_bands_image[all_bands_image == no_data_value] = np.nan
    

   
    no_data_mask, valid_mask = generate_no_data_mask(all_bands_image, sensor, no_data_value=np.nan)
    
    # Auxiliary folder
    curr_aux_folder = os.path.join(scene_folder, "auxiliary")
    os.makedirs(curr_aux_folder, exist_ok=True)


    
    
    # Generate cloud mask or use default if clouds are not computed
    path_cloud_mask, cc_perc = create_omnicloudmask(data, 
                                                    scene_id, auxiliary_folder_path, curr_aux_folder)

    

    
    no_data_percentage = np.sum(no_data_mask) / (data.sizes["y"] * data.sizes["x"])
    cloud_perc_corr = cloud_cover_percentage / (1 - no_data_percentage)


    if np.sum(no_data_mask) / len(valid_mask.flatten()) > 1 or cloud_perc_corr > 0.6:
        print('TOO MANY INVALID PIXELS...')
        
        # Save the scene in the log file
        with open(cloud_scenes_file, "a") as f:
            f.write(f"{scene_id}\n")
        
        # delete folder!!
        shutil.rmtree(scene_folder)
        return
    

    # Compute spectral indices: NDVI, NDSI, band difference, and shadow index
    bands = define_bands(data, sensor)
    
    spectral_idx_computer(bands['GREEN'], bands['NIR'], 'normDiff', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NDWI.tif", data)
    spectral_idx_computer(bands['NIR'], bands['RED'], 'normDiff', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NDVI.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'normDiff', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NDSI.tif", data)
    spectral_idx_computer(bands['BLUE'], bands['NIR'], 'band_diff', no_data_mask, 
                          curr_aux_folder,sensor, f"{scene_id}_diffBNIR.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'shad_idx', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_shad_idx.tif", data)
    spectral_idx_computer(bands['BLUE'], bands['NIR'], 'normDiff', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NormDiffBNIR.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['RED'], 'normDiff', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NormDiffGreenRed.tif", data)
    spectral_idx_computer(bands['NIR'], bands['RED'], 'EVI', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_EVI.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['RED'], 'idx6', no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_idx6.tif", data, B3=bands['NIR'])
    spectral_idx_computer(bands['RED'], bands['SWIR'], 'bandRatioGlaciers', no_data_mask,
                          curr_aux_folder, sensor, f"{scene_id}_bandRatioGlaciers.tif", data)
    
    
    # Calculate solar incidence angle
    solar_incidence_angle, sun_altitude, sun_azimuth = solar_incidence_angle_calculator(
        data,
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
                                            auxiliary_folder_path, 
                                            no_data_mask, 
                                            bands['NIR'])
   
    # adiecency map
    adiacency_indexes(scene_id, curr_aux_folder, auxiliary_folder_path, no_data_mask, bands)
    
    
    # Collect training data and train the SVM model if no pretrained model exists
    predefined_model = config.get('Predefined_model', 'no')
    
    
    if predefined_model == 'no':

        print("Generating training shapefile.")
        try:
            shapefile_path = collect_trainings(scene_id, 
                                               all_bands_image, 
                                               curr_aux_folder, 
                                               auxiliary_folder_path,
                                               SCF_folder,
                                               no_data_mask, 
                                               bands,
                                               sun_altitude)
        except:
            print("Error for training collection")
            return

            
            
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)

        # Check if the shapefile has both values (assuming they are in a column named 'class')
        unique_values = set(gdf['value'].unique())
        print(unique_values)
        thematic_map_path = thematic_map_classifier(scene_id, data, curr_aux_folder, auxiliary_folder_path,
                                                    no_data_mask, SCF_folder, classify_glaciers,
                                                    date_time)
    

        if unique_values != {1, 2}:
            print(
                f"Skipping scene {scene_id} due to missing value 1 or 2. Produced just default map")
            
            output_path = os.path.join(SCF_folder, f'{scene_id}_SnowFLAKES_GLACIERS.tif')
            
            if config["simple_class"]: 
                # Open the raster
                with rasterio.open(path_cloud_mask) as src:
                    meta = src.meta.copy()
    
                # Open the raster
                with rasterio.open(thematic_map_path) as src:
                    thematic_map = src.read(1)
    
                # Save the modified raster
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(thematic_map, 1)
    
                # Save the scene in the log file
                with open(skipped_scenes_file, "a") as f:
                    f.write(f"{scene_id}\n")
    
                return  # Skip to the next scene
        
            else:
            
                # Look for the closest date with representative trainings
                closest = find_closest_valid_scf(working_folder, date)
                scene_id_closest = os.path.basename(closest).split("_SnowFLAKES.tif")[0]
    
                
                _, date_closest = define_datetime(scene_id_closest, config)
                date_closest = dt.strptime(date_closest, "%Y%m%d").strftime("%Y-%m-%d")
                
                # take the model of the closest image
                svm_model_filename = os.path.join(os.path.dirname(closest), "svm_model.p")
                
                
                
                # Run SCF prediction
                FSC_SVM_map_path = SCF_dist_SV(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                               svm_model_filename, Nprocesses=1, overwrite=True)
                
                # check if there is snow around the glacier
                if classify_glaciers == 'yes': # and snow_around_glacier(FSC_SVM_map_path, curr_aux_folder, auxiliary_folder_path):
                    
                    
                    snow_mask, ice_mask = get_pixels_ice(bands, curr_aux_folder, auxiliary_folder_path, FSC_SVM_map_path, no_data_mask)
                    
                    results_glacier = run_snow_ice_classification(
                        data=data,
                        snow_mask=snow_mask,
                        ice_mask=ice_mask,
                        output_folder=None,
                        max_samples_per_class=10000,
                        prediction_mask=None,
                    )

                    # glacier_map = glacier_classifier(scene_id, data, no_data_mask, curr_aux_folder, auxiliary_folder_path)
                    # model_path = r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Azufre/training_checked/model_ice.p'

                    # glacier_map = glacier_xgboost(model_path, data, no_data_mask, curr_aux_folder, auxiliary_folder_path)
                    mask_raster_with_glacier(results_glacier, FSC_SVM_map_path, auxiliary_folder_path, curr_aux_folder, no_data_mask)
                    
                    
                return
                
            
        
        # else SnowFLAKES
        print('TRAINING')
        svm_model_filename = model_training(scene_id, all_bands_image, data, 
                                            shapefile_path, curr_aux_folder, gamma=None)

        # Run SCF prediction
        FSC_SVM_map_path = SCF_dist_SV(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                       svm_model_filename, Nprocesses=1, overwrite=True)
        
        

        
        remove_low_scf(FSC_SVM_map_path, bands, curr_aux_folder, dem_path)
        

        if classify_glaciers == 'yes': # and snow_around_glacier(FSC_SVM_map_path, curr_aux_folder, auxiliary_folder_path):
            
            snow_mask, ice_mask = get_pixels_ice(bands, curr_aux_folder, auxiliary_folder_path, FSC_SVM_map_path, no_data_mask)
            
            results_glacier = run_snow_ice_classification(
                data=data,
                snow_mask=snow_mask,
                ice_mask=ice_mask,
                output_folder=None,
                max_samples_per_class=10000,
                prediction_mask=None,
            )

            # # glacier_map = glacier_classifier(scene_id, data, no_data_mask, curr_aux_folder, auxiliary_folder_path)
            # model_path = r'/mnt/CEPH_PROJECTS/SNOWCOP/Glaciers/Azufre/training_checked/model_ice.p'

            # glacier_map = glacier_xgboost(model_path, data, no_data_mask, curr_aux_folder, auxiliary_folder_path)
            mask_raster_with_glacier(results_glacier, FSC_SVM_map_path, auxiliary_folder_path, curr_aux_folder, no_data_mask)


        print("Process completed. Condition met, and no points found where SCF > 0 and NDSI < 0.")

    # Prediction if predefined SVM model exists
    else:
        svm_model_filename = config["Predefined_SVM_model"]
        print(f"Using predefined model {svm_model_filename}")
        FSC_SVM_map_path = SCF_dist_SV(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                       svm_model_filename, Nprocesses=1, overwrite=True)


    
    

    

  

if __name__ == "__main__":
    print('Running SnowFLAKES')
