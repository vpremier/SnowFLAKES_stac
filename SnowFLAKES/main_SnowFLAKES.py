#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:59:20 2024

@author: rbarella
"""
import os
import numpy as np
import glob
from datetime import datetime
import time
from scipy.ndimage import binary_dilation


from SnowFLAKES.auxiliary_folder_population import *
from SnowFLAKES.utilities import *
from SnowFLAKES.training_collection import *
from SnowFLAKES.SCF_functions import *
from SnowFLAKES.xgboost_functions import *
from SnowFLAKES.shadow_mask_gen import *


from stac.load_stac import load_cdse_collection


def run_snowflakes(config, data, scene_id):

    # Config info
    working_folder = config['output_directory']
    scene_folder = os.path.join(config['output_directory'], scene_id)
    os.makedirs(scene_folder, exist_ok=True)
    
    satellite = config['satellite']
    sensor = get_sensor(scene_id)
    ow = config['overwrite']
    
    try:
        # Extract date and time from the folder name and sensor type
        date_time, date = define_datetime(sensor, scene_id)
    except Exception:
        raise ValueError("Non valid scene id")

    
    # print("Creating auxiliary folder for static data...")
    auxiliary_folder_path = create_auxiliary_folder(working_folder)
    
    
    SVM_folder_name = config['SVM_folder_name']
    XGB_folder_name = SVM_folder_name + '_XGB'
    
    
    # log files
    create_empty_files(working_folder)

    scenes_to_skip = scenes_skip(working_folder)
    scenes_to_skip_clouds = cloud_mask_to_skip(working_folder)
    
    # Ensure the directory exists
    skipped_scenes_file = os.path.join(working_folder, "skipped_scenes.log")
    if not os.path.exists(skipped_scenes_file):
        open(skipped_scenes_file, "w").close()

    
    # No data value
    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)

    print(f"no data value: {no_data_value}.")
    
    
    # Generate water mask
    print("Generating water mask...")
    external_water_mask_path = config['External_water_mask']
    if not external_water_mask_path:
        water_mask_path = water_identifier(data, auxiliary_folder_path)
    else:
        # to be updated
        water_mask_path = water_mask_cutting(external_water_mask_path, ref_img_path, auxiliary_folder_path)

    print(f"Water mask saved at {water_mask_path}")
    
    
    # Generate glacier mask
    print("Generating glacier mask...")
    classify_glaciers = config['classify_glaciers']
    if classify_glaciers == 'yes':
        external_glacier_mask_path = config['external_glacier_mask_path']
        glaciers_model_svm = config['glaciers_model_name']
        glaciers_mask_path = glacier_mask_cutting(external_glacier_mask_path, water_mask_path)

        start_glaciers_month = int(config['start_glaciers_month'])
        end_glaciers_month = int(config['end_glaciers_month'])

        dt_start_glaciers_month = datetime(1900, start_glaciers_month, 1)
        dt_end_glaciers_month = datetime(1900, end_glaciers_month, 1)

        print(f"Glacier mask saved at {glaciers_mask_path}")
    else:
        print("No glacier mask created.")
    
    
    
    # Load DEM, and compute slope and aspect
    dem_path = os.path.join(auxiliary_folder_path, "DEM.tif")
    
    if not os.path.exists(dem_path):
        print("Downloading DEM from CDSE...")
        dem = load_cdse_collection("cop-dem-glo-30-dged-cog", 
                                   auxiliary_folder_path,
                                   resolution=config['resampling_params']['resolution'], 
                                   extent_target=config['resampling_params']['extent_target'], 
                                   epsg_target=config['resampling_params']['epsg_target'])

    slopePath, aspectPath = calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)
    
    
    # Snow Cover Fraction
    SCF_folder = os.path.join(scene_folder, "SCF")
    os.makedirs(SCF_folder, exist_ok=True)
    
    date_str = data.start_datetime.item()[:10].replace("-", "")
    SCF_path = os.path.join(SCF_folder, "{scene_id}.tif")

    # Skip already processed scenes
    if os.path.exists(SCF_path) and not ow:
        print(f"Scene {scene_id} already processed. Set overwrite as True in the config file.")
        return

    start = time.time()

    print(f"Running SnowFLAKES for {scene_id}")
    
    # loading in the memory the STAC
    data.load()
    
    # Cloud Masking settings
    cloud_prob = float(config.get('Cloud cover probability', 0.6))
    average_over = int(config.get('average_over', 3))
    dilation_size = int(config.get('dilation_cloud_cover', 3))
    overwrite_cloud = int(config.get('Overwrite_cloud', 0))
    


    # load bands -> all_bands
    curr_bands = select_band_names(sensor, 'scfT') 
    curr_image = np.squeeze(data.sel(band=curr_bands).values)
    curr_image[curr_image == no_data_value] = np.nan
    
    
    all_bands = select_band_names(sensor, 'scf') # curr_band_stack_path
    all_bands_image = np.squeeze(data.sel(band=all_bands).values)
    all_bands_image[all_bands_image == no_data_value] = np.nan
    
    cloud_bands = select_band_names(sensor, 'cloud')
    cloud_bands_image = np.squeeze(data.sel(band=cloud_bands).values)
    cloud_bands_image[cloud_bands_image == no_data_value] = np.nan
    

    no_data_mask, valid_mask = generate_no_data_mask(all_bands_image, sensor, no_data_value=np.nan)
    
    # Auxiliary folder
    curr_aux_folder = os.path.join(scene_folder, "auxiliary")
    os.makedirs(curr_aux_folder, exist_ok=True)

    # da rivedere come strutturo le cartelle!!!!
    
    # Cloud mask
    path_cloud_mask = os.path.join(curr_aux_folder, f'{scene_id}_cloud_Mask.tif')
    Compute_clouds = config.get('Compute_clouds', 'no') == 'yes'
    
    
    # Generate cloud mask or use default if clouds are not computed
    if not Compute_clouds:
        create_default_cloud_mask(data, path_cloud_mask)
        cloud_cover_percentage = 0
    elif sensor == 'S2':
        cloud_prob = float(config.get('Cloud_cover_probability', 60))
        average_over = int(config.get('average_over', 3))
        dilation_size = int(config.get('dilation_cloud_cover', 3))
        overwrite_cloud = int(config.get('Overwrite_cloud', 0))
  
    
        cloud_mask_path, cloud_cover_percentage = S2_clouds_classifier(data, cloud_bands, 
                                                                       no_data_value, path_cloud_mask, 
                                                                       cloud_prob, overwrite_cloud=0,
                                                                       average_over=2, dilation_size=3)      
        
    elif sensor == 'L7' or sensor == 'L8':
        # to be changed!!!
        path_cloud_mask, cloud_cover_percentage = landsat_cloud_classifier(curr_aux_folder, path_cloud_mask,
                                                                           ref_img_path, sensor, valid_mask,
                                                                           Nprocesses=8, dilate_iterations=5)
    
    
    
    no_data_percentage = np.sum(no_data_mask) / (data.sizes["y"] * data.sizes["x"])
    cloud_perc_corr = cloud_cover_percentage / (1 - no_data_percentage)

    # Compute spectral indices: NDVI, NDSI, band difference, and shadow index
    valid_mask = np.logical_not(no_data_mask)

    if np.sum(no_data_mask) / len(valid_mask.flatten()) > 1 or cloud_perc_corr > 0.6:
        print('TOO MANY INVALID PIXELS...')
        return

    bands = define_bands(curr_image, valid_mask, sensor)
    
    spectral_idx_computer(bands['NIR'], bands['RED'], 'normDiff', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NDVI.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'normDiff', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NDSI.tif", data)
    spectral_idx_computer(bands['BLUE'], bands['NIR'], 'band_diff', curr_image, no_data_mask, 
                          curr_aux_folder,sensor, f"{scene_id}_diffBNIR.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'shad_idx', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_shad_idx.tif", data)
    spectral_idx_computer(bands['BLUE'], bands['NIR'], 'normDiff', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NormDiffBNIR.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['RED'], 'normDiff', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NormDiffGreenRed.tif", data)
    spectral_idx_computer(bands['NIR'], bands['RED'], 'EVI', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_EVI.tif", data)
    spectral_idx_computer(bands['GREEN'], bands['RED'], 'NDSIplus', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_NDSIplus.tif",
                          data, B3=bands['NIR'], B4=bands['SWIR'])
    spectral_idx_computer(bands['GREEN'], bands['RED'], 'idx6', curr_image, no_data_mask, 
                          curr_aux_folder, sensor, f"{scene_id}_idx6.tif", data, B3=bands['NIR'])
    spectral_idx_computer(bands['RED'], bands['SWIR'], 'bandRatioGlaciers', curr_image, no_data_mask,
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
    shadow_mask_path = generate_shadow_mask(curr_aux_folder, auxiliary_folder_path, no_data_mask, bands['NIR'])
    
    # adiecency map
    adiacency_indexes(scene_id, curr_aux_folder, auxiliary_folder_path, no_data_mask, bands)
    
    
    # Collect training data and train the SVM model if no pretrained model exists
    predefined_model = config.get('Predefined_model', 'no')
    
    
    if predefined_model == 'no':
        if (classify_glaciers == 'yes' and
            dt_start_glaciers_month is not None and
            dt_end_glaciers_month is not None and
            is_month_in_range(date_time.month, dt_start_glaciers_month.month, dt_end_glaciers_month.month)):
    
            with rasterio.open(glaciers_mask_path) as src:
                glaciers_mask = src.read(1)  # Read the cloud mask (first band)
    
            # Apply an N-pixel buffer using binary dilation
            N = 3  # Replace this with the desired buffer size
            structure = np.ones((2 * N + 1, 2 * N + 1))  # Define the dilation kernel
            glaciers_mask = binary_dilation(glaciers_mask, structure=structure).astype(int)
            training_collection_no_data_mask = np.logical_or(no_data_mask, glaciers_mask == 1)
            
            
        else:
            training_collection_no_data_mask = no_data_mask
            dt_start_glaciers_month = None
            dt_end_glaciers_month = None

        shapefile_path = os.path.join(scene_folder, scene_id, SVM_folder_name,
                                      'representative_pixels_for_training_samples.shp')
        
        
        if not os.path.exists(shapefile_path):
            print("Generating training shapefile.")
            # shapefile_path, training_mask_path = collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path,
            #                                                        SVM_folder_name, training_collection_no_data_mask, bands,
            #                                                        shadow_mask_path)

            try:
                shapefile_path = collect_trainings(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path,
                                                   SVM_folder_name, training_collection_no_data_mask, bands,
                                                   shadow_mask_path)
            except:
                print("Error for training collection")
                return
        else:
            print("Shapefile already present, skipping.")
            
            
            
        # Controlla se il file shapefile esiste ed è valido
        if not os.path.exists(shapefile_path) or os.path.getsize(shapefile_path) == 0:
            print(f"Skipping scene {scene_id} due to missing geometries.")

            # Save the scene in the log file
            with open(skipped_scenes_file, "a") as f:
                f.write(f"{scene_id}\n")
                
            return  # Skip to the next scene
            
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)

        # Check if the shapefile has both values (assuming they are in a column named 'class')
        unique_values = set(gdf['value'].unique())
        print(unique_values)
        thematic_map_path = thematic_map_classifier(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path,
                                                    no_data_mask, SVM_folder_name, classify_glaciers,
                                                    date_time, dt_start_glaciers_month, dt_end_glaciers_month)
    


        if unique_values != {1, 2}:
            print(
                f"Skipping scene {scene_id} due to missing value 1 or 2. Produced just default map")
            
            output_path = os.path.join(SCF_folder, f'{scene_id}_SnowFLAKES_GLACIERS.tif')

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
                f.write(f"{scene_id} - missing class 1 or 2\n")

            return  # Skip to the next scene
        
        ## Preclassification with xgboost
        NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
        with rasterio.open(NDSI_path) as ndsi_src:
            ndsi_data = ndsi_src.read(1)  # Reading first band
        counter_to_exit = 0

    
        while True:
            print('TRAINING')
            svm_model_filename = model_training(scene_id, all_bands_image, data, 
                                                shapefile_path, SVM_folder_name, gamma=None)
            # xgb_model_filename = model_training_xgb(curr_acquisition, shapefile_path, XGB_folder_name, perform_pca=False, grid_search=True)

            # Run SCF prediction
            FSC_SVM_map_path = SCF_dist_SV(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                           svm_model_filename, Nprocesses=1, overwrite=True)
            # FSC_XGB_map_path = snow_class_XGB(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, xgb_model_filename,
            #                  Nprocesses=8, overwrite=true, perform_pca=False)

            # Result check
            shapefile_path = check_scf_results(scene_id, all_bands_image, FSC_SVM_map_path, 
                                               shapefile_path, curr_aux_folder, k=5, n_closest=5)


            # Load SCF and NDSI data to check the condition
            with rasterio.open(FSC_SVM_map_path) as scf_src:
                scf_data = scf_src.read(1)  # Reading first band

            # Check condition
            if np.sum((scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0)) == 0 or counter_to_exit >= 10:
                break  # Exit the loop if no points meet the condition
   

            counter_to_exit += 1

            if (classify_glaciers == 'yes' and
                dt_start_glaciers_month is not None and
                dt_end_glaciers_month is not None and
                is_month_in_range(date_time.month, dt_start_glaciers_month.month, dt_end_glaciers_month.month)):

                mask_raster_with_glacier(FSC_SVM_map_path, thematic_map_path, auxiliary_folder_path)


            ## Glacier_classification
            hemisphere = get_hemisphere(FSC_SVM_map_path)

            print("Process completed. Condition met, and no points found where SCF > 0 and NDSI < 0.")

    # Prediction if predefined SVM model exists
    else:
        svm_model_filename = config["Predefined_SVM_model"]
        print(f"Using predefined model {svm_model_filename}")
        FSC_SVM_map_path = SCF_dist_SV(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                       svm_model_filename, Nprocesses=1, overwrite=True)



    # Initialize an empty list to track scenes without cloud masks
    # scenes_not_to_cloud_mask = []
    
    
    # check the log files
    # remove PCA
    # check folder structure
    # add time duration

  

if __name__ == "__main__":
    print('ciao')

# aggiungere buffer per laghi

# aggiornare doc string

# si vede bordo tra le tile

# soglie shadow - non shadow alpi e ande
