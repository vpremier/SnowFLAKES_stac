#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:59:20 2024

@author: rbarella
"""
import os
import glob
from datetime import datetime
import time
from scipy.ndimage import binary_dilation


from auxiliary_folder_population import *
from utilities import *
from training_collection import *
from SCF_functions import *
from xgboost_functions import *
from shadow_mask_gen import *



def run_snowflakes(config):


    # Replace CSV-related code
    working_folder = config['output_directory']
    satellite = config['satellite']
    
    
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
    
    
    
    
    
    
    
    
    
    
    create_empty_files(working_folder)

    scenes_to_skip = scenes_skip(working_folder)
    scenes_to_skip_clouds = cloud_mask_to_skip(working_folder)

    print("Checking satellite mission...")


    if 'skipped_scenes.log' in acquisitions:
        acquisitions.remove('skipped_scenes.log')

    if not acquisitions:
        raise ValueError("No acquisitions found in the working folder.")

    acquisition_name = acquisitions[1]
    sensor = get_sensor(acquisition_name)

    print("Creating auxiliary folder for static data...")
    auxiliary_folder_path = create_auxiliary_folder(working_folder)
    



    resolution = float(config['resampling_params']['resolution'])
    print(f"Using resolution: {resolution}.")

    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)

    print(f"no data value: {no_data_value}.")

   

    # Step 6: Generate VRT (Virtual Raster Table) files with stacked bands for selected acquisitions
    print("Creating VRT files...")

    if sensor != 'PRISMA':
        create_vrt_files(acquisitions_filtered, sensor, resolution)
        ref_img_path = glob.glob(acquisitions_filtered[0] + os.sep + "*scf.vrt")[0]
        
        print("VRT creation process completed.")









   # Step 9: Download or crop DEM, and compute slope and aspect
    subimage_extents = process_image(img_info)
    print("Extent computed...")

    external_dem_path = input_data.get('External_Dem_path')
    if not external_dem_path:
        print("Preparing DEM download...")
        dem_path = dem_downloader(subimage_extents, ref_img_path, resolution, auxiliary_folder_path, buffer=0.02)
    else:
        print("Preparing DEM cropping...")
        dem_path = crop_predefined_DEM(ref_img_path, external_dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)

    slopePath, aspectPath = calc_slope_aspect(dem_path, auxiliary_folder_path, reproj_type='bilinear', overwrite=False)

    # Initialize an empty list to track scenes without cloud masks
    scenes_not_to_cloud_mask = []

    SVM_folder_name = input_data['SVM_folder_name']
    XGB_folder_name = SVM_folder_name + '_XGB'

    # Ensure the directory exists
    skipped_scenes_file = os.path.join(working_folder, "skipped_scenes.log")
    if not os.path.exists(skipped_scenes_file):
        open(skipped_scenes_file, "w").close()

    overwrite_scenes = input_data.get('overwrite', 'no')

    for curr_acquisition in acquisitions_filtered:
        folder = curr_acquisition + os.sep + SVM_folder_name

        # Skip already processed scenes
        if glob.glob(folder + os.sep + "*FLAKES*") or glob.glob(
                working_folder + os.sep + "*" + SVM_folder_name + "*FLAKES*"):
            if overwrite_scenes == 'no':
                print("Scene already processed " + os.path.basename(working_folder))
                continue
            elif overwrite_scenes == 'yes':
                print("Scene already processed. Overwriting scene.")

        start = time.time()
        print(os.path.basename(curr_acquisition))

        try:
            # Extract date and time from the folder name and sensor type
            date_time, date = define_datetime(sensor, curr_acquisition)
        except:
            continue
        
        # Handle no-data mask
        curr_band_stack_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
        all_band_stack_path = glob.glob(os.path.join(curr_acquisition, '*scfT.vrt'))

        if all_band_stack_path == []:
            glob.glob(os.path.join(curr_acquisition, '*cloud.vrt'))

        if curr_band_stack_path == []:
            curr_band_stack_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            curr_band_stack_path = curr_band_stack_path[0]

        if all_band_stack_path == []:
            all_band_stack_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            all_band_stack_path = all_band_stack_path[0]

        curr_image, curr_image_info = open_image(curr_band_stack_path)
        curr_image[curr_image == no_data_value] = np.nan

        all_bands_image = open_image(all_band_stack_path)[0]
        all_bands_image[all_bands_image == no_data_value] = np.nan
        no_data_mask, valid_mask = generate_no_data_mask(all_bands_image, sensor, no_data_value=np.nan)

        # Create auxiliary folder for this acquisition, for storing cloud mask, spectral indices, etc.
        curr_aux_folder = create_auxiliary_folder(curr_acquisition, folder_name='00_auxiliary_folder_' + date)

        # Cloud Masking
        cloud_prob = float(input_data.get('Cloud cover probability', 0.6))
        average_over = int(input_data.get('average_over', 3))
        dilation_size = int(input_data.get('dilation_cloud_cover', 3))
        overwrite_cloud = int(input_data.get('Overwrite_cloud', 0))

        path_cloud_mask = os.path.join(curr_aux_folder, os.path.basename(curr_acquisition) + '_cloud_Mask.tif')
        Compute_clouds = input_data.get('Compute_clouds', 'no') == 'yes'


        # Generate cloud mask or use default if clouds are not computed
        if not Compute_clouds or date in scenes_to_skip_clouds:
            create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
            cloud_cover_percentage = 0
        else:
            if date in scenes_not_to_cloud_mask:
                create_default_cloud_mask(vrt[0, :, :], path_cloud_mask)
                cloud_cover_percentage = 0
            elif sensor == 'S2':
                cloud_prob = float(input_data.get('Cloud_cover_probability', 60))
                average_over = int(input_data.get('average_over', 3))
                dilation_size = int(input_data.get('dilation_cloud_cover', 3))
                overwrite_cloud = int(input_data.get('Overwrite_cloud', 0))
                stack_clouds_path = \
                [os.path.join(curr_acquisition, f) for f in os.listdir(curr_acquisition) if 'cloud.vrt' in f][0]
                cloud_mask_path, cloud_cover_percentage = S2_clouds_classifier(
                    stack_clouds_path, path_cloud_mask, ref_img_path, cloud_prob, overwrite_cloud=overwrite_cloud,
                    average_over=average_over, dilation_size=dilation_size)
            elif sensor == 'L7' or sensor == 'L8':
                path_cloud_mask, cloud_cover_percentage = landsat_cloud_classifier(curr_aux_folder, path_cloud_mask,
                                                                                   ref_img_path, sensor, valid_mask,
                                                                                   Nprocesses=8, dilate_iterations=5)

        no_data_percentage = np.sum(no_data_mask) / (
                    curr_image_info['X_Y_raster_size'][0] * curr_image_info['X_Y_raster_size'][1])
        cloud_perc_corr = cloud_cover_percentage / (1 - no_data_percentage)

        if perform_pca:
            perform_pca_on_geotiff(curr_band_stack_path, valid_mask)

        # Compute spectral indices: NDVI, NDSI, band difference, and shadow index
        valid_mask = np.logical_not(no_data_mask)

        if np.sum(no_data_mask) / len(valid_mask.flatten()) > 1 or cloud_perc_corr > 0.6:
            print('TOO MANY INVALID PIXELS...')
            continue

        bands = define_bands(curr_image, valid_mask, sensor)

        spectral_idx_computer(bands['NIR'], bands['RED'], 'normDiff', curr_image, no_data_mask, curr_aux_folder, sensor,
                              f"{sensor}_{date}_NDVI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'normDiff', curr_image, no_data_mask, curr_aux_folder,
                              sensor, f"{sensor}_{date}_NDSI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['BLUE'], bands['NIR'], 'band_diff', curr_image, no_data_mask, curr_aux_folder,
                              sensor, f"{sensor}_{date}_diffBNIR.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['SWIR'], 'shad_idx', curr_image, no_data_mask, curr_aux_folder,
                              sensor, f"{sensor}_{date}_shad_idx.tif", curr_band_stack_path)

        spectral_idx_computer(bands['BLUE'], bands['NIR'], 'normDiff', curr_image, no_data_mask, curr_aux_folder,
                              sensor, f"{sensor}_{date}_NormDiffBNIR.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['RED'], 'normDiff', curr_image, no_data_mask, curr_aux_folder,
                              sensor, f"{sensor}_{date}_NormDiffGreenRed.tif", curr_band_stack_path)

        spectral_idx_computer(bands['NIR'], bands['RED'], 'EVI', curr_image, no_data_mask, curr_aux_folder, sensor,
                              f"{sensor}_{date}_EVI.tif", curr_band_stack_path)
        spectral_idx_computer(bands['GREEN'], bands['RED'], 'NDSIplus', curr_image, no_data_mask, curr_aux_folder,
                              sensor, f"{sensor}_{date}_NDSIplus.tif", curr_band_stack_path, B3=bands['NIR'],
                              B4=bands['SWIR'])
        spectral_idx_computer(bands['GREEN'], bands['RED'], 'idx6', curr_image, no_data_mask, curr_aux_folder, sensor,
                              f"{sensor}_{date}_idx6.tif", curr_band_stack_path, B3=bands['NIR'])

        spectral_idx_computer(bands['RED'], bands['SWIR'], 'bandRatioGlaciers', curr_image, no_data_mask,
                              curr_aux_folder, sensor, f"{sensor}_{date}_bandRatioGlaciers.tif", curr_band_stack_path)

        # Calculate solar incidence angle
        solar_incidence_angle, sun_altitude, sun_azimuth = solar_incidence_angle_calculator(curr_image_info, date_time,
                                                                                            slopePath, aspectPath,
                                                                                            curr_aux_folder, date)

        # shadow mask
        shadow_mask_path = generate_shadow_mask(curr_aux_folder, auxiliary_folder_path, no_data_mask, bands['NIR'])

        ## adiecency map
        adiacency_indexes(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, bands)

        # Step 10a: Collect training data and train the SVM model if no pretrained model exists
        predefined_model = input_data.get('Predefined_model', 'no')


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

            shapefile_path = os.path.join(working_folder, curr_acquisition, SVM_folder_name,
                                          'representative_pixels_for_training_samples.shp')

            if not os.path.exists(shapefile_path):
                print("Generating training shapefile.")
                # shapefile_path, training_mask_path = collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path,
                #                                                        SVM_folder_name, training_collection_no_data_mask, bands,
                #                                                        shadow_mask_path)

                try:
                    shapefile_path = collect_trainings(curr_acquisition, curr_aux_folder, auxiliary_folder_path,
                                                       SVM_folder_name, training_collection_no_data_mask, bands,
                                                       shadow_mask_path)
                except:
                    print("Error for training collection")
                    continue
            else:
                print("Shapefile already present, skipping.")

            # Controlla se il file shapefile esiste ed è valido
            if not os.path.exists(shapefile_path) or os.path.getsize(shapefile_path) == 0:
                print(f"Skipping scene {os.path.basename(curr_acquisition)} due to missing geometries.")

                # Save the scene in the log file
                with open(skipped_scenes_file, "a") as f:
                    f.write(f"{os.path.basename(curr_acquisition)}\n")

                continue  # Skip to the next scene

            # Load the shapefile
            gdf = gpd.read_file(shapefile_path)
            # print(gdf.columns)

            # Check if the shapefile has both values (assuming they are in a column named 'class')
            unique_values = set(gdf['value'].unique())
            print(unique_values)
            thematic_map_path = thematic_map_classifier(curr_acquisition, curr_aux_folder, auxiliary_folder_path,
                                                        no_data_mask, SVM_folder_name, classify_glaciers,
                                                        date_time, dt_start_glaciers_month, dt_end_glaciers_month)
            if unique_values != {1, 2}:
                print(
                    f"Skipping scene {os.path.basename(curr_acquisition)} due to missing value 1 or 2. Produced just default map")
                scf_folder = os.path.join(curr_acquisition, SVM_folder_name)
                output_path = os.path.join(scf_folder, os.path.basename(curr_acquisition) + '_SnowFLAKES_GLACIERS.tif')

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
                    f.write(f"{os.path.basename(curr_acquisition)} - missing class 1 or 2\n")

                continue  # Skip to the next scene
            ## Preclassification with xgboost

            NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
            with rasterio.open(NDSI_path) as ndsi_src:
                ndsi_data = ndsi_src.read(1)  # Reading first band
            counter_to_exit = 0

            while True:
                print('TRAINING')
                svm_model_filename = model_training(curr_acquisition, shapefile_path, SVM_folder_name, gamma=None,
                                                    perform_pca=perform_pca)
                # xgb_model_filename = model_training_xgb(curr_acquisition, shapefile_path, XGB_folder_name, perform_pca=False, grid_search=True)

                # Step 11: Run SCF prediction
                FSC_SVM_map_path = SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                               svm_model_filename, Nprocesses=1, overwrite=True,
                                               perform_pca=perform_pca)
                # FSC_XGB_map_path = snow_class_XGB(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, xgb_model_filename,
                #                  Nprocesses=8, overwrite=true, perform_pca=False)

                # Step 12: Result check
                shapefile_path = check_scf_results(FSC_SVM_map_path, shapefile_path, curr_aux_folder, curr_acquisition,
                                                   k=5, n_closest=5, perform_pca=perform_pca)

                # Load SCF and NDSI data to check the condition
                with rasterio.open(FSC_SVM_map_path) as scf_src:
                    scf_data = scf_src.read(1)  # Reading first band

                # Load SM and NDSI data to check the condition
                # with rasterio.open(FSC_XGB_map_path) as sm_src:
                #     sm_data = sm_src.read(1)  # Reading first band

                # Check condition
                # if np.sum((scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0) & (sm_data <= 100) & (sm_data > 0)) == 0 or counter_to_exit >= 10:

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


        # Step 10b: Prediction if predefined SVM model exists
        else:
            svm_model_filename = get_input_param(input_data, "Predefined_SVM_model")
            print(f"Using predefined model {svm_model_filename}")
            FSC_SVM_map_path = SCF_dist_SV(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask,
                                           svm_model_filename, Nprocesses=1, overwrite=True, perform_pca=perform_pca)

        give_time(start)


if __name__ == "__main__":
    print('ciao')


