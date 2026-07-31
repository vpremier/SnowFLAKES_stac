#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:59:20 2024

@author: rbarella
"""
import os
import numpy as np
from datetime import datetime as dt
import geopandas as gpd

from SnowFLAKES.auxiliary_folder_population import create_auxiliary_information

from SnowFLAKES.utilities import (
    create_log,
    create_folder,
    define_datetime,
    find_closest_valid_scf,
    snow_around_glacier,
    remove_low_scf,
    get_uncertainty
)

from SnowFLAKES.training_collection import (
    collect_trainings,
    get_pixels_ice,
)

from SnowFLAKES.SCF_functions import (
    model_training, 
    SCF_dist_SV,
    mask_raster_with_glacier,
)

from SnowFLAKES.ice import run_snow_ice_classification




def run_snowflakes(config, data, scene_id):
    
    print(f"Running SnowFLAKES for {scene_id}")

    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")

    # Scene's auxiliary folder
    curr_aux_folder = create_folder(scene_folder, "auxiliary")
    
    
    # log files: create log files
    skipped_scenes_file = create_log(wd, '00_scenes_to_skip')
    closest_models_file = create_log(wd, '00_closest_models_used')
    
    # overwrite
    ow = config['overwrite']
    
    # uncertainty
    unc = config['uncertainty']

    # whether to classify glaciers or not
    classify_glaciers = config['classify_glaciers']

    # Extract date and time from the folder name
    date_time, date = define_datetime(scene_id, config)

    # No data value
    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)

    # Create all auxiliary information
    valid_scene = create_auxiliary_information(scene_id, data, config)

    if not valid_scene:
        print("Ending process. No valid data for this image.")
        return

    # Collect training data and train the SVM model if no pretrained model exists

    print("Generating training shapefile.")
    shapefile_path = None
    training_error = None
    unique_values = set()
    class_counts = {}

    try:
        shapefile_path = collect_trainings(
            data,
            scene_id,
            config,
        )
    except Exception as error:
        training_error = error
        print(
            "Training collection failed: "
            f"{type(error).__name__}: {error}"
        )

    if shapefile_path is None:
        if training_error is None:
            training_reason = (
                "Training collection did not return a shapefile."
            )
        else:
            training_reason = (
                "Training collection failed before a shapefile was created: "
                f"{type(training_error).__name__}: {training_error}"
            )
    else:
        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)

        if "value" not in gdf.columns:
            training_reason = (
                "The training shapefile has no 'value' column."
            )
        else:
            class_counts = gdf["value"].value_counts().to_dict()
            unique_values = set(class_counts)

            if not unique_values:
                training_reason = (
                    "The training shapefile contains no training samples."
                )
            elif unique_values == {1}:
                training_reason = (
                    f"Only snow training samples were found: {class_counts}"
                )
            elif unique_values == {2}:
                training_reason = (
                    f"Only snow-free training samples were found: {class_counts}"
                )
            elif unique_values != {1, 2}:
                training_reason = (
                    f"Unexpected training classes were found: {class_counts}"
                )
            else:
                training_reason = None

    if training_reason is not None:
        print(
            f"Cannot train a model for {scene_id}. "
            f"Reason: {training_reason}"
        )
        
        if config["find_closest_model"]: 
            
            # Look for the closest date with representative trainings
            closest = find_closest_valid_scf(wd, date)

            if closest is None:
                print("No valid model from another scene was found.")
                with open(skipped_scenes_file, "a") as f:
                    f.write(f"{scene_id}\n")
                return

            scene_id_closest = os.path.basename(closest).split("_SnowFLAKES.tif")[0]

            
            _, date_closest = define_datetime(scene_id_closest, config)
            date_closest = dt.strptime(date_closest, "%Y%m%d").strftime("%Y-%m-%d")
            
            # take the model of the closest image
            svm_model_filename = os.path.join(
                os.path.dirname(closest),
                "auxiliary",
                "svm_model.p",
            )

            # Run SCF prediction
            FSC_SVM_map_path = SCF_dist_SV(data, scene_id, config, svm_model_filename, 
                                           Nprocesses=1, overwrite=ow)

            # Record which existing model was used for this scene.
            target_date = dt.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
            write_header = os.path.getsize(closest_models_file) == 0

            with open(closest_models_file, "a") as f:
                if write_header:
                    f.write(
                        "target_scene,target_date,"
                        "model_scene,model_date\n"
                    )

                f.write(
                    f"{scene_id},{target_date},"
                    f"{scene_id_closest},{date_closest}\n"
                )
            
        else:

            # Save the scene in the log file
            with open(skipped_scenes_file, "a") as f:
                f.write(f"{scene_id}\n")

            return  # Skip to the next scene

    else:
        print(f"Training class counts: {class_counts}")

        # SCF map creation
        print('TRAINING')
        svm_model_filename = model_training(data, scene_id, shapefile_path,
                                            curr_aux_folder, no_data_value, gamma=None)

        # Run SCF prediction
        FSC_SVM_map_path = SCF_dist_SV(data, scene_id, config, svm_model_filename, Nprocesses=1, overwrite=ow)
        
    
    # post-processing map cleaning
    remove_low_scf(scene_id, data, FSC_SVM_map_path, curr_aux_folder)
    
    # add uncertainty layer
    if unc:
        get_uncertainty(scene_id, config)
        
    # check if there is snow around the glacier
    if classify_glaciers == 'yes':
        
        if snow_around_glacier(wd, scene_id):
     
            snow_mask, ice_mask = get_pixels_ice(scene_id, data, config)
            
            results_glacier = run_snow_ice_classification(
                data=data,
                snow_mask=snow_mask,
                ice_mask=ice_mask,
                output_folder=None,
                max_samples_per_class=10000,
                prediction_mask=None,
            )
            
            mask_raster_with_glacier(scene_id, data, config, results_glacier)

            
        # else:
            
        #     get_blue_ice(wd, data, scene_id)
            

  
            

    print("Process completed. Condition met, and no points found where SCF > 0 and NDSI < 0.")



if __name__ == "__main__":
    print('Running SnowFLAKES')
