#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:32:49 2024

@author: rbarella
"""

import os
import glob
import pickle
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from training_collection import *
from joblib import Parallel, delayed
import glob
import pickle


def model_training_xgb(curr_acquisition, shapefile_path, XGB_folder_name, perform_pca=False, grid_search=False):
    if perform_pca:
        bands_path = glob.glob(os.path.join(curr_acquisition, '*PCA.tif'))[0]
    else:
        bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
        if bands_path == []:
            bands_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            bands_path = bands_path[0]

    # Load the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Open the bands raster to align with the shapefile
    with rasterio.open(bands_path) as src:
        # Create masks for snow (1) and no-snow (2) points
        mask_snow = geometry_mask(
            [geom for geom in shapefile.geometry[shapefile['value'] == 1]],
            transform=src.transform,
            invert=True,
            out_shape=src.shape
        )
        mask_no_snow = geometry_mask(
            [geom for geom in shapefile.geometry[shapefile['value'] == 2]],
            transform=src.transform,
            invert=True,
            out_shape=src.shape
        )

    # Extract training values using the masks
    snow_training = read_masked_values(bands_path, mask_snow)
    no_snow_training = read_masked_values(bands_path, mask_no_snow)

    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    class_array = np.concatenate((np.ones(snow_training.shape[0]), np.zeros(no_snow_training.shape[0])), axis=0)

    # Normalize features
    normalizer = preprocessing.StandardScaler().fit(training_array)
    Samples_train_normalized = normalizer.transform(training_array)

    if grid_search:
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb = XGBClassifier(eval_metric='logloss')
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(Samples_train_normalized, class_array)
        best_model = grid_search.best_estimator_
        print("Best parameters found: ", grid_search.best_params_)
    else:
        # Train XGBoost model with default parameters
        best_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8
        )
        best_model.fit(Samples_train_normalized, class_array)

    pred = best_model.predict(Samples_train_normalized)

    xgb_model = {
        'xgbModel': best_model,
        'normalizer': normalizer,
        'classes': class_array,
        'trainings': training_array
    }

    # Save the XGBoost model
    xgb_folder = os.path.join(curr_acquisition, XGB_folder_name)
    if not os.path.exists(xgb_folder):
        os.makedirs(xgb_folder)

    xgb_model_filename = os.path.join(xgb_folder, 'xgb_model.p')
    pickle.dump(xgb_model, open(xgb_model_filename, "wb"))

    return xgb_model_filename


def snow_class_XGB(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, xgb_model_filename,
                   Nprocesses=8, overwrite=False, perform_pca=False):
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]

    if perform_pca:
        bands_path = glob.glob(os.path.join(curr_acquisition, '*PCA.tif'))[0]
    else:
        bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
        if bands_path == []:
            bands_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
        else:
            bands_path = bands_path[0]

    scf_folder = os.path.dirname(xgb_model_filename)

    # Load the XGBoost model
    xgb_model = pickle.load(open(xgb_model_filename, 'rb'))

    with rasterio.open(path_cloud_mask) as src:
        cloud_mask = src.read(1)  # Read the cloud mask (first band)

    with rasterio.open(path_water_mask) as src:
        water_mask = src.read(1)  # Read the water mask (first band)

    with rasterio.open(bands_path) as src:
        bands = src.read()
        profile = src.profile

    profile.update(dtype='uint8', count=1, compress='lzw', nodata=255, driver='GTiff')

    valid_mask = np.logical_not(no_data_mask)

    FSC_XGB_map_path = os.path.join(scf_folder, os.path.basename(bands_path)[:-11] + '_xgb_snowMap.tif')

    # Check if the map file exists and overwrite if specified
    if os.path.exists(FSC_XGB_map_path) and not overwrite:
        print(f"{FSC_XGB_map_path} already exists. Skipping creation.")
        return FSC_XGB_map_path

    print('Image classification...\n')
    Image_array_to_classify = bands[:, valid_mask].transpose()
    normalizer = xgb_model['normalizer']
    Samples_to_classify = normalizer.transform(Image_array_to_classify)

    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)

    def classify_block(model, block):
        return model.predict(block)

    # Perform classification
    snowImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(classify_block)(xgb_model['xgbModel'], samplesBlocks[i]) for i in range(len(samplesBlocks))
    )

    snowImage_array = np.concatenate(snowImage_arrayBlocks, axis=0)
    SM_Image_array = (snowImage_array * 100).astype('uint8')  # Scale to thematic map range

    SM_map = 255 * np.ones(np.shape(valid_mask))
    SM_map[valid_mask] = SM_Image_array.flatten()

    SM_map[cloud_mask == 2] = 205
    SM_map[water_mask == 1] = 210
    SM_map[water_mask == 255] = 210

    valid_mask[np.logical_not(valid_mask)] = 255

    # Write the SCF map to a file, overwriting if necessary
    with rasterio.open(FSC_XGB_map_path, 'w', **profile) as dst:
        dst.write(SM_map, 1)

    print(f"SCF map saved to {FSC_XGB_map_path}")
    return FSC_XGB_map_path
