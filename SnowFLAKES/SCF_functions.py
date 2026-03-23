#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:31:29 2024

@author: rbarella
"""

import numpy as np
import os
import geopandas as gpd
import glob
import rasterio
import pandas as pd
import pickle
from joblib import Parallel, delayed
from rasterio.features import geometry_mask
import rioxarray

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

from SnowFLAKES.training_collection import *
from SnowFLAKES.utilities import *




def model_training(scene_id, all_bands_image, data, shapefile_path, gamma=None):
    gamma_range = np.logspace(-2, 2, 100)

    # Load the shapefile
    shapefile = gpd.read_file(shapefile_path)
    SCF_folder = os.path.dirname(shapefile_path)

    # Create a mask with the same dimensions as the raster, setting snow (1) and no-snow (2) points
    mask_snow = geometry_mask([geom for geom in shapefile.geometry[shapefile['value'] == 1]],
                              transform=data.rio.transform(),
                              invert=True,
                              out_shape=(data.sizes["y"], data.sizes["x"]))

    mask_no_snow = geometry_mask([geom for geom in shapefile.geometry[shapefile['value'] == 2]],
                                 transform=data.rio.transform(),
                                 invert=True,
                                 out_shape=(data.sizes["y"], data.sizes["x"]))

    # Extract training values using the masks
    snow_training = read_masked_values(all_bands_image, mask_snow)
    no_snow_training = read_masked_values(all_bands_image, mask_no_snow)

    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    class_array = np.concatenate((np.ones(snow_training.shape[0]), np.zeros(no_snow_training.shape[0])), axis=0)

    # Rescale: standardization between 0 and 1
    normalizer = preprocessing.StandardScaler().fit(training_array)
    Samples_train_normalized = normalizer.transform(training_array)

    if gamma == None:
        # Gamma selection by examining the kernel
        std_list = []
        for curr_gamma in gamma_range:
            rbf = rbf_kernel(Samples_train_normalized, Samples_train_normalized, gamma=curr_gamma)
            std_list.append(rbf.std())

        idx_max_std = np.argmax(std_list)
        best_gamma = gamma_range[idx_max_std]

        print('The best Gamma is: ' + str(best_gamma))

    else:
        best_gamma = gamma

    svm = SVC(C=2000000, kernel='rbf', gamma=best_gamma, probability=False,
              decision_function_shape='ovo', cache_size=8000)

    svm.fit(Samples_train_normalized, class_array)
    pred = svm.predict(Samples_train_normalized)

    svm_model = {'svmModel': svm, 'normalizer': normalizer, 'classes': class_array,
                 'trainings': training_array, 'SV': svm.support_vectors_}


    svm_model_filename = os.path.join(SCF_folder, 'svm_model.p')
    pickle.dump(svm_model, open(svm_model_filename, "wb"))

    return svm_model_filename



def hyp_disatance(svmModel, svmMatrix):
    return svmModel.decision_function(svmMatrix)



def SCF_dist_SV(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, no_data_mask, svm_model_filename,
                Nprocesses=8, overwrite=False):
    
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    diff_B_NIR_path = glob.glob(os.path.join(curr_aux_folder, '*diffBNIR.tif'))[0]
    shadow_mask_path = glob.glob(os.path.join(curr_aux_folder, '*shadow_mask.tif'))[0]
    distance_index_path = glob.glob(os.path.join(curr_aux_folder, '*distance.tif'))[0]


    scf_folder = os.path.dirname(svm_model_filename)

    # Load the SVM model
    svm_model = pickle.load(open(svm_model_filename, 'rb'), encoding='latin1')

    with rasterio.open(path_cloud_mask) as src:
        cloud_mask = src.read(1)  # Read the cloud mask (first band)

    with rasterio.open(path_water_mask) as src:
        water_mask = src.read(1)  # Read the water mask (first band)

    with rasterio.open(shadow_mask_path) as src:
        shadow_mask = src.read(1)  # Read the shadow mask (first band)

    with rasterio.open(diff_B_NIR_path) as src:
        diff_B_NIR = src.read(1)
        
    with rasterio.open(distance_index_path) as src:
        dist_idx = src.read(1)  # Reading first band
        profile = src.profile


    profile.update(dtype='uint8', count=1, compress='lzw', nodata=255, driver='GTiff')

    SV = svm_model['SV']
    svm = svm_model['svmModel']

    min_score_ns = -1
    max_score_s = 1

    valid_mask = np.logical_not(no_data_mask)
    FSC_SVM_map_path = os.path.join(scf_folder, f'{scene_id}_SnowFLAKES.tif')

    # Check if the map file exists and overwrite if specified
    if os.path.exists(FSC_SVM_map_path) and not overwrite:
        print(f"{FSC_SVM_map_path} already exists. Skipping creation.")
        return FSC_SVM_map_path
    else:
        print(f"Saving prediction to {FSC_SVM_map_path}.")

    print('Image classification...\n')
    Image_array_to_classify = all_bands_image[:, valid_mask].transpose()
    normalizer = svm_model['normalizer']

    Samples_to_classify = normalizer.transform(Image_array_to_classify)

    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)

    # Calculate the score
    scoreImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(hyp_disatance)(svm, samplesBlocks[i]) for i in range(len(samplesBlocks))
    )

    scoreImage_array = np.concatenate(scoreImage_arrayBlocks, axis=0)
    Score_map = 255 * np.ones(np.shape(valid_mask)).astype(float)
    Score_map[valid_mask] = scoreImage_array.flatten()

    scoreImage_array[scoreImage_array < min_score_ns] = min_score_ns
    scoreImage_array[scoreImage_array > max_score_s] = max_score_s

    SCF_Image_array = (scoreImage_array * 50 + 50).astype('uint8')

    # Create the SCF map
    SCF_map = 255 * np.ones(np.shape(valid_mask))
    SCF_map[valid_mask] = SCF_Image_array.flatten()

    # scf correction based on diff B NIR and shadow mask

    pixels_to_correct = np.logical_and.reduce(
        (valid_mask, diff_B_NIR > 0, diff_B_NIR < 0.06, shadow_mask == 1, SCF_map > 0, SCF_map < 50))

    SCF_map[np.logical_and(valid_mask, SCF_map < 10)] = 0

    SCF_map[pixels_to_correct] = 0
    SCF_map[cloud_mask == 2] = 205
    SCF_map[water_mask == 1] = 210
    SCF_map[water_mask == 255] = 210
    SCF_map[np.logical_and.reduce((SCF_map > 0, SCF_map <= 100, dist_idx == 255))] = 0

    valid_mask[np.logical_not(valid_mask)] = 255

    # Write the SCF map to a file, overwriting if necessary
    with rasterio.open(FSC_SVM_map_path, 'w', **profile) as dst:
        dst.write(SCF_map, 1)

    print(f"SCF map saved to {FSC_SVM_map_path}")
    return FSC_SVM_map_path



def glaciers_svm(svmModel, svmMatrix):
    return svmModel.predict(svmMatrix)



def check_scf_results(scene_id, all_bands_image, FSC_SVM_map_path, shapefile_path, curr_aux_folder, k=5, n_closest=5):
    # Define paths for NDSI and bands data
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]



    # Load the SCF map and NDSI map
    with rasterio.open(FSC_SVM_map_path) as scf_src:
        scf_data = scf_src.read(1)  # Reading first band

    with rasterio.open(NDSI_path) as ndsi_src:
        ndsi_data = ndsi_src.read(1)  # Reading first band

    # Identify points where SCF > 0 and NDSI < 0
    valid_mask = (scf_data > 0) & (scf_data <= 100) & (ndsi_data < 0)

    # Check if there are any valid points; exit if none
    if np.sum(valid_mask) == 0:
        print("No valid points found; exiting function.")
        return shapefile_path  # Optionally return the original shapefile path without modification

    print(np.sum(valid_mask))


    # Use the representative pixel selection function to get new training samples
    transposed_bands = all_bands_image.reshape(all_bands_image.shape[0], -1).T
    representative_pixels_mask = get_representative_pixels(transposed_bands,
                                                           valid_mask.flatten(),
                                                           k=min(k, np.sum(valid_mask.flatten())),
                                                           n_closest=n_closest).reshape(scf_data.shape)

    # Load the original shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Convert the representative pixels mask to points in the shapefile's CRS
    new_samples = []
    for row, col in np.argwhere(representative_pixels_mask == 1):
        x, y = scf_src.xy(row, col)  # Get the coordinates
        new_samples.append({'geometry': gpd.points_from_xy([x], [y])[0], 'value': 2})  # 2 for "no snow" label

    # Add new "no snow" samples to the shapefile's GeoDataFrame
    new_samples_gdf = gpd.GeoDataFrame(new_samples, crs=shapefile.crs)
    updated_shapefile = gpd.GeoDataFrame(pd.concat([shapefile, new_samples_gdf], ignore_index=True))

    print(f"Additional rows in updated shapefile: {len(updated_shapefile) - len(shapefile)}")

    # Save the updated shapefile
    updated_shapefile.to_file(shapefile_path)

    return shapefile_path



def mask_raster_with_glacier(glacier_map, FSC_SVM_map_path, thematic_map_path, auxiliary_folder_path):
    # Define output path
    output_path = FSC_SVM_map_path.replace('.tif', '_GLACIERS.tif')

    # Open the raster
    with rasterio.open(FSC_SVM_map_path) as src:
        meta = src.meta.copy()
        fsc_data = src.read(1)  # Read first band

    # Open the raster
    with rasterio.open(thematic_map_path) as src:
        thematic_map = src.read(1)


    # Apply mask: Set FSC values to NoData where glacier_mask is not 255
    fsc_data[glacier_map == 215] = thematic_map[glacier_map == 215]
    fsc_data[glacier_map == 100] = thematic_map[glacier_map == 100]


    # Save the modified raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(fsc_data, 1)

    print(f"Modified raster saved at: {output_path}")
    return output_path





# to be updated : not used in the current version!!


def glaciers_classifier(FSC_SVM_map_path, auxiliary_folder_path, glaciers_model_svm, curr_acquisition, Nprocesses=8,
                        overwrite=False):
    glaciers_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier_mask.tif'))[0]

    emisphere = get_hemisphere(FSC_SVM_map_path)

    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))[0]

    with rasterio.open(bands_path) as src:
        bands = src.read()

    with rasterio.open(glaciers_mask_path) as src:
        glaciers_mask = src.read(1)  # Read the cloud mask (first band)

    with rasterio.open(FSC_SVM_map_path) as src:
        SCF_map = src.read(1)  # Read the cloud mask (first band)
        profile = src.profile

    # Load the SVM model
    svm_dict = pickle.load(open(glaciers_model_svm, 'rb'), encoding='latin1')

    valid_mask_glaciers = np.logical_and(glaciers_mask == 1, SCF_map <= 100)

    scf_folder = os.path.dirname(FSC_SVM_map_path)
    FSC_glaciers_SVM_map_path = os.path.join(scf_folder, os.path.basename(bands_path)[:-11] + '_SnowFLAKES_GL.tif')

    # Check if the map file exists and overwrite if specified
    if os.path.exists(FSC_glaciers_SVM_map_path) and not overwrite:
        print(f"{FSC_glaciers_SVM_map_path} already exists. Skipping creation.")
        return FSC_glaciers_SVM_map_path

    print('Image classification...\n')

    Image_array_to_classify = bands[:, valid_mask_glaciers].transpose()
    normalizer = svm_dict['normalizer']
    Samples_to_classify = normalizer.transform(Image_array_to_classify)

    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)

    # Calculate the score
    glImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(glaciers_svm)(svm_dict['svmModel'], samplesBlocks[i]) for i in range(len(samplesBlocks))
    )

    glImage_array = np.concatenate(glImage_arrayBlocks, axis=0)

    glImage_array[glImage_array == 1] = 100
    glImage_array[glImage_array == 2] = 215
    glImage_array[glImage_array == 3] = 0

    SCF_map[valid_mask_glaciers] = glImage_array
    SCF_map[valid_mask_glaciers] = glImage_array

    # Write the SCF map to a file, overwriting if necessary
    with rasterio.open(FSC_glaciers_SVM_map_path, 'w', **profile) as dst:
        dst.write(SCF_map, 1)

    print(f"SCF map saved to {FSC_glaciers_SVM_map_path}")
    return FSC_glaciers_SVM_map_path
























