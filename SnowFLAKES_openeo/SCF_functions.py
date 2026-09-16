#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:31:29 2024

@author: rbarella
"""

import numpy as np
import os
import geopandas as gpd
import pandas as pd
import rasterio
import pickle
from joblib import Parallel, delayed
from rasterio.features import geometry_mask
from rasterio.transform import rowcol

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

from SnowFLAKES.utilities import (
    load_map,
    save_tif,
    get_sensor,
    select_band_names,
    valid_mask,
    create_folder
)




def build_feature_matrix(all_bands_image, mask, curr_aux_folder):
    
    solar_incidence_angle = load_map(curr_aux_folder, '*solar_incidence_angle.tif')
    hillshade = np.cos(np.deg2rad(solar_incidence_angle))
    
    shadow = load_map(curr_aux_folder, '*shad_idx.tif')
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')
    
    X = all_bands_image[:, mask].T

    hillshade_feat = hillshade[mask].reshape(-1, 1)
    shadow_feat = shadow[mask].reshape(-1, 1)
    diff_B_NIR_feat = diff_B_NIR[mask].reshape(-1, 1)


    # concatenate all features
    return np.hstack((
        X,
        hillshade_feat,
        shadow_feat,
        diff_B_NIR_feat
    ))


def _save_training_samples_csv(shapefile, shapefile_path, data,
                               all_bands_image, all_bands, curr_aux_folder):
    """Save the sampled training features and labels beside the shapefile."""
    # Training points are written by ``training_collection`` in the same CRS
    # and grid as the image.  Reproject here as a safeguard for externally
    # supplied training shapefiles.
    image_crs = getattr(data.rio, "crs", None)
    if image_crs is not None and shapefile.crs is not None and shapefile.crs != image_crs:
        shapefile = shapefile.to_crs(image_crs)

    rows, cols = rowcol(
        data.rio.transform(),
        shapefile.geometry.x.to_numpy(),
        shapefile.geometry.y.to_numpy(),
    )
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_bounds = (
        (rows >= 0) & (rows < all_bands_image.shape[-2]) &
        (cols >= 0) & (cols < all_bands_image.shape[-1])
    )

    rows = rows[in_bounds]
    cols = cols[in_bounds]
    samples = shapefile.iloc[np.flatnonzero(in_bounds)].copy()

    solar_incidence_angle = load_map(curr_aux_folder, '*solar_incidence_angle.tif')
    hillshade = np.cos(np.deg2rad(solar_incidence_angle))
    shadow = load_map(curr_aux_folder, '*shad_idx.tif')
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')

    feature_array = np.hstack((
        all_bands_image[:, rows, cols].T,
        hillshade[rows, cols, None],
        shadow[rows, cols, None],
        diff_B_NIR[rows, cols, None],
    ))
    feature_names = [str(name) for name in all_bands]
    feature_names.extend(["hillshade", "shad_idx", "diffBNIR"])
    training_df = pd.DataFrame(feature_array, columns=feature_names)

    training_df["class"] = samples["value"].map({1: "snow", 2: "snow_free"}).to_numpy()
    illumination_column = next(
        (column for column in ("illum", "illumination") if column in samples.columns),
        None,
    )
    if illumination_column is None:
        training_df["illumination"] = "unknown"
    else:
        training_df["illumination"] = samples[illumination_column].map(
            {1: "sun", 2: "shadow"}
        ).fillna(samples[illumination_column].astype(str)).to_numpy()

    csv_path = os.path.splitext(shapefile_path)[0] + ".csv"
    training_df.to_csv(csv_path, index=False)
    print(f"Training samples saved to {csv_path}")
    return csv_path


   
def model_training(data, scene_id, shapefile_path, curr_aux_folder,
                   no_data_value, gamma=None):
    

    # load bands used for SCF retrieval
    sensor = get_sensor(scene_id)
    all_bands = select_band_names(sensor, 'scf') 
    selected_data = data.sel(band=all_bands)
    
    validMask = valid_mask(data, no_data_value=no_data_value)

    all_bands_image = np.squeeze(selected_data.where(validMask, no_data_value).values)


    gamma_range = np.logspace(-2, 2, 1000)

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
    snow_training = build_feature_matrix(
        all_bands_image,
        mask_snow,
        curr_aux_folder
    )
    
    no_snow_training = build_feature_matrix(
        all_bands_image,
        mask_no_snow,
        curr_aux_folder
    )



    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    class_array = np.concatenate((np.ones(snow_training.shape[0]), np.zeros(no_snow_training.shape[0])), axis=0)

    _save_training_samples_csv(
        shapefile,
        shapefile_path,
        data,
        all_bands_image,
        all_bands,
        curr_aux_folder,
    )

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

    svm_model = {'svmModel': svm, 'normalizer': normalizer, 'classes': class_array,
                 'trainings': training_array, 'SV': svm.support_vectors_}


    svm_model_filename = os.path.join(SCF_folder, 'svm_model.p')
    pickle.dump(svm_model, open(svm_model_filename, "wb"))

    return svm_model_filename



def hyp_disatance(svmModel, svmMatrix):
    return svmModel.decision_function(svmMatrix)



def SCF_dist_SV(data, scene_id, config, svm_model_filename, Nprocesses=8, overwrite=False):
    
    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")

    # Scene's auxiliary folder
    curr_aux_folder = create_folder(scene_folder, "auxiliary")
    
    # No data value
    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)
        
        
    # load bands used for SCF retrieval
    sensor = get_sensor(scene_id)
    all_bands = select_band_names(sensor, 'scf') 
    selected_data = data.sel(band=all_bands)
    
    validMask = valid_mask(data, no_data_value=no_data_value)

    all_bands_image = np.squeeze(selected_data.where(validMask, no_data_value).values)
    
    
    
    # Load DEM
    dem_path = os.path.join(auxiliary_folder, "DEM.tif")

    
    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder, '*Water_Mask.tif')

    
    # Load the SVM model
    svm_model = pickle.load(open(svm_model_filename, 'rb'), encoding='latin1')

    svm = svm_model['svmModel']

    min_score_ns = -1
    max_score_s = 1

    FSC_SVM_map_path = os.path.join(scene_folder, f'{scene_id}_SnowFLAKES.tif')

    # Check if the map file exists and overwrite if specified
    if os.path.exists(FSC_SVM_map_path) and not overwrite:
        print(f"{FSC_SVM_map_path} already exists. Skipping creation.")
        return FSC_SVM_map_path
    else:
        print(f"Saving prediction to {FSC_SVM_map_path}.")

    print('Image classification...\n')
    Image_array_to_classify = build_feature_matrix(
        all_bands_image,
        validMask,
        curr_aux_folder
    )
    

    normalizer = svm_model['normalizer']

    Samples_to_classify = normalizer.transform(Image_array_to_classify)

    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)

    # Calculate the score
    scoreImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(hyp_disatance)(svm, samplesBlocks[i]) for i in range(len(samplesBlocks))
    )

    scoreImage_array = np.concatenate(scoreImage_arrayBlocks, axis=0)
    Score_map = 255 * np.ones(np.shape(validMask)).astype(float)
    Score_map[validMask] = scoreImage_array.flatten()

    scoreImage_array[scoreImage_array < min_score_ns] = min_score_ns
    scoreImage_array[scoreImage_array > max_score_s] = max_score_s

    SCF_Image_array = (scoreImage_array * 50 + 50).astype('uint8')

    # Create the SCF map
    SCF_map = 255 * np.ones(np.shape(validMask))
    SCF_map[validMask] = SCF_Image_array.flatten()


    SCF_map[cloud_mask == 1] = 205
    SCF_map[cloud_mask == 2] = 205
    SCF_map[water_mask == 1] = 210
    SCF_map[water_mask == 255] = 210

    SCF_map[np.logical_not(validMask)] = 255

    # save output tif file
    save_tif(SCF_map, dem_path, FSC_SVM_map_path, dtype=rasterio.uint8)
    

    return FSC_SVM_map_path





def mask_raster_with_glacier(scene_id, data, config, results_glacier):
    

    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")

    # Scene's auxiliary folder
    curr_aux_folder = create_folder(scene_folder, "auxiliary")
    
    # No data value
    no_data_value = config['no_data_value']
    if no_data_value is None or 'nan' in str(no_data_value).lower():
        no_data_value = np.nan
    else:
        no_data_value = float(no_data_value)
    
    validMask = valid_mask(data, no_data_value=no_data_value)


    # Load masks and maps
    glacier_mask = load_map(auxiliary_folder, '*glacier*.tif')
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    fsc_data, FSC_SVM_map_path = load_map(scene_folder, '*SnowFLAKES.tif', return_path=True)
        
    # Define output path
    output_path = os.path.join(scene_folder, f"{scene_id}_SnowFLAKES_GLACIERS.tif")

    # Open the raster
    with rasterio.open(FSC_SVM_map_path) as src:
        meta = src.meta.copy()
        fsc_data = src.read(1)  # Read first band

    
    ice = np.logical_and.reduce((validMask, 
                                 cloud_mask==0, 
                                 glacier_mask == 1, 
                                 results_glacier["classification"] == 0,
                                 fsc_data >0))


    # Apply mask: Set FSC values to NoData where glacier_mask is not 255
    fsc_data[ice] = 215
    # fsc_data[glacier_map == 100] = thematic_map[glacier_map == 100]
    # glacier_map[glacier_map == 1] = 215 
    # glacier_map[glacier_map == 2] = 100 
    # glacier_map[glacier_map == 3] = 0 

    # mask = np.logical_and.reduce([
    #     fsc_data > 0,
    #     fsc_data <= 100,
    #     glacier_map > 0
    # ])
    

    # fsc_data[mask] = glacier_map[mask]

    # Save the modified raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(fsc_data, 1)

    print(f"Modified raster saved at: {output_path}")
    return output_path














