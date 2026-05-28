#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:07:46 2024

@author: rbarella
"""
import numpy as np
import os
import pickle
import glob
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial import distance
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import silhouette_score
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import binary_dilation, binary_erosion
from joblib import Parallel, delayed

from SnowFLAKES.utilities import *



def calculate_training_samples(solar_incidence_angle, ranges, total_samples):
    """
    Calculate the number of training samples for each angle range proportional to the pixel distribution.

    Parameters:
        solar_incidence_angle (np.ndarray): 2D array representing the solar incidence angle map.
        ranges (list of tuple): List of angle ranges (start, end).
        total_samples (int): Total number of training samples to distribute.

    Returns:
        dict: A dictionary with ranges as keys and the number of training samples as values.
    """
    # Flatten the angle map for easier processing
    flattened_map = solar_incidence_angle.flatten()

    # Initialize a dictionary to store the count for each range
    range_pixel_counts = {r: 0 for r in ranges}

    # Count pixels in each range
    for r in ranges:
        range_pixel_counts[r] = np.sum((flattened_map >= r[0]) & (flattened_map < r[1]))

    # Calculate the total number of pixels considered
    total_pixels = sum(range_pixel_counts.values())

    threshold = 0.01  # 1%

    range_samples = {
        r: int(total_samples * (count / total_pixels)) + 20
        if total_pixels > 0 and (count / total_pixels) > threshold
        else 0
        for r, count in range_pixel_counts.items()
    }

    return range_samples



def get_pixels_shadow(diff_B_NIR, shad_idx, NDSI, distance_idx, mask_shadow):
    
    # Compute 2nd and 95th percentiles
    diff_B_NIR_low_perc, diff_B_NIR_high_perc = np.percentile(diff_B_NIR[mask_shadow], [2, 95])
    shad_idx_low_perc, shad_idx_high_perc = np.percentile(shad_idx[mask_shadow], [2, 95])
    
    # Normalize indices 
    diff_B_NIR_norm = np.clip((diff_B_NIR - diff_B_NIR_low_perc) / 
                              (diff_B_NIR_high_perc - diff_B_NIR_low_perc), 0, 1)
    shad_idx_norm = np.clip((shad_idx - shad_idx_low_perc) / 
                            (shad_idx_high_perc - shad_idx_low_perc), 0, 1)
    
    # Compute shadow metric
    score_snow_shadow = diff_B_NIR_norm - shad_idx_norm
    
    # threshold
    threshold_snow = np.percentile(score_snow_shadow[mask_shadow], 99)
    
    snow = np.logical_and.reduce((mask_shadow,
                                  score_snow_shadow >= threshold_snow,
                                  NDSI > 0.9, 
                                  distance_idx != 255))
    
    threshold_no_snow = np.percentile(score_snow_shadow[mask_shadow], 5)
    snowfree = (mask_shadow & (score_snow_shadow <= threshold_no_snow))
    
    return snow, snowfree



def get_pixels_sun(NDSI, NDVI, green, distance_idx, NDWI, mask_sun):
    
    # Compute 1th and 99th percentiles
    NDSI_low_perc, NDSI_high_perc = np.percentile(NDSI[mask_sun], [1, 99])
    NDVI_low_perc, NDVI_high_perc = np.percentile(NDVI[mask_sun], [1, 99])
    green_low_perc, green_high_perc = np.percentile(green[mask_sun], [1, 99])
    
    # Normalize indices 
    NDSI_norm = np.clip((NDSI - NDSI_low_perc) / (NDSI_high_perc - NDSI_low_perc), 0, 1)
    NDVI_norm = np.clip((NDVI - NDVI_low_perc) / (NDVI_high_perc - NDVI_low_perc), 0, 1)
    green_norm = np.clip((green - green_low_perc) / (green_high_perc - green_low_perc), 0, 1)
    
    # Compute sun metric
    score_snow_sun = NDSI_norm - NDVI_norm + green_norm
 
    # get the threshold
    threshold = np.percentile(score_snow_sun[mask_sun], 95)
    
    # conditions of val
    snow = np.logical_and.reduce((mask_sun,
                                  score_snow_sun >= threshold, 
                                  NDSI > 0.7, 
                                  distance_idx != 255,
                                  NDWI < 0.5))
    
    # modificare threshold
    snowfree = (mask_sun & (NDSI < 0))
    
    return snow, snowfree



def collect_trainings(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, 
                      SVM_folder_name, no_data_mask, bands, total_samples=500):
    
    # working directory
    wd = Path(curr_aux_folder).parent
    
    # subdirectory SCF
    scf_folder = wd / SVM_folder_name
    scf_folder.mkdir(exist_ok=True)


    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder_path, '*Water_Mask.tif')
    solar_incidence_angle = load_map(curr_aux_folder, '*solar_incidence_angle.tif')
    glacier_mask = load_map(auxiliary_folder_path, '*glacier*.tif')
    shadow_mask = load_map(curr_aux_folder, '*shadow_mask.tif')
    NDSI = load_map(curr_aux_folder, '*NDSI.tif')
    NDVI = load_map(curr_aux_folder, '*NDVI.tif')
    NDWI = load_map(curr_aux_folder, '*NDWI.tif')
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')
    shad_idx = load_map(curr_aux_folder, '*shad_idx.tif')
    distance_idx = load_map(curr_aux_folder, '*distance.tif')
    green = bands["GREEN"]
    
    NDSI_path = find_path(curr_aux_folder, '*NDSI.tif')
    
    # validity mask: a binary dilation is applied (avoid training collection
    # near water bodies, clouds, etc).. glaciers??
    curr_scene_valid = build_valid_scene(no_data_mask,
                                         cloud_mask == 2,
                                         water_mask == 1,
                                         glacier_mask == 1)
    
    
    # enlarge shadow - sun masks to create a buffer where training collection
    # is avoided
    shadow_mask_eroded = binary_erosion(
        shadow_mask == 1,
        iterations=5
    )

    sun_mask_eroded = binary_erosion(
        shadow_mask == 0,
        iterations=5
    )
    

    # define solar incidence angle ranges
    max_SIA = np.nanmax(solar_incidence_angle[curr_scene_valid])
    
    if max_SIA <= 90: 
        ranges = ((0, 20), (20, 45), (45, 70), (70, max_SIA))
    else:
        ranges = ((0, 20), (20, 45), (45, 70), (70, 90), (90, 180))


    # get a number of training proportional to the area belonging to that range 
    range_samples = calculate_training_samples(solar_incidence_angle, ranges, total_samples)




    empty = np.zeros(curr_scene_valid.shape, dtype='uint8')
    percentage_per_angles_list = []
    
    # collect training for each SIA range 
    for curr_range, sample_count in range_samples.items():
        
        # if curr_range[0] == 0:
        #     break

        
        curr_angle_valid = np.logical_and.reduce((curr_scene_valid, 
                                                  solar_incidence_angle >= curr_range[0],
                                                  solar_incidence_angle < curr_range[1]))
        
        percentage_of_scene_valid = np.sum(curr_angle_valid) / np.sum(curr_scene_valid)
        print(percentage_of_scene_valid)

        percentage_per_angles_list.append(percentage_of_scene_valid)

        # SHADOW --------------------------------------------------------------

        # mask angles and shadow
        mask_shadow = curr_angle_valid & shadow_mask_eroded
                
        if np.any(mask_shadow):
            
            # initialize empty masks
            representative_pixels_mask_snow = np.zeros(empty.shape, dtype='uint8')
            representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype='uint8')
            
            print('Collecting trainings in shadow')
            
            snow_shad, snowfree_shad = get_pixels_shadow(diff_B_NIR, 
                                                         shad_idx, 
                                                         NDSI, 
                                                         distance_idx,
                                                         mask_shadow)
            # erosion
            # Shrink mask by 3 pixels
            # snow_shad_eroded = binary_erosion(
            #     snow_shad,
            #     iterations=3
            # )
            
            
            # # Shrink mask by 3 pixels
            # snowfree_shad_eroded = binary_erosion(
            #     snowfree_shad,
            #     iterations=3
            # )
            
            if np.sum(snow_shad) > 10:
                representative_pixels_mask_snow = get_representative_pixels(all_bands_image, 
                                                                            snow_shad,
                                                                            sample_count=int(sample_count / 2), 
                                                                            k=5,
                                                                            n_closest='auto')
                
            if np.sum(snowfree_shad) > 10:
                representative_pixels_mask_noSnow = get_representative_pixels(all_bands_image,
                                                                              snowfree_shad,
                                                                              sample_count=int(sample_count / 2), 
                                                                              k=5,
                                                                              n_closest='auto') * 2
            # merge the two masks
            representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
            empty[mask_shadow] = representative_pixels_mask[mask_shadow]
            
            print(str(np.sum(representative_pixels_mask_snow.flatten())) + ' SNOW PIXELS')
            print(str(np.sum(representative_pixels_mask_noSnow.flatten() / 2)) + ' NO SNOW PIXELS')
            
            
            
        
        # SUN --------------------------------------------------------------

        # mask angles and sun
        mask_sun = curr_angle_valid & sun_mask_eroded
    
        if np.any(mask_sun):
            
            # initialize empty masks
            representative_pixels_mask_snow = np.zeros(empty.shape, dtype='uint8')
            representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype='uint8')
            
            print('Collecting trainings in sun')

            snow_sun, snowfree_sun = get_pixels_sun(NDSI, 
                                                    NDVI, 
                                                    green, 
                                                    distance_idx,
                                                    NDWI,
                                                    mask_sun)

            
            
            # Shrink mask by 3 pixels
            snow_sun_eroded = binary_erosion(
                snow_sun,
                iterations=1
            )
            
            
            # Shrink mask by 3 pixels
            snowfree_sun_eroded = binary_erosion(
                snowfree_sun,
                iterations=1
            )
    
            if np.sum(snow_sun_eroded) > 10:
                representative_pixels_mask_snow = get_representative_pixels(all_bands_image, 
                                                                            snow_sun_eroded,
                                                                            sample_count=int(sample_count / 2), 
                                                                            k=5,
                                                                            n_closest='auto')
    
            if np.sum(snowfree_sun_eroded) > 10:
                representative_pixels_mask_noSnow = get_representative_pixels(all_bands_image, 
                                                                              snowfree_sun_eroded,
                                                                              sample_count=int(sample_count / 2), 
                                                                              k=10,
                                                                              n_closest='auto') * 2
    
            # merge the two masks
            representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
            empty[mask_sun] = representative_pixels_mask[mask_sun]

            print(str(np.sum(representative_pixels_mask_snow.flatten())) + ' SNOW PIXELS')
            print(str(np.sum(representative_pixels_mask_noSnow.flatten() / 2)) + ' NO SNOW PIXELS')

    # Convert points where result == 1 or 2 to a shapefile
    points = []
    values = []
    with rasterio.open(NDSI_path) as src:
        for row, col in zip(*np.where((empty == 1) | (empty == 2))):
            x, y = src.xy(row, col)
            points.append(Point(x, y))
            values.append(empty[row, col])

    gdf = gpd.GeoDataFrame({"value": values}, geometry=points, crs=src.crs)

    plot_valid_pixels_percentage(ranges, percentage_per_angles_list, scf_folder)

    shapefile_path = os.path.join(scf_folder, 'representative_pixels_for_training_samples.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")

    # training_mask_path = os.path.join(svm_folder_path, 'representative_pixels_for_training_samples.tif')

    # Update the profile and save the representative mask
    with rasterio.open(NDSI_path) as src:
        profile = src.profile
    profile.update(dtype='uint8', count=1, compress='lzw', nodata=0)

    # with rasterio.open(training_mask_path, 'w', **profile) as dst:
    #     dst.write(empty, 1)

    # return shapefile_path , training_mask_path
    return shapefile_path




def glacier_xgboost(model_path, data, no_data_mask, curr_aux_folder, 
                    auxiliary_folder_path, Nprocesses=8):
    
    
    # Load the model
    with open(model_path, 'rb') as model_file:
        svm_dict = pickle.load(model_file)
    xgboost_model = svm_dict['xgboostModel']
    normalizer = svm_dict['normalizer']
    feature_names = svm_dict['feature_names']
    
    
    
    glacier_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier*.tif'))[0]
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]

    glacier_mask = open_image(glacier_mask_path)[0]
    cloud_mask = open_image(path_cloud_mask)[0]
    valid_mask = np.logical_not(no_data_mask)
    
    mask = valid_mask & (cloud_mask==1) & (glacier_mask == 1)

    # Extract valid pixels
    features = np.column_stack([
        np.squeeze(data.sel(band=band).values)[mask]
        for band in feature_names
    ])
    
    
    # Normalize features
    features = np.nan_to_num(features)
    features = normalizer.transform(features)

    # Split features for parallel processing
    feature_blocks = np.array_split(features, Nprocesses)

    # Classify in parallel using XGBoost
    print("Starting XGBoost classification...")
    def classify_block(block):
        return xgboost_model.predict(block)
    
    predictions_blocks = Parallel(n_jobs=Nprocesses, verbose=10)(
        delayed(classify_block)(block) for block in feature_blocks
    )
    predictions = np.concatenate(predictions_blocks) + 1  # Adjust class indices
    
    # Create the output raster
    class_map = np.zeros((data.sizes['y'], data.sizes['x']), dtype='uint8')
    class_map[mask] = predictions
    
    return class_map
        
        
    
    
def glacier_classifier(scene_id, data, no_data_mask, curr_aux_folder, auxiliary_folder_path):
    
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    glacier_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier*.tif'))[0]

    
    sensor = get_sensor(scene_id)

    # Create valid mask from no_data_mask (True means valid)
    valid_mask = np.logical_not(no_data_mask)
    cloud_mask = open_image(path_cloud_mask)[0]
    glacier_mask = open_image(glacier_mask_path)[0]

    # Load the image bands using your open_image and define_bands functions.
    bands = define_bands(data, valid_mask, sensor)
    
    # Expected band ordering: blue, red, nir, swir
    green = bands['GREEN']


    nir = bands['NIR']
    swir = bands['SWIR']
    
    # Load indices
    ndsi = open_image(NDSI_path)[0]
    ndvi = open_image(NDVI_path)[0]
    

    # NSIR
    nsir = nir * nir/swir
    
    # NDWI
    ndwi = (green - nir)/(green + nir)
    
    # Select NDSI > 0.7
    ndwi[(ndsi>=0.7) & (cloud_mask==1)]

    
    nsir_vals = nsir[((ndsi >= 0.7) & (cloud_mask == 1) & (glacier_mask == 1))]

    glacier_map = np.zeros_like(nir, dtype=np.uint8)

    try:
        nsir_threshold = threshold_otsu(nsir_vals)
        
        snow = (
            (ndsi >= 0.7) &
            (cloud_mask == 1) &
            (glacier_mask == 1) &
            (nsir >= nsir_threshold) &
            (ndwi <= 0.1)
        )
        
        ice = (
            (ndsi >= 0.7) &
            (cloud_mask == 1) &
            (glacier_mask == 1) &
            ((nsir < nsir_threshold) |
            (ndwi > 0.1))
        )
        

        
    except:
        candidate_mask = valid_mask & (ndsi > 0.4) & (ndvi < 0.5)
        
        if np.any(candidate_mask):
            red = bands['RED']
            red_swir = red / (swir + 1e-10)
    
            red_swir_dynamic_threshold = threshold_otsu(red_swir[candidate_mask])
        else:
            red_swir_dynamic_threshold = 0.9  # fallback if candidate_mask is empty
            
        ice = np.logical_and.reduce((candidate_mask, red_swir <= red_swir_dynamic_threshold, glacier_mask == 1))
        snow = np.logical_and.reduce((candidate_mask, red_swir > red_swir_dynamic_threshold, glacier_mask == 1))

    glacier_map[snow] = 100
    glacier_map[ice] = 215
    
    return glacier_map
    
    
    
def thematic_map_classifier(scene_id, data, curr_aux_folder, auxiliary_folder_path,
                            no_data_mask, SVM_folder_name, classify_glaciers,
                            date_time, dt_start_glaciers_month, dt_end_glaciers_month):
    """
    Generate a thematic map using precomputed indices and bands.
    The output thematic map uses:
      100 = snow
      215 = ice
        0 = snow free
      205 = clouds (optional)
      210 = water (optional)
      255 = invalid/no-data

    Parameters:
      curr_acquisition: str, directory containing the current acquisition
      curr_aux_folder: str, directory with auxiliary files (e.g., cloud mask, indices)
      auxiliary_folder_path: str, directory for additional auxiliary files (e.g., water mask, glacier mask)
      no_data_mask: numpy array, boolean mask where True indicates no-data pixels
      SVM_folder_name: str, name of the folder to store intermediate outputs if needed
      classify_glaciers: str, if 'yes', then glacier classification will be applied
      date_time: datetime, acquisition date and time
      dt_start_glaciers_month: datetime, start month for glacier classification
      dt_end_glaciers_month: datetime, end month for glacier classification
    """

    wd = Path(curr_aux_folder).parent

    # Create folder to store outputs if it doesn't exist
    thematic_folder = os.path.join(wd, SVM_folder_name)
    if not os.path.exists(thematic_folder):
        os.makedirs(thematic_folder)


    # Define paths for auxiliary files
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]

    
    # Load external masks for clouds, water and glaciers
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    valid_mask = np.logical_not(no_data_mask)

    
    # Set a fixed NDSI threshold (candidate pixels) and a NDVI threshold to avoid vegetation
    ndsi_threshold = 0.4
    ndvi_threshold = 0.5
    
    # Load indices
    ndsi = open_image(NDSI_path)[0]
    ndvi = open_image(NDVI_path)[0]
    
    # Mark invalid pixels as 255 (no-data, clouds, or water)
    thematic_map = np.zeros_like(ndsi, dtype=np.uint8)

    

    # Build candidate mask: valid pixels with sufficient NDSI and low NDVI.
    snow_mask = valid_mask & (ndsi > ndsi_threshold) & (ndvi < ndvi_threshold)
    thematic_map[snow_mask] = 100

    
    # Glacier reclassification: only if classify_glaciers == 'yes' and date within glacier season.
    if (classify_glaciers.lower() == 'yes' and
        dt_start_glaciers_month is not None and dt_end_glaciers_month is not None and
        is_month_in_range(date_time.month, dt_start_glaciers_month.month, dt_end_glaciers_month.month)):
        
        glacier_map = glacier_classifier(scene_id, data, no_data_mask, 
                                         curr_aux_folder, 
                                         auxiliary_folder_path)
        
        thematic_map[glacier_map == 100] = 100
        thematic_map[glacier_map == 215] = 215


    thematic_map[np.logical_not(valid_mask)] = 255
    # Optionally mark cloud and water areas with distinct codes:
    thematic_map[cloud_mask == 2] = 205
    thematic_map[water_mask == 1] = 210
    
    
    # if np.sum(np.logical_and(thematic_map == 100, glacier_mask == 0)) > np.sum(glacier_mask == 1):
    #     thematic_map[thematic_map == 215] = 100

    # Define output path
    output_path = os.path.join(wd, SVM_folder_name,f'{scene_id}_simple_class.tif')

    # Open one of the auxiliary rasters (e.g., cloud mask) to copy metadata
    with rasterio.open(path_cloud_mask) as src:
        meta = src.meta.copy()
    meta.update(dtype=rasterio.uint8, count=1)

    # Save the modified raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(thematic_map, 1)

    return output_path



def get_representative_pixels(bands_data, valid_mask, sample_count=50, k='auto', n_closest='auto'):
    """
    Selects representative "no snow" pixels by clustering and distance to cluster centroids.
    Saves the output as a raster.

    Parameters
    ----------
    bands_data : numpy.ndarray
        3D array (bands, height, width) containing spectral data for each band.
    valid_mask : numpy.ndarray
        2D mask of valid pixels for selection.
    k : int, optional
        Number of clusters for K-means, by default 5.
    n_closest : int, optional
        Number of closest pixels to each centroid to select, by default 5.

    Returns
    -------
    representative_pixels_mask : numpy.ndarray
        2D mask with representative pixels marked as 1.
    """
    # Extract "valid" pixels for clustering
    valid_pixels = bands_data[:, valid_mask].T   # (pixels, bands)
    coords = np.argwhere(valid_mask)
    # valid_pixels = bands_data[valid_mask, :]  # Shape (pixels, bands)

    # Normalize the valid pixels
    scaler = StandardScaler()
    normalized_pixels = scaler.fit_transform(valid_pixels)

    # find optimal K
    if k == 'auto':
        k = find_optimal_k(normalized_pixels, max_k=10, method="elbow")
    if n_closest == 'auto':
        n_closest = int(sample_count / k)

    # Perform K-means clustering on "no snow" pixels
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(normalized_pixels)

    # Get cluster centroids and labels
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Initialize an empty mask for representative pixels
    representative_pixels_mask = np.zeros(valid_mask.shape, dtype='uint8')

    # Find the n_closest pixels to each centroid
    for cluster_idx in range(k):
        # Select pixels in the current cluster
        cluster_indices = np.where(labels == cluster_idx)[0]
        cluster_pixels = normalized_pixels[cluster_indices]

        # Compute distances to the centroid for these pixels
        distances = distance.cdist(cluster_pixels, [centroids[cluster_idx]], 'euclidean').flatten()

        # Get the indices of the n_closest pixels in the cluster
        closest_indices = np.argsort(distances)[:n_closest]

        # Map the closest indices back to the original image coordinates
        # original_indices = np.argwhere(valid_mask)[cluster_indices]
        # selected_pixels = original_indices[closest_indices]
        selected_pixels = coords[cluster_indices][closest_indices]

        # Set these pixels in the representative mask
        # representative_pixels_mask[selected_pixels] = 1
        representative_pixels_mask[selected_pixels[:, 0], selected_pixels[:, 1]] = 1
        
    return representative_pixels_mask



def find_optimal_k(data, max_k=10, method="elbow", random_state=42):
    """
    Find the optimal number of clusters using the Elbow or Silhouette method.

    Parameters:
    - data (array-like): The dataset to cluster.
    - max_k (int): The maximum number of clusters to evaluate.
    - method (str): "elbow" for WCSS-based elbow method or "silhouette" for silhouette score.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - int: The optimal number of clusters.
    """
    wcss = []  # Within-Cluster Sum of Squares
    silhouette_scores = []  # Silhouette Scores
    k_values = range(2, max_k + 1)  # Start from 2 clusters for silhouette

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    if method == "elbow":
        # Calculate second derivative to find the "elbow"
        wcss_diff = np.diff(wcss)
        wcss_diff2 = np.diff(wcss_diff)
        optimal_k = k_values[np.argmin(wcss_diff2) + 1]  # Offset for the diff
    elif method == "silhouette":
        # Choose k with the highest silhouette score
        optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        raise ValueError("Invalid method. Choose 'elbow' or 'silhouette'.")

    return optimal_k





def plot_valid_pixels_percentage(ranges, percentage_per_angles_list, svm_folder_path):
    """
    Plots the percentage of valid pixels per angle range and saves the plot as a PNG file.

    Parameters:
    - ranges (tuple of tuples): Angle ranges for the x-axis.
    - percentage_per_angles_list (list): Percentage values corresponding to the ranges.
    - svm_folder_path (str): Directory to save the plot.
    """
    # Ensure ranges and percentage lists match
    if len(ranges) != len(percentage_per_angles_list):
        raise ValueError("Length of ranges and percentage_per_angles_list must match.")

    # Create the bar plot
    x_labels = [f"{r[0]}-{r[1]}" for r in ranges]
    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, percentage_per_angles_list, color='skyblue')

    # Add title and labels
    plt.title("Percentage of Valid Pixels per Solar Incidence Angle Range", fontsize=14)
    plt.xlabel("Angle Ranges (degrees)", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    output_path = os.path.join(svm_folder_path, 'valid_pixels_per_angle.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to avoid display issues in non-interactive environments
    print(f"Plot saved to: {output_path}")
    
    















