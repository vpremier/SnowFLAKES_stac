#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:07:46 2024

@author: rbarella
"""
import numpy as np
import os
from pathlib import Path

from .utilities import *
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import binary_erosion
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import distance
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from scipy.ndimage import distance_transform_edt
from skimage import exposure
from rasterio.transform import from_origin
from skimage.filters import threshold_otsu


def read_masked_values(data_source, mask, bands=None):
    """
    Reads pixel values corresponding to a logical mask from either
    a multispectral GeoTIFF or a NumPy array stack.

    Parameters
    ----------
    data_source : str or numpy.ndarray
        Path to a GeoTIFF file OR a NumPy array with shape (bands, height, width).
    mask : numpy.ndarray
        2D boolean mask (True where values should be extracted).
    bands : list of int, optional
        List of band indices to read (1-based for GeoTIFF, 0-based for NumPy).
        If None, all bands are used.

    Returns
    -------
    masked_values : numpy.ndarray
        2D array with shape (num_pixels, num_bands).
    """

    masked_values_per_band = []

    # Case 1: GeoTIFF path
    if isinstance(data_source, str):
        with rasterio.open(data_source) as src:

            if bands is None:
                bands = list(range(1, src.count + 1))

            for band in bands:
                data = src.read(band)
                masked_values_per_band.append(data[mask])

    # Case 2: NumPy stack
    elif isinstance(data_source, np.ndarray):

        n_bands = data_source.shape[0]

        if bands is None:
            bands = list(range(n_bands))

        for band in bands:
            data = data_source[band]
            masked_values_per_band.append(data[mask])

    else:
        raise TypeError("data_source must be either a file path or a numpy array")

    masked_values = np.stack(masked_values_per_band, axis=-1)

    return masked_values



def collect_trainings(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, 
                      SVM_folder_name, no_data_mask, bands, total_samples=500):
    
    wd = Path(curr_aux_folder).parent
    
    scf_folder = os.path.join(wd, SVM_folder_name)
    if not os.path.exists(scf_folder):
        os.makedirs(scf_folder)

    sensor = get_sensor(scene_id)

    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    diff_B_NIR_path = glob.glob(os.path.join(curr_aux_folder, '*diffBNIR.tif'))[0]
    shad_idx_path = glob.glob(os.path.join(curr_aux_folder, '*shad_idx.tif'))[0]
    distance_index_path = glob.glob(os.path.join(curr_aux_folder, '*distance.tif'))[0]


    valid_mask = np.logical_not(no_data_mask)

    # Load masks and other necessary data
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    solar_incidence_angle = open_image(solar_incidence_angle_path)[0]
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))

    new_90 = min(np.nanmax(solar_incidence_angle[curr_scene_valid])-1, 90)

    ranges = ((0, 20), (20, 45), (45, 70), (70, new_90), (new_90, 180))
    range_samples = calculate_training_samples(solar_incidence_angle, ranges, total_samples)
    # ranges = ((70, 90))

    # ranges = ((0,20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 180))
    empty = np.zeros(curr_scene_valid.shape, dtype='uint8')

    percentage_per_angles_list = []
    for curr_range, sample_count in range_samples.items():
        print(curr_range)
        print(sample_count)

        # Initialize as empty arrays
        representative_pixels_mask_snow = np.array([])
        representative_pixels_mask_noSnow = np.array([])

        curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0],
                                                                           solar_incidence_angle < curr_range[1]))

        percentage_of_scene_valid = np.sum(curr_angle_valid) / np.sum(curr_scene_valid)

        percentage_per_angles_list.append(percentage_of_scene_valid)

        curr_NDSI = read_masked_values(NDSI_path, curr_angle_valid)
        curr_NDVI = read_masked_values(NDVI_path, curr_angle_valid)
        curr_green = read_masked_values(all_bands_image, curr_angle_valid, bands=[2])
        curr_bands = read_masked_values(all_bands_image, curr_angle_valid)
        curr_diff_B_NIR = read_masked_values(diff_B_NIR_path, curr_angle_valid)
        curr_shad_idx = read_masked_values(shad_idx_path, curr_angle_valid)
        curr_distance_idx = read_masked_values(distance_index_path, curr_angle_valid)

        # SNOW TRAINING
        if curr_range[0] >= new_90:
            # Normalize indices and compute shadow metric
            diff_B_NIR_low_perc, diff_B_NIR_high_perc = np.percentile(curr_diff_B_NIR, [2, 95])
            shad_idx_low_perc, shad_idx_high_perc = np.percentile(curr_shad_idx, [2, 95])
            curr_diff_B_NIR_norm = np.clip(
                (curr_diff_B_NIR - diff_B_NIR_low_perc) / (diff_B_NIR_high_perc - diff_B_NIR_low_perc), 0, 1)
            curr_shad_idx_norm = np.clip((curr_shad_idx - shad_idx_low_perc) / (shad_idx_high_perc - shad_idx_low_perc),
                                         0, 1)
            curr_score_snow_shadow = curr_diff_B_NIR_norm - curr_shad_idx_norm
            threshold_shadow = np.percentile(curr_score_snow_shadow, 95)
            curr_valid_snow_mask_shadow = np.logical_and.reduce(
                (curr_score_snow_shadow >= threshold_shadow, curr_NDSI > 0.7, curr_distance_idx != 255)).flatten()
            if np.sum(curr_valid_snow_mask_shadow) > 10:
                representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask_shadow,
                                                                            sample_count=int(sample_count / 2), k=5,
                                                                            n_closest='auto')
        else:
            # Normalize indices and compute sun metric
            NDSI_low_perc, NDSI_high_perc = np.percentile(curr_NDSI[np.logical_not(np.isnan(curr_NDSI))], [1, 99])
            NDVI_low_perc, NDVI_high_perc = np.percentile(curr_NDVI[np.logical_not(np.isnan(curr_NDVI))], [1, 99])
            green_low_perc, green_high_perc = np.percentile(curr_green, [1, 99])
            curr_NDSI_norm = np.clip((curr_NDSI - NDSI_low_perc) / (NDSI_high_perc - NDVI_low_perc), 0, 1)
            curr_NDVI_norm = np.clip((curr_NDVI - NDVI_low_perc) / (NDVI_high_perc - NDVI_low_perc), 0, 1)
            curr_green_norm = np.clip((curr_green - green_low_perc) / (green_high_perc - green_low_perc), 0, 1)
            curr_score_snow_sun = curr_NDSI_norm - curr_NDVI_norm + curr_green_norm
            threshold = np.percentile(curr_score_snow_sun, 95)
            curr_valid_snow_mask = np.logical_and.reduce(
                (curr_score_snow_sun >= threshold, curr_NDSI > 0.7, curr_distance_idx != 255)).flatten()

            if np.sum(curr_valid_snow_mask) > 10:
                representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask,
                                                                            sample_count=int(sample_count / 2), k=5,
                                                                            n_closest='auto')

        ## NO snow TRAINING
        if curr_range[0] >= new_90:
            threshold_shadow_no_snow = np.percentile(curr_score_snow_shadow, 5)
            curr_valid_no_snow_mask_shadow = (curr_score_snow_shadow <= threshold_shadow_no_snow).flatten()

            if np.sum(curr_valid_no_snow_mask_shadow) > 10:
                representative_pixels_mask_noSnow = get_representative_pixels(curr_bands,
                                                                              curr_valid_no_snow_mask_shadow,
                                                                              sample_count=int(sample_count / 2), k=5,
                                                                              n_closest='auto') * 2
        else:
            curr_valid_no_snow_mask = (curr_NDSI < 0).flatten()

            if np.sum(curr_valid_no_snow_mask) > 10:
                representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask,
                                                                              sample_count=int(sample_count / 2), k=10,
                                                                              n_closest='auto') * 2

        # Check if masks have been assigned; if not, set as zeros
        if representative_pixels_mask_snow.size == 0:
            representative_pixels_mask_snow = np.zeros(curr_angle_valid.sum(), dtype='uint8')
        if representative_pixels_mask_noSnow.size == 0:
            representative_pixels_mask_noSnow = np.zeros(curr_angle_valid.sum(), dtype='uint8')

        representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
        empty[curr_angle_valid] = representative_pixels_mask

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



def thematic_map_classifier(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path,
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

    sensor = get_sensor(scene_id)

    # Define paths for auxiliary files
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    if classify_glaciers == 'yes':
        glacier_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier*.tif'))[0]
    else:
        pass


    # Create valid mask from no_data_mask (True means valid)
    valid_mask = np.logical_not(no_data_mask)

    # Load the image bands using your open_image and define_bands functions.
    bands = define_bands(all_bands_image, valid_mask, sensor)
    # Expected band ordering: blue, red, nir, swir
    blue = bands['BLUE']
    red = bands['RED']
    nir = bands['NIR']
    swir = bands['SWIR']

    # Load indices
    ndsi = open_image(NDSI_path)[0]
    ndvi = open_image(NDVI_path)[0]

    # Load external masks for clouds, water and glaciers
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    if classify_glaciers == 'yes':
        glacier_mask = open_image(glacier_mask_path)[0]
    else:
        glacier_mask = np.zeros_like(water_mask)

    # Compute red/SWIR ratio (avoid division by zero)
    red_swir = red / (swir + 1e-10)

    # Set a fixed NDSI threshold (candidate pixels) and a NDVI threshold to avoid vegetation
    ndsi_threshold = 0.4
    ndvi_threshold = 0.5

    # Build candidate mask: valid pixels with sufficient NDSI and low NDVI.
    candidate_mask = valid_mask & (ndsi > ndsi_threshold) & (ndvi < ndvi_threshold)

    # Compute dynamic red/SWIR threshold using Otsu's method on candidate pixels.
    if np.any(candidate_mask):
        red_swir_dynamic_threshold = threshold_otsu(red_swir[candidate_mask])
    else:
        red_swir_dynamic_threshold = 0.9  # fallback if candidate_mask is empty

    # Swapped condition: Snow if red/SWIR ratio is lower than dynamic threshold, ice if higher.
    snow_mask = candidate_mask

    # Glacier reclassification: only if classify_glaciers == 'yes' and date within glacier season.
    if (classify_glaciers.lower() == 'yes' and
        dt_start_glaciers_month is not None and dt_end_glaciers_month is not None and
        is_month_in_range(date_time.month, dt_start_glaciers_month.month, dt_end_glaciers_month.month)):

        ice_mask = np.logical_and.reduce((candidate_mask, red_swir <= red_swir_dynamic_threshold, glacier_mask == 1))
    else:
        ice_mask = np.zeros_like(snow_mask).astype('bool')



    # Initialize the thematic map with snow free (0)
    thematic_map = np.zeros_like(blue, dtype=np.uint8)
    thematic_map[snow_mask] = 100
    thematic_map[ice_mask] = 215

    # Mark invalid pixels as 255 (no-data, clouds, or water)

    thematic_map[np.logical_not(valid_mask)] = 255
    # Optionally mark cloud and water areas with distinct codes:
    thematic_map[cloud_mask == 2] = 205
    thematic_map[water_mask == 1] = 210

    if np.sum(np.logical_and(thematic_map == 100, glacier_mask == 0)) > np.sum(glacier_mask == 1):
        thematic_map[thematic_map == 215] = 100

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
    valid_pixels = bands_data[valid_mask, :]  # Shape (pixels, bands)

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
        original_indices = np.argwhere(valid_mask)[cluster_indices]
        selected_pixels = original_indices[closest_indices]

        # Set these pixels in the representative mask
        representative_pixels_mask[selected_pixels] = 1

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


# to be updated







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

    # Calculate the proportion of samples for each range
    range_samples = {
        r: int(total_samples * (count / total_pixels)) + 20 if total_pixels > 0 else 0
        for r, count in range_pixel_counts.items()
    }

    return range_samples


def collect_trainings2(curr_acquisition, curr_aux_folder, auxiliary_folder_path, SVM_folder_name, no_data_mask, bands,
                       PCA=False, total_samples=500):
    # TODO: This function is not used --> remove?
    scf_folder = os.path.join(curr_acquisition, SVM_folder_name)
    if not os.path.exists(scf_folder):
        os.makedirs(scf_folder)

    sensor = get_sensor(os.path.basename(curr_acquisition))

    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    diff_B_NIR_path = glob.glob(os.path.join(curr_aux_folder, '*diffBNIR.tif'))[0]
    shad_idx_path = glob.glob(os.path.join(curr_aux_folder, '*shad_idx.tif'))[0]
    shadow_mask_path = glob.glob(os.path.join(curr_aux_folder, '*shadow_mask.tif'))[0]

    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))

    if bands_path == []:
        bands_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
    else:
        bands_path = bands_path[0]

    valid_mask = np.logical_not(no_data_mask)

    # Load masks and other necessary data
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    solar_incidence_angle = open_image(solar_incidence_angle_path)[0]
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))

    ranges = ((0, 20), (20, 45), (45, 70), (70, 90), (90, 180))
    range_samples = calculate_training_samples(solar_incidence_angle, ranges, total_samples)
    # ranges = ((70, 90))

    # ranges = ((0,20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 180))
    empty = np.zeros(curr_scene_valid.shape, dtype='uint8')

    percentage_per_angles_list = []
    for curr_range, sample_count in range_samples.items():
        print(curr_range)
        print(sample_count)

        # Initialize as empty arrays
        representative_pixels_mask_snow = np.array([])
        representative_pixels_mask_noSnow = np.array([])

        curr_angle_valid = np.logical_and(curr_scene_valid, np.logical_and(solar_incidence_angle >= curr_range[0],
                                                                           solar_incidence_angle < curr_range[1]))

        percentage_of_scene_valid = np.sum(curr_angle_valid) / np.sum(curr_scene_valid)

        percentage_per_angles_list.append(percentage_of_scene_valid)

        curr_NDSI = read_masked_values(NDSI_path, curr_angle_valid)
        curr_NDVI = read_masked_values(NDVI_path, curr_angle_valid)
        curr_green = read_masked_values(bands_path, curr_angle_valid, bands=[2])
        curr_bands = read_masked_values(bands_path, curr_angle_valid)
        curr_diff_B_NIR = read_masked_values(diff_B_NIR_path, curr_angle_valid)
        curr_shad_idx = read_masked_values(shad_idx_path, curr_angle_valid)
        curr_shadow_mask = read_masked_values(shadow_mask_path, curr_angle_valid)

        # SNOW TRAINING SHADOW

        # Normalize indices and compute shadow metric
        diff_B_NIR_low_perc, diff_B_NIR_high_perc = np.percentile(curr_diff_B_NIR, [2, 95])
        shad_idx_low_perc, shad_idx_high_perc = np.percentile(curr_shad_idx, [2, 95])
        curr_diff_B_NIR_norm = np.clip(
            (curr_diff_B_NIR - diff_B_NIR_low_perc) / (diff_B_NIR_high_perc - diff_B_NIR_low_perc), 0, 1)
        curr_shad_idx_norm = np.clip((curr_shad_idx - shad_idx_low_perc) / (shad_idx_high_perc - shad_idx_low_perc), 0,
                                     1)
        curr_score_snow_shadow = curr_diff_B_NIR_norm - curr_shad_idx_norm
        threshold_shadow = np.percentile(curr_score_snow_shadow, 95)
        curr_valid_snow_mask_shadow = np.logical_and.reduce(
            (curr_score_snow_shadow >= threshold_shadow, curr_NDSI > 0.7, curr_shadow_mask == 1,
             curr_diff_B_NIR > 0.1)).flatten()
        if np.sum(curr_valid_snow_mask_shadow) > 10:
            representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask_shadow,
                                                                        sample_count=int(sample_count / 3), k=5,
                                                                        n_closest='auto')

        # SNOW TRAINING SUN

        # Normalize indices and compute sun metric
        NDSI_low_perc, NDSI_high_perc = np.percentile(curr_NDSI[np.logical_not(np.isnan(curr_NDSI))], [1, 99])
        NDVI_low_perc, NDVI_high_perc = np.percentile(curr_NDVI[np.logical_not(np.isnan(curr_NDVI))], [1, 99])
        green_low_perc, green_high_perc = np.percentile(curr_green, [1, 99])
        curr_NDSI_norm = np.clip((curr_NDSI - NDSI_low_perc) / (NDSI_high_perc - NDVI_low_perc), 0, 1)
        curr_NDVI_norm = np.clip((curr_NDVI - NDVI_low_perc) / (NDVI_high_perc - NDVI_low_perc), 0, 1)
        curr_green_norm = np.clip((curr_green - green_low_perc) / (green_high_perc - green_low_perc), 0, 1)
        curr_score_snow_sun = curr_NDSI_norm - curr_NDVI_norm + curr_green_norm
        threshold = np.percentile(curr_score_snow_sun, 95)
        curr_valid_snow_mask = np.logical_and.reduce(
            (curr_score_snow_sun >= threshold, curr_NDSI > 0.7, curr_shadow_mask == 0)).flatten()

        if np.sum(curr_valid_snow_mask) > 10:
            representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask,
                                                                        sample_count=int(sample_count / 4), k=5,
                                                                        n_closest='auto')

        ## NO snow TRAINING SHADOW

        threshold_shadow_no_snow = np.percentile(curr_score_snow_shadow, 5)
        curr_valid_no_snow_mask_shadow = np.logical_and.reduce(
            (curr_score_snow_shadow <= threshold_shadow_no_snow, curr_diff_B_NIR < 0.08,
             curr_shadow_mask == 1)).flatten()

        if np.sum(curr_valid_no_snow_mask_shadow) > 10:
            representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask_shadow,
                                                                          sample_count=int(sample_count / 4), k=5,
                                                                          n_closest='auto') * 2

        ## NO snow TRAINING SUN
        curr_valid_no_snow_mask = (curr_NDSI < 0).flatten()

        if np.sum(curr_valid_no_snow_mask) > 10:
            representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask,
                                                                          sample_count=int(sample_count / 4), k=10,
                                                                          n_closest='auto') * 2

        # Check if masks have been assigned; if not, set as zeros
        if representative_pixels_mask_snow.size == 0:
            representative_pixels_mask_snow = np.zeros(curr_angle_valid.sum(), dtype='uint8')
        if representative_pixels_mask_noSnow.size == 0:
            representative_pixels_mask_noSnow = np.zeros(curr_angle_valid.sum(), dtype='uint8')

        representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
        empty[curr_angle_valid] = representative_pixels_mask

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
    svm_folder_path = os.path.join(curr_acquisition, SVM_folder_name)

    plot_valid_pixels_percentage(ranges, percentage_per_angles_list, svm_folder_path)

    shapefile_path = os.path.join(svm_folder_path, 'representative_pixels_for_training_samples.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")

    training_mask_path = os.path.join(svm_folder_path, 'representative_pixels_for_training_samples.tif')

    # Update the profile and save the representative mask
    with rasterio.open(NDSI_path) as src:
        profile = src.profile
    profile.update(dtype='uint8', count=1, compress='lzw', nodata=0)

    with rasterio.open(training_mask_path, 'w', **profile) as dst:
        dst.write(empty, 1)

    return shapefile_path, training_mask_path




def rbf_kernel(X, gamma):
    """
    Computes the RBF (Gaussian) kernel matrix for the dataset X.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data.
    - gamma: float
        The parameter for the RBF kernel, defined as 1 / (2 * sigma^2).

    Returns:
    - K: np.ndarray, shape (n_samples, n_samples)
        The RBF kernel matrix.
    """
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    K = np.exp(-gamma * pairwise_sq_dists)
    return K


def kernel_kmeans(X, n_clusters, gamma, max_iter=100):
    """
    Performs k-means clustering using an RBF kernel instead of Euclidean distance.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data.
    - n_clusters: int
        Number of clusters.
    - gamma: float
        Parameter for the RBF kernel, defined as 1 / (2 * sigma^2).
    - max_iter: int, optional, default=100
        Maximum number of iterations for the algorithm.

    Returns:
    - labels: np.ndarray, shape (n_samples,)
        The cluster labels for each sample.
    - centroids: np.ndarray, shape (n_clusters, n_features)
        The approximated centroids in the original feature space.
    """
    # Step 1: Compute the kernel matrix
    K = rbf_kernel(X, gamma)

    # Step 2: Initialize cluster assignments randomly
    n_samples = X.shape[0]
    labels = np.random.randint(0, n_clusters, n_samples)

    # Step 3: Kernel k-means iteration
    for _ in range(max_iter):
        # Compute distances to centroids in the transformed space
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            cluster_k = K[:, labels == k]
            N_k = np.sum(labels == k)
            if N_k > 0:
                # Compute the distance to the "centroid" in the kernel space
                distances[:, k] = np.diag(K) - 2 * np.sum(cluster_k, axis=1) / N_k + \
                                  np.sum(K[labels == k][:, labels == k]) / (N_k ** 2)

        # Update labels based on minimal distance
        new_labels = np.argmin(distances, axis=1)

        # Check for convergence
        if np.all(labels == new_labels):
            break

        labels = new_labels

    # Step 4: Compute approximate centroids in the original space
    centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

    return labels, centroids


def cop_k_means_vectorized(X, fixed_labels, k=3, max_iter=100, tol=1e-4):
    """
    A simplified, vectorized COP-k-means implementation.
    Fixed points (fixed_labels != -1) are not reassigned.
    Unconstrained points (fixed_labels == -1) are assigned via a vectorized distance computation.

    Parameters:
      X: array of shape (n_samples, n_features)
      fixed_labels: array of shape (n_samples,) with values:
          0 for sure snow (class 1),
         -1 for unconstrained,
          2 for sure no snow (class 3)
      k: number of clusters (should be 3)
      max_iter: maximum iterations

    Returns:
      assignments: cluster assignments (0, 1, or 2) for each sample
      centroids: array of shape (k, n_features)
    """
    n_samples, n_features = X.shape
    assignments = np.full(n_samples, -1, dtype=int)

    # Set fixed assignments:
    assignments[fixed_labels == 0] = 0
    assignments[fixed_labels == 2] = 2

    # Initialize centroids using a seeded approach:
    centroids = np.zeros((k, n_features))
    idx0 = np.flatnonzero(fixed_labels == 0)
    centroids[0] = X[idx0[0]] if len(idx0) > 0 else X[np.random.choice(n_samples)]

    idx2 = np.flatnonzero(fixed_labels == 2)
    centroids[2] = X[idx2[0]] if len(idx2) > 0 else X[np.random.choice(n_samples)]

    idx_unconstrained = np.flatnonzero(fixed_labels == -1)
    centroids[1] = X[np.random.choice(idx_unconstrained)] if len(idx_unconstrained) > 0 else X[
        np.random.choice(n_samples)]

    for iteration in range(max_iter):
        prev_assignments = assignments.copy()
        # Get indices of unconstrained points
        unconstrained_idx = np.flatnonzero(fixed_labels == -1)
        if unconstrained_idx.size > 0:
            # Extract unconstrained points
            U = X[unconstrained_idx]  # shape: (n_unconstrained, n_features)
            # Compute squared Euclidean distances using vectorized dot-product formula:
            # dist(i, j) = ||U[i] - centroids[j]||^2
            #             = sum(U[i]**2) + sum(centroids[j]**2) - 2 * U[i] dot centroids[j]
            U_sq = np.sum(U ** 2, axis=1)[:, np.newaxis]  # shape: (n_unconstrained, 1)
            C_sq = np.sum(centroids ** 2, axis=1)[np.newaxis, :]  # shape: (1, k)
            dists = U_sq + C_sq - 2 * np.dot(U, centroids.T)  # shape: (n_unconstrained, k)

            # Assign each unconstrained point to the closest centroid
            assignments[unconstrained_idx] = np.argmin(dists, axis=1)

        # Update centroids for each cluster (k is small, so a loop is fine)
        for c in range(k):
            indices = np.flatnonzero(assignments == c)
            if len(indices) > 0:
                centroids[c] = np.mean(X[indices], axis=0)

        # Check for convergence
        if np.array_equal(assignments, prev_assignments):
            print(f"Converged at iteration {iteration}")
            break

    return assignments, centroids


def glacier_classifier(curr_acquisition, curr_aux_folder, auxiliary_folder_path, SVM_folder_name, no_data_mask, bands):
    scf_folder = os.path.join(curr_acquisition, SVM_folder_name)
    if not os.path.exists(scf_folder):
        os.makedirs(scf_folder)

    sensor = get_sensor(os.path.basename(curr_acquisition))

    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
    solar_incidence_angle_path = glob.glob(os.path.join(curr_aux_folder, '*solar_incidence_angle.tif'))[0]
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    diff_B_NIR_path = glob.glob(os.path.join(curr_aux_folder, '*diffBNIR.tif'))[0]
    shad_idx_path = glob.glob(os.path.join(curr_aux_folder, '*shad_idx.tif'))[0]
    dem_path = glob.glob(os.path.join(auxiliary_folder_path, '*DEM.tif'))[0]
    band_ratio_glaciers_path = glob.glob(os.path.join(curr_aux_folder, '*Glaciers.tif'))[0]

    glacier_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier*.tif'))[0]

    bands_path = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))

    if bands_path == []:
        bands_path = [f for f in glob.glob(curr_acquisition + os.sep + "PRS*.tif") if 'PCA' not in f][0]
    else:
        bands_path = bands_path[0]

    valid_mask = np.logical_not(no_data_mask)

    # Load masks and other necessary data
    cloud_mask = open_image(path_cloud_mask)[0]
    water_mask = open_image(path_water_mask)[0]
    glacier_mask = open_image(glacier_mask_path)[0]
    solar_incidence_angle = open_image(solar_incidence_angle_path)[0]
    glacier_index = open_image(band_ratio_glaciers_path)[0]
    dem = open_image(dem_path)[0]
    ndsi = open_image(NDSI_path)[0]

    curr_scene_valid_gl = np.logical_not(
        np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask, glacier_mask == 0)))
    curr_scene_valid = np.logical_not(np.logical_or.reduce((cloud_mask == 2, water_mask == 1, no_data_mask)))

    green = define_bands(open_image(bands_path)[0], valid_mask, sensor)['GREEN']

    ### COP K means

    bands = open_image(bands_path)[0]

    sure_snow = np.logical_and.reduce((curr_scene_valid, ndsi > 0.7, green > 0.5))

    sure_no_snow = np.logical_and(curr_scene_valid, ndsi < 0.1)

    Nfeatures, Nrows, Ncols = bands.shape

    # Convert bands to a 2D data matrix where each row is a pixel's feature vector.
    # Since bands is (Nfeatures, Nrows, Ncols), we transpose to (Nrows, Ncols, Nfeatures) and then reshape.
    X = bands.transpose(1, 2, 0).reshape(-1, Nfeatures)
    n_pixels = Nrows * Ncols

    # Build fixed_labels for the full image:
    # - Pixels in sure_snow get label 0.
    # - Pixels in sure_no_snow get label 2.
    # - All others get -1.
    fixed_labels = - np.ones(n_pixels, dtype=int)
    fixed_labels[sure_snow.flatten()] = 0
    fixed_labels[sure_no_snow.flatten()] = 2

    # Run the vectorized COP-k-means on the full dataset.
    assignments, centroids = cop_k_means_vectorized(X, fixed_labels, k=3, max_iter=100)

    # Optionally re-enforce fixed assignments
    # assignments[sure_snow.flatten()] = 100
    # assignments[sure_no_snow.flatten()] = 0
    assignments[assignments == 0] = 100
    assignments[assignments == 1] = 215
    assignments[assignments == 2] = 0
    assignments[assignments == -1] = 255

    # Reshape the assignments to (Nrows, Ncols)
    full_cluster_map = assignments.reshape(Nrows, Ncols)
    # plt.imshow(full_cluster_map, cmap='viridis')

    # Create a classified map over only the valid region (curr_scene_valid_gl)
    valid_class_map = 255 * np.ones((Nrows, Ncols), dtype=int)
    valid_class_map[curr_scene_valid_gl] = full_cluster_map[curr_scene_valid_gl]

    # valid_class_map[valid_class_map == -1] = 2

    # Visualization of the valid region classification
    # plt.figure(figsize=(8, 6))
    # plt.imshow(valid_class_map, cmap='viridis')
    # plt.title("COP-k-means Cluster Assignment (Valid Region)")
    # plt.colorbar(label='Cluster Label')
    # plt.show()

    return valid_class_map


# def thematic_map_classifier(curr_acquisition, curr_aux_folder, auxiliary_folder_path, no_data_mask, SVM_folder_name):
#     """
#     Generate a thematic map using precomputed indices and bands.
#     The output thematic map uses:
#       100 = snow
#       215 = ice
#         0 = snow free
#       255 = invalid/no-data

#     Parameters:
#       curr_acquisition: str, directory containing the current acquisition
#       curr_aux_folder: str, directory with auxiliary files (e.g., cloud mask, indices)
#       auxiliary_folder_path: str, directory for additional auxiliary files (e.g., water mask)
#       no_data_mask: numpy array, boolean mask where True indicates no-data pixels
#       SVM_folder_name: str, name of the folder to store intermediate outputs if needed
#     """

#     # Create folder to store outputs if it doesn't exist
#     thematic_folder = os.path.join(curr_acquisition, SVM_folder_name)
#     if not os.path.exists(thematic_folder):
#         os.makedirs(thematic_folder)
#     sensor = get_sensor(os.path.basename(curr_acquisition))
#     # Define paths for auxiliary files
#     path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
#     path_water_mask = glob.glob(os.path.join(auxiliary_folder_path, '*Water_Mask.tif'))[0]
#     NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
#     NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
#     glacier_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier*.tif'))[0]

#     # Find the main bands file. Look for a VRT first; if not found, fallback to a TIF.
#     bands_path_list = glob.glob(os.path.join(curr_acquisition, '*scf.vrt'))
#     if bands_path_list == []:
#         bands_path = [f for f in glob.glob(os.path.join(curr_acquisition, "PRS*.tif")) if 'PCA' not in f][0]
#     else:
#         bands_path = bands_path_list[0]

#     # Load the image bands using your open_image function.
#     # Assume the returned bands array has shape (Nfeatures, Nrows, Ncols)
#     bands = define_bands(open_image(bands_path)[0], valid_mask, sensor)
#     # For this example, assume the ordering is:
#     # blue = bands[0], red = bands[1], nir = bands[2], swir = bands[3]
#     blue = bands['BLUE']
#     red  = bands['RED']
#     nir  = bands['NIR']
#     swir = bands['SWIR']

#     # Load indices
#     ndsi  = open_image(NDSI_path)[0]
#     ndvi  = open_image(NDVI_path)[0]

#     # Load external masks for clouds and water
#     cloud_mask = open_image(path_cloud_mask)[0]
#     water_mask = open_image(path_water_mask)[0]
#     glacier_mask = open_image(glacier_mask_path)[0]

#     # Create a valid data mask from the no_data_mask (True means valid)
#     valid_mask = np.logical_not(no_data_mask)

#     # Compute red/SWIR ratio (adding a small constant to avoid division by zero)
#     red_swir = red / (swir + 1e-10)

#     # Set classification thresholds (you can fine-tune these values)
#     ndsi_threshold = 0.4      # Only consider pixels with ndsi above this value
#     red_swir_threshold = 0.9  # Differentiate snow (>= threshold) from ice (< threshold)

#     # Build a candidate mask: pixels with sufficient NDSI are considered for snow/ice classification.
#     candidate_mask = ndsi > ndsi_threshold


#     snow_mask = candidate_mask & (red_swir < red_swir_threshold)
#     ice_mask  = candidate_mask & (red_swir >= red_swir_threshold)

#     # Initialize the thematic map with snow free (0)
#     thematic_map = np.zeros_like(blue, dtype=np.uint8)
#     thematic_map[snow_mask] = 100
#     thematic_map[ice_mask]  = 215

#     # Optionally, mark pixels that are invalid (no-data, clouds, or water) as 255
#     invalid_mask = np.logical_or.reduce((np.logical_not(valid_mask),
#                                            cloud_mask == 2,
#                                            water_mask == 1))
#     thematic_map[cloud_mask == 2] = 205
#     thematic_map[water_mask == 1] = 210
#     thematic_map[np.logical_and(glacier_mask == 0, thematic_map == 215)] = 100
#     # Define output path

#     output_path = os.path.join(curr_acquisition, SVM_folder_name, os.path.basename(curr_acquisition) + '_simple_class.tif')
#     # Open the raster
#     with rasterio.open(path_cloud_mask) as src:
#         meta = src.meta.copy()

#     # Save the modified raster
#     with rasterio.open(output_path, 'w', **meta) as dst:
#         dst.write(thematic_map, 1)


#     return thematic_map



