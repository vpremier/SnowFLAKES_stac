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
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import silhouette_score
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

from SnowFLAKES.utilities import (
    load_map,
    open_image,
    build_valid_scene,
    get_sensor,
    define_bands,
    scene_valid_mask,
    scene_no_data_value,
    create_folder,
    define_datetime
)

from SnowFLAKES.auxiliary_folder_population import get_altitude_azimuth


from SnowFLAKES.fit_distribution import fit_distribution_and_median



def plot_trainings(training_stats, pixel_stats, outfolder):
    

    df_train = pd.DataFrame(training_stats)
    df_pixels = pd.DataFrame(pixel_stats)


    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_labels = []
    
    for _, row in df_train.iterrows():
        x_labels.append(
            f"{row['angle_range']}\n{row['illumination']}"
        )
    
    x = np.arange(len(df_train))
    width = 0.4
    
    ax.bar(
        x - width/2,
        df_train["snow_train"],
        width,
        label="Snow"
    )
    
    ax.bar(
        x + width/2,
        df_train["nosnow_train"],
        width,
        label="Snow-free"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel("Training samples")
    ax.set_title("Selected training samples per angle range")
    ax.legend()
    
    # Save the plot
    output_path = os.path.join(outfolder, 'valid_trainings_per_angle.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to avoid display issues in non-interactive environments
    print(f"Plot saved to: {output_path}")

    ############################################
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_labels = []
    
    for _, row in df_pixels.iterrows():
        x_labels.append(
            f"{row['angle_range']}\n{row['illumination']}"
        )
    
    x = np.arange(len(df_pixels))
    width = 0.4
    
    ax.bar(
        x - width/2,
        df_pixels["pixels"],
        width
    )
    
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel("Pixels")
    ax.set_title("Available pixels per angle range")
    ax.legend()
    
    
    # Save the plot
    output_path = os.path.join(outfolder, 'valid_pixels_per_angle.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to avoid display issues in non-interactive environments
    print(f"Plot saved to: {output_path}")



def save_histogram(
    data,
    output_path,
    bins=50,
    xlabel="Value",
    ylabel="Count",
    title=None,
    density=False,
    alpha=0.6
):
    """
    Save one or more histograms.

    Parameters
    ----------
    data : dict
        Dictionary {label: values}. For example:
        {
            "Snow": green[snow_mask],
            "No snow": green[nosnow_mask]
        }

    output_path : str
        Output image path.

    bins : int
        Number of histogram bins.

    xlabel : str
        X-axis label.

    ylabel : str
        Y-axis label.

    title : str or None
        Figure title.

    density : bool
        Plot probability density instead of counts.

    alpha : float
        Histogram transparency.
    """


    plt.figure()
    plt.hist(
        data,
        bins=bins,
        alpha=alpha
    )

    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else ylabel)

    if title is not None:
        plt.title(title)

    if len(data) > 1:
        plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
    
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

    total_pixels = sum(range_pixel_counts.values())
    range_samples = {r: 0 for r in ranges}
    if total_pixels == 0 or total_samples <= 0:
        return range_samples

    non_empty = [r for r in ranges if range_pixel_counts[r] > 0]
    raw = {
        r: total_samples * range_pixel_counts[r] / total_pixels
        for r in non_empty
    }
    for r in non_empty:
        range_samples[r] = int(np.floor(raw[r]))
    remainder = int(total_samples - sum(range_samples.values()))
    for r in sorted(non_empty, key=lambda item: raw[item] - np.floor(raw[item]), reverse=True)[:remainder]:
        range_samples[r] += 1
    return range_samples



def define_threshold(feature,
                     mask,
                     feature_name,
                     outfolder,
                     threshold=(0.08, 0.12),
                     tolerance=0.5):
    """
    Parameters
    ----------
    feature : ndarray
    mask : ndarray(bool)
    threshold : tuple
        Default Gaussian means (low, high)
    tolerance : float
        Relative difference allowed before replacing defaults.
        0.15 = 15%
    """

    values = np.asarray(feature[mask], dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        
        return np.asarray(threshold)
        
    
    values = values.reshape(-1, 1)


    
    values_df = pd.DataFrame({
        feature_name: values.flatten()
    })
    
    values_df.to_csv(os.path.join(outfolder, f"{feature_name}_values.csv"),
        index=False
    )


    # -------------------------
    # Fit two-component GMM
    # -------------------------
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        means_init=np.array(threshold).reshape(-1, 1),
        max_iter=1000,
        tol = 1e-4
    )

    try:
        gmm.fit(values)
    except ValueError:
        return np.asarray(threshold)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    # Sort from low to high
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    weights = weights[order]

    default = np.asarray(threshold)

    rel_diff = np.abs(means - default) / default

    final_means = default.copy()

    for i in range(2):
        if rel_diff[i] <= tolerance:
            final_means[i] = means[i]

    print(f"Default means : {default}")
    print(f"GMM means     : {means}")
    print(f"Relative diff : {100*rel_diff}%")
    print(f"Using means   : {final_means}")

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(7,4))

    counts, bins, _ = plt.hist(
        values,
        bins='auto',
        density=True,
        alpha=0.5,
        color='lightgray',
        edgecolor='k'
    )

    x = np.linspace(values.min(), values.max(), 500)

    for mu, sigma, w in zip(means, stds, weights):
        pdf = (
            w
            * 1/(sigma*np.sqrt(2*np.pi))
            * np.exp(-(x-mu)**2/(2*sigma**2))
        )
        plt.plot(x, pdf, lw=2)

    plt.axvline(final_means[0], color='r', ls='--',
                label=f'Low={final_means[0]:.3f}')
    plt.axvline(final_means[1], color='b', ls='--',
                label=f'High={final_means[1]:.3f}')

    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.title("Two-component Gaussian Mixture")
    plt.legend()

    plt.savefig(
        os.path.join(outfolder, feature_name + "_histogram.png"),
        dpi=300,
        bbox_inches="tight"
    )

    # The histogram is a diagnostic file; do not open an interactive window
    # when SnowFLAKES is run from the command line.
    plt.close()

    return final_means

    
    
def get_pixels_shadow(bands, curr_aux_folder, curr_scene_valid, mask_shadow):
    green = bands["GREEN"]
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')
    
    
    shadow_training_mask = np.asarray(mask_shadow, dtype=bool)
    valid_shadow_values = green[shadow_training_mask]
    if valid_shadow_values.size == 0:
        return np.zeros(green.shape, dtype=bool), np.zeros(green.shape, dtype=bool)

    mask = np.logical_and.reduce((
        shadow_training_mask,
        green < np.nanpercentile(valid_shadow_values, 95),
        curr_scene_valid,
    ))
    
    green_thresholds = define_threshold(green, mask, "green_shadow", curr_aux_folder, threshold=(0.075, 0.1))
    BNIR_thresholds = define_threshold(diff_B_NIR, mask, "diffBNIR_shadow", curr_aux_folder, threshold=(0.08, 0.12))



    # conditions of val
    # snow = np.logical_and.reduce((mask_snow, green>green_threshold_snow))
    snow = np.logical_and.reduce((mask_shadow, 
                                  green>max(green_thresholds),
                                  diff_B_NIR>max(BNIR_thresholds)))

    
    
    # snowfree = np.logical_and.reduce((mask_sf, green<green_threshold_sf))
    snowfree = np.logical_and.reduce((mask_shadow, 
                                  green<min(green_thresholds),
                                  diff_B_NIR<min(BNIR_thresholds)))
    
    return snow, snowfree
    
    
    
def get_pixels_sun(bands, curr_aux_folder, mask_sun, curr_range, sun_altitude):
    
    NDSI = load_map(curr_aux_folder, '*NDSI.tif')
    NDWI = load_map(curr_aux_folder, '*NDWI.tif')

    green = bands["GREEN"]
    swir = bands["SWIR"]

    distance_idx = load_map(curr_aux_folder, '*distance.tif')
    
    NDSI_thresholds = define_threshold(
        NDSI,
        mask_sun,
        f"NDSI_sun_{curr_range[0]}-{curr_range[1]}",
        curr_aux_folder,
        threshold=(0.2, 0.7),
    )

    
    
    # fixed conditions for being a snow pixel
    mask_snow = np.logical_and.reduce((mask_sun, 
                                        NDWI<0.1, 
                                        NDSI>max(NDSI_thresholds)))
    
    # fixed conditions for being a snowfree pixel
    mask_sf = np.logical_and.reduce((mask_sun, 
                                     NDSI<min(NDSI_thresholds)))
    

    # find dynamic thresholds
    fit_green_snow = fit_distribution_and_median(green, 
                                                 f"green_sun_snow_{curr_range[0]}-{curr_range[1]}", 
                                                 mask_snow, 
                                                 curr_aux_folder,
                                                 default_median=0.6)
    
    fit_green_sf = fit_distribution_and_median(green, 
                                               f"green_sun_sf_{curr_range[0]}-{curr_range[1]}", 
                                               mask_sf, 
                                               curr_aux_folder,
                                               default_median=0.5)
    
    fit_swir_snow = fit_distribution_and_median(swir, 
                                                f"swir_sun_snow_{curr_range[0]}-{curr_range[1]}", 
                                                mask_snow, 
                                                curr_aux_folder,
                                                default_median=0.2)
    
    fit_swir_sf = fit_distribution_and_median(swir, 
                                             f"swir_sun_sf_{curr_range[0]}-{curr_range[1]}",
                                             mask_sf, 
                                             curr_aux_folder,
                                             default_median=0.1)
    
    green_threshold_snow = fit_green_snow['fitted_median'] #- fit_green_snow['parameters']['std']
    
    green_threshold_sf = fit_green_sf['fitted_median'] #+ fit_green_sf['parameters']["std"]
    
    swir_threshold_snow = fit_swir_snow['fitted_median'] #+ fit_swir_snow['parameters']["std"]
    
    swir_threshold_sf = fit_swir_sf['fitted_median'] #- fit_swir_sf['parameters']["std"]


    # conditions of val
    snow = np.logical_and.reduce((mask_snow,
                                  green > green_threshold_snow,
                                  swir < swir_threshold_snow))
    
    # snowfree = snowfree_1 | snowfree_2
    snowfree = np.logical_and.reduce((mask_sf,
                                      green < green_threshold_sf,
                                      swir > swir_threshold_sf))

    return snow, snowfree



def get_pixels_ice(scene_id, data, config):
    
    # load information for current scene
    sensor = get_sensor(scene_id)
    bands = define_bands(data, sensor)
    
    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")

    # Scene's auxiliary folder
    curr_aux_folder = create_folder(scene_folder, "auxiliary")
    
    # No data value
    no_data_value = scene_no_data_value(config)
        
        
    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder, '*Water_Mask.tif')
    glacier_mask = load_map(auxiliary_folder, '*glacier*.tif')
    shadow_mask = load_map(curr_aux_folder, '*shadow_mask.tif')
    NDSI = load_map(curr_aux_folder, '*NDSI.tif')
    nir = bands["NIR"]
    red = bands["RED"]


    SCF = load_map(scene_folder, '*SnowFLAKES.tif')
    swir = bands["SWIR"]
        
    # validity mask: a binary dilation is applied by default (avoid training 
    # collection near water bodies, clouds, etc)
    validMask = scene_valid_mask(data, config)

    curr_scene_valid = build_valid_scene(~validMask,
                                         cloud_mask == 1,
                                         cloud_mask == 2,
                                         water_mask == 1)


    # fixed conditions for being an ice pixel   
    mask_snow = np.logical_and.reduce((glacier_mask==1, 
                                      shadow_mask == 0,
                                      NDSI > 0.7, 
                                      curr_scene_valid,
                                      nir>0.5))
    
    mask_potential_ice = np.logical_and.reduce((glacier_mask==1, 
                                      shadow_mask == 0,
                                      NDSI > 0.7, 
                                      curr_scene_valid,
                                      nir<0.5))
    
    red_thresholds = define_threshold(red, mask_potential_ice, "red_ice", curr_aux_folder, threshold=(0.45, 0.55))
    """
    if min(nir_thresholds) > 0.5:
        ice = None
        snow = None
        
    elif max(nir_thresholds) < 0.5:
        
        

    
    plt.hist(nir[mask_ice_snow])
    plt.hist(green[mask_ice_snow])

    
    mask_ice = np.logical_and.reduce((mask_ice_snow, 
                                      SCF > 0,
                                      diff_B_NIR > 0.15, 
                                      curr_scene_valid))
    
    # fixed conditions for being a snow pixel
    mask_snow = np.logical_and.reduce((glacier_mask==1, 
                                      SCF > 0,
                                      diff_B_NIR < 0.1, 
                                      curr_scene_valid))
    
    fit_swir_ice = fit_distribution_and_median(swir, 
                                              "swir_ice", 
                                               mask_ice, 
                                               curr_aux_folder,
                                               default_median=0.05)
    
    fit_swir_snow = fit_distribution_and_median(swir, 
                                              "swir_snow", 
                                               mask_snow, 
                                               curr_aux_folder,
                                               default_median=0.05)
    
    swir_threshold_snow = fit_swir_snow['fitted_median'] #- 2*fit_green_snow['parameters']['std']
    
    swir_threshold_ice = min(0.05, fit_swir_ice['fitted_median']) #+ 2*fit_green_snow['parameters']['std']
    

    # conditions of val
    snow = np.logical_and.reduce((mask_snow, swir>swir_threshold_snow))
    
    ice = np.logical_and.reduce((mask_ice, swir<swir_threshold_ice))

    return snow, ice

"""



def sample_histogram_equal(mask, values, n_samples, n_bins=20, seed=None):
    """
    Sample pixels approximately uniformly over the histogram.

    Parameters
    ----------
    mask : 2D bool array
        Pixels eligible for sampling (e.g. snow).
    values : 2D array
        Variable whose histogram should be sampled (e.g. green_corr).
    n_samples : int
        Total number of pixels to return.
    n_bins : int
        Number of histogram bins.
    seed : int or None
        Random seed.

    Returns
    -------
    sample_mask : bool array
        Boolean mask of selected pixels.
    sample_idx : ndarray
        Flat indices of selected pixels.
    sample_values : ndarray
        Values of selected pixels.
    """
    rng = np.random.default_rng(seed)

    # Eligible pixels
    valid_idx = np.flatnonzero(mask)
    valid_values = np.asarray(values).flat[valid_idx]
    finite = np.isfinite(valid_values)
    valid_idx = valid_idx[finite]
    valid_values = valid_values[finite]
    if n_samples <= 0 or valid_values.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    if np.all(valid_values == valid_values[0]):
        selected = rng.choice(
            valid_idx, min(int(n_samples), valid_idx.size), replace=False
        )
        result = np.zeros(mask.shape, dtype=bool)
        result.flat[selected] = True
        return result

    # Histogram bins
    bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1)

    # Desired samples per bin
    target = int(np.ceil(n_samples / n_bins))

    selected = []

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = np.where((valid_values >= bins[i]) &
                              (valid_values <= bins[i+1]))[0]
        else:
            in_bin = np.where((valid_values >= bins[i]) &
                              (valid_values < bins[i+1]))[0]

        if len(in_bin) == 0:
            continue

        n = min(target, len(in_bin))
        chosen = rng.choice(in_bin, n, replace=False)
        selected.extend(valid_idx[chosen])

    selected = np.array(selected)

    # If too many samples, randomly reduce
    if len(selected) > n_samples:
        selected = rng.choice(selected, n_samples, replace=False)

    # Build output mask
    sample_mask = np.zeros(mask.shape, dtype=bool)
    sample_mask.flat[selected] = True


    return sample_mask



    
    
    
def collect_trainings(data, scene_id, config, total_samples=500):
    
    # load information for current scene
    sensor = get_sensor(scene_id)
    bands = define_bands(data, sensor)
    
    # Create output directory for the scene
    wd = config['output_directory']
    scene_folder = create_folder(wd, scene_id)   

    # auxiliary folder with common features (dem, slope, etc..)
    auxiliary_folder = create_folder(wd, "01_TEST_auxiliary_folder")

    # Scene's auxiliary folder
    curr_aux_folder = create_folder(scene_folder, "auxiliary")
    
    # Extract date and time from the folder name
    date_time, date = define_datetime(scene_id, config)
    
    # No data value
    no_data_value = scene_no_data_value(config)
    training_seed = config.get("training_seed", 0)
        
       
    # get sun altitude
    sun_altitude, _ = get_altitude_azimuth(data, date_time)

    # Load masks and other necessary data
    water_mask = load_map(auxiliary_folder, '*Water_Mask.tif')
    glacier_mask = load_map(auxiliary_folder, '*glacier*.tif')
    cloud_mask, cloud_path = load_map(curr_aux_folder, '*cloud_Mask.tif', return_path=True)
    solar_incidence_angle = load_map(curr_aux_folder, '*solar_incidence_angle.tif')
    shadow_mask = load_map(curr_aux_folder, '*shadow_mask.tif')
    green = bands["GREEN"]
    
    # validity mask: a binary dilation is applied by default (avoid training 
    # collection near water bodies, clouds, etc)
    validMask = scene_valid_mask(data, config)

    curr_scene_valid = build_valid_scene(~validMask,
                                         cloud_mask == 1,
                                         cloud_mask == 2,
                                         water_mask == 1)
    
    # enlarge shadow - sun masks to create a buffer where training collection
    # is avoided

    sun_mask_eroded = binary_erosion(
        binary_dilation((shadow_mask == 0) & curr_scene_valid),
        iterations=3
    ) & curr_scene_valid
    

    # define solar incidence angle ranges
    valid_angles = np.asarray(solar_incidence_angle[curr_scene_valid], dtype=float)
    valid_angles = valid_angles[np.isfinite(valid_angles)]
    if valid_angles.size == 0:
        print(f"No valid solar-incidence angles for {scene_id}; skipping training collection")
        return None
    max_SIA = float(np.max(valid_angles))
    
    if max_SIA <= 90: 
        ranges = ((0, 20), (20, 45), (45, 70), (70, np.nextafter(max_SIA, np.inf)))
    else:
        ranges = ((0, 20), (20, 45), (45, 70), (70, 90), (90, 180))


    # get a number of training proportional to the area belonging to that range 
    range_samples = calculate_training_samples(solar_incidence_angle, ranges, total_samples)




    empty = np.zeros(curr_scene_valid.shape, dtype='uint8')
    illumination = np.zeros(curr_scene_valid.shape, dtype='uint8')
    
    training_stats = []
    pixel_stats = []

    # Shadow training is intentionally independent of solar-incidence angle.
    # Compute the thresholds and representative pixels once from the complete
    # valid shadow population.
    mask_shadow = np.logical_and.reduce((
        curr_scene_valid,
        shadow_mask == 1,
        glacier_mask == 0,
    ))
    shadow_pixels = int(np.sum(mask_shadow))
    shadow_percentage = int(
        shadow_pixels * 100 / max(np.sum(curr_scene_valid), 1)
    )
    pixel_stats.append({
        "angle_range": "all",
        "illumination": "Shadow",
        "pixels": shadow_percentage,
    })
    representative_pixels_mask_snow = np.zeros(empty.shape, dtype="uint8")
    representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype="uint8")
    if shadow_pixels > 0:
        print("Collecting trainings in shadow (all solar-angle ranges)")
        snow_shad, snowfree_shad = get_pixels_shadow(
            bands, curr_aux_folder, curr_scene_valid, mask_shadow
        )
        if np.sum(snow_shad) > 10:
            representative_pixels_mask_snow = sample_histogram_equal(
                snow_shad, green, int(total_samples / 2), n_bins=20, seed=training_seed
            )
        if np.sum(snowfree_shad) > 10:
            representative_pixels_mask_noSnow = sample_histogram_equal(
                snowfree_shad, green, int(total_samples / 2), n_bins=20, seed=training_seed
            ) * 2

    representative_pixels_mask = (
        representative_pixels_mask_noSnow + representative_pixels_mask_snow
    )
    empty[mask_shadow] = representative_pixels_mask[mask_shadow]
    illumination[representative_pixels_mask > 0] = 2
    print(f"{np.sum(representative_pixels_mask_snow)} SNOW PIXELS (shadow)")
    print(f"{np.sum(representative_pixels_mask_noSnow) / 2} NO SNOW PIXELS (shadow)")
    training_stats.append({
        "angle_range": "all",
        "illumination": "Shadow",
        "snow_train": int(np.sum(representative_pixels_mask_snow)),
        "nosnow_train": int(np.sum(representative_pixels_mask_noSnow) / 2),
    })
    
    # collect training for each SIA range 
    for curr_range, sample_count in range_samples.items():

        
        curr_angle_valid = np.logical_and.reduce((curr_scene_valid, 
                                                  solar_incidence_angle >= curr_range[0],
                                                  solar_incidence_angle < curr_range[1]))
        
    
        print(f"SIA range: {curr_range}")

        # # SUN --------------------------------------------------------------

        # mask angles and sun
        mask_sun = curr_angle_valid & sun_mask_eroded
        
        pixel_perc_sun = int(
            np.sum(mask_sun) * 100 / max(np.sum(curr_scene_valid), 1)
        )

        pixel_stats.append({
                            "angle_range": f"{curr_range[0]}-{curr_range[1]}",
                            "illumination": "Sun",
                            "pixels": pixel_perc_sun
                        })


        if np.sum(mask_sun) > 10:
            
            # initialize empty masks
            representative_pixels_mask_snow = np.zeros(empty.shape, dtype='uint8')
            representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype='uint8')
            
            print('Collecting trainings in sun')

            snow_sun, snowfree_sun = get_pixels_sun(
                bands, curr_aux_folder, mask_sun,
                curr_range, sun_altitude
            )

    
            if np.sum(snow_sun) > 10:
                representative_pixels_mask_snow  = sample_histogram_equal(snow_sun, green, int(sample_count / 2), n_bins=20, seed=training_seed)
                
                save_histogram(
                    green[representative_pixels_mask_snow],
                    os.path.join(curr_aux_folder, f"hist_snow_selected_{curr_range[0]}-{curr_range[1]}.png"),
                    bins=50,
                    xlabel="Value",
                    ylabel="Count"
                )
                
      
    
            if np.sum(snowfree_sun) > 10:
                representative_pixels_mask_noSnow  = sample_histogram_equal(snowfree_sun, green, int(sample_count / 2), n_bins=20, seed=training_seed) * 2
                
                save_histogram(
                    green[representative_pixels_mask_noSnow==2],
                    os.path.join(curr_aux_folder, f"hist_sf_selected_{curr_range[0]}-{curr_range[1]}.png"),
                    bins=50,
                    xlabel="Value",
                    ylabel="Count"
                )
                
      
            
       
    
            # merge the two masks
            representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
            empty[mask_sun] = representative_pixels_mask[mask_sun]
            
            # mark selected training pixels as sun
            illumination[representative_pixels_mask > 0] = 1

            print(str(np.sum(representative_pixels_mask_snow.flatten())) + ' SNOW PIXELS')
            print(str(np.sum(representative_pixels_mask_noSnow.flatten() / 2)) + ' NO SNOW PIXELS')
            
            training_stats.append({
                        "angle_range": f"{curr_range[0]}-{curr_range[1]}",
                        "illumination": "Sun",
                        "snow_train": np.sum(representative_pixels_mask_snow.flatten()),
                        "nosnow_train": np.sum(representative_pixels_mask_noSnow.flatten() / 2)
                    })
            

    # Convert points where result == 1 or 2 to a shapefile
    points = []
    values = []
    illum_values = []
    with rasterio.open(cloud_path) as src:
        cloud_crs = src.crs
        for row, col in zip(*np.where((empty == 1) | (empty == 2))):
            x, y = src.xy(row, col)
            points.append(Point(x, y))
            values.append(empty[row, col])
            illum_values.append(illumination[row, col])
            
    gdf = gpd.GeoDataFrame({"value": values, 
                            "illum": illum_values}, 
                            geometry=points, crs=cloud_crs)

    if not points:
        print(f"No representative training pixels found for {scene_id}")
        return None


    shapefile_path = os.path.join(curr_aux_folder, 'representative_pixels_for_training_samples.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")

    plot_trainings(training_stats, pixel_stats, curr_aux_folder)

    return shapefile_path


    
    
    

def glacier_xgboost(model_path, data, no_data_mask, curr_aux_folder, 
                    auxiliary_folder_path, Nprocesses=8):
    
    
    # Load the model
    with open(model_path, 'rb') as model_file:
        svm_dict = pickle.load(model_file)
    xgboost_model = svm_dict['xgboostModel']
    normalizer = svm_dict['normalizer']
    feature_names = svm_dict['feature_names']
    
    

    glacier_mask = load_map(auxiliary_folder_path, '*glacier*.tif')
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    valid_mask = np.logical_not(no_data_mask)
    
    mask = valid_mask & (cloud_mask==0) & (glacier_mask == 1) # cambiare

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
        
        
    
    
