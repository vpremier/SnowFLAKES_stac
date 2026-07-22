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
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

from SnowFLAKES.utilities import (
    load_map,
    open_image,
    build_valid_scene
)
from SnowFLAKES.fit_distribution import fit_distribution_and_median


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

    range_samples = {
        r: int(total_samples * (count / total_pixels)) + 20
        if total_pixels > 0
        else 0
        for r, count in range_pixel_counts.items()
    }
    
    
    return range_samples



def apply_topographic_correction(band, band_name, SIA, sun_altitude, correction="weak", g=0.22):
    
    sun_zenith = 90 - sun_altitude
    
    if band_name in ['GREEN', 'BLUE', 'RED']:
        b = 0.75
    elif band_name in ['NIR', 'SWIR']:
        if correction == "weak":
            b = 0.33
        elif correction == "strong":
            b = 1
            
    # cos of the SIA
    cos_beta = np.cos(np.deg2rad(SIA))
    
    # CALCULATE SEN2COR THRESHOLD:
    if sun_zenith < 45:
        beta_threshold = sun_zenith + 20
    elif sun_zenith <= 55:
        beta_threshold = sun_zenith + 15
    else:
        beta_threshold = sun_zenith + 10
        
    
    # Prevent a zero or negative denominator.
    beta_threshold_deg = min(beta_threshold, 89.0)
    cos_beta_threshold = np.cos(np.deg2rad(beta_threshold_deg))
    
    # valid application range between betaT and 90
    valid = cos_beta > 0
    faint_illumination = valid & (SIA > beta_threshold_deg)

    
    # geometric function G Eq. 0.32 
    G = np.clip((cos_beta/cos_beta_threshold)**b, g, 1.0)

    band_corr = band.copy()
    band_corr[faint_illumination] = band[faint_illumination]*G[faint_illumination]
    
    return band_corr 



def define_threshold(feature,
                     mask,
                     feature_name,
                     curr_aux_folder,
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
    # working directory
    wd = Path(curr_aux_folder).parent
    
    # subdirectory SCF
    scf_folder = wd / "SCF"
    scf_folder.mkdir(exist_ok=True)
    
    
    values = feature[mask].reshape(-1, 1)


    
    values_df = pd.DataFrame({
        feature_name: values.flatten()
    })
    
    values_df.to_csv(
        scf_folder / f"{feature_name}_values.csv",
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

    gmm.fit(values)

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
        os.path.join(scf_folder, feature_name + "_histogram.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return final_means

    
    
def get_pixels_shadow(bands, curr_aux_folder, curr_scene_valid, mask_shadow):
    
    shadow_mask = load_map(curr_aux_folder, '*shadow_mask.tif')
    green = bands["GREEN"]
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')
    distance_idx = load_map(curr_aux_folder, '*distance.tif')
    
    mask = np.logical_and.reduce((shadow_mask==1, 
                                       green <0.25, 
                                       curr_scene_valid))
    
    green_thresholds = define_threshold(green, mask, "green_shadow", curr_aux_folder, threshold=(0.075, 0.1))
    BNIR_thresholds = define_threshold(diff_B_NIR, mask, "diffBNIR_shadow", curr_aux_folder, threshold=(0.08, 0.12))



    # # fixed conditions for being a snow pixel
    # mask_snow = np.logical_and.reduce((shadow_mask==1, 
    #                                    diff_B_NIR>0.12, 
    #                                    green <0.25, 
    #                                    curr_scene_valid))
    
    # # fixed conditions for being a snowfree pixel
    # mask_sf = np.logical_and.reduce((shadow_mask==1, 
    #                                  diff_B_NIR>0, 
    #                                  diff_B_NIR<0.08, 
    #                                  green <0.25, 
    #                                  curr_scene_valid)) 
    
    # fit_green_snow = fit_distribution_and_median(green, 
    #                                             "green_shadow_snow", 
    #                                              mask_snow, 
    #                                              curr_aux_folder,
    #                                              default_median=0.1)
    
    # fit_green_sf = fit_distribution_and_median(green, 
    #                                            "green_shadow_sf", 
    #                                             mask_sf, 
    #                                             curr_aux_folder,
    #                                             default_median=0.075)
    
    # green_threshold_snow = fit_green_snow['fitted_median'] #- 2*fit_green_snow['parameters']['std']
    
    # green_threshold_sf = fit_green_sf['fitted_median'] #+ 2*fit_green_snow['parameters']['std']
    

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
    
    # fixed conditions for being a snow pixel
    mask_snow = np.logical_and.reduce((mask_sun, 
                                          NDWI<0.1, 
                                          distance_idx != 255, 
                                          NDSI>0.6)) 
    
    # fixed conditions for being a snowfree pixel
    mask_sf = np.logical_and.reduce((mask_sun, 
                                     NDSI<0)) 
    

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
                                             f"green_sun_sf_{curr_range[0]}-{curr_range[1]}", 
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
    snowfree = np.logical_and.reduce((mask_sun,
                                      green < green_threshold_sf,
                                      swir > swir_threshold_sf))

        
    
    return snow, snowfree




def get_pixels_ice(bands, curr_aux_folder, auxiliary_folder_path, FSC_SVM_map_path, no_data_mask):
    

    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder_path, '*Water_Mask.tif')
    glacier_mask = load_map(auxiliary_folder_path, '*glacier*.tif')
    diff_B_NIR = load_map(curr_aux_folder, '*diffBNIR.tif')
    swir = bands["SWIR"]

    SCF, _ = open_image(FSC_SVM_map_path)
    
    curr_scene_valid = build_valid_scene(no_data_mask,
                                         cloud_mask == 1,
                                         water_mask == 1)
    
    # fixed conditions for being an ice pixel
    mask_ice = np.logical_and.reduce((glacier_mask==1, 
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
    
    swir_threshold_ice = fit_swir_ice['fitted_median'] #+ 2*fit_green_snow['parameters']['std']
    

    # conditions of val
    snow = np.logical_and.reduce((mask_snow, swir>swir_threshold_snow))
    
    ice = np.logical_and.reduce((mask_ice, swir<swir_threshold_ice))

    return snow, ice



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
    valid_values = values.flat[valid_idx]

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
    
    
    
def collect_trainings(scene_id, all_bands_image, curr_aux_folder, auxiliary_folder_path, 
                      SCF_folder, no_data_mask, bands, sun_altitude, FSC_SVM_map_path = None, total_samples=500):
    


    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder_path, '*Water_Mask.tif')
    solar_incidence_angle = load_map(curr_aux_folder, '*solar_incidence_angle.tif')
    glacier_mask = load_map(auxiliary_folder_path, '*glacier*.tif')
    shadow_mask = load_map(curr_aux_folder, '*shadow_mask.tif')
    shad_idx = load_map(curr_aux_folder, '*shad_idx.tif')
    distance_idx = load_map(curr_aux_folder, '*distance.tif')
    green = bands["GREEN"]
    
    
    _, NDSI_path = load_map(curr_aux_folder, '*NDSI.tif', return_path=True)
    
    # validity mask: a binary dilation is applied (avoid training collection
    # near water bodies, clouds, etc).. glaciers??
    curr_scene_valid = build_valid_scene(no_data_mask,
                                         cloud_mask == 1,
                                         water_mask == 1)
    


    # enlarge shadow - sun masks to create a buffer where training collection
    # is avoided
    shadow_mask_eroded = binary_erosion(
        shadow_mask == 1,
        iterations=3
    )

    sun_mask_eroded = binary_erosion(
        shadow_mask == 0,
        iterations=3
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
    illumination = np.zeros(curr_scene_valid.shape, dtype='uint8')
    
    training_stats = []
    pixel_stats = []
    percentage_per_angles_list = []
    
    # collect training for each SIA range 
    for curr_range, sample_count in range_samples.items():


        curr_angle_valid = np.logical_and.reduce((curr_scene_valid, 
                                                  solar_incidence_angle >= curr_range[0],
                                                  solar_incidence_angle < curr_range[1]))
        
        percentage_of_scene_valid = np.sum(curr_angle_valid) / np.sum(curr_scene_valid)
        print(f"SIA range: {curr_range}")
        percentage_per_angles_list.append(percentage_of_scene_valid)

        # SHADOW --------------------------------------------------------------

        # mask angles and shadow
        mask_shadow = np.logical_and.reduce((curr_angle_valid, 
                                             shadow_mask_eroded,
                                             glacier_mask==0)) # no dilation applied for glacier here
          
                  
        pixel_stats.append({
                            "angle_range": f"{curr_range[0]}-{curr_range[1]}",
                            "illumination": "Shadow",
                            "pixels": int(np.sum(mask_shadow) *100/ np.sum(curr_scene_valid))
                            })
            
        if np.sum(mask_shadow) > 10:
            
            # initialize empty masks
            representative_pixels_mask_snow = np.zeros(empty.shape, dtype='uint8')
            representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype='uint8')
            
            print('Collecting trainings in shadow')
            
            # snow_shad, snowfree_shad = get_pixels_shadow(diff_B_NIR, 
            #                                              shad_idx, 
            #                                              NDSI, 
            #                                              distance_idx,
            #                                              mask_shadow)
            
            snow_shad, snowfree_shad = get_pixels_shadow(bands, curr_aux_folder, curr_scene_valid, mask_shadow)

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
                representative_pixels_mask_snow  = sample_histogram_equal(snow_shad, green, int(sample_count / 2), n_bins=20, seed=None)


                # representative_pixels_mask_snow = get_representative_pixels(all_bands_image, 
                #                                                             snow_shad,
                #                                                             sample_count=int(sample_count / 2), 
                #                                                             k=3,
                #                                                             n_closest='auto')
                
            if np.sum(snowfree_shad) > 10:
                representative_pixels_mask_noSnow  = sample_histogram_equal(snowfree_shad, green, int(sample_count / 2), n_bins=20, seed=None) * 2

                # representative_pixels_mask_noSnow = get_representative_pixels(all_bands_image,
                #                                                               snowfree_shad,
                #                                                               sample_count=int(sample_count / 2), 
                #                                                               k=3,
                #                                                               n_closest='auto') * 2
            # merge the two masks
            representative_pixels_mask = representative_pixels_mask_noSnow + representative_pixels_mask_snow
            empty[mask_shadow] = representative_pixels_mask[mask_shadow]
            
            # mark selected training pixels as shadow
            illumination[representative_pixels_mask > 0] = 2
            
            print(str(np.sum(representative_pixels_mask_snow.flatten())) + ' SNOW PIXELS')
            print(str(np.sum(representative_pixels_mask_noSnow.flatten() / 2)) + ' NO SNOW PIXELS')
            

                            
            training_stats.append({
                "angle_range": f"{curr_range[0]}-{curr_range[1]}",
                "illumination": "Shadow",
                "snow_train": int(np.sum(representative_pixels_mask_snow)),
                "nosnow_train": int(np.sum(representative_pixels_mask_noSnow) / 2)
            })


            
            
            
        
        # # SUN --------------------------------------------------------------

        # mask angles and sun
        mask_sun = curr_angle_valid & sun_mask_eroded
        
        pixel_stats.append({
                            "angle_range": f"{curr_range[0]}-{curr_range[1]}",
                            "illumination": "Sun",
                            "pixels": int(np.sum(mask_sun)*100/ np.sum(curr_scene_valid))
                        })


        if np.sum(mask_sun) > 10:
            
            # initialize empty masks
            representative_pixels_mask_snow = np.zeros(empty.shape, dtype='uint8')
            representative_pixels_mask_noSnow = np.zeros(empty.shape, dtype='uint8')
            
            print('Collecting trainings in sun')

            snow_sun, snowfree_sun = get_pixels_sun(bands, curr_aux_folder, mask_sun, curr_range, sun_altitude)

    
            if np.sum(snow_sun) > 10:
                representative_pixels_mask_snow  = sample_histogram_equal(snow_sun, green, int(sample_count / 2), n_bins=20, seed=None)

                # representative_pixels_mask_snow = get_representative_pixels(all_bands_image, 
                #                                                             snow_sun,
                #                                                             sample_count=int(sample_count / 2), 
                #                                                             k=5,
                #                                                             n_closest='auto')
                
                save_histogram(
                    green[representative_pixels_mask_snow],
                    os.path.join(SCF_folder, f"hist_snow_selected_{curr_range[0]}-{curr_range[1]}.png"),
                    bins=50,
                    xlabel="Value",
                    ylabel="Count"
                )
                
      
    
            if np.sum(snowfree_sun) > 10:
                representative_pixels_mask_noSnow  = sample_histogram_equal(snowfree_sun, green, int(sample_count / 2), n_bins=20, seed=None) * 2

                # representative_pixels_mask_noSnow = get_representative_pixels(all_bands_image, 
                #                                                               snowfree_sun,
                #                                                               sample_count=int(sample_count / 2), 
                #                                                               k=10,
                #                                                               n_closest='auto') * 2
                
                save_histogram(
                    green[representative_pixels_mask_noSnow==2],
                    os.path.join(SCF_folder, f"hist_sf_selected_{curr_range[0]}-{curr_range[1]}.png"),
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
    with rasterio.open(NDSI_path) as src:
        for row, col in zip(*np.where((empty == 1) | (empty == 2))):
            x, y = src.xy(row, col)
            points.append(Point(x, y))
            values.append(empty[row, col])
            illum_values.append(illumination[row, col])
            
    gdf = gpd.GeoDataFrame({"value": values, 
                            "illum": illum_values}, 
                           geometry=points, crs=src.crs)

    #plot_valid_pixels_percentage(ranges, percentage_per_angles_list, scf_folder)

    shapefile_path = os.path.join(SCF_folder, 'representative_pixels_for_training_samples.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")

    plot_trainings(training_stats, pixel_stats, SCF_folder)

    return shapefile_path


def plot_trainings(training_stats, pixel_stats, scf_folder):
    

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
    output_path = os.path.join(scf_folder, 'valid_trainings_per_angle.png')
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
    output_path = os.path.join(scf_folder, 'valid_pixels_per_angle.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to avoid display issues in non-interactive environments
    print(f"Plot saved to: {output_path}")
    
    
    

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
    bands = define_bands(data, sensor)
    
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
    ndwi[(ndsi>=0.7) & (cloud_mask==1)] # cambiare

    
    nsir_vals = nsir[((ndsi >= 0.7) & (cloud_mask == 1) & (glacier_mask == 1))] # cambiare

    glacier_map = np.zeros_like(nir, dtype=np.uint8)

    try:
        nsir_threshold = threshold_otsu(nsir_vals)
        
        snow = (
            (ndsi >= 0.7) &
            (cloud_mask == 1) & # cambiare
            (glacier_mask == 1) &
            (nsir >= nsir_threshold) &
            (ndwi <= 0.1)
        )
        
        ice = (
            (ndsi >= 0.7) &
            (cloud_mask == 1) & #cambiare
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
    


def glacier_classifier2(scene_id, data, no_data_mask, curr_aux_folder, auxiliary_folder_path):
    # working directory
    wd = Path(curr_aux_folder).parent
    
    
    # subdirectory SCF
    scf_folder = wd / SVM_folder_name
    scf_folder.mkdir(exist_ok=True)
    
    NDSI_path = glob.glob(os.path.join(curr_aux_folder, '*NDSI.tif'))[0]
    NDVI_path = glob.glob(os.path.join(curr_aux_folder, '*NDVI.tif'))[0]
    path_cloud_mask = glob.glob(os.path.join(curr_aux_folder, '*cloud_Mask.tif'))[0]
    glacier_mask_path = glob.glob(os.path.join(auxiliary_folder_path, '*glacier*.tif'))[0]

    
    sensor = get_sensor(scene_id)

    # Create valid mask from no_data_mask (True means valid)
    valid_mask = np.logical_not(no_data_mask)
    cloud_mask = open_image(path_cloud_mask)[0]
    glacier_mask = open_image(glacier_mask_path)[0]
    glacier_mask =  binary_dilation(glacier_mask==1, iterations=5)

    # Load the image bands using your open_image and define_bands functions.
    bands = define_bands(data, sensor)
    
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
    nsir_vals = nsir[((ndsi >= 0.7) & (cloud_mask == 1) & (glacier_mask == 1))]

    glacier_map = np.zeros_like(nir, dtype=np.uint8)
    
    sample_count=50
    
    empty = np.zeros(ndsi.shape, dtype='uint8')

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


    representative_pixels_snow = get_representative_pixels(all_bands_image, 
                                                                snow,
                                                                sample_count=int(sample_count / 2), 
                                                                k=3,
                                                                n_closest='auto')
    
    
    representative_pixels_ice = get_representative_pixels(all_bands_image, 
                                                                ice,
                                                                sample_count=int(sample_count / 2), 
                                                                k=3,
                                                                n_closest='auto') * 2
    
    representative_pixels_mask = representative_pixels_snow + representative_pixels_ice

    
    # Convert points where result == 1 or 2 to a shapefile
    points = []
    values = []
    with rasterio.open(NDSI_path) as src:
        for row, col in zip(*np.where((representative_pixels_mask == 1) | (representative_pixels_mask == 2))):
            x, y = src.xy(row, col)
            points.append(Point(x, y))
            values.append(representative_pixels_mask[row, col])

    gdf = gpd.GeoDataFrame({"value": values}, geometry=points, crs=src.crs)
    
    
    shapefile_path = os.path.join(scf_folder, 'representative_pixels_for_glaciers.shp')
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")
    

    
    glacier_map[snow] = 100
    glacier_map[ice] = 215
    
    return glacier_map



    
    
def thematic_map_classifier(scene_id, data, curr_aux_folder, auxiliary_folder_path,
                            no_data_mask, SCF_folder, classify_glaciers,
                            date_time):
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


    # Load masks and other necessary data
    cloud_mask = load_map(curr_aux_folder, '*cloud_Mask.tif')
    water_mask = load_map(auxiliary_folder_path, '*Water_Mask.tif')
    NDSI, NDSI_path = load_map(curr_aux_folder, '*NDSI.tif', return_path=True)
    NDVI = load_map(curr_aux_folder, '*NDVI.tif')

    valid_mask = np.logical_not(no_data_mask)

    
    # Set a fixed NDSI threshold (candidate pixels) and a NDVI threshold to avoid vegetation
    ndsi_threshold = 0.4
    ndvi_threshold = 0.5
    

    
    # Mark invalid pixels as 255 (no-data, clouds, or water)
    thematic_map = np.zeros_like(NDSI, dtype=np.uint8)

    

    # Build candidate mask: valid pixels with sufficient NDSI and low NDVI.
    snow_mask = valid_mask & (NDSI > ndsi_threshold) & (NDVI < ndvi_threshold)
    thematic_map[snow_mask] = 100

    
    # Glacier reclassification: only if classify_glaciers == 'yes' and date within glacier season.
    # if (classify_glaciers.lower() == 'yes' and
    #     dt_start_glaciers_month is not None and dt_end_glaciers_month is not None and
    #     is_month_in_range(date_time.month, dt_start_glaciers_month.month, dt_end_glaciers_month.month)):
        
    #     glacier_map = glacier_classifier(scene_id, data, no_data_mask, 
    #                                      curr_aux_folder, 
    #                                      auxiliary_folder_path)
        
    #     thematic_map[glacier_map == 100] = 100
    #     thematic_map[glacier_map == 215] = 215


    thematic_map[np.logical_not(valid_mask)] = 255
    
    # Optionally mark cloud and water areas with distinct codes:
    thematic_map[cloud_mask == 1] = 205  # cambiare
    thematic_map[water_mask == 1] = 210
    

    # Define output path
    output_path = os.path.join(SCF_folder,f'{scene_id}_simple_class.tif')

    # save output tif file
    save_tif(thematic_map, NDSI_path, output_path, dtype=rasterio.uint8)

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
    
    













