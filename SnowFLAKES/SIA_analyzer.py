#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:07:41 2024

@author: rbarella
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os
import rasterio


def plot_solar_incidence(files):
    """
    Plot average solar incidence angle with standard deviation as a buffer.

    Parameters:
        files (list of str): List of file paths to solar incidence angle files.
                             Filenames must contain the date in a parseable format (e.g., YYYYMMDD).
    """
    dates = []
    means = []
    stds = []

    for file in files:
        # Extract date from filename (assuming date is in format YYYYMMDD in the filename)
        date_str = os.path.basename(file).split('_')[-4]  # Adjust based on your filename format
        date = datetime.strptime(date_str, '%Y%m%d')
        dates.append(date)

        # Load solar incidence angle data from GeoTIFF
        with rasterio.open(file) as src:
            data = src.read(1)  # Read the first band

            # Mask invalid (no-data) values
            data = data[data != src.nodata]

        # Compute mean and standard deviation
        means.append(np.mean(data))
        stds.append(np.std(data))

    # Combine and sort data by date
    sorted_data = sorted(zip(dates, means, stds))
    sorted_dates, sorted_means, sorted_stds = zip(*sorted_data)

    # Convert to numpy arrays
    sorted_dates = np.array(sorted_dates)
    sorted_means = np.array(sorted_means)
    sorted_stds = np.array(sorted_stds)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_dates, sorted_means, label='Average Solar Incidence Angle', color='blue')
    plt.fill_between(sorted_dates, sorted_means - sorted_stds, sorted_means + sorted_stds, color='blue', alpha=0.3,
                     label='Std Dev')
    plt.xlabel('Date')
    plt.ylabel('Average Solar Incidence Angle (degrees)')
    plt.title('Solar Incidence Angle Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_solar_incidence_with_quartiles(files):
    """
    Plot the 50th percentile (median) solar incidence angle with quartiles as shaded areas.

    Parameters:
        files (list of str): List of file paths to solar incidence angle files.
                             Filenames must contain the date in a parseable format (e.g., YYYYMMDD).
    """
    dates = []
    medians = []
    q1_values = []
    q3_values = []
    min_values = []
    max_values = []

    for file in files:
        # Extract date from filename (assuming date is in format YYYYMMDD in the filename)
        date_str = os.path.basename(file).split('_')[-4]  # Adjust based on your filename format
        date = datetime.strptime(date_str, '%Y%m%d')
        dates.append(date)

        # Load solar incidence angle data from GeoTIFF
        with rasterio.open(file) as src:
            data = src.read(1)  # Read the first band

            # Mask invalid (no-data) values
            data = data[data != src.nodata]

        # Compute statistics
        medians.append(np.median(data))
        q1_values.append(np.percentile(data, 25))
        q3_values.append(np.percentile(data, 75))
        min_values.append(np.min(data))
        max_values.append(np.max(data))

    # Combine and sort data by date
    sorted_data = sorted(zip(dates, medians, q1_values, q3_values, min_values, max_values))
    sorted_dates, sorted_medians, sorted_q1, sorted_q3, sorted_min, sorted_max = zip(*sorted_data)

    # Convert to numpy arrays
    sorted_dates = np.array(sorted_dates)
    sorted_medians = np.array(sorted_medians)
    sorted_q1 = np.array(sorted_q1)
    sorted_q3 = np.array(sorted_q3)
    sorted_min = np.array(sorted_min)
    sorted_max = np.array(sorted_max)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_dates, sorted_medians, label='Median Solar Incidence Angle', color='blue', linewidth=2)
    plt.fill_between(sorted_dates, sorted_q1, sorted_q3, color='blue', alpha=0.3,
                     label='Interquartile Range (25th-75th Percentile)')
    plt.fill_between(sorted_dates, sorted_min, sorted_q1, color='green', alpha=0.2,
                     label='1st Quartile (Min-25th Percentile)')
    plt.fill_between(sorted_dates, sorted_q3, sorted_max, color='red', alpha=0.2,
                     label='4th Quartile (75th Percentile-Max)')

    plt.xlabel('Date')
    plt.ylabel('Solar Incidence Angle (degrees)')
    plt.title('Solar Incidence Angle Over Time with Quartiles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage
# ts_folder = '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Sentinel2/Maipo/merged'
ts_folder = '/mnt/CEPH_PROJECTS/PROSNOW/SENTINEL-2/32TPS/Reprojected_Senales_catchment'

# Adjust the glob pattern to match your file structure
sia_list = glob.glob(os.path.join(ts_folder, '*', '00*', '*_solar_incidence_angle.tif'))

# plot_solar_incidence(sia_list)

plot_solar_incidence_with_quartiles(sia_list)





