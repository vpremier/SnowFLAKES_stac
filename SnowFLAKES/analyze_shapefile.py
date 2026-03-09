#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:02:52 2025

@author: rbarella
"""
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


# Function to extract raster values within geometries
def extract_raster_values(raster_path, geometries):
    """Extract raster values within geometries."""
    with rasterio.open(raster_path) as src:
        values = []
        for geom in geometries:
            try:
                out_image, _ = rasterio.mask.mask(src, [geom], crop=True)
                out_data = out_image[0]
                values.append(out_data[out_data != src.nodata])
            except Exception as e:
                print(f"Error processing geometry: {e}")
                values.append([])
    return values


# Function to prepare data for boxplot
def prepare_data(shapefile_path, raster_paths, class_field):
    """Prepare data for boxplots from shapefile and rasters."""
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)

    if class_field not in gdf.columns:
        raise ValueError(f"The field '{class_field}' is not found in the shapefile.")

    # Extract geometries and classes
    geometries = gdf.geometry
    classes = gdf[class_field]

    # Collect raster values for each geometry
    data = {}
    for raster_path in raster_paths:
        raster_name = raster_path.split('/')[-1]
        raster_values = extract_raster_values(raster_path, geometries)
        data[raster_name] = []

        for i, values in enumerate(raster_values):
            if values.size > 0:
                class_name = classes.iloc[i]
                data[raster_name].append((class_name, values))

    return data


# Function to plot all boxplots in the same plot
def plot_combined_boxplots(data):
    """Plot all boxplots in the same plot with debugging information."""
    combined_data = []
    combined_labels = []
    raster_labels = []

    for raster_name, values in data.items():
        for class_name, class_values in values:
            combined_data.extend(class_values)
            combined_labels.extend([class_name] * len(class_values))
            raster_labels.extend([raster_name] * len(class_values))

    # Debugging: Check lengths and unique values
    print(f"Total data points: {len(combined_data)}")
    print(f"Unique class labels: {set(combined_labels)}")
    print(f"Unique raster labels: {set(raster_labels)}")

    if not combined_data:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 8))
    sns.boxplot(x=combined_labels, y=combined_data, hue=raster_labels)
    plt.title("Combined Boxplot for All Rasters")
    plt.xlabel("Class")
    plt.ylabel("Raster Values")
    plt.legend(title="Raster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    shapefile_path = "/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-9/LC09_L1TP_merged_20220413_20230422_02_T1/20220413_class_check_shapefile.shp"
    interest_names_list = ['NDSI', 'B8']
    names_to_exclude = ['plus', 'shadow_mask']

    path_rasters = glob.glob(os.path.join(os.path.dirname(shapefile_path), '00*', '*'))
    raster_paths = [f for f in path_rasters if any(name in f for name in interest_names_list) and not any(
        exclude in f for exclude in names_to_exclude)]

    class_field = "class"  # Field in shapefile with class names

    try:
        data = prepare_data(shapefile_path, raster_paths, class_field)
        plot_combined_boxplots(data)
    except Exception as e:
        print(f"Error: {e}")
