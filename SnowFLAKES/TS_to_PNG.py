#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 07:38:39 2021

@author: rbarella
"""

import os
import glob
from osgeo import gdal
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib
import pandas as pd
from matplotlib import colormaps

matplotlib.use('Agg')


def save_scene_png(time_series_folder, png_folder, start_date, end_date, scf_subfolder_name):
    """
    Generates and saves PNGs of snow cover maps and corresponding RGB composites.

    Parameters:
        time_series_folder (str): Path to the folder containing time series data.
        png_folder (str): Path to the folder where PNGs will be saved.
        start_date (str): Start date for filtering images (format: YYYYMMDD).
        end_date (str): End date for filtering images (format: YYYYMMDD).
        scf_subfolder_name (str): Subfolder name containing the snow cover maps.
    """

    def create_colormap():
        """Creates a custom colormap for snow cover visualization."""
        viridis = colormaps['cool']
        top = colormaps['cool']
        bottom = colormaps['Blues']
        newcolors = np.vstack((
            top(np.linspace(0, 1, 101)),
            bottom(np.linspace(0, 1, 155))
        ))
        newcolors[:1, :] = np.array([0, 0.3, 0, 1])
        newcolors[205, :] = [150 / 256, 150 / 256, 150 / 256, 1]  # Grey for clouds
        newcolors[255, :] = [0, 0, 0, 1]  # Black for no data
        newcolors[210, :] = [0, 0, 1, 1]  # Blue for water
        newcolors[215, :] = [153 / 256, 1, 1, 1]  # Cyan for glaciers
        return ListedColormap(newcolors)

    def extract_date(sensor, filename):
        """Extracts the acquisition date from the filename based on the sensor type."""
        if sensor == 'S2':
            return os.path.basename(filename).split('_')[2].split('T')[0]
        elif sensor == 'L8' or sensor == 'L7':
            return os.path.basename(filename).split('_')[3]

    def validate_cloud_coverage(map_data):
        """Calculates cloud coverage percentage and validates it."""
        cloud_percentage = np.sum(map_data == 205) / np.sum(map_data != 255)
        return cloud_percentage < 1, cloud_percentage

    # Prepare folders and colormap
    os.makedirs(png_folder, exist_ok=True)
    colormap = create_colormap()

    # Find images of interest

    images_all = sorted(glob.glob(os.path.join(time_series_folder, '*', scf_subfolder_name, '*SnowFLAKES.tif')))
    
    if len(images_all ) == 0:
        
        return
    
    # images_all = sorted(glob.glob(os.path.join(time_series_folder, '*', scf_subfolder_name, '*GLACIER*.tif')))
    sensor = check_Mission(images_all[0])
    if sensor == 'S2':
        images_filtered = [f for f in images_all if
                           start_date <= os.path.basename(f).split('_')[2].split('T')[0] <= end_date]

    if sensor == 'L7' or sensor == 'L8':
        images_filtered = [f for f in images_all if start_date <= os.path.basename(f).split('_')[3] <= end_date]

    for curr_map in images_filtered:

        date = extract_date(sensor, curr_map)

        save_path = os.path.join(png_folder, f"{date}_scene.png")

        if os.path.exists(save_path):
            print('SCENE ALREADY SAVED')
            continue

        try:
            corrispondent_vrt = glob.glob(os.path.join(time_series_folder, '*', f'*{date}*scf.vrt'))[0]
        except IndexError:
            print(f"No corresponding VRT found for date: {date}")
            continue

        map_data = open_image(curr_map)[0].astype('uint8')
        is_valid, cloud_percentage = validate_cloud_coverage(map_data)

        if not is_valid:
            print(f"Skipping {curr_map} due to high cloud coverage ({cloud_percentage:.2%}).")
            continue

        vrt_data = open_image(corrispondent_vrt)[0]
        rgb_bands = {"S2": [8, 7, 1], "L8": [5, 4, 2], "L7": [4, 3, 1]}
        RGB_stack = vrt_data[rgb_bands[sensor], :, :]

        # Create and save the figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Snow cover map
        # Snow cover map
        map_data[0, 0] = 255
        axes[0].imshow(map_data.astype('float') - 1, cmap=colormap, interpolation='nearest')
        axes[0].set_title(f"Snow Cover Map - {date}")
        axes[0].axis('off')
        colorbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=axes[0])
        colorbar.set_label("Snow Cover Classes")

        RGB_stack[np.isnan(RGB_stack)] = 0

        # RGB composite
        RGB_stack_normalized = RGB_stack / np.percentile(RGB_stack, 99, axis=(1, 2), keepdims=True)
        RGB_stack_normalized = np.clip(RGB_stack_normalized, 0, 1).transpose(1, 2, 0)
        axes[1].imshow(RGB_stack_normalized)
        axes[1].set_title(f"RGB Composite - {date}")
        axes[1].axis('off')

        # Save the figure

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        print(f"Saved PNG for {date} to {save_path}")


def open_image(image_path):
    if os.path.exists(image_path) is False:
        raise Exception('[Errno 2] No such file or directory: \'' + image_path + '\'')

    image = gdal.Open(image_path)

    if image == None:
        raise Exception("Unable to read the data file")

    cols = image.RasterXSize
    rows = image.RasterYSize
    geotransform = image.GetGeoTransform()
    proj = image.GetProjection()
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows
    X_Y_raster_size = [cols, rows]
    extent = [minx, miny, maxx, maxy]
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj
    image_array = np.array(image.ReadAsArray(0, 0, cols, rows))

    return image_array, information;


def check_Mission(acquisition_name):
    basename = os.path.basename(acquisition_name)

    if 'LT05' in acquisition_name:
        sensor = 'L5'
    elif 'LE07' in acquisition_name:
        sensor = 'L7'
    elif 'LC08' in acquisition_name:
        sensor = 'L8'

    elif 'LC09' in acquisition_name:
        sensor = 'L8'

    elif 'S2' in acquisition_name:
        sensor = 'S2'
    else:
        print('This is not a Landsat!')

    return sensor;


def Plot_TS(time_series_folder, png_folder, start_date, end_date, scf_subfolder_name):
    # SCF maps

    scf_maps_list_all = glob.glob(os.path.join(time_series_folder, 'S2*', scf_subfolder_name, '*SVM_FSC_map.tif'))
    scf_maps_list = sorted([f for f in scf_maps_list_all if
                            os.path.basename(f).split('_')[2].split('T')[0] >= start_date and
                            os.path.basename(f).split('_')[2].split('T')[0] <= end_date])

    print('{} SCF maps detected for the period {} - {} in {}'.format(len(scf_maps_list), start_date, end_date,
                                                                     os.path.basename(time_series_folder)))

    TS_data = {'date_list': [],

               'SCA_perc': [],
               'SCA_km2': [],
               'cloud_cover': []}

    for curr_SCF in scf_maps_list:
        date = os.path.basename(curr_SCF).split('_')[2].split('T')[0]
        # print(date)

        curr_snow_map, curr_snow_map_info = open_image(curr_SCF)

        resolution = curr_snow_map_info['geotransform'][1]

        valid_mask = curr_snow_map <= 100

        # SCA km2
        curr_sca_km2 = np.round(
            np.sum((curr_snow_map[valid_mask] / 100) * curr_snow_map_info['geotransform'][1] ** 2) * 10 ** -6, 2)
        # SCA km2
        curr_sca_perc = np.round(np.sum((curr_snow_map[valid_mask] / 100)), 2) / np.sum(valid_mask)

        # cloud cover

        curr_cloud_cover = np.round(np.sum(curr_snow_map == 205) / np.sum(curr_snow_map != 255), 2)

        TS_data['SCA_perc'].append(curr_sca_perc)
        TS_data['SCA_km2'].append(curr_sca_km2)
        TS_data['date_list'].append(date)
        TS_data['cloud_cover'].append(curr_cloud_cover)

    pixel_size = curr_snow_map_info['geotransform'][1] ** 2
    Total_area_km2 = np.sum(curr_snow_map != 255) * pixel_size * 10 ** -6
    TS_data_df = pd.DataFrame.from_dict(TS_data)
    TS_data_df['date_list'] = pd.to_datetime(TS_data_df['date_list'])

    # Plot SCA

    path_imge_km2 = os.path.join(png_folder, '00_SCF_km2_{}_{}_{}.png'.format(start_date, end_date,
                                                                              os.path.basename(time_series_folder)))
    path_imge_perc = os.path.join(png_folder, '00_SCF_perc_{}_{}_{}.png'.format(start_date, end_date,
                                                                                os.path.basename(time_series_folder)))

    TS_data_df.plot(x='date_list', y=['SCA_km2'], style=['bo'])
    plt.errorbar(TS_data_df['date_list'], TS_data_df['SCA_km2'],
                 yerr=[np.zeros_like(TS_data_df['cloud_cover']),
                       TS_data_df['cloud_cover'] * Total_area_km2], linestyle='None', color='y', label='Cloud')
    plt.ylabel('SCA km2')
    plt.title('SCF km2 maps detected for the period {} - {}\n in {}'.format(start_date, end_date,
                                                                            os.path.basename(time_series_folder)))
    plt.savefig(path_imge_km2)

    TS_data_df.plot(x='date_list', y=['SCA_perc'], style=['ro'])
    plt.errorbar(TS_data_df['date_list'], TS_data_df['SCA_perc'],
                 yerr=[np.zeros_like(TS_data_df['cloud_cover']),
                       TS_data_df['cloud_cover']], linestyle='None', color='y', label='Cloud')
    plt.ylabel('SCA %')
    plt.title('SCF % maps detected for the period {} - {}\n in {}'.format(start_date, end_date,
                                                                          os.path.basename(time_series_folder)))
    plt.savefig(path_imge_perc)

    return



if __name__ == "__main__":



    time_series_folder = '/mnt/CEPH_PROJECTS/FRAM3S/Sentinel-2/Results/32TPT'
    png_folder = os.path.join('/mnt/CEPH_PROJECTS/FRAM3S/Sentinel-2/PNG')
    scf_subfolder_name = 'SCF'
    
    start_date = '20101223'
    end_date = '20251223'
    
    plot_ts = False
    save_png = True
    #######################################################
    
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    
    if plot_ts:
        Plot_TS(time_series_folder, png_folder, start_date, end_date, scf_subfolder_name)
    
    if save_png:
        save_scene_png(time_series_folder, png_folder, start_date, end_date, scf_subfolder_name)








