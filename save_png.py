#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:58:28 2026

@author: vpremier
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colormaps


def create_colormap():
    """Creates a custom colormap for snow cover visualization."""
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


def read_rgb(path):
    with rasterio.open(path) as src:
        img = src.read()

        rgb = np.dstack([img[0], img[1], img[2]]).astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
        return rgb


def read_class(path):
    with rasterio.open(path) as src:
        return src.read(1)


def save_side_by_side(rgb_path, class_path, out_path):
    rgb = read_rgb(rgb_path)
    cls = read_class(class_path)

    cmap = create_colormap()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[0].axis("off")

    ax[1].imshow(cls, cmap=cmap, vmin=0, vmax=255)
    ax[1].set_title("Classification")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    
    import glob    
    import os
    # ----------------------------
    # Base paths
    # ----------------------------
    base_dir = r"/mnt/CEPH_PROJECTS/PROSNOW/Careser/Sentinel2/"
    out_dir = r"/mnt/CEPH_PROJECTS/PROSNOW/Careser/PNG/"
    
    # ----------------------------
    # Find all classification files
    # ----------------------------
    class_files = list(glob.glob(base_dir + os.sep + "*/SCF/*SnowFLAKES.tif"))
    
    print(f"Found {len(class_files)} classification files")
    
    # ----------------------------
    # Process each classification
    # ----------------------------
    for cls_path in class_files:
        
        scene_id = os.path.basename(cls_path).replace("_SnowFLAKES.tif", "")

        print(f"Processing: {scene_id}")
        
        # ----------------------------
        # Find matching RGB (search by scene_id)
        # ----------------------------
        rgb_candidates = list(glob.glob(base_dir + os.sep + f"{scene_id}/false_color_composite*.tif")).pop()


        save_side_by_side(
            rgb_candidates,
            cls_path,
            os.path.join(out_dir, scene_id + '.png')
        )