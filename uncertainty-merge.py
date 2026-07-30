#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:14:05 2026

@author: vpremier
"""

from pathlib import Path

import numpy as np
import rasterio


source_dir = Path("/mnt/CEPH_PROJECTS/SNOWCOP/Paloma/Area06/S2-new")
target_dir = Path("/mnt/CEPH_PROJECTS/SNOWCOP/Vale/SCF_Diego")

target_dir.mkdir(parents=True, exist_ok=True)


for folder in source_dir.rglob("*"):
    if not folder.is_dir():
        continue

    # Prefer glacier-corrected SCF
    glaciers = list(folder.glob("*_SnowFLAKES_GLACIERS.tif"))

    if glaciers:
        scf_path = glaciers[0]

        # Adjust this pattern if your uncertainty filename differs
        uncertainty_path = folder / scf_path.name.replace(
            "_SnowFLAKES_GLACIERS.tif",
            "_SnowFLAKES_UNC.tif",
        )
    else:
        normal = list(folder.glob("*_SnowFLAKES.tif"))

        # Exclude uncertainty files if the pattern could match them
        normal = [
            path for path in normal
            if not path.name.endswith("_SnowFLAKES_UNC.tif")
        ]

        if not normal:
            continue

        scf_path = normal[0]
        uncertainty_path = folder / scf_path.name.replace(
            "_SnowFLAKES.tif",
            "_SnowFLAKES_UNC.tif",
        )

    if not uncertainty_path.exists():
        print(f"Missing uncertainty: {uncertainty_path}")
        continue

    with rasterio.open(scf_path) as scf_src, rasterio.open(
        uncertainty_path
    ) as unc_src:

        # Check that both rasters have the same grid
        if scf_src.shape != unc_src.shape:
            print(f"Shape mismatch: {scf_path.name}")
            continue

        if scf_src.transform != unc_src.transform:
            print(f"Transform mismatch: {scf_path.name}")
            continue

        if scf_src.crs != unc_src.crs:
            print(f"CRS mismatch: {scf_path.name}")
            continue

        scf = scf_src.read(1)
        uncertainty = unc_src.read(1)

        # Both bands in one GeoTIFF must use the same dtype
        output_dtype = np.result_type(scf.dtype, uncertainty.dtype)

        scf = scf.astype(output_dtype, copy=False)
        uncertainty = uncertainty.astype(output_dtype, copy=False)

        profile = scf_src.profile.copy()
        profile.update(
            count=2,
            dtype=np.dtype(output_dtype).name,
            compress="deflate",
        )

        # Remove GLACIERS from the output name, when present
        output_name = scf_path.name.replace(
            "_SnowFLAKES_GLACIERS.tif",
            "_SnowFLAKES.tif",
        )

        output_path = target_dir / output_name

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(scf, 1)
            dst.write(uncertainty, 2)

            dst.set_band_description(1, "Snow Cover Fraction")
            dst.set_band_description(2, "SCF Uncertainty")

            # Preserve optional metadata from each input band
            dst.update_tags(1, **scf_src.tags(1))
            dst.update_tags(2, **unc_src.tags(1))

    print(f"Created: {output_path}")

print("Done!")