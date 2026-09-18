#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:56:32 2026

@author: vpremier
"""

import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation


SENSOR_BANDS = {
    "S2": ["B04", "B03", "B8A"],
    "L5": ["red", "green", "nir08"],
    "L7": ["red", "green", "nir08"],
    "L8": ["red", "green", "nir08"],
}


def compute_omnicloudmask(
    data: xr.DataArray,
    sensor: str,
    no_data_value=np.nan,
    dilation_iterations: int = 3,
    band_dimension: str = "band",
) -> xr.DataArray:
    """Compute an OmniCloudMask classification without file-system operations."""

    if dilation_iterations < 0:
        raise ValueError("dilation_iterations must be non-negative")

    try:
        required_bands = SENSOR_BANDS[sensor]
    except KeyError:
        raise ValueError(
            f"Cloud masking is not supported for sensor {sensor!r}"
        )

    missing_bands = [
        band
        for band in required_bands
        if band not in data.coords[band_dimension]
    ]

    if missing_bands:
        raise ValueError(
            f"Missing bands for sensor {sensor!r}: {missing_bands}"
        )

    selected = data.sel({band_dimension: required_bands})

    # Make the band dimension explicit instead of relying on its position.
    selected = selected.transpose(
        band_dimension,
        *[dim for dim in selected.dims if dim != band_dimension],
    )

    input_array = selected.values.astype(np.float32)

    mask = predict_from_array(
        input_array,
        no_data_value=no_data_value,
    )
    mask = np.squeeze(mask)

    thick_cloud = mask == 1

    if dilation_iterations > 0:
        thick_cloud = binary_dilation(
            thick_cloud,
            iterations=dilation_iterations,
        )

    # Avoid modifying data potentially owned by the prediction function.
    mask = mask.copy()
    mask[thick_cloud] = 1

    spatial_dimensions = [
        dim for dim in selected.dims if dim != band_dimension
    ]

    if mask.ndim != len(spatial_dimensions):
        raise ValueError(
            "Unexpected OmniCloudMask output shape: "
            f"{mask.shape}. Expected dimensions {spatial_dimensions}."
        )

    coordinates = {
        dim: selected.coords[dim]
        for dim in spatial_dimensions
    }

    return xr.DataArray(
        mask.astype(np.uint8),
        dims=spatial_dimensions,
        coords=coordinates,
        name="cloud_mask",
        attrs={
            "long_name": "OmniCloudMask classification",
            "classes": "0=clear, 1=thick cloud, 2=thin cloud",
        },
    )