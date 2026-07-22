#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:10:29 2026

@author: vpremier
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import Affine
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def prepare_band_data(
    data: xr.DataArray,
    selected_bands: Sequence[str] | None = None,
) -> xr.DataArray:
    """
    Prepare an xarray DataArray for pixel-based classification.

    The output dimensions are always:

        (band, y, x)

    Parameters
    ----------
    data : xr.DataArray
        Input Sentinel-2 data.

        Supported examples:
            (band, y, x)
            (day, band, y, x)
            (time, band, y, x)

        Singleton time/day dimensions are removed.

    selected_bands : sequence of str or None
        Bands to use. If None, all bands are used.

    Returns
    -------
    xr.DataArray
        Data with dimensions (band, y, x).
    """

    if not isinstance(data, xr.DataArray):
        raise TypeError(
            "data must be an xarray.DataArray with a 'band' dimension."
        )

    if "band" not in data.dims:
        raise ValueError(
            f"data must contain a 'band' dimension. Found: {data.dims}"
        )

    prepared = data

    # Remove singleton non-spatial dimensions such as day or time.
    for dimension in list(prepared.dims):
        if dimension not in {"band", "y", "x"}:
            if prepared.sizes[dimension] != 1:
                raise ValueError(
                    f"Dimension {dimension!r} contains "
                    f"{prepared.sizes[dimension]} values. Select or aggregate "
                    "this dimension before classification."
                )

            prepared = prepared.squeeze(dimension, drop=True)

    if "y" not in prepared.dims or "x" not in prepared.dims:
        raise ValueError(
            f"data must contain spatial dimensions 'y' and 'x'. "
            f"Found: {prepared.dims}"
        )

    if selected_bands is not None:
        selected_bands = list(selected_bands)

        available_bands = set(
            str(value) for value in prepared.band.values
        )

        missing_bands = [
            band for band in selected_bands
            if band not in available_bands
        ]

        if missing_bands:
            raise ValueError(
                f"Requested bands are missing: {missing_bands}. "
                f"Available bands: {list(prepared.band.values)}"
            )

        prepared = prepared.sel(band=selected_bands)

    return prepared.transpose("band", "y", "x")


def build_training_samples(
    data: xr.DataArray,
    snow_mask: np.ndarray,
    ice_mask: np.ndarray,
    selected_bands: Sequence[str] | None = None,
    max_samples_per_class: int | None = None,
    random_state: int = 42,
):
    """
    Build a feature matrix from snow and ice masks.

    Parameters
    ----------
    data : xr.DataArray
        Feature cube with a band dimension.

    snow_mask : ndarray of bool
        Pixels labelled as snow.

    ice_mask : ndarray of bool
        Pixels labelled as ice.

    selected_bands : sequence of str or None
        Bands to use.

    max_samples_per_class : int or None
        Maximum number of samples retained for each class. If None, all
        available valid pixels are used.

        Setting this value also balances the classes by taking the same number
        from snow and ice.

    random_state : int
        Random seed.

    Returns
    -------
    X : ndarray
        Feature matrix with shape (n_samples, n_bands).

    y : ndarray
        Labels, where 0 is ice and 1 is snow.

    feature_names : list[str]
        Input band names.

    training_mask : ndarray of bool
        Mask of all valid labelled pixels before optional subsampling.

    sample_flat_indices : ndarray
        Flat image indices of the selected training pixels.
    """

    prepared = prepare_band_data(data, selected_bands)

    snow_mask = np.asarray(snow_mask, dtype=bool)
    ice_mask = np.asarray(ice_mask, dtype=bool)

    expected_shape = (
        prepared.sizes["y"],
        prepared.sizes["x"],
    )

    if snow_mask.shape != expected_shape:
        raise ValueError(
            f"snow_mask has shape {snow_mask.shape}, but the image shape is "
            f"{expected_shape}."
        )

    if ice_mask.shape != expected_shape:
        raise ValueError(
            f"ice_mask has shape {ice_mask.shape}, but the image shape is "
            f"{expected_shape}."
        )

    overlap = snow_mask & ice_mask

    if np.any(overlap):
        raise ValueError(
            f"The snow and ice masks overlap at "
            f"{np.count_nonzero(overlap)} pixels."
        )

    feature_cube = prepared.values.astype(np.float32)

    finite_features = np.all(
        np.isfinite(feature_cube),
        axis=0,
    )

    valid_snow = snow_mask & finite_features
    valid_ice = ice_mask & finite_features

    snow_indices = np.flatnonzero(valid_snow)
    ice_indices = np.flatnonzero(valid_ice)

    if snow_indices.size == 0:
        raise ValueError("No valid snow training pixels were found.")

    if ice_indices.size == 0:
        raise ValueError("No valid ice training pixels were found.")

    rng = np.random.default_rng(random_state)

    if max_samples_per_class is not None:
        if max_samples_per_class <= 0:
            raise ValueError(
                "max_samples_per_class must be greater than zero."
            )

        n_per_class = min(
            int(max_samples_per_class),
            snow_indices.size,
            ice_indices.size,
        )

        snow_indices = rng.choice(
            snow_indices,
            size=n_per_class,
            replace=False,
        )

        ice_indices = rng.choice(
            ice_indices,
            size=n_per_class,
            replace=False,
        )

    sample_flat_indices = np.concatenate(
        [ice_indices, snow_indices]
    )

    y = np.concatenate([
        np.zeros(ice_indices.size, dtype=np.uint8),
        np.ones(snow_indices.size, dtype=np.uint8),
    ])

    # Convert cube from (band, y, x) to (pixel, band).
    all_pixels = feature_cube.reshape(
        feature_cube.shape[0],
        -1,
    ).T

    X = all_pixels[sample_flat_indices]

    # Randomize the order of the samples.
    order = rng.permutation(X.shape[0])

    X = X[order]
    y = y[order]
    sample_flat_indices = sample_flat_indices[order]

    feature_names = [
        str(value) for value in prepared.band.values
    ]

    training_mask = valid_snow | valid_ice

    return (
        X,
        y,
        feature_names,
        training_mask,
        sample_flat_indices,
    )


def train_snow_ice_classifier(
    data: xr.DataArray,
    snow_mask: np.ndarray,
    ice_mask: np.ndarray,
    selected_bands: Sequence[str] | None = None,
    max_samples_per_class: int | None = None,
    test_size: float = 0.30,
    random_state: int = 42,
    model_parameters: dict | None = None,
):
    """
    Train and evaluate an XGBoost snow-versus-ice classifier.

    Returns
    -------
    result : dict
        Contains the trained model, metrics, test predictions, feature names,
        and sample information.
    """

    (
        X,
        y,
        feature_names,
        training_mask,
        sample_flat_indices,
    ) = build_training_samples(
        data=data,
        snow_mask=snow_mask,
        ice_mask=ice_mask,
        selected_bands=selected_bands,
        max_samples_per_class=max_samples_per_class,
        random_state=random_state,
    )

    if np.unique(y).size != 2:
        raise ValueError(
            "Both snow and ice classes are required for training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    default_parameters = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
    }

    if model_parameters is not None:
        default_parameters.update(model_parameters)

    model = XGBClassifier(**default_parameters)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test).astype(np.uint8)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, y_pred)
        ),
        "precision_snow": float(
            precision_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_snow": float(
            recall_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_snow": float(
            f1_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        ),
        "n_total_samples": int(X.shape[0]),
        "n_training_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_ice_samples": int(np.count_nonzero(y == 0)),
        "n_snow_samples": int(np.count_nonzero(y == 1)),
    }

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["ice", "snow"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "feature_names": feature_names,
        "selected_bands": feature_names,
        "training_mask": training_mask,
        "sample_flat_indices": sample_flat_indices,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_probability": y_probability,
    }


def predict_snow_ice_map(
    data: xr.DataArray,
    model: XGBClassifier,
    selected_bands: Sequence[str],
    prediction_mask: np.ndarray | None = None,
    probability_threshold: float = 0.5,
    invalid_class_value: int = 255,
):
    """
    Apply the trained model to an image.

    Parameters
    ----------
    data : xr.DataArray
        Sentinel-2 feature cube.

    model : XGBClassifier
        Trained binary classifier.

    selected_bands : sequence of str
        Bands used during training, in the same order.

    prediction_mask : ndarray of bool or None
        Optional mask limiting where predictions are made. For example, this
        could be a glacier mask. If None, every finite pixel is classified.

    probability_threshold : float
        Snow is assigned when P(snow) >= this value.

    invalid_class_value : int
        Value assigned to invalid or excluded pixels.

    Returns
    -------
    classification : xr.DataArray
        Integer classification map:
            0 = ice
            1 = snow
            invalid_class_value = invalid

    snow_probability : xr.DataArray
        Probability of the snow class.
    """

    if not 0 <= probability_threshold <= 1:
        raise ValueError(
            "probability_threshold must be between 0 and 1."
        )

    prepared = prepare_band_data(
        data,
        selected_bands=selected_bands,
    )

    feature_cube = prepared.values.astype(np.float32)

    height = prepared.sizes["y"]
    width = prepared.sizes["x"]

    X_all = feature_cube.reshape(
        feature_cube.shape[0],
        -1,
    ).T

    valid = np.all(np.isfinite(X_all), axis=1)

    if prediction_mask is not None:
        prediction_mask = np.asarray(
            prediction_mask,
            dtype=bool,
        )

        if prediction_mask.shape != (height, width):
            raise ValueError(
                f"prediction_mask has shape {prediction_mask.shape}, "
                f"but the expected shape is {(height, width)}."
            )

        valid &= prediction_mask.ravel()

    probability_flat = np.full(
        height * width,
        np.nan,
        dtype=np.float32,
    )

    class_flat = np.full(
        height * width,
        invalid_class_value,
        dtype=np.uint8,
    )

    if np.any(valid):
        snow_probability = model.predict_proba(
            X_all[valid]
        )[:, 1]

        probability_flat[valid] = snow_probability.astype(
            np.float32
        )

        class_flat[valid] = (
            snow_probability >= probability_threshold
        ).astype(np.uint8)

    classification = xr.DataArray(
        class_flat.reshape(height, width),
        dims=("y", "x"),
        coords={
            "y": prepared.y,
            "x": prepared.x,
        },
        name="snow_ice_class",
        attrs={
            "class_0": "ice",
            "class_1": "snow",
            "invalid_value": invalid_class_value,
            "probability_threshold": probability_threshold,
            "grid_mapping": "spatial_ref",
        },
    )

    snow_probability = xr.DataArray(
        probability_flat.reshape(height, width),
        dims=("y", "x"),
        coords={
            "y": prepared.y,
            "x": prepared.x,
        },
        name="snow_probability",
        attrs={
            "long_name": "Probability of snow",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "grid_mapping": "spatial_ref",
        },
    )

    # Preserve CRS and spatial reference when available.
    if "spatial_ref" in prepared.coords:
        classification = classification.assign_coords(
            spatial_ref=prepared.spatial_ref
        )
        snow_probability = snow_probability.assign_coords(
            spatial_ref=prepared.spatial_ref
        )

    try:
        if prepared.rio.crs is not None:
            classification = classification.rio.write_crs(
                prepared.rio.crs,
                inplace=False,
            )
            snow_probability = snow_probability.rio.write_crs(
                prepared.rio.crs,
                inplace=False,
            )

            classification = classification.rio.write_transform(
                prepared.rio.transform(),
                inplace=False,
            )
            snow_probability = snow_probability.rio.write_transform(
                prepared.rio.transform(),
                inplace=False,
            )
    except AttributeError:
        # rioxarray may not be imported or spatial metadata may be unavailable.
        pass

    return classification, snow_probability


def save_classification_geotiff(
    classification: xr.DataArray,
    output_path: str | Path,
    reference_data: xr.DataArray,
    nodata: int = 255,
):
    """
    Save the snow/ice class map as a uint8 GeoTIFF.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepare_band_data(reference_data)

    crs = prepared.rio.crs
    transform = prepared.rio.transform()

    if crs is None:
        raise ValueError(
            "No CRS is attached to reference_data."
        )

    array = np.asarray(classification.values, dtype=np.uint8)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(array, 1)
        destination.set_band_description(
            1,
            "Snow/ice class: 0=ice, 1=snow, 255=invalid",
        )


def save_probability_geotiff(
    probability: xr.DataArray,
    output_path: str | Path,
    reference_data: xr.DataArray,
):
    """
    Save the snow probability map as a float32 GeoTIFF.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepare_band_data(reference_data)

    crs = prepared.rio.crs
    transform = prepared.rio.transform()

    if crs is None:
        raise ValueError(
            "No CRS is attached to reference_data."
        )

    array = np.asarray(probability.values, dtype=np.float32)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(array, 1)
        destination.set_band_description(
            1,
            "Probability of snow",
        )


def save_classifier_outputs(
    training_result: dict,
    output_folder: str | Path,
):
    """
    Save model, metrics, confusion matrix, and feature importance.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = training_result["model"]
    metrics = training_result["metrics"]
    report = training_result["classification_report"]
    feature_importance = training_result["feature_importance"]

    joblib.dump(
        model,
        output_folder / "snow_ice_xgboost_model.joblib",
    )

    with open(
        output_folder / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        output_folder / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    feature_importance.to_csv(
        output_folder / "feature_importance.csv",
        header=True,
    )

    figure, axis = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=training_result["confusion_matrix"],
        display_labels=["Ice", "Snow"],
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title("Snow–ice confusion matrix")

    figure.tight_layout()
    figure.savefig(
        output_folder / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))

    feature_importance.sort_values().plot.barh(ax=axis)

    axis.set_xlabel("XGBoost feature importance")
    axis.set_ylabel("Band")
    axis.set_title("Snow–ice classifier feature importance")

    figure.tight_layout()
    figure.savefig(
        output_folder / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def run_snow_ice_classification(
    data: xr.DataArray,
    snow_mask: np.ndarray,
    ice_mask: np.ndarray,
    output_folder: str | Path | None = None,
    selected_bands: Sequence[str] | None = None,
    prediction_mask: np.ndarray | None = None,
    max_samples_per_class: int | None = 10000,
    test_size: float = 0.30,
    probability_threshold: float = 0.5,
    random_state: int = 42,
    model_parameters: dict | None = None,
):
    """
    Complete snow-versus-ice classification workflow.
    """

    training_result = train_snow_ice_classifier(
        data=data,
        snow_mask=snow_mask,
        ice_mask=ice_mask,
        selected_bands=selected_bands,
        max_samples_per_class=max_samples_per_class,
        test_size=test_size,
        random_state=random_state,
        model_parameters=model_parameters,
    )

    classification, snow_probability = predict_snow_ice_map(
        data=data,
        model=training_result["model"],
        selected_bands=training_result["feature_names"],
        prediction_mask=prediction_mask,
        probability_threshold=probability_threshold,
    )

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        save_classifier_outputs(
            training_result,
            output_folder,
        )

        save_classification_geotiff(
            classification=classification,
            output_path=output_folder / "snow_ice_classification.tif",
            reference_data=data,
        )

        save_probability_geotiff(
            probability=snow_probability,
            output_path=output_folder / "snow_probability.tif",
            reference_data=data,
        )

    return {
        **training_result,
        "classification": classification,
        "snow_probability": snow_probability,
    }


