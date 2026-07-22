#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:02:50 2026

@author: vpremier
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import os


import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def fit_distribution_and_median(
    feature,
    feature_name,
    mask,
    curr_aux_folder,
    distribution="auto",
    bins=50,
    plot=True,
    ax=None,
    default_median=np.nan
):
    """
    Fit a Gaussian or lognormal distribution to masked feature values.

    If there are not enough valid values, or the distribution cannot be fitted,
    return a dictionary containing default values.

    Parameters
    ----------
    feature : ndarray
        Input feature array.

    feature_name : str
        Feature name used for labels and the output filename.

    mask : ndarray of bool
        Boolean mask with the same shape as feature.

    curr_aux_folder : str or Path
        Auxiliary directory. The histogram is saved in a sibling SCF folder.

    distribution : {"gaussian", "lognormal", "auto"}
        Distribution to fit.

    bins : int
        Number of histogram bins.

    plot : bool
        Whether to create and save the histogram.

    ax : matplotlib.axes.Axes or None
        Existing plotting axes.

    default_median : float
        Median returned when fitting is not possible.

    Returns
    -------
    result : dict
        Distribution fit information.
    """

    def make_default_result(default_median, n_samples=0, reason=None):
        return {
            "distribution": "gaussian",
            "empirical_median": default_median,
            "fitted_median": default_median,
            "parameters": {
                "mean": default_median,
                "std": 0.0,
            },
            "log_likelihood": np.nan,
            "aic": np.nan,
            "n_samples": int(n_samples),
            "fit_success": False,
            "reason": reason,
        }

    # Output directory
    wd = Path(curr_aux_folder).parent
    scf_folder = wd / "SCF"
    scf_folder.mkdir(parents=True, exist_ok=True)

    feature = np.asarray(feature)
    mask = np.asarray(mask, dtype=bool)

    if feature.shape != mask.shape:
        raise ValueError(
            "feature and mask must have the same shape. "
            f"Received {feature.shape} and {mask.shape}."
        )

    # Extract finite masked values
    values = feature[mask].astype(np.float64).ravel()
    values = values[np.isfinite(values)]

    # Return defaults when there are insufficient valid observations
    if values.size < 20:
        return make_default_result(
            default_median=default_median,
            n_samples=values.size,
            reason="Fewer than 20 finite masked values are available."
        )

    empirical_median = float(np.median(values))

    def fit_gaussian(data):
        mean, std = stats.norm.fit(data)

        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(
                "Gaussian fit produced invalid parameters or zero variance."
            )

        log_likelihood = np.sum(
            stats.norm.logpdf(data, loc=mean, scale=std)
        )

        aic = 2 * 2 - 2 * log_likelihood
        fitted_median = stats.norm.median(loc=mean, scale=std)

        return {
            "distribution": "gaussian",
            "parameters": {
                "mean": float(mean),
                "std": float(std),
            },
            "fitted_median": float(fitted_median),
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "n_fit_samples": int(data.size),
            "pdf": lambda x: stats.norm.pdf(
                x,
                loc=mean,
                scale=std
            ),
        }

    def fit_lognormal(data):
        positive_data = data[data > 0]

        if positive_data.size < 3:
            raise ValueError(
                "A lognormal fit requires at least three positive values."
            )

        shape, loc, scale = stats.lognorm.fit(
            positive_data,
            floc=0
        )

        if (
            not np.isfinite(shape)
            or not np.isfinite(loc)
            or not np.isfinite(scale)
            or shape <= 0
            or scale <= 0
        ):
            raise ValueError(
                "Lognormal fit produced invalid parameters."
            )

        log_likelihood = np.sum(
            stats.lognorm.logpdf(
                positive_data,
                shape,
                loc=loc,
                scale=scale
            )
        )

        aic = 2 * 2 - 2 * log_likelihood

        fitted_median = stats.lognorm.median(
            shape,
            loc=loc,
            scale=scale
        )

        return {
            "distribution": "lognormal",
            "parameters": {
                "std": float(shape),
                "loc": float(loc),
                "scale": float(scale),
            },
            "fitted_median": float(fitted_median),
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "n_fit_samples": int(positive_data.size),
            "pdf": lambda x: stats.lognorm.pdf(
                x,
                shape,
                loc=loc,
                scale=scale
            ),
        }

    distribution = distribution.lower()

    if distribution not in {"gaussian", "lognormal", "auto"}:
        raise ValueError(
            "distribution must be 'gaussian', 'lognormal', or 'auto'."
        )

    # Fit the requested distribution while safely handling failures
    try:
        if distribution == "gaussian":
            selected_fit = fit_gaussian(values)

        elif distribution == "lognormal":
            selected_fit = fit_lognormal(values)

        else:
            fits = []

            try:
                fits.append(fit_gaussian(values))
            except (ValueError, RuntimeError, FloatingPointError):
                pass

            try:
                fits.append(fit_lognormal(values))
            except (ValueError, RuntimeError, FloatingPointError):
                pass

            if not fits:
                return make_default_result(
                    n_samples=values.size,
                    reason="Neither Gaussian nor lognormal fitting succeeded."
                )

            selected_fit = min(fits, key=lambda fit: fit["aic"])

    except (ValueError, RuntimeError, FloatingPointError) as error:
        result = make_default_result(
            n_samples=values.size,
            reason=str(error)
        )

        # The empirical median is still available
        result["empirical_median"] = empirical_median

        return result

    result = {
        key: value
        for key, value in selected_fit.items()
        if key != "pdf"
    }

    result.update({
        "empirical_median": empirical_median,
        "n_samples": int(values.size),
        "fit_success": True,
        "reason": None,
    })

    if plot:
        created_figure = ax is None

        if created_figure:
            _, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Observed distribution"
        )

        x_min, x_max = np.percentile(values, [0.1, 99.9])

        if not np.isfinite(x_min) or not np.isfinite(x_max):
            x_min = values.min()
            x_max = values.max()

        if x_min == x_max:
            padding = max(abs(x_min) * 0.01, 1e-6)
            x_min -= padding
            x_max += padding

        if selected_fit["distribution"] == "lognormal":
            x_min = max(x_min, np.finfo(float).eps)

        x = np.linspace(x_min, x_max, 1000)
        fitted_pdf = selected_fit["pdf"](x)

        ax.plot(
            x,
            fitted_pdf,
            linewidth=2,
            label=f"Fitted {selected_fit['distribution']}"
        )

        ax.axvline(
            empirical_median,
            linestyle="--",
            linewidth=2,
            label=f"Empirical median = {empirical_median:.4f}"
        )

        ax.axvline(
            selected_fit["fitted_median"],
            linestyle=":",
            linewidth=2,
            label=(
                f"Fitted median = "
                f"{selected_fit['fitted_median']:.4f}"
            )
        )

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Probability density")
        ax.set_title(
            f"{selected_fit['distribution'].capitalize()} distribution fit"
        )
        ax.legend()
        ax.grid(alpha=0.25)

        if created_figure:
            plt.tight_layout()

            safe_feature_name = str(feature_name).replace(os.sep, "_")

            output_path = (
                scf_folder /
                f"{safe_feature_name}_histogram.png"
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    return result