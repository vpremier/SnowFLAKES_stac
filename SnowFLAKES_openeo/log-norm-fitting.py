#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:05:18 2026

@author: cmarin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import lognorm, norm
from scipy.special import logsumexp
from scipy.optimize import minimize

from pathlib import Path


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def aic_bic(loglik, n_params, n_obs):
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n_obs) - 2 * loglik
    return aic, bic


def robust_tail_endmember(values, tail="high", fraction=0.05):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    values = np.sort(values)
    k = max(1, int(np.ceil(len(values) * fraction)))

    if tail == "high":
        return np.median(values[-k:])
    elif tail == "low":
        return np.median(values[:k])
    else:
        raise ValueError("tail must be 'high' or 'low'")


def shifted_lognormal_pdf(x, shift, shape, scale):
    """
    scipy lognorm:
        shape = sigma
        scale = exp(mu)

    Shifted lognormal:
        y = x - shift
        y > 0
    """
    y = x - shift
    pdf = np.zeros_like(x, dtype=float)

    valid = y > 0
    pdf[valid] = lognorm.pdf(y[valid], s=shape, loc=0, scale=scale)

    return pdf


def shifted_lognormal_logpdf(x, shift, shape, scale):
    y = x - shift
    logpdf = np.full_like(x, -np.inf, dtype=float)

    valid = y > 0
    logpdf[valid] = lognorm.logpdf(y[valid], s=shape, loc=0, scale=scale)

    return logpdf


def shifted_lognormal_mean(shift, shape, scale):
    """
    For Y ~ LogNormal(mu, sigma):
        scale = exp(mu)
        E[Y] = scale * exp(sigma^2 / 2)

    X = shift + Y
    """
    return shift + scale * np.exp(shape**2 / 2)


def shifted_lognormal_mode(shift, shape, scale):
    """
    Mode of shifted lognormal:
        shift + scale * exp(-sigma^2)
    """
    return shift + scale * np.exp(-shape**2)


def unpack(theta):
    w_free = 1.0 / (1.0 + np.exp(-theta[0]))
    w_snow = 1.0 - w_free

    shift = theta[1]
    ln_shape = np.exp(theta[2])
    ln_scale = np.exp(theta[3])

    mu_snow = theta[4]
    sigma_snow = np.exp(theta[5])

    return w_free, w_snow, shift, ln_shape, ln_scale, mu_snow, sigma_snow







def fit_log_normal(feature, feat_name, mask, curr_aux_folder, trim_outliers=False):
    
    
    # working directory
    wd = Path(curr_aux_folder).parent
    
    # subdirectory SCF
    scf_folder = wd / "SCF"
    scf_folder.mkdir(exist_ok=True)
    
    
    values = feature[mask].reshape(-1, 1)
    
    
    # Optional outlier trimming
    low_quantile = 0.001
    high_quantile = 0.999      
    
    if trim_outliers:
        q_low, q_high = np.quantile(values, [low_quantile, high_quantile])
        values = values[(values >= q_low) & (values <= q_high)]
    
    n = len(values)
    
    print("Data summary")
    print("------------")
    print(f"Valid pixels: {n}")
    print(f"Minimum: {values.min():.6f}")
    print(f"Maximum: {values.max():.6f}")
    print(f"Mean:    {values.mean():.6f}")
    print(f"Std:     {values.std():.6f}")
    
    

    # centers from ppt
    snowfree_center_init = {
        "green_shadow": 0.075,
        "diffBNIR_shadow" : 0.008,
        "green_sun": 0.5,
        "swir_sun": 0.1
    }

    snow_center_init = {
        "green_shadow": 0.1,
        "diffBNIR_shadow" : 0.12,
        "green_sun": 0.6,
        "swir_sun": 0.2
    }

    
    # Approximate shift for log-normal (min detectable value?)
    shift_init = 0.055
    
    # Snow Gaussian initial sigma
    sigma_snow_init = 0.015
    
    # Shift must be below the minimum meaningful snow-free value.
    shift_min = 0.035
    shift_max = 0.070
    
    # Lognormal shape sigma bounds
    ln_sigma_min = 0.005
    ln_sigma_max = 1
    
    # Lognormal scale bounds.
    # scipy lognorm scale = exp(mu)
    ln_scale_min = 1e-4
    ln_scale_max = 0.20
    
    # Snow Gaussian bounds
    mu_snow_min = 0.06
    mu_snow_max = 0.20
    sigma_snow_min = 0.0003
    sigma_snow_max = 0.1
    
    # Mixture weight initialization 50 - 50?
    w_free_init = 0.5
    
    # Parameters for the end member selection
    
    # Posterior threshold for high-confidence pixels
    posterior_threshold = 0.95
    
    # Endmember extraction fraction
    tail_fraction = 0.05
    
    
    
    # Bounds
    w_min = 0.05
    w_max = 0.95



    # ============================================================
    # 4. INITIALIZATION
    # ============================================================
    
    # We want the shifted lognormal mean close to snowfree_center_init.
    # Approximate:
    # mean = shift + scale * exp(sigma^2 / 2)
    #
    # Choose a moderate lognormal sigma.
    ln_shape_init = 0.45
    ln_scale_init = (snowfree_center_init[feat_name] - shift_init) / np.exp(ln_shape_init**2 / 2)
    
    ln_scale_init = max(ln_scale_init, ln_scale_min)
    
    mu_snow_init = snow_center_init[feat_name]
    w_free_init = np.clip(w_free_init, w_min, w_max)
    
    theta0 = np.array([
        np.log(w_free_init / (1 - w_free_init)),  # logit weight snow-free
        shift_init,                               # shift x0
        np.log(ln_shape_init),                    # log lognormal sigma
        np.log(ln_scale_init),                    # log lognormal scale
        mu_snow_init,                             # Gaussian snow mean
        np.log(sigma_snow_init)                   # log Gaussian snow sigma
    ])
    
    
    
    
    
    
    # ============================================================
    # 6. OPTIMIZATION
    # ============================================================
    
    bounds = [
        # w_free
        (np.log(w_min / (1 - w_min)), np.log(w_max / (1 - w_max))),
    
        # shift
        (shift_min, shift_max),
    
        # lognormal shape sigma
        (np.log(ln_sigma_min), np.log(ln_sigma_max)),
    
        # lognormal scale
        (np.log(ln_scale_min), np.log(ln_scale_max)),
    
        # snow Gaussian mean
        (mu_snow_min, mu_snow_max),
    
        # snow Gaussian sigma
        (np.log(sigma_snow_min), np.log(sigma_snow_max))
    ]
    
    
    
    def negative_loglik(theta):
        w_free, w_snow, shift, ln_shape, ln_scale, mu_snow, sigma_snow = unpack(theta)
    
        mean_free = shifted_lognormal_mean(shift, ln_shape, ln_scale)
    
        penalty = 0.0
    
        # Physical constraint:
        # snow-free mean should be left of snow mean.
        if mean_free >= mu_snow:
            penalty += 1e8 * (mean_free - mu_snow + 1e-8) ** 2
    
        # Shift must remain lower than the smallest valid x.
        # This is mostly already handled by bounds, but this adds safety.
        if shift >= values.min():
            penalty += 1e10 * (shift - values.min() + 1e-8) ** 2
    
        log_p_free = (
            np.log(w_free)
            + shifted_lognormal_logpdf(values, shift, ln_shape, ln_scale)
        )
    
        log_p_snow = (
            np.log(w_snow)
            + norm.logpdf(values, loc=mu_snow, scale=sigma_snow)
        )
    
        log_mix = logsumexp(
            np.vstack([log_p_free, log_p_snow]),
            axis=0
        )
    
        return -np.sum(log_mix) + penalty



    result = minimize(
        negative_loglik,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 50000,
            "ftol": 1e-12
        }
    )
    
    if not result.success:
        print("\nWarning: optimization did not fully converge.")
        print(result.message)

    
    # ============================================================
    # 7. EXTRACT FITTED PARAMETERS
    # ============================================================
    
    w_free, w_snow, shift, ln_shape, ln_scale, mu_snow, sigma_snow = unpack(result.x)
    
    mean_free = shifted_lognormal_mean(shift, ln_shape, ln_scale)
    mode_free = shifted_lognormal_mode(shift, ln_shape, ln_scale)
    median_free = shift + ln_scale
    
    log_p_free = (
        np.log(w_free)
        + shifted_lognormal_logpdf(values, shift, ln_shape, ln_scale)
    )
    
    log_p_snow = (
        np.log(w_snow)
        + norm.logpdf(values, loc=mu_snow, scale=sigma_snow)
    )
    
    log_probs = np.vstack([log_p_free, log_p_snow])
    log_den = logsumexp(log_probs, axis=0)
    
    p_free = np.exp(log_p_free - log_den)
    p_snow = np.exp(log_p_snow - log_den)
    
    loglik = np.sum(log_den)
    
    # parameters:
    # 1 weight + shift + lognormal sigma + lognormal scale + Gaussian mean + Gaussian sigma
    n_params = 6
    aic, bic = aic_bic(loglik, n_params, n)

    
    # ============================================================
    # 8. CLASSIFICATION AND ENDMEMBERS
    # ============================================================
    
    snowfree_conf = values[p_free > posterior_threshold]
    snow_conf = values[p_snow > posterior_threshold]
    
    snowfree_endmember = robust_tail_endmember(
        snowfree_conf,
        tail="low",
        fraction=tail_fraction
    )
    
    snow_endmember = robust_tail_endmember(
        snow_conf,
        tail="high",
        fraction=tail_fraction
    )
    
    
    # Decision boundary
    xx_boundary = np.linspace(values.min(), values.max(), 10000)
    
    pdf_free_boundary = (
        w_free
        * shifted_lognormal_pdf(xx_boundary, shift, ln_shape, ln_scale)
    )
    
    pdf_snow_boundary = (
        w_snow
        * norm.pdf(xx_boundary, loc=mu_snow, scale=sigma_snow)
    )
    
    diff = pdf_free_boundary - pdf_snow_boundary
    crossings = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    
    if len(crossings) > 0:
        candidate_boundaries = xx_boundary[crossings]
    
        between = candidate_boundaries[
            (candidate_boundaries > mean_free) &
            (candidate_boundaries < mu_snow)
        ]
    
        if len(between) > 0:
            boundary = between[0]
        else:
            boundary = candidate_boundaries[0]
    else:
        boundary = np.nan

    
    # ============================================================
    # 9. PRINT RESULTS
    # ============================================================
    
    print("\nFitted shifted-lognormal + Gaussian mixture")
    print("-------------------------------------------")
    
    print("\nSnow-free component: shifted lognormal")
    print(f"  weight:        {w_free:.4f}")
    print(f"  shift x0:      {shift:.6f}")
    print(f"  sigma shape:   {ln_shape:.6f}")
    print(f"  scale exp(mu): {ln_scale:.6f}")
    print(f"  mode:          {mode_free:.6f}")
    print(f"  median:        {median_free:.6f}")
    print(f"  mean:          {mean_free:.6f}")
    
    print("\nSnow component: Gaussian")
    print(f"  weight:        {w_snow:.4f}")
    print(f"  mean mu:       {mu_snow:.6f}")
    print(f"  sigma:         {sigma_snow:.6f}")
    
    print("\nModel quality")
    print(f"  log-likelihood: {loglik:.3f}")
    print(f"  AIC:            {aic:.3f}")
    print(f"  BIC:            {bic:.3f}")
    
    print("\nPosterior classification")
    print(f"  threshold: {posterior_threshold}")
    print(f"  high-confidence snow-free pixels: {len(snowfree_conf)}")
    print(f"  high-confidence snow pixels:      {len(snow_conf)}")
    
    print("\nEstimated endmembers")
    print(f"  snow-free endmember: {snowfree_endmember:.6f}")
    print(f"  snow endmember:      {snow_endmember:.6f}")
    
    print("\nDecision boundary")
    if np.isfinite(boundary):
        print(f"  p(snow-free | x) = p(snow | x) at x = {boundary:.6f}")
    else:
        print("  no boundary found")




    
    # ============================================================
    # 11. PLOT FIT
    # ============================================================
    
    xx = np.linspace(values.min(), values.max(), 2000)
    
    pdf_free = w_free * shifted_lognormal_pdf(xx, shift, ln_shape, ln_scale)
    pdf_snow = w_snow * norm.pdf(xx, loc=mu_snow, scale=sigma_snow)
    pdf_total = pdf_free + pdf_snow
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(
        values,
        bins=120,
        density=True,
        alpha=0.35,
        label="Observed histogram"
    )
    
    plt.plot(
        xx,
        pdf_total,
        linewidth=2.5,
        label="Total mixture"
    )
    
    plt.plot(
        xx,
        pdf_free,
        "--",
        linewidth=2,
        label="Snow-free shifted lognormal"
    )
    
    plt.plot(
        xx,
        pdf_snow,
        "--",
        linewidth=2,
        label="Snow Gaussian"
    )
    
    plt.axvline(
        shift,
        linestyle=":",
        linewidth=2,
        label="Lognormal shift"
    )
    
    plt.axvline(
        mean_free,
        linestyle=":",
        linewidth=2,
        label="Snow-free mean"
    )
    
    plt.axvline(
        mu_snow,
        linestyle=":",
        linewidth=2,
        label="Snow mean"
    )
    
    plt.axvline(
        snowfree_endmember,
        linestyle="-.",
        linewidth=2,
        label="Snow-free endmember"
    )
    
    plt.axvline(
        snow_endmember,
        linestyle="-.",
        linewidth=2,
        label="Snow endmember"
    )
    
    if np.isfinite(boundary):
        plt.axvline(
            boundary,
            linestyle="-",
            linewidth=2,
            label="Posterior boundary"
        )
    
    plt.xlabel("Green - NIR")
    plt.ylabel("Density")
    plt.title("Shifted-lognormal + Gaussian mixture fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

