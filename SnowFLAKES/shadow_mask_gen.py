#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:28:37 2025

@author: rbarella
"""

import numpy as np
from numpy import sin, cos, tan, arcsin
from pyproj import Transformer
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
from datetime import timezone
import sys
import rasterio


def get_central_coords_(curr_image_info, date_time):
    E_min, N_min, E_max, N_max = curr_image_info['extent']
    epsg_code = curr_image_info['EPSG']

    # Transform the coordinates to WGS84
    transformer = Transformer.from_crs(f"epsg:{epsg_code}", "epsg:4326", always_xy=True)
    central_E = E_min + (E_max - E_min) / 2
    central_N = N_min + (N_max - N_min) / 2
    Central_WGS84 = transformer.transform(central_E, central_N)

    # Convert date_time to UTC timezone
    datetime_object = date_time.replace(tzinfo=timezone.utc)

    return Central_WGS84[1], Central_WGS84[0], datetime_object


def get_timezone_from_coordinates(latitude, longitude):
    """Determine the timezone based on latitude and longitude."""
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
    if timezone_str is None:
        raise ValueError("Could not determine timezone for the given coordinates.")
    return timezone_str


def calculate_sun_vector_with_auto_timezone(date_time, curr_image_info):
    """Calculate sun vector with automatic timezone determination."""

    latitude, longitude, datetime_object = get_central_coords_(curr_image_info, date_time)

    timezone_str = get_timezone_from_coordinates(latitude, longitude)
    timezone = pytz.timezone(timezone_str)
    local_offset = timezone.utcoffset(date_time).total_seconds() / 3600.0

    # Convert datetime to Julian date
    julian_date = to_juliandate(date_time)

    # Compute the sun vector
    sun_vec = sun_vector(julian_date, latitude, longitude, local_offset)
    return sun_vec


def to_juliandate(d):
    """Convert a datetime object to a julian date.

    A Julian date is the decimal number of days since January 1, 4713 BCE."""
    seconds_per_day = 86400
    return d.timestamp() / seconds_per_day + 2440587.5


# TODO TdR 07/10/16: test
def sun_vector(julian_date, latitude, longitude, timezone):
    """Calculate a unit vector in the direction of the sun.

    :param date_time: utc timestamp, time of observation
    :param latitude: latitude of observation
    :param longitude: longitude of observation
    :param timezone: timezone hour offset relative to UTC
    :return: 3-dimensional unit vector.
    """
    # TODO TdR 07/10/16: verify computations
    omega_r = _hour_angle(julian_date, longitude, timezone)
    delta_r = np.deg2rad(sun_declination(julian_date))
    lambda_r = np.deg2rad(latitude)

    # TODO TdR 27/09/16: factor out common computation
    svx = - sin(omega_r) * cos(delta_r)
    svy = sin(lambda_r) * cos(omega_r) * cos(delta_r) \
          - cos(lambda_r) * sin(delta_r)
    svz = cos(lambda_r) * cos(omega_r) * cos(delta_r) \
          + sin(lambda_r) * sin(delta_r)
    return np.array([svx, svy, svz])


def sun_declination(julian_date):
    """Compute the declination of the sun on a given day."""
    # TODO TdR 27/09/16: verify calculations.
    jdc = (julian_date - 2451545.0) / 36525.0
    sec = 21.448 - jdc * (46.8150 + jdc * (0.00059 - jdc * .001813))
    e0 = 23.0 + (26.0 + (sec / 60.0)) / 60.0
    oblcorr = e0 + 0.00256 * cos(np.deg2rad(125.04 - 1934.136 * jdc))
    l0 = 280.46646 + jdc * (36000.76983 + jdc * 0.0003032)
    l0 = (l0 - 360 * (l0 // 360)) % 360
    gmas = 357.52911 + jdc * (35999.05029 - 0.0001537 * jdc)
    gmas = np.deg2rad(gmas)
    seqcent = sin(gmas) * (1.914602 - jdc * (0.004817 + 0.000014 * jdc)) + \
              sin(2 * gmas) * (0.019993 - 0.000101 * jdc) + sin(3 * gmas) * 0.000289

    suntl = l0 + seqcent
    sal = suntl - 0.00569 - 0.00478 * sin(np.deg2rad(125.04 - 1934.136 * jdc))
    delta = arcsin(sin(np.deg2rad(oblcorr)) * sin(np.deg2rad(sal)))
    return np.rad2deg(delta)


def _equation_of_time(julian_date):
    """Calculate the equation of time.

    See https://en.wikipedia.org/wiki/Equation_of_time.
    """
    # TODO TdR 07/10/16: verify computations.
    jdc = (julian_date - 2451545.0) / 36525.0
    sec = 21.448 - jdc * (46.8150 + jdc * (0.00059 - jdc * 0.001813))
    e0 = 23.0 + (26.0 + (sec / 60.0)) / 60.0
    oblcorr = e0 + 0.00256 * cos(np.deg2rad(125.04 - 1934.136 * jdc))
    l0 = 280.46646 + jdc * (36000.76983 + jdc * 0.0003032)
    l0 = (l0 - 360 * (l0 // 360)) % 360
    gmas = 357.52911 + jdc * (35999.05029 - 0.0001537 * jdc)
    gmas = np.deg2rad(gmas)

    ecc = 0.016708634 - jdc * (0.000042037 + 0.0000001267 * jdc)
    y = (tan(np.deg2rad(oblcorr) / 2)) ** 2
    rl0 = np.deg2rad(l0)
    EqTime = y * sin(2 * rl0) \
             - 2.0 * ecc * sin(gmas) \
             + 4.0 * ecc * y * sin(gmas) * cos(2 * rl0) \
             - 0.5 * y * y * sin(4 * rl0) \
             - 1.25 * ecc * ecc * sin(2 * gmas)
    return np.rad2deg(EqTime) * 4


# TODO TdR 07/10/16: test
def _hour_angle(julian_date, longitude, timezone):
    """Internal function for solar position calculation."""
    # TODO TdR 07/10/16: verify computations
    hour = ((julian_date - np.floor(julian_date)) * 24 + 12) % 24
    time_offset = _equation_of_time(julian_date)
    standard_meridian = timezone * 15
    delta_longitude_time = (longitude - standard_meridian) * 24.0 / 360.0
    omega_r = np.pi * (
            ((hour + delta_longitude_time + time_offset / 60) / 12.0) - 1.0)
    return omega_r


##### SHADOEW


def project_shadows_from_path(dem_path, sun_vector):
    """Cast shadows on the DEM from a given sun position.

    :param dem_path: str, path to the DEM file
    :param sun_vector: np.array, 3D sun vector
    :return: np.array, shadow mask (1 = in shadow, 0 = in sun)
    """
    # Load DEM from file
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        dx = transform[0]  # Pixel size in x direction
        dy = -transform[4]  # Pixel size in y direction (negative due to rasterio conventions)

    inverse_sun_vector = _invert_sun_vector(sun_vector)
    normal_sun_vector = _normalize_sun_vector(sun_vector)

    rows, cols = dem.shape
    z = dem.T

    # Determine sun direction.
    if sun_vector[0] < 0:
        # The sun shines from the West.
        start_col = 1
    else:
        # The sun shines from the East.
        start_col = cols - 1

    if sun_vector[1] < 0:
        # The sun shines from the North.
        start_row = 1
    else:
        # The sun shines from the South.
        start_row = rows - 1

    in_shadow = np.zeros_like(z)
    # Project West-East
    row = start_row
    for col in range(cols):
        _cast_shadow(row, col, rows, cols, dx, in_shadow, inverse_sun_vector,
                     normal_sun_vector, z)

    # Project North-South
    col = start_col
    for row in range(rows):
        _cast_shadow(row, col, rows, cols, dy, in_shadow, inverse_sun_vector,
                     normal_sun_vector, z)

    # Convert shadow mask to 1 for shadowed and 0 for non-shadowed pixels
    shadow_mask = (in_shadow == 0).astype(int)

    return shadow_mask.T


def _normalize_sun_vector(sun_vector):
    normal_sun_vector = np.zeros(3)
    normal_sun_vector[2] = np.sqrt(sun_vector[0] ** 2 + sun_vector[1] ** 2)
    normal_sun_vector[0] = -sun_vector[0] * sun_vector[2] / normal_sun_vector[2]
    normal_sun_vector[1] = -sun_vector[1] * sun_vector[2] / normal_sun_vector[2]
    return normal_sun_vector


def _invert_sun_vector(sun_vector):
    return -sun_vector / max(abs(sun_vector[:2]))


def _cast_shadow(row, col, rows, cols, dl, in_shadow, inverse_sun_vector,
                 normal_sun_vector, z):
    n = 0
    z_previous = -sys.float_info.max

    while True:
        # Calculate projection offset
        dx = inverse_sun_vector[0] * n
        dy = inverse_sun_vector[1] * n
        col_dx = int(round(col + dx))
        row_dy = int(round(row + dy))
        if (col_dx < 0) or (col_dx >= cols) or (row_dy < 0) or (row_dy >= rows):
            break

        vector_to_origin = np.zeros(3)
        vector_to_origin[0] = dx * dl
        vector_to_origin[1] = dy * dl
        vector_to_origin[2] = z[col_dx, row_dy]
        z_projection = np.dot(vector_to_origin, normal_sun_vector)

        if z_projection < z_previous:
            in_shadow[col_dx, row_dy] = 1
        else:
            z_previous = z_projection
        n += 1


def project_shadows_from_path(dem_path, sun_vector):
    """Cast shadows on the DEM from a given sun position.

    :param dem_path: str, path to the DEM file
    :param sun_vector: np.array, 3D sun vector
    :return: np.array, shadow mask (1 = in shadow, 0 = in sun)
    """
    # Load DEM from file
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        dx = transform[0]  # Pixel size in x direction
        dy = -transform[4]  # Pixel size in y direction (negative due to rasterio conventions)

    inverse_sun_vector = _invert_sun_vector(sun_vector)
    normal_sun_vector = _normalize_sun_vector(sun_vector)

    rows, cols = dem.shape
    z = dem.T

    # Determine sun direction.
    if sun_vector[0] < 0:
        # The sun shines from the West.
        start_col = 1
    else:
        # The sun shines from the East.
        start_col = cols - 1

    if sun_vector[1] < 0:
        # The sun shines from the North.
        start_row = 1
    else:
        # The sun shines from the South.
        start_row = rows - 1

    in_shadow = np.zeros_like(z)
    # Project West-East
    row = start_row
    for col in range(cols):
        _cast_shadow(row, col, rows, cols, dx, in_shadow, inverse_sun_vector,
                     normal_sun_vector, z)

    # Project North-South
    col = start_col
    for row in range(rows):
        _cast_shadow(row, col, rows, cols, dy, in_shadow, inverse_sun_vector,
                     normal_sun_vector, z)

    # Convert shadow mask to 1 for shadowed and 0 for non-shadowed pixels
    shadow_mask = (in_shadow == 0).astype(int)

    return shadow_mask.T


def hillshade_from_path(dem_path, sun_vector):
    """Generate a hillshade map from the DEM and sun position.

    :param dem_path: str, path to the DEM file
    :param sun_vector: np.array, 3D sun vector
    :return: np.array, hillshade map
    """
    # Load DEM from file
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        dx = transform[0]  # Pixel size in x direction
        dy = -transform[4]  # Pixel size in y direction

    # Calculate gradient
    grad = gradient(dem, dx, dy)

    # Compute hillshade
    hillshade_map = hill_shade(grad, sun_vector)

    return hillshade_map


def gradient(grid, length_x, length_y=None):
    """
    Calculate the numerical gradient of a matrix in X, Y and Z directions.

    :param grid: Matrix
    :param length_x: Length between two columns
    :param length_y: Length between two rows
    :return:
    """
    if length_y is None:
        length_y = length_x

    assert len(grid.shape) == 2, "Grid should be a matrix."

    grad = np.empty((*grid.shape, 3))
    grad[:] = np.nan
    grad[:-1, :-1, 0] = 0.5 * length_y * (
            grid[:-1, :-1] - grid[:-1, 1:] + grid[1:, :-1] - grid[1:, 1:]
    )
    grad[:-1, :-1, 1] = 0.5 * length_x * (
            grid[:-1, :-1] + grid[:-1, 1:] - grid[1:, :-1] - grid[1:, 1:]
    )
    grad[:-1, :-1, 2] = length_x * length_y

    # Copy last row and column
    grad[-1, :, :] = grad[-2, :, :]
    grad[:, -1, :] = grad[:, -2, :]

    area = np.sqrt(
        grad[:, :, 0] ** 2 +
        grad[:, :, 1] ** 2 +
        grad[:, :, 2] ** 2
    )
    for i in range(3):
        grad[:, :, i] /= area
    return grad


def hill_shade(grad, sun_vector):
    """
    Compute the intensity of illumination on a surface given the sun position.
    :param grad:
    :param sun_vector:
    :return:
    """
    check_gradient(grad)

    hsh = (
            grad[:, :, 0] * sun_vector[0] +
            grad[:, :, 1] * sun_vector[1] +
            grad[:, :, 2] * sun_vector[2]
    )
    # Remove negative incidence angles - indicators for self-shading.
    hsh = (hsh + abs(hsh)) / 2.

    return hsh


def _normalize_sun_vector(sun_vector):
    normal_sun_vector = np.zeros(3)
    normal_sun_vector[2] = np.sqrt(sun_vector[0] ** 2 + sun_vector[1] ** 2)
    normal_sun_vector[0] = -sun_vector[0] * sun_vector[2] / normal_sun_vector[2]
    normal_sun_vector[1] = -sun_vector[1] * sun_vector[2] / normal_sun_vector[2]
    return normal_sun_vector


def _invert_sun_vector(sun_vector):
    return -sun_vector / max(abs(sun_vector[:2]))


def _cast_shadow(row, col, rows, cols, dl, in_shadow, inverse_sun_vector,
                 normal_sun_vector, z):
    n = 0
    z_previous = -sys.float_info.max

    while True:
        # Calculate projection offset
        dx = inverse_sun_vector[0] * n
        dy = inverse_sun_vector[1] * n
        col_dx = int(round(col + dx))
        row_dy = int(round(row + dy))
        if (col_dx < 0) or (col_dx >= cols) or (row_dy < 0) or (row_dy >= rows):
            break

        vector_to_origin = np.zeros(3)
        vector_to_origin[0] = dx * dl
        vector_to_origin[1] = dy * dl
        vector_to_origin[2] = z[col_dx, row_dy]
        z_projection = np.dot(vector_to_origin, normal_sun_vector)

        if z_projection < z_previous:
            in_shadow[col_dx, row_dy] = 1
        else:
            z_previous = z_projection
        n += 1


def check_gradient(grad):
    assert len(grad.shape) == 3 and grad.shape[2] == 3, \
        "Gradient should be a tensor with 3 layers."



