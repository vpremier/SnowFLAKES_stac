#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:02:49 2026

@author: vpremier
"""
import math
import hydra
from omegaconf import DictConfig, OmegaConf

import openeo
import shapely
import json
from pathlib import Path
from pyproj import Transformer
from openeo.processes import (
    arccos,
    array_append,
    array_create,
    cos,
    max as openeo_max,
    min as openeo_min,
    quantiles,
    sin,
)


def elevation_mask(region, conn: openeo.Connection, cfg:DictConfig):
    elevation = conn.load_collection("COPERNICUS_30", spatial_extent=region).max_time()
    percentile10 = elevation.aggregate_spatial(region, reducer=lambda x: x.quantiles(probabilities = [0.1])).vector_to_raster(target=elevation).rename_labels(dimension="bands", target=["percentile10"])


    return elevation.merge_cubes(percentile10).reduce_dimension(dimension="bands", reducer=lambda x: x[0] < x[1] - 200)

#%%


def slope_aspect(aoi, connection, cfg):
    dem_spacetime = connection.load_collection("COPERNICUS_30", spatial_extent=aoi)
    dem = dem_spacetime.reduce_dimension(dimension='t', reducer='mean')

    aspect = dem.aspect()
    slope = dem.slope()

    aspect_slope = aspect.merge_cubes(slope).rename_labels(
        dimension="bands", target=["aspect", "slope"]
    )
    return aspect_slope



def cloud_water_mask(region, time_period, conn: openeo.Connection, cfg:DictConfig):
    scl = conn.load_collection(
        cfg.sentinel2_l2a.collection,
        spatial_extent=region,
        temporal_extent=time_period,

        bands=[cfg.sentinel2_l2a.scl_band])

    cloud_mask = scl.reduce_dimension(dimension="bands", reducer=lambda x: any([ x == cloud_value for cloud_value in cfg.sentinel2_l2a.cloud_values]))

    water = conn.load_collection(
        cfg.water_mask.collection,
        spatial_extent=region,
        bands=[cfg.water_mask.band]).max_time()

    water_mask = water.reduce_dimension(dimension="bands", reducer=lambda x: any(
        [x == cloud_value for cloud_value in cfg.water_mask.water_values]))

    return cloud_mask | water_mask

#%%

def shadow_mask(s2_cube, region):
    """Add the scene-wide composite shadow mask as a new band.
    """

    # Bands that need spatial min-max normalization. Aggregating them together
    # in a single ``aggregate_spatial`` call (per statistic) is much faster than
    # doing it band-by-band.
    norm_band_names = ["idx6", "SI", "NDVI", "EVI", "B08"]
    norm_bands = s2_cube.filter_bands(norm_band_names)

    band_min = norm_bands.aggregate_spatial(
        geometries=region,
        reducer=lambda data: openeo_min(data, ignore_nodata=True),
    ).vector_to_raster(target=norm_bands)
    band_max = norm_bands.aggregate_spatial(
        geometries=region,
        reducer=lambda data: openeo_max(data, ignore_nodata=True),
    ).vector_to_raster(target=norm_bands)
    band_range = band_max - band_min
    # Adding one only for a zero-width range reproduces the local
    # ``zeros_like`` branch without an eager Python/NumPy conditional.
    safe_range = band_range + (band_range == 0)
    normed = (norm_bands - band_min) / safe_range

    def select_band(cube, name):
        # ``DataCube.band`` enables a special client-side band-math mode that
        # cannot be combined with the aggregated statistic cubes above.
        return cube.filter_bands([name]).reduce_dimension(
            dimension="bands", reducer="first"
        )

    nir_n = select_band(normed, "B08")
    ndvi_n = select_band(normed, "NDVI")
    idx6_n = select_band(normed, "idx6")
    shad_idx_n = select_band(normed, "SI")
    evi_n = select_band(normed, "EVI")
    sia = select_band(s2_cube, "local_solar_incidence_angle")

    shadow_score = (
        (idx6_n + shad_idx_n)
        / (ndvi_n + evi_n + nir_n + 1e-6)
    )

    threshold = shadow_score.aggregate_spatial(
        geometries=region,
        reducer=lambda data: quantiles(
            data, probabilities=[0.85], ignore_nodata=True
        )
    ).vector_to_raster(target=shadow_score)

    curr_angle_valid = (sia >= 70) & (sia < 180)
    self_shadow = sia >= 90
    # Comparing two cubes with ``>`` triggers a ``merge_cubes`` with a ``gt``
    # overlap resolver, which the geopyspark backend does not support. Merging
    # via ``subtract`` (supported) and then comparing to a scalar sidesteps this.
    spectral_shadow = (shadow_score - threshold) > 0
    shadow = (spectral_shadow & curr_angle_valid) | self_shadow

    shadow = shadow.process_with_node(shadow.result_node(), metadata=sia.metadata)
    shadow_band = shadow.add_dimension(
        name="bands", label="shadow_mask", type="bands"
    )
    return s2_cube.merge_cubes(shadow_band)



def local_incidence_angle(s2_cube, aoi, connection, cfg, bands_to_retain = ["B02", "B03", "B04", "B08", "B11"]):
    slope_aspect_cube = slope_aspect(aoi, connection, cfg)
    combined = s2_cube.merge_cubes(slope_aspect_cube)

    # degree - radians conversions
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi

    # define the solar incidence angle function per-pixel
    def compute_sia(data):
        zenith_rad = data["sunZenithAngles"] * deg2rad
        azimuth_rad = data["sunAzimuthAngles"] * deg2rad
        slope_rad = data["slope"] * deg2rad
        aspect_rad = data["aspect"] * deg2rad

        cos_theta = (
            cos(zenith_rad) * cos(slope_rad) +
                sin(zenith_rad) * sin(slope_rad) *
                cos(aspect_rad - azimuth_rad)
        )

        cos_theta_clipped = cos_theta.max(-1).min(1)
        local_angle = arccos(cos_theta_clipped) * rad2deg

        bands = [data[b] for b in bands_to_retain]
        bands.append(local_angle)

        return array_create( bands )

    # apply the function
    extended_cube = combined.apply_dimension(
        dimension="bands",
        process=lambda data: compute_sia(data)
    )
    extended_cube = extended_cube.rename_labels(dimension="bands", target=bands_to_retain + ["local_solar_incidence_angle"])

    return extended_cube



def snowflake_inputs_cube(aoi, time_period, connection, cfg):
    # mask = cloud_water_mask(aoi, time_period, connection, cfg)
    dem_mask = elevation_mask(aoi, connection, cfg)
    sentinel2_bands = connection.load_collection(
        cfg.sentinel2_l1c.collection,
        spatial_extent=aoi,
        temporal_extent=time_period,
        bands=cfg.sentinel2_l1c.bands)
    # masked_s2 = sentinel2_bands.mask(mask | dem_mask)
    masked_s2 = sentinel2_bands.mask(dem_mask)

    s2_with_local_angle = local_incidence_angle(masked_s2, aoi, connection, cfg)


    from openeo.processes import normalized_difference
    def compute_indices(bands):
        nir = bands["B08"]
        ndvi = normalized_difference(nir, bands["B04"])
        updated = array_append(bands, ndvi, "NDVI")
        
        green = bands["B03"]
        swir = bands["B11"]
        ndsi = normalized_difference(green, swir)
        updated = array_append(updated, ndsi, "NDSI")
        
        ndwi = normalized_difference(green, nir)
        updated = array_append(updated, ndwi, "NDWI")

        blue = bands["B02"]
        diff_B_NIR = blue - nir
        updated = array_append(updated, diff_B_NIR, "diff_B_NIR")

        SI = ((green - swir) / (green + swir) / green)
        updated = array_append(updated, SI, "SI")
        
        red = bands["B04"]
        idx6 = 2 * (2 * green - red - nir) / (2 * green + red + nir)
        updated = array_append(updated, idx6, "idx6")
        
        EVI = 2.5 * (nir - red) / (nir + 2.4 * red + 1)
        updated = array_append(updated, EVI, "EVI")
        return updated

    bands_indices = s2_with_local_angle.apply_dimension(dimension="bands", process=compute_indices).rename_labels(
        dimension="bands", target=s2_with_local_angle.metadata.band_names + ["NDVI", "NDSI", "NDWI", "diff_B_NIR", "SI", "idx6", "EVI"])
    
    with_shadow_mask = shadow_mask(bands_indices, aoi)

    return with_shadow_mask

#%%

#%%

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_openeo(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    c = openeo.connect("https://openeo.dataspace.copernicus.eu/").authenticate_oidc()

    job_options = {
        "executor-memory": "2G",
        "executor-memoryOverhead": "4G",
        "executor-cores": 1,
        "image-name": "openeo-docker-ci.artifactory.vgt.vito.be/openeo-yarn:20251014-4204"
        #"stac-version-experimental": "1.1"
    }
    
    
    if cfg.experiment.aoi is None:
        aoi = json.load(open(Path(__file__).parent / "senales_wgs84.geojson"))
    else:
        aoi = shapely.box(*cfg.experiment.aoi)
        if cfg.experiment.aoi_crs is not None:
            transformer = Transformer.from_crs(cfg.experiment.aoi_crs, "EPSG:4326", always_xy=True)
            from shapely.ops import transform
            aoi = transform(transformer.transform, aoi)

    # define time period
    time_period = list(cfg.experiment.temporal_extent)

    representative_pixels = snow_cover_fraction_cube(aoi,time_period, c, cfg )

    job_options = dict(cfg.experiment.job_options)
    representative_pixels.execute_batch( "representative_pixels_senales_multirange_classified.nc", 
                                        title=(cfg.experiment.title_prefix or "") , 
                                        filename_prefix=cfg.experiment.title_prefix , 
                                        job_options=job_options)



def snow_cover_fraction_cube(aoi,time_period , c, cfg ):
    bands_indices = snowflake_inputs_cube(aoi, time_period, c, cfg)

    return bands_indices    





    
if "__main__" == __name__:
    import sys
    # Strip Jupyter kernel args (e.g. --f=...kernel.json) so hydra's argparse
    # doesn't choke when this file is executed inside an Interactive Window.
    sys.argv = [sys.argv[0]]
    run_openeo()
