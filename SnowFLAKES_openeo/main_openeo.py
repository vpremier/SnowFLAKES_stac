#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:02:49 2026

@author: vpremier
"""
import math
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import openeo
import shapely
import json
from pathlib import Path
from pyproj import Transformer
from openeo.processes import cos, sin, arccos, array_create, array_append


def elevation_mask(region, conn: openeo.Connection, cfg:DictConfig):
    elevation = conn.load_collection("COPERNICUS_30", spatial_extent=region).max_time()
    percentile10 = elevation.aggregate_spatial(region, reducer=lambda x: x.quantiles(probabilities = [0.1])).vector_to_raster(target=elevation).rename_labels(dimension="bands", target=["percentile10"])


    return elevation.merge_cubes(percentile10).reduce_dimension(dimension="bands", reducer=lambda x: x[0] < x[1] - 200)



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



def shadow_mask(s2_cube):
    """Generate the shadow mask."""

    def normalize(arr):
        # Normalize indices to range [0, 1]
        arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
        return (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)
   
    def compute_shadow_mask(data):
        NIR = data["B08"]
        ndvi = data["NDVI"] 
        SIA = data["SIA"]
        idx6 = data["idx6"] 
        shad_idx = data["SI"] 
        evi = data["EVI"] 
        
        idx6_norm = normalize(idx6)
        shad_idx_norm = normalize(shad_idx)
        ndvi_norm = normalize(ndvi)
        evi_norm = normalize(evi)
        nir_norm = normalize(NIR)
        
        # SIA between 70 and 180
        curr_angle_valid = np.logical_and(SIA >= 70,
                                          SIA < 180)
        
        # Combine indices to create a composite shadow score
        shadow_score = (
            (idx6_norm + shad_idx_norm) /
            (ndvi_norm + evi_norm + nir_norm + 1e-6)
        )
        
        threshold = np.nanpercentile(shadow_score, 85)
        
        # DIFFERENT SHADOWS
        self_shadow = SIA >= 90
        # cloud_shadow = cloud_mask == 3
        spectral_shadow = shadow_score > threshold
        casted_shadow = np.logical_and(spectral_shadow, curr_angle_valid)
        # shadow_mask = np.logical_or.reduce((casted_shadow, self_shadow, cloud_shadow))
        shadow_mask = np.logical_or.reduce((casted_shadow, self_shadow))

        return shadow_mask
    
    # apply the function
    extended_cube = s2_cube.apply_dimension(
        dimension="bands",
        process=lambda data: compute_shadow_mask(data)
    )
    extended_cube = extended_cube.rename_labels(dimension="bands", target=s2_cube.metadata.band_names + ["shadow_mask"])
    
    return extended_cube



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
        
        idx6 = 2 * (2 * bands["B01"] - blue - green) / (2 * bands["B01"] + blue + green)
        updated = array_append(updated, idx6, "idx6")
        
        EVI =  2.5 * (bands["B01"] - blue) / (bands["B01"] + 2.4 * blue + 1)
        updated = array_append(updated, EVI, "EVI")
        return updated

    bands_indices = s2_with_local_angle.apply_dimension(dimension="bands", process=compute_indices).rename_labels(
        dimension="bands", target=s2_with_local_angle.metadata.band_names + ["NDVI", "NDSI", "NDWI", "diff_B_NIR", "SI", "idx6", "EVI"])
    
    with_shadow_mask = shadow_mask(bands_indices)

    return with_shadow_mask



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
    run_openeo()