#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:13:15 2026

@author: vpremier
"""
from ipyleaflet import (
    Map,
    GeomanDrawControl,
    LayersControl,
    FullScreenControl,
    basemaps,
    ImageOverlay,
    Marker,
)
import json
from pathlib import Path
from io import BytesIO
from pathlib import Path
import base64
import numpy as np
import geopandas as gpd
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point
import rioxarray

def create_training_map(
    rgb_display,
    data_stack,
    output_path="analysis_outputs/training_samples.shp",
    zoom=11,
):
    """Create an interactive map for collecting snow training points."""

    # Convert RGB array to a PNG data URL
    rgb_uint8 = (np.clip(rgb_display, 0, 1) * 255).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(rgb_uint8).save(buffer, format="PNG")

    image_url = (
        "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode()
    )

    # Convert image bounds to WGS84
    image_crs = data_stack.rio.crs
    transformer = Transformer.from_crs(
        image_crs,
        "EPSG:4326",
        always_xy=True,
    )

    west, east = float(data_stack.x.min()), float(data_stack.x.max())
    south, north = float(data_stack.y.min()), float(data_stack.y.max())

    west_lon, south_lat = transformer.transform(west, south)
    east_lon, north_lat = transformer.transform(east, north)

    bounds = (
        (south_lat, west_lon),
        (north_lat, east_lon),
    )

    # Create map
    training_map = Map(
        center=(
            (south_lat + north_lat) / 2,
            (west_lon + east_lon) / 2,
        ),
        zoom=zoom,
        scroll_wheel_zoom=True,
    )

    training_map.add_layer(
        ImageOverlay(
            url=image_url,
            bounds=bounds,
        )
    )

    # Widgets
    class_selector = widgets.Dropdown(
        options=[
            ("Snow", 1),
            ("Snow-free", 2),
        ],
        value=1,
        description="Class:",
    )

    save_button = widgets.Button(
        description="Save training samples",
        button_style="success",
    )

    message = widgets.Output()
    collected = []

    # Collect points
    def collect_point(**kwargs):
        if kwargs.get("type") != "click":
            return

        latitude, longitude = kwargs["coordinates"]

        collected.append({
            "class": class_selector.value,
            "geometry": Point(longitude, latitude),
        })

        training_map.add_layer(
            Marker(location=(latitude, longitude))
        )

        with message:
            message.clear_output()
            print(
                f"Collected {class_selector.label}: "
                f"{len(collected)} point(s)"
            )

    training_map.on_interaction(collect_point)

    # Save points
    def save_points(_):
        with message:
            message.clear_output()

            if not collected:
                print("Click at least one point first.")
                return

            samples = gpd.GeoDataFrame(
                collected,
                crs="EPSG:4326",
            ).to_crs(image_crs)

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            samples.to_file(output)

            print(
                f"Saved {len(samples)} training points to:\n"
                f"{output}"
            )

    save_button.on_click(save_points)

    display(
        widgets.VBox([
            widgets.HBox([class_selector, save_button]),
            message,
            training_map,
        ])
    )

    return training_map, collected

    
def create_aoi_map(
    center=(46.5, 11.3),
    zoom=8,
):
    state = {
        "geometry": None,
        "feature": None,
    }

    m = Map(
        center=center,
        zoom=zoom,
        basemap=basemaps.Esri.WorldImagery,
        scroll_wheel_zoom=True,
    )

    draw_control = GeomanDrawControl(
        marker={},
        circlemarker={},
        circle={},
        polyline={},
        rectangle={},
        text={},
        polygon={
            "pathOptions": {
                "color": "#ff7800",
                "fillColor": "#ff7800",
                "fillOpacity": 0.2,
                "weight": 3,
            },
            "allowSelfIntersection": False,
        },
        edit=True,
        remove=True,
    )

    def handle_draw(target, action, geo_json):
        print("Action:", action)

        if action == "remove":
            state["geometry"] = None
            state["feature"] = None
            return

        if isinstance(geo_json, list):
            if not geo_json:
                return
            feature = geo_json[-1]
        else:
            feature = geo_json

        if feature.get("type") == "Feature":
            geometry = feature["geometry"]
        else:
            geometry = feature
            feature = {
                "type": "Feature",
                "properties": {},
                "geometry": geometry,
            }

        state["feature"] = feature
        state["geometry"] = geometry

        print("AOI stored successfully")

    draw_control.on_draw(handle_draw)

    m.add(draw_control)

    return m, state

def save_aoi_geojson(aoi, output_path="aoi.geojson"):
    geometry = aoi.get("geometry")

    if geometry is None:
        raise ValueError("No AOI has been drawn yet.")

    feature = {
        "type": "Feature",
        "properties": {
            "name": "Area of interest"
        },
        "geometry": geometry
    }

    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature, f, indent=2)

    return output_path