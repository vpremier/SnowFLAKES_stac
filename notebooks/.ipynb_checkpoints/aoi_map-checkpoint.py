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
)
import json
from pathlib import Path
from ipyleaflet import Map, GeomanDrawControl, basemaps


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