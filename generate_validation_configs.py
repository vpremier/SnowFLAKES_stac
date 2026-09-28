#!/usr/bin/env python3
"""Generate SnowFLAKES configurations centred on validation VHR imagery."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import sys

from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import transform_bounds


DEFAULT_ITEMS = Path(
    "/mnt/CEPH_PROJECTS/FRAM3S/reference_RGB/reference_dataset/items"
)
DEFAULT_CONFIG_ROOT = Path(
    "/mnt/CEPH_PROJECTS/FRAM3S/validation_SnowFLAKES/configs"
)
DEFAULT_RESULTS_ROOT = Path(
    "/mnt/CEPH_PROJECTS/FRAM3S/validation_SnowFLAKES/results"
)
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "config/config_validation.json"
EXTENT_KM = (2, 5, 7, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the inputs and report what would be generated.",
    )
    return parser.parse_args()


def choose_vhr_raster(dataset_dir: Path) -> Path | None:
    geotiffs = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    rgb = [path for path in geotiffs if "rgb" in path.name.lower()]
    if rgb:
        return rgb[0]

    # Some delivered VHR products use a sensor-derived name instead of RGB.
    imagery = [
        path
        for path in geotiffs
        if "snow" not in path.name.lower() and "class" not in path.name.lower()
    ]
    return imagery[0] if imagery else None


def target_crs_for_raster(source_crs: CRS, bounds: tuple[float, ...]) -> CRS:
    """Choose a metre-based local CRS suitable for the raster centre."""
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    centre_x = (bounds[0] + bounds[2]) / 2
    centre_y = (bounds[1] + bounds[3]) / 2
    longitude, latitude = transformer.transform(centre_x, centre_y)
    zone = int((longitude + 180) // 6) + 1

    if latitude < 0:
        return CRS.from_epsg(32700 + zone)  # WGS 84 / UTM, southern hemisphere
    if -16 <= longitude <= 33 and 25 <= latitude <= 84:
        return CRS.from_epsg(25800 + zone)  # ETRS89 / UTM for Europe
    return CRS.from_epsg(32600 + zone)  # WGS 84 / UTM, northern hemisphere


def aligned_centre_extent(
    bounds: tuple[float, ...], size_km: int, resolution: int
) -> list[int]:
    centre_x = round(((bounds[0] + bounds[2]) / 2) / resolution) * resolution
    centre_y = round(((bounds[1] + bounds[3]) / 2) / resolution) * resolution
    half_size = size_km * 1000 // 2
    return [
        int(centre_x - half_size),
        int(centre_y - half_size),
        int(centre_x + half_size),
        int(centre_y + half_size),
    ]


def aligned_vhr_extent(bounds: tuple[float, ...], resolution: int) -> list[int]:
    """Expand the transformed VHR bounds to complete output pixels."""
    return [
        math.floor(bounds[0] / resolution) * resolution,
        math.floor(bounds[1] / resolution) * resolution,
        math.ceil(bounds[2] / resolution) * resolution,
        math.ceil(bounds[3] / resolution) * resolution,
    ]


def footprint_geojson(
    bounds: tuple[float, ...], source_crs: CRS, site: str, date: str
) -> dict:
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    left, bottom, right, top = bounds
    source_ring = [
        (left, bottom),
        (right, bottom),
        (right, top),
        (left, top),
        (left, bottom),
    ]
    ring = [list(transformer.transform(x, y)) for x, y in source_ring]
    return {
        "type": "FeatureCollection",
        "name": f"{site}_{date}_vhr_footprint",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {"site": site, "date": date},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8"))
    resolution = int(template["resampling_params"]["resolution"])
    manifest_rows: list[dict[str, object]] = []
    skipped: list[str] = []

    dataset_dirs = sorted(path for path in args.items.glob("*/*") if path.is_dir())
    for dataset_dir in dataset_dirs:
        site = dataset_dir.parent.name
        date_match = re.match(r"(\d{8})", dataset_dir.name)
        if not date_match:
            skipped.append(f"{dataset_dir}: folder name has no YYYYMMDD date")
            continue
        date_token = date_match.group(1)
        acquisition = datetime.strptime(date_token, "%Y%m%d").date()
        vhr_path = choose_vhr_raster(dataset_dir)
        if vhr_path is None:
            skipped.append(f"{dataset_dir}: no VHR imagery GeoTIFF")
            continue

        with rasterio.open(vhr_path) as source:
            if source.crs is None:
                skipped.append(f"{dataset_dir}: {vhr_path.name} has no CRS")
                continue
            source_crs = CRS.from_user_input(source.crs)
            source_bounds = tuple(source.bounds)

        target_crs = target_crs_for_raster(source_crs, source_bounds)
        target_bounds = transform_bounds(
            source_crs, target_crs, *source_bounds, densify_pts=21
        )
        config_dir = args.config_root / site / date_token
        aoi_path = config_dir / "aoi_vhr.geojson"
        if not args.dry_run:
            write_json(
                aoi_path,
                footprint_geojson(source_bounds, source_crs, site, date_token),
            )

        # date_end is exclusive in SnowFLAKES, hence +3 includes date +2.
        date_start = (acquisition - timedelta(days=2)).isoformat()
        date_end = (acquisition + timedelta(days=3)).isoformat()
        variants = [
            (
                f"{size_km}km",
                size_km,
                aligned_centre_extent(target_bounds, size_km, resolution),
            )
            for size_km in EXTENT_KM
        ]
        variants.append(
            (
                "vhr_extent",
                None,
                aligned_vhr_extent(target_bounds, resolution),
            )
        )
        for label, size_km, extent_target in variants:
            config = copy.deepcopy(template)
            config["study_area"] = f"{site}_{date_token}_{label}"
            config["working_directory"] = str(
                args.results_root / site / date_token
            )
            config["shapefile"] = str(aoi_path)
            config["date_start"] = date_start
            config["date_end"] = date_end
            config["resampling_params"]["extent_target"] = extent_target
            config["resampling_params"]["epsg_target"] = target_crs.to_epsg()

            config_path = config_dir / f"config_{site}_{date_token}_{label}.json"
            if not args.dry_run:
                write_json(config_path, config)
            manifest_rows.append(
                {
                    "site": site,
                    "date": date_token,
                    "roi": label,
                    "extent_km": size_km,
                    "epsg": target_crs.to_epsg(),
                    "extent_target": " ".join(
                        str(value)
                        for value in config["resampling_params"]["extent_target"]
                    ),
                    "date_start": date_start,
                    "date_end_exclusive": date_end,
                    "vhr_raster": str(vhr_path),
                    "config": str(config_path),
                }
            )

    extent_labels = "_".join(f"{size_km}km" for size_km in EXTENT_KM)
    extent_labels += "_vhr_extent"
    manifest_path = args.config_root / f"config_manifest_{extent_labels}.csv"
    if manifest_rows and not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

    action = "Would generate" if args.dry_run else "Generated"
    print(
        f"{action} {len(manifest_rows)} configs for "
        f"{len(manifest_rows) // (len(EXTENT_KM) + 1)} datasets."
    )
    if not args.dry_run:
        print(f"Manifest: {manifest_path}")
    for reason in skipped:
        print(f"Skipped: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
