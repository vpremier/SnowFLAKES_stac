#!/usr/bin/env python3
"""Benchmark SnowFLAKES STAC loading against SAFE and S2DL download methods.

The STAC side calls ``loading.load_stac.convert_sentinel2_bands`` exactly as
the SnowFLAKES workflow does, including calibration and loading the resulting
xarray object into memory.  The comparison side downloads the zipped SAFE
archive advertised by the same CDSE STAC item, optionally through S2DL.  Note
that S2DL downloads image JP2 files from Google's public Sentinel-2 bucket; it
does not download a complete CDSE SAFE archive.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import zipfile
from datetime import date as date_type
from pathlib import Path

import geopandas as gpd
import numpy as np
import pystac_client
import requests
from affine import Affine
from dotenv import load_dotenv
from pyproj import CRS
from pystac_client.stac_api_io import StacApiIO
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from shapely.geometry import box, mapping
from urllib3 import Retry

from loading import load_stac


CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
DEFAULT_BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B11",
    "B12",
    "B8A",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current SnowFLAKES Sentinel-2 STAC workflow with "
            "downloading the complete zipped SAFE product for the same scene."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--date", required=True, type=date_type.fromisoformat)
    parser.add_argument(
        "--tile",
        help="MGRS tile such as T32TPT, 32TPT, or MGRS-32TPT. If omitted, "
        "the first tile intersecting the configured AOI is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs"),
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="STAC output resolution in metres (default: config value).",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=DEFAULT_BANDS,
        help="STAC assets to load (default: the workflow's analysis bands).",
    )
    parser.add_argument(
        "--tile-area-fractions",
        nargs="+",
        type=float,
        default=[1.0],
        metavar="FRACTION",
        help="Centered fractions of tile area to load; 1 means the full tile.",
    )
    parser.add_argument(
        "--max-cloud-cover",
        type=float,
        help="Override the maximum cloud cover from the config.",
    )
    parser.add_argument(
        "--overwrite-safe",
        action="store_true",
        help="Replace an existing SAFE zip so transfer time can be measured.",
    )
    parser.add_argument(
        "--skip-safe",
        action="store_true",
        help="Run only the STAC side of the benchmark.",
    )
    parser.add_argument(
        "--with-s2dl",
        action="store_true",
        help="Also download S2DL image JP2 files from Google (pip install s2dl).",
    )
    parser.add_argument(
        "--with-odata",
        action="store_true",
        help="Also download the complete product through the CDSE OData zipper endpoint.",
    )
    parser.add_argument(
        "--extract-safe",
        action="store_true",
        help="Extract the OData SAFE archive and include extraction time.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PNG plot of STAC versus full-product timings.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot interactively (implies --plot).",
    )
    return parser.parse_args()


def normalize_tile(tile: str | None) -> str | None:
    if tile is None:
        return None
    normalized = tile.strip().upper()
    if normalized.startswith("MGRS-"):
        normalized = normalized[5:]
    if normalized.startswith("T"):
        normalized = normalized[1:]
    if not re.fullmatch(r"\d{2}[A-Z]{3}", normalized):
        raise ValueError(f"Invalid MGRS tile: {tile!r}")
    return normalized


def stac_client() -> pystac_client.Client:
    retry = Retry(
        total=5,
        backoff_factor=8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    client = pystac_client.Client.open(
        CDSE_STAC_URL,
        stac_io=StacApiIO(max_retries=retry),
    )
    client.add_conforms_to("ITEM_SEARCH")
    return client


def configured_geometry(config: dict) -> dict:
    shapefile = config.get("shapefile")
    if shapefile and Path(shapefile).exists():
        frame = gpd.read_file(shapefile).to_crs(4326)
        geometry = frame.union_all().simplify(0.05, preserve_topology=True)
        return mapping(geometry)

    extent = config["resampling_params"]["extent_target"]
    epsg = config["resampling_params"]["epsg_target"]
    frame = gpd.GeoSeries([box(*extent)], crs=f"EPSG:{epsg}").to_crs(4326)
    return mapping(frame.iloc[0])


def baseline(item: dict) -> int:
    match = re.search(r"_N(\d{4})_", item["id"])
    return int(match.group(1)) if match else -1


def discover_scene(
    acquisition_date: date_type,
    tile: str | None,
    geometry: dict | None,
    max_cloud_cover: float,
) -> tuple[dict, float]:
    query: dict = {"eo:cloud_cover": {"lte": max_cloud_cover}}
    if tile is not None:
        query["grid:code"] = {"eq": f"MGRS-{tile}"}

    started = time.perf_counter()
    search_parameters = {
        "collections": ["sentinel-2-l1c"],
        "datetime": acquisition_date.isoformat(),
        "query": query,
    }
    if geometry is not None:
        search_parameters["intersects"] = geometry
    items = list(
        stac_client().search(**search_parameters).items_as_dicts()
    )
    elapsed = time.perf_counter() - started
    if tile is not None:
        items = [
            item
            for item in items
            if item.get("properties", {}).get("grid:code") == f"MGRS-{tile}"
        ]
    if not items:
        tile_message = f" for tile {tile}" if tile else ""
        raise RuntimeError(
            f"No Sentinel-2 L1C item found on {acquisition_date}{tile_message}"
        )

    # Match the workflow's preference for the newest processing baseline.
    selected = max(
        items,
        key=lambda item: (
            baseline(item),
            item.get("properties", {}).get("created", ""),
        ),
    )
    return selected, elapsed


def tile_grid(item: dict, reference_band: str = "B02") -> tuple[int, tuple]:
    try:
        metadata = item["assets"][reference_band]
        shape = metadata["proj:shape"]
        transform = Affine(*metadata["proj:transform"][:6])
        crs_text = metadata.get("proj:code") or item["properties"]["proj:code"]
    except KeyError as error:
        raise RuntimeError(
            f"STAC item lacks grid metadata for reference band {reference_band}"
        ) from error

    epsg = CRS.from_user_input(crs_text).to_epsg()
    if epsg is None:
        raise RuntimeError(f"Cannot resolve an EPSG code from {crs_text!r}")
    bounds = array_bounds(int(shape[0]), int(shape[1]), transform)
    return epsg, bounds


def extent_for_area_fraction(
    full_bounds: tuple,
    area_fraction: float,
    resolution: float,
) -> list[float]:
    if not 0 < area_fraction <= 1:
        raise ValueError("Tile area fractions must be greater than 0 and at most 1")
    xmin, ymin, xmax, ymax = map(float, full_bounds)
    scale = math.sqrt(area_fraction)
    width_pixels = max(1, round((xmax - xmin) * scale / resolution))
    height_pixels = max(1, round((ymax - ymin) * scale / resolution))
    width = width_pixels * resolution
    height = height_pixels * resolution
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return [
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    ]


def benchmark_stac(
    item: dict,
    acquisition_date: date_type,
    output_dir: Path,
    epsg: int,
    extent: list[float],
    resolution: float,
    bands: list[str],
) -> dict:
    load_stac.setup_cdse_credentials()
    started = time.perf_counter()
    data, scene_id = load_stac.convert_sentinel2_bands(
        outdir=str(output_dir),
        date=acquisition_date.isoformat(),
        resolution=resolution,
        extent_target=extent,
        epsg_target=epsg,
        reproj_type=Resampling.bilinear,
        idList=[item["id"]],
        filter_by_geometry=False,
        save=False,
        bands=bands,
    )
    graph_seconds = time.perf_counter() - started
    if data is None:
        raise RuntimeError("The SnowFLAKES STAC loader returned no data")

    load_started = time.perf_counter()
    data.load()
    load_seconds = time.perf_counter() - load_started
    total_seconds = time.perf_counter() - started
    result = {
        "scene_id": scene_id,
        "graph_seconds": graph_seconds,
        "load_seconds": load_seconds,
        "total_seconds": total_seconds,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "memory_bytes": int(data.nbytes),
    }
    del data
    gc.collect()
    return result


def get_access_token(username: str, password: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["access_token"], time.perf_counter() - started


def benchmark_safe_download(
    item: dict,
    output_dir: Path,
    overwrite: bool,
    extract: bool = False,
) -> dict:
    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "CDSE_USERNAME and CDSE_PASSWORD are required for the SAFE download"
        )
    try:
        product = item["assets"]["Product"]
        download_url = product["href"]
        filename = product.get("file:local_path") or f"{item['id']}.SAFE.zip"
    except KeyError as error:
        raise RuntimeError("The selected STAC item has no Product/SAFE asset") from error

    destination = output_dir / Path(filename).name
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; use --overwrite-safe for a fresh timing"
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    token, authentication_seconds = get_access_token(username, password)

    started = time.perf_counter()
    bytes_written = 0
    with requests.get(
        download_url,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(60, 300),
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    bytes_written += len(chunk)
    download_seconds = time.perf_counter() - started
    os.replace(temporary, destination)
    extraction_seconds = 0.0
    extracted_path = None
    if extract:
        extracted_path = destination.with_suffix("")
        if extracted_path.exists() and any(extracted_path.iterdir()):
            raise FileExistsError(
                f"SAFE extraction directory is not empty: {extracted_path}"
            )
        extraction_started = time.perf_counter()
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(extracted_path)
        extraction_seconds = time.perf_counter() - extraction_started
    return {
        "authentication_seconds": authentication_seconds,
        "download_seconds": download_seconds,
        "extraction_seconds": extraction_seconds,
        "total_seconds": authentication_seconds + download_seconds + extraction_seconds,
        "bytes": bytes_written,
        "catalogue_bytes": product.get("file:size"),
        "path": str(destination.resolve()),
        "extracted_path": str(extracted_path.resolve()) if extracted_path else None,
    }


def benchmark_s2dl(item: dict, output_dir: Path) -> dict:
    """Time S2DL's image-file download from Google's public bucket."""
    try:
        from s2dl import fetch_single_sentinel_product
    except ImportError as error:
        raise RuntimeError(
            "S2DL is not installed; install it with `pip install s2dl` "
            "or omit --with-s2dl"
        ) from error

    target = output_dir / "s2dl" / item["id"]
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"S2DL output directory is not empty: {target}. "
            "Remove it or choose a new --output-dir."
        )
    target.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    returned_path = fetch_single_sentinel_product(item["id"], target)
    total_seconds = time.perf_counter() - started
    returned = Path(returned_path) if returned_path is not None else target
    root = returned if returned.exists() else target
    bytes_on_disk = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    return {
        "product_id": item["id"],
        "total_seconds": total_seconds,
        "bytes_on_disk": bytes_on_disk,
        "path": str(root.resolve()),
        "source": "Google Cloud public Sentinel-2 bucket",
        "base_url": (
            "https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles"
            if "_L1C_" in item["id"]
            else "https://storage.googleapis.com/gcp-public-data-sentinel-2/L2/tiles"
        ),
        "complete_safe": False,
        "image_files_only": True,
        "includes_extraction": False,
    }


def benchmark_odata_download(
    item: dict,
    output_dir: Path,
    username: str,
    password: str,
    extract: bool = False,
) -> dict:
    """Download using the project's exact ``download_cdse`` implementation."""
    from data_download.sentinel2_query_download import download_cdse

    product_name = item["id"] + ".SAFE"
    query_started = time.perf_counter()
    response = requests.get(
        CDSE_ODATA_URL,
        params={"$filter": f"startswith(Name,'{product_name}')", "$top": 5},
        timeout=60,
    )
    response.raise_for_status()
    products = response.json().get("value", [])
    query_seconds = time.perf_counter() - query_started
    if not products:
        raise RuntimeError(f"OData could not find product {product_name}")
    product = next(
        (candidate for candidate in products if candidate.get("Name") == product_name),
        products[0],
    )
    product_id = product["Id"]
    name = product.get("Name", product_name)
    import pandas as pd

    # download_cdse expects the same Id/Name columns returned by query_cdse.
    products_frame = pd.DataFrame([{"Id": product_id, "Name": name}])
    odata_root = output_dir / "odata_exact"
    tile = name.split("_")[5]
    destination = odata_root / "Sentinel2" / tile / name.replace(".SAFE", ".zip")
    if destination.exists() and destination.stat().st_size > 0:
        raise FileExistsError(f"OData output already exists: {destination}")

    started = time.perf_counter()
    download_cdse(products_frame, str(odata_root), username, password)
    download_seconds = time.perf_counter() - started
    if not destination.exists():
        raise RuntimeError(f"OData downloader did not create {destination}")
    bytes_written = destination.stat().st_size
    extraction_seconds = 0.0
    extracted_path = None
    if extract:
        extracted_path = destination.with_suffix("")
        if extracted_path.exists() and any(extracted_path.iterdir()):
            raise FileExistsError(f"SAFE extraction directory is not empty: {extracted_path}")
        extraction_started = time.perf_counter()
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(extracted_path)
        extraction_seconds = time.perf_counter() - extraction_started
    return {
        "product_id": product_id,
        "product_name": name,
        "query_seconds": query_seconds,
        "authentication_seconds": None,
        "download_seconds": download_seconds,
        "extraction_seconds": extraction_seconds,
        "total_seconds": query_seconds + download_seconds + extraction_seconds,
        "bytes": bytes_written,
        "path": str(destination.resolve()),
        "extracted_path": str(extracted_path.resolve()) if extracted_path else None,
    }


def plot_report(report: dict, destination: Path, show: bool = False) -> None:
    """Save and optionally display the timing comparison plot."""
    import matplotlib.pyplot as plt

    stac = report["stac"]
    fractions = [entry["tile_area_fraction"] * 100 for entry in stac]
    stac_times = [entry["total_seconds"] for entry in stac]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(fractions, stac_times, marker="o", label="STAC window read")
    if report.get("safe") is not None:
        axis.axhline(
            report["safe"]["total_seconds"],
            color="tab:orange",
            linestyle="--",
            label="OData SAFE download"
            + (" + extraction" if report["safe"].get("extracted_path") else ""),
        )
    if report.get("odata") is not None:
        axis.axhline(
            report["odata"]["total_seconds"],
            color="tab:red",
            linestyle="--",
            label="OData zipper"
            + (" + extraction" if report["odata"].get("extracted_path") else ""),
        )
    if report.get("s2dl") is not None:
        axis.axhline(
            report["s2dl"]["total_seconds"],
            color="tab:green",
            linestyle=":",
            label="S2DL image JP2 files (Google)",
        )
    axis.set_xlabel("Requested tile area (%)")
    axis.set_ylabel("Elapsed time (s)")
    axis.set_title(f"Sentinel-2 access benchmark\n{report['item_id']}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    load_dotenv()
    with args.config.open() as source:
        config = json.load(source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile = normalize_tile(args.tile)
    max_cloud_cover = (
        args.max_cloud_cover
        if args.max_cloud_cover is not None
        else float(config.get("max_cloudcover", 90))
    )
    resolution = (
        args.resolution
        if args.resolution is not None
        else float(config["resampling_params"]["resolution"])
    )

    item, discovery_seconds = discover_scene(
        args.date,
        tile,
        configured_geometry(config) if tile is None else None,
        max_cloud_cover,
    )
    epsg, full_bounds = tile_grid(item)
    selected_tile = item.get("properties", {}).get("grid:code")
    print(f"Selected {item['id']} ({selected_tile})")
    print(f"Native tile bounds: {full_bounds}, EPSG:{epsg}")

    report = {
        "date": args.date.isoformat(),
        "item_id": item["id"],
        "tile": selected_tile,
        "tile_bounds": list(full_bounds),
        "target_epsg": epsg,
        "target_resolution": resolution,
        "bands": args.bands,
        "discovery_seconds": discovery_seconds,
        "stac": [],
        "safe": None,
        "odata": None,
        "s2dl": None,
        "notes": [
            "STAC loads only the requested analysis bands and calibrates them.",
            "SAFE timing downloads the complete compressed product; --extract-safe also times extraction.",
            "OData timing queries the product and downloads it through zipper /$value.",
            "S2DL downloads only image JP2 files from Google's public Sentinel-2 bucket; it is not a complete SAFE download.",
            "Run-to-run network and provider caching can affect timings.",
        ],
    }

    for fraction in args.tile_area_fractions:
        extent = extent_for_area_fraction(full_bounds, fraction, resolution)
        print(f"Benchmarking STAC at tile area fraction {fraction:g}: {extent}")
        result = benchmark_stac(
            item,
            args.date,
            args.output_dir,
            epsg,
            extent,
            resolution,
            args.bands,
        )
        result["tile_area_fraction"] = fraction
        result["extent"] = extent
        report["stac"].append(result)
        print(f"STAC total: {result['total_seconds']:.2f} s")

    if not args.skip_safe:
        print("Downloading the complete zipped SAFE product...")
        report["safe"] = benchmark_safe_download(
            item,
            args.output_dir,
            args.overwrite_safe,
            args.extract_safe,
        )
        print(f"SAFE download: {report['safe']['download_seconds']:.2f} s")
        if args.extract_safe:
            print(f"SAFE extraction: {report['safe']['extraction_seconds']:.2f} s")

    if args.with_s2dl:
        print("Downloading the complete product with S2DL...")
        report["s2dl"] = benchmark_s2dl(item, args.output_dir)
        print(f"S2DL total: {report['s2dl']['total_seconds']:.2f} s")

    if args.with_odata:
        username = os.getenv("CDSE_USERNAME")
        password = os.getenv("CDSE_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "CDSE_USERNAME and CDSE_PASSWORD are required for OData download"
            )
        print("Downloading the complete product through CDSE OData...")
        report["odata"] = benchmark_odata_download(
            item,
            args.output_dir,
            username,
            password,
            args.extract_safe,
        )
        print(f"OData total: {report['odata']['total_seconds']:.2f} s")

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    report_path = args.output_dir / f"benchmark_{item['id']}_{timestamp}.json"
    with report_path.open("w") as output:
        json.dump(report, output, indent=2)
    print(f"Benchmark report saved to {report_path}")
    if args.plot or args.show_plot:
        plot_path = args.output_dir / f"benchmark_{item['id']}_{timestamp}.png"
        plot_report(report, plot_path, show=args.show_plot)
        print(f"Timing plot saved to {plot_path}")


if __name__ == "__main__":
    main()
