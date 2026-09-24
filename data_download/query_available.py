#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query available Sentinel-2 and Landsat scenes for a study area.

The output CSV files are compatible with the selected download helper. Google
mode produces S2DL fields, while OData/S3 mode preserves the CDSE ``Id`` and
``S3Path`` fields required by the corresponding downloaders. Dates use the
same half-open interval as the query functions:
``date_start <= acquisition_date < date_end``.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_download.landsat_query_download import query_landsat
from data_download.sentinel2_query_download import query_cdse
from data_download.sentinel2_google_query import (
    normalize_tile,
    query_google_sentinel2,
)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, *args, **kwargs):
        """Small fallback for environments without python-dotenv."""
        path = Path(dotenv_path or ".env")
        if not path.is_file():
            return False
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
        return True


LANDSAT_CODES = ("LT05", "LE07", "LC08", "LC09")


def month_ranges(date_start, date_end):
    """Split a half-open date interval into periods of at most one month."""
    start = datetime.strptime(date_start, "%Y-%m-%d").date()
    end = datetime.strptime(date_end, "%Y-%m-%d").date()
    if start >= end:
        raise ValueError("date_start must be before date_end")

    current = start
    periods = []
    while current < end:
        next_month = date(
            current.year + (current.month == 12),
            1 if current.month == 12 else current.month + 1,
            1,
        )
        period_end = min(end, next_month)
        periods.append((current.isoformat(), period_end.isoformat()))
        current = period_end
    return periods


def _split_values(values):
    """Flatten repeated CLI values and comma-separated values."""
    result = []
    for value in values or []:
        result.extend(item.strip() for item in str(value).split(","))
    return [item for item in result if item]


def _normalise_sentinel_tiles(values):
    return {normalize_tile(value) for value in _split_values(values)}


def _normalise_landsat_tiles(values):
    return {str(value).strip() for value in _split_values(values)}


def _filter_sentinel2(products, skip_tiles):
    if products.empty or not skip_tiles:
        return products
    return products[~products["tile"].isin(skip_tiles)].reset_index(drop=True)


def _filter_landsat(products, skip_pathrows):
    if products.empty or not skip_pathrows:
        return products
    pathrows = products["Name"].astype(str).str.split("_").str[2]
    return products[~pathrows.isin(skip_pathrows)].reset_index(drop=True)


def _csv_path(query_dir, satellite, date_start, date_end):
    return query_dir / f"{satellite}_{date_start}_{date_end}.csv"


def _previous_day(value):
    return (
        datetime.strptime(value, "%Y-%m-%d").date() - timedelta(days=1)
    ).isoformat()


def _existing_sentinel2_query_is_compatible(path, source):
    if not path.exists():
        return False
    if source == "google":
        return True
    try:
        columns = pd.read_csv(path, nrows=0).columns
        if source == "s3":
            return "Id" in columns and "S3Path" in columns
        return "Id" in columns
    except (OSError, pd.errors.ParserError):
        return False


def query_period(
    aoi,
    query_dir,
    date_start,
    date_end,
    max_cloudcover,
    satellite,
    skip_sentinel2_tiles=None,
    skip_landsat_pathrows=None,
    sentinel2_source="google",
):
    """Query one period, skipping each CSV that already exists."""
    outputs = []
    skip_sentinel2_tiles = skip_sentinel2_tiles or set()
    skip_landsat_pathrows = skip_landsat_pathrows or set()

    if satellite in ("sentinel2", "both"):
        output = _csv_path(query_dir, "Sentinel2", date_start, date_end)
        if _existing_sentinel2_query_is_compatible(output, sentinel2_source):
            print(f"Skipping existing query: {output}")
        else:
            if sentinel2_source == "google":
                products = query_google_sentinel2(
                    date_start,
                    date_end,
                    shp=aoi,
                    max_cc=max_cloudcover,
                )
            elif sentinel2_source in {"odata", "s3"}:
                username = os.getenv("CDSE_USERNAME")
                password = os.getenv("CDSE_PASSWORD")
                if not username or not password:
                    raise ValueError(
                        "CDSE_USERNAME and CDSE_PASSWORD are required "
                        "for Sentinel-2 OData queries. Expected them in "
                        f"{PROJECT_ROOT / '.env'} or beside the config file."
                    )
                products = query_cdse(
                    date_start,
                    date_end,
                    username,
                    password,
                    shp=aoi,
                    max_cc=max_cloudcover,
                )
            else:
                raise ValueError(
                    "sentinel2_source must be 'google', 'odata', or 's3'"
                )
            products = _filter_sentinel2(products, skip_sentinel2_tiles)
            products.to_csv(output, index=False)
            print(f"Saved {len(products)} Sentinel-2 scenes: {output}")
        outputs.append(output)

    if satellite in ("landsat", "both"):
        output = _csv_path(query_dir, "Landsat", date_start, date_end)
        if output.exists():
            print(f"Skipping existing query: {output}")
        else:
            username = os.getenv("ERS_USERNAME")
            token = os.getenv("ERS_TOKEN")
            if not username or not token:
                raise ValueError(
                    "ERS_USERNAME and ERS_TOKEN are required for Landsat "
                    f"queries. Expected them in {PROJECT_ROOT / '.env'} or "
                    "beside the config file."
                )
            products = query_landsat(
                date_start,
                _previous_day(date_end),
                username,
                token,
                shp=aoi,
                max_cc=max_cloudcover,
                sat=list(LANDSAT_CODES),
            )
            products = products.rename(columns={"displayId": "Name"})
            products = _filter_landsat(products, skip_landsat_pathrows)
            products.to_csv(output, index=False)
            print(f"Saved {len(products)} Landsat scenes: {output}")
        outputs.append(output)

    return outputs


def run_queries(
    study_area,
    working_directory,
    aoi,
    date_start,
    date_end,
    max_cloudcover=90,
    satellite="both",
    skip_sentinel2_tiles=None,
    skip_landsat_pathrows=None,
    sentinel2_source="google",
):
    """Create monthly query CSV files beneath ``<work>/<study>/QUERY``."""
    aoi = Path(aoi)
    if not aoi.is_file():
        raise FileNotFoundError(f"AOI file does not exist: {aoi}")
    satellite_aliases = {
        "sentinel-2": "sentinel2",
        "sentinel2": "sentinel2",
        "landsat": "landsat",
        "landsat-5": "landsat",
        "landsat-7": "landsat",
        "landsat-8": "landsat",
        "landsat-9": "landsat",
        "both": "both",
    }
    satellite = satellite_aliases.get(str(satellite).lower())
    if satellite not in {"sentinel2", "landsat", "both"}:
        raise ValueError(
            "satellite must be Sentinel-2, Landsat, or both"
        )
    if not 0 <= max_cloudcover <= 100:
        raise ValueError("max_cloudcover must be between 0 and 100")
    sentinel2_source = str(sentinel2_source).lower()
    if sentinel2_source not in {"google", "odata", "s3"}:
        raise ValueError("sentinel2_source must be 'google', 'odata', or 's3'")

    query_dir = Path(working_directory) / study_area / "QUERY"
    query_dir.mkdir(parents=True, exist_ok=True)
    periods = month_ranges(date_start, date_end)
    skip_sentinel2_tiles = _normalise_sentinel_tiles(skip_sentinel2_tiles)
    skip_landsat_pathrows = _normalise_landsat_tiles(skip_landsat_pathrows)

    outputs = []
    for period_start, period_end in periods:
        outputs.extend(
            query_period(
                aoi,
                query_dir,
                period_start,
                period_end,
                max_cloudcover,
                satellite,
                skip_sentinel2_tiles,
                skip_landsat_pathrows,
                sentinel2_source,
            )
        )
    return outputs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Query monthly Sentinel-2 and/or Landsat scene lists"
    )
    parser.add_argument(
        "--config",
        help="JSON configuration containing the query settings",
    )
    parser.add_argument("--study-area")
    parser.add_argument("--aoi", help="Shapefile or GeoJSON")
    parser.add_argument("--working-directory")
    parser.add_argument("--date-start", help="YYYY-MM-DD")
    parser.add_argument("--date-end", help="YYYY-MM-DD (exclusive)")
    parser.add_argument("--max-cloudcover", type=float, default=90)
    parser.add_argument(
        "--satellite",
        default="both",
        help="both, sentinel2/Sentinel-2, or landsat/Landsat-8",
    )
    parser.add_argument(
        "--sentinel2-source",
        choices=("google", "odata", "s3"),
        default="google",
        help="Sentinel-2 source; OData produces CSVs for download_cdse",
    )
    parser.add_argument(
        "--skip-sentinel2-tiles",
        nargs="*",
        default=[],
        help="MGRS tiles such as T19HDE; values may also be comma-separated",
    )
    parser.add_argument(
        "--skip-landsat-pathrows",
        nargs="*",
        default=[],
        help="WRS-2 path/rows such as 193027; values may also be comma-separated",
    )
    return parser


def _config_value(config, *keys, default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _run_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    aoi = _config_value(config, "shapefile", "aoi")
    if not aoi:
        raise ValueError("The config must define 'shapefile' or 'aoi'")

    study_area = _config_value(
        config,
        "study_area",
        "study_area_name",
        default=Path(aoi).stem,
    )
    working_directory = _config_value(
        config,
        "working_directory",
        default=None,
    )
    if working_directory is None:
        output_directory = config.get("output_directory")
        if output_directory:
            # Backward-compatible fallback for the existing configurations:
            # their output directory is normally ``<work>/<study>``.
            working_directory = str(Path(output_directory).parent)
        else:
            raise ValueError(
                "The config must define 'working_directory' or "
                "'output_directory'"
            )

    satellite = str(config.get("satellite", "both"))
    if satellite.lower().startswith("sentinel"):
        satellite = "sentinel2"
    elif satellite.lower().startswith("landsat"):
        satellite = "landsat"
    else:
        satellite = "both"

    return run_queries(
        study_area=study_area,
        working_directory=working_directory,
        aoi=aoi,
        date_start=config["date_start"],
        date_end=config["date_end"],
        max_cloudcover=float(config.get("max_cloudcover", 90)),
        satellite=satellite,
        skip_sentinel2_tiles=_config_value(
            config,
            "skip_sentinel2_tiles",
            "s2_tile_skip",
            "exclude_tiles",
            default=[],
        ),
        skip_landsat_pathrows=_config_value(
            config,
            "skip_landsat_pathrows",
            "landsat_tile_skip",
            default=[],
        ),
        sentinel2_source=config.get("sentinel2_source", "google"),
    )


def main():
    args = build_parser().parse_args()
    # Load the repository-level .env even when the script is launched from a
    # different working directory. A config-local .env is also supported.
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()

    if args.config:
        load_dotenv(Path(args.config).resolve().parent / ".env")
        _run_from_config(args.config)
        return

    required = {
        "study_area": args.study_area,
        "aoi": args.aoi,
        "working_directory": args.working_directory,
        "date_start": args.date_start,
        "date_end": args.date_end,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(
            "Without --config, these arguments are required: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    run_queries(
        study_area=args.study_area,
        working_directory=args.working_directory,
        aoi=args.aoi,
        date_start=args.date_start,
        date_end=args.date_end,
        max_cloudcover=args.max_cloudcover,
        satellite=args.satellite,
        skip_sentinel2_tiles=args.skip_sentinel2_tiles,
        skip_landsat_pathrows=args.skip_landsat_pathrows,
        sentinel2_source=args.sentinel2_source,
    )


if __name__ == "__main__":
    main()
