"""Download Sentinel-2 SAFE products from Copernicus Data Space S3."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _product_id_and_tile(name):
    product_id = Path(str(name).strip()).name
    product_id = product_id.removesuffix(".zip").removesuffix(".SAFE")
    parts = product_id.split("_")
    if len(parts) < 6 or not parts[0].startswith("S2"):
        raise ValueError(f"Invalid Sentinel-2 product name: {name}")
    return product_id, parts[5]


def _credentials(config):
    access = config.get("sentinel2_s3_access_key") or os.getenv("CDSE_S3_ACCESS_KEY")
    secret = config.get("sentinel2_s3_secret_key") or os.getenv("CDSE_S3_SECRET_KEY")
    profile = config.get("sentinel2_s3_profile") or os.getenv("CDSE_S3_PROFILE")
    return access, secret, profile


def download_sentinel2_s3(products, outdir, config=None):
    """Download queried Sentinel-2 products recursively from CDSE S3.

    The query DataFrame must contain ``Name`` and ``S3Path``.  ``S3Path`` is
    returned by the CDSE OData query and points to the product prefix, for
    example ``/eodata/Sentinel-2/MSI/L2A/...SAFE``.  Credentials can be
    supplied through config keys, ``CDSE_S3_ACCESS_KEY``/
    ``CDSE_S3_SECRET_KEY``, or a boto3 profile (usually ``cdse``).
    """
    config = config or {}
    if not isinstance(products, pd.DataFrame) or "Name" not in products:
        raise ValueError("products must be a DataFrame containing a Name column")
    if "S3Path" not in products.columns:
        raise ValueError(
            "The Sentinel-2 query does not contain S3Path. Re-run the CDSE "
            "query so the S3 product path is saved in the query CSV."
        )

    try:
        import boto3
    except ImportError as error:  # pragma: no cover - dependency is in snow.yml
        raise ImportError("S3 download requires boto3") from error

    access, secret, profile = _credentials(config)
    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    client_kwargs = {
        "endpoint_url": config.get(
            "sentinel2_s3_endpoint", "https://eodata.dataspace.copernicus.eu"
        ),
        "region_name": config.get("sentinel2_s3_region", "default"),
    }
    if access and secret:
        client_kwargs.update(
            aws_access_key_id=access, aws_secret_access_key=secret
        )
    s3 = session.resource("s3", **client_kwargs)

    downloaded = []
    for _, product in products.drop_duplicates("Name").iterrows():
        product_id, tile = _product_id_and_tile(product["Name"])
        s3_path = str(product["S3Path"]).strip().lstrip("/")
        if not s3_path or "/" not in s3_path:
            raise ValueError(f"Invalid S3Path for {product_id}: {s3_path!r}")
        bucket_name, prefix = s3_path.split("/", 1)
        prefix = prefix.rstrip("/") + "/"
        scene_dir = Path(outdir) / tile / product_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        objects = list(s3.Bucket(bucket_name).objects.filter(Prefix=prefix))
        if not objects:
            raise FileNotFoundError(
                f"No S3 objects found for {product_id} ({s3_path})"
            )
        existing = any(path.is_file() for path in scene_dir.rglob("*"))
        if existing:
            print(f"{product_id} already downloaded: {scene_dir}")
            downloaded.append(str(scene_dir))
            continue
        print(f"Downloading {product_id} from CDSE S3")
        for obj in objects:
            relative = obj.key[len(prefix):]
            if not relative:
                continue
            destination = scene_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.Bucket(bucket_name).download_file(obj.key, str(destination))
        downloaded.append(str(scene_dir))
    return downloaded
