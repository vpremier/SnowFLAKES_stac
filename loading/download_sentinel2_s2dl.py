#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download Sentinel-2 query results with S2DL, grouped by MGRS tile."""

import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def get_product_id_and_tile(file_name):
    """Extract the S2DL product ID and MGRS tile from a product name."""
    product_id = os.path.basename(str(file_name).strip())

    if product_id.lower().endswith('.zip'):
        product_id = product_id[:-4]
    if product_id.lower().endswith('.safe'):
        product_id = product_id[:-5]

    parts = product_id.split('_')
    if len(parts) < 6 or not parts[0].startswith('S2'):
        raise ValueError(f'Invalid Sentinel-2 product name: {file_name}')

    tile = parts[5]
    if not tile.startswith('T') or len(tile) != 6:
        raise ValueError(f'Invalid Sentinel-2 tile in: {file_name}')

    return product_id, tile


def _existing_product_path(tile_dir, product_id):
    """Return an existing non-empty SAFE/archive path, if available."""
    candidates = (
        tile_dir / product_id,
        tile_dir / f'{product_id}.SAFE',
        tile_dir / f'{product_id}.zip',
        tile_dir / f'{product_id}.SAFE.zip',
    )
    for path in candidates:
        if _product_is_valid(path):
            return path
    return None


def _product_is_valid(path):
    """Check that a downloaded SAFE/archive contains non-empty content."""
    path = Path(path)
    if path.is_file():
        if path.stat().st_size == 0:
            return False
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    return archive.testzip() is None and bool(archive.namelist())
            except (OSError, zipfile.BadZipFile):
                return False
        return True
    if path.is_dir():
        return any(item.is_file() and item.stat().st_size > 0 for item in path.rglob("*"))
    return False


def _log_and_remove_failed(tile_dir, product_id, reason):
    """Remove a failed product directory and append a shared error log."""
    tile_dir = Path(tile_dir)
    raw_dir = tile_dir.parent.parent
    log_path = raw_dir / "download_errors.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{timestamp}\tSentinel-2\t{product_id}\t{reason}\n")

    for candidate in (
        tile_dir / product_id,
        tile_dir / f"{product_id}.SAFE",
        tile_dir / f"{product_id}.zip",
        tile_dir / f"{product_id}.SAFE.zip",
    ):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        elif candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass
    print(f"Removed failed Sentinel-2 product {product_id}; logged to {log_path}")


def download_s2dl(s2List, outdir, return_status=False):
    """Download scenes returned by ``query_cdse`` using S2DL.

    ``s2List`` can be the DataFrame returned by ``query_cdse`` or the path to
    its CSV file. Scenes are saved under ``outdir/<tile>/<scene>``.
    """
    try:
        from s2dl import fetch_single_sentinel_product
    except ImportError as error:
        raise ImportError('S2DL is not installed. Run: pip install s2dl') from error

    if isinstance(s2List, (str, os.PathLike)):
        s2List = pd.read_csv(s2List)

    if not isinstance(s2List, pd.DataFrame) or 'Name' not in s2List.columns:
        raise ValueError("s2List must contain a 'Name' column")

    # Keep the query order while removing duplicate products.
    scene_names = s2List['Name'].drop_duplicates().tolist()
    downloaded_scenes = []
    failed_scenes = []

    for file_name in tqdm(scene_names, desc='Downloading Sentinel-2'):
        try:
            product_id, tile = get_product_id_and_tile(file_name)
        except Exception as error:
            failed_scenes.append(str(file_name))
            print(f'Invalid Sentinel-2 product {file_name}: {error}')
            continue
        tile_dir = Path(outdir) / tile
        scene_dir = tile_dir / product_id

        existing_path = _existing_product_path(tile_dir, product_id)
        if existing_path is not None:
            print(f'{product_id} already downloaded: {existing_path}')
            downloaded_scenes.append(str(existing_path))
            continue

        tile_dir.mkdir(parents=True, exist_ok=True)
        if scene_dir.exists() and not _product_is_valid(scene_dir):
            _log_and_remove_failed(tile_dir, product_id, "empty product directory from previous attempt")
        print(f'Downloading {product_id} in {tile}')
        try:
            downloaded_path = fetch_single_sentinel_product(product_id, tile_dir)
            candidate = Path(downloaded_path) if downloaded_path else scene_dir
            existing = _existing_product_path(tile_dir, product_id)
            if not _product_is_valid(candidate) and existing is not None:
                candidate = existing
            if not _product_is_valid(candidate):
                _log_and_remove_failed(tile_dir, product_id, "empty or corrupted download")
                failed_scenes.append(str(file_name))
                continue
            downloaded_scenes.append(str(candidate))
        except Exception as error:
            _log_and_remove_failed(tile_dir, product_id, str(error))
            failed_scenes.append(str(file_name))
            continue

    if return_status:
        return {"downloaded": downloaded_scenes, "failed": failed_scenes}
    return downloaded_scenes


def main():
    if len(sys.argv) != 3:
        print(
            'Usage: python loading/download_sentinel2_s2dl.py '
            'path_to_query.csv output_directory'
        )
        return 1

    query_csv = sys.argv[1]
    outdir = sys.argv[2]
    scenes = download_s2dl(query_csv, outdir)
    print(f'\nDownload completed: {len(scenes)} scenes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
