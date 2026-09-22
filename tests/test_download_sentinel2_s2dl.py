import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from loading.download_sentinel2_s2dl import (
    download_s2dl,
    get_product_id_and_tile,
)


SCENE_32TPS = 'S2A_MSIL1C_20240601T102601_N0510_R108_T32TPS_20240601T123456'
SCENE_32TPT = 'S2B_MSIL2A_20240604T103619_N0510_R008_T32TPT_20240604T130001'


def test_product_name_parsing():
    assert get_product_id_and_tile(f'{SCENE_32TPS}.SAFE') == (
        SCENE_32TPS,
        'T32TPS',
    )
    assert get_product_id_and_tile(f'{SCENE_32TPS}.zip') == (
        SCENE_32TPS,
        'T32TPS',
    )


def test_downloads_scenes_into_their_tile_directories(tmp_path, monkeypatch):
    calls = []

    def fake_fetcher(product_id, target):
        calls.append((product_id, target))
        scene_dir = os.path.join(target, product_id)
        os.makedirs(scene_dir)
        with open(os.path.join(scene_dir, 'B02.jp2'), 'wb') as file:
            file.write(b'image')
        return scene_dir

    monkeypatch.setitem(
        sys.modules,
        's2dl',
        types.SimpleNamespace(fetch_single_sentinel_product=fake_fetcher),
    )
    products = pd.DataFrame(
        {'Name': [f'{SCENE_32TPS}.SAFE', f'{SCENE_32TPT}.SAFE']}
    )

    scenes = download_s2dl(products, tmp_path)

    assert calls == [
        (SCENE_32TPS, Path(tmp_path) / 'T32TPS'),
        (SCENE_32TPT, Path(tmp_path) / 'T32TPT'),
    ]
    assert all(os.path.isdir(scene) for scene in scenes)


def test_removes_duplicates_and_skips_existing_scene(tmp_path, monkeypatch):
    scene_dir = os.path.join(tmp_path, 'T32TPS', SCENE_32TPS)
    os.makedirs(scene_dir)
    with open(os.path.join(scene_dir, 'B02.jp2'), 'wb') as file:
        file.write(b'image')

    def unexpected_fetcher(product_id, target):
        raise AssertionError('Existing scene must not be downloaded')

    monkeypatch.setitem(
        sys.modules,
        's2dl',
        types.SimpleNamespace(fetch_single_sentinel_product=unexpected_fetcher),
    )
    products = pd.DataFrame(
        {'Name': [f'{SCENE_32TPS}.SAFE', f'{SCENE_32TPS}.SAFE']}
    )

    scenes = download_s2dl(products, tmp_path)

    assert scenes == [str(scene_dir)]


def test_skips_existing_safe_archive(tmp_path, monkeypatch):
    archive = tmp_path / 'T32TPS' / f'{SCENE_32TPS}.SAFE'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'existing archive')

    def unexpected_fetcher(product_id, target):
        raise AssertionError('Existing SAFE archive must not be downloaded')

    monkeypatch.setitem(
        sys.modules,
        's2dl',
        types.SimpleNamespace(fetch_single_sentinel_product=unexpected_fetcher),
    )
    products = pd.DataFrame({'Name': [f'{SCENE_32TPS}.SAFE']})

    scenes = download_s2dl(products, tmp_path)

    assert scenes == [str(archive)]


def test_rejects_query_without_name_column(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        's2dl',
        types.SimpleNamespace(fetch_single_sentinel_product=lambda *args: None),
    )
    with pytest.raises(ValueError, match="'Name' column"):
        download_s2dl(pd.DataFrame({'Id': ['uuid']}), tmp_path)
