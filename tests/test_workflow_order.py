"""Exercise orchestration without importing geospatial/download dependencies."""
import ast
import contextlib
import gc
import io
import json
from datetime import date, datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import ModuleType

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def workflow_namespace():
    tree = ast.parse((ROOT / 'main.py').read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(np=np, pd=pd, Path=Path, json=json, gc=gc, PROJECT_ROOT=ROOT)
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'main.py', 'exec'), namespace)
    return namespace


class WorkflowOrderTests(unittest.TestCase):
    def test_query_and_processing_interleave_with_partial_months(self):
        ns = workflow_namespace()
        query_module = ModuleType('data_download.query_available')
        tree = ast.parse((ROOT / 'data_download/query_available.py').read_text())
        ranges = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == 'month_ranges')
        scope = dict(date=date, datetime=datetime)
        exec(compile(ast.Module(body=[ranges], type_ignores=[]), 'query', 'exec'), scope)
        query_module.month_ranges = scope['month_ranges']
        events = []
        query_module.run_from_settings = lambda config: events.append(
            ('query', config['date_start'], config['date_end']))
        ns['load_dotenv'] = Mock()

        def process(path, *, config_overrides):
            events.append(('process', config_overrides['date_start'],
                           config_overrides['date_end']))
            return []

        ns['run'] = process
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            original = json.dumps(dict(date_start='2025-12-20', date_end='2026-02-10'))
            path.write_text(original)
            with patch.dict('sys.modules', {'data_download.query_available': query_module}):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ns['run_monthly_workflow'](path), [])
            self.assertEqual(path.read_text(), original)
        self.assertEqual(events, [
            (stage, start, end)
            for start, end in [('2025-12-20', '2026-01-01'),
                               ('2026-01-01', '2026-02-01'),
                               ('2026-02-01', '2026-02-10')]
            for stage in ('query', 'process')
        ])

    def test_empty_query_csv_returns_empty_products(self):
        ns = workflow_namespace()
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / 'QUERY').mkdir()
            (study / 'QUERY' / 'Sentinel2_empty.csv').write_text('')
            with contextlib.redirect_stdout(io.StringIO()):
                products = ns['_read_query_files'](study, 'Sentinel2')
            self.assertTrue(products.empty)
            self.assertIn('Name', products.columns)

    def exercise(self, *, crop=True, source='s3', download_only=False,
                 cached=False, invalid=False, local=False, empty=False):
        ns = workflow_namespace()
        sentinel = pd.DataFrame({'Name': [
            'S2A_MSIL1C_20260201T100000_N0000_R000_T32TNS',
            'S2A_MSIL1C_20260102T100000_N0000_R000_T32TNS',
            'S2A_MSIL1C_20260101T100000_N0000_R000_T32TNS',
            'S2A_MSIL1C_20260101T100000_N0000_R000_T32TNT',
            'S2A_MSIL1C_20251231T100000_N0000_R000_T32TNS',
        ]})
        landsat = pd.DataFrame({'Name': [
            'LC08_L1TP_001001_20260101_20260102_02_T1',
            'LC09_L1TP_001001_20260102_20260103_02_T1',
            'LC08_L1TP_001001_20251231_20260102_02_T1',
            'LC09_L1TP_001001_20260201_20260202_02_T1',
        ]})
        if empty:
            sentinel, landsat = sentinel.iloc[:0], landsat.iloc[:0]
        events = []
        ns['load_dotenv'] = Mock()
        ns['_read_query_files'] = lambda study, sensor, *args: (
            sentinel.copy() if sensor == 'Sentinel2' else landsat.copy()
        )
        for name in ('_remove_snowflakes_dates', '_remove_logged_scenes',
                     '_remove_processed', '_remove_cached_tiles'):
            ns[name] = lambda products, *args, **kwargs: products
        ns['_remove_merged_dates'] = lambda products, *args: (
            products.iloc[:0] if cached else products
        )

        def download(config, study, s2, ls, **kwargs):
            names = list(s2.Name) + list(ls.Name)
            if names:
                dates = ns['_dates'](s2, 'Sentinel-2') + ns['_dates'](ls, 'Landsat')
                self.assertEqual(len({date[:7] for date in dates}), 1)
                events.append(('download', min(dates), names))
            return (list(s2.Name) if invalid else []), []

        def load_raw(config, sensor, products, date, *args, **kwargs):
            events.append(('load', date, list(products.Name)))
            return object(), products.iloc[0].Name

        def load_stac(config, sensor, date, save, products, **kwargs):
            return load_raw(config, sensor, products, date)

        def process(config, sensor, date, data, scene, output, collected, **kwargs):
            events.append(('process', date, [scene]))
            if not config['run_snowflakes']:
                collected.append(scene)

        ns['_download_raw'] = download
        ns['_load_raw_date'] = load_raw
        ns['_load_stac_date'] = load_stac
        ns['_rgb_cache_exists'] = lambda *args: True
        ns['_rgb_uses_analysis_data'] = lambda *args: False
        ns['_run_or_collect'] = process
        with tempfile.TemporaryDirectory() as directory:
            config = dict(
                working_directory=directory, study_area='study', satellite='both',
                DOWNLOAD_SENTINEL=False if local else source,
                DOWNLOAD_LANDSAT=False if local else 'usgs-m2m',
                CROP=crop, SAVE=not download_only,
                run_snowflakes=not download_only, SAVE_RGB=False,
                SAVE_FALSE_COLOR=False,
                resampling_params=dict(resolution=20, epsg_target=32632,
                                       extent_target=[0, 0, 100, 100]),
            )
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                result = ns['run'](path)
        self.assertEqual(result, [])
        self.assertEqual([e[1] for e in events], sorted(e[1] for e in events))
        if not empty:
            self.assertTrue(events)
        if download_only:
            self.assertTrue(all(e[0] == 'download' for e in events))
            self.assertEqual(len(events), 3)
        else:
            self.assertEqual({e[1] for e in events if e[0] == 'process'},
                             set() if empty else {'2025-12-31', '2026-01-01',
                                                  '2026-01-02', '2026-02-01'})
        if cached:
            self.assertFalse(any(e[0] == 'download' for e in events))
        if invalid:
            self.assertFalse(any(name.startswith('S2') for e in events
                                 if e[0] == 'process' for name in e[2]))
        return events

    def test_cropped_dates_keep_all_tiles(self):
        events = self.exercise()
        loads = [e for e in events if e[0] == 'load' and e[2][0].startswith('S2')]
        self.assertEqual([len(e[2]) for e in loads], [1, 2, 1, 1])

    def test_downloads_are_grouped_by_month(self):
        events = self.exercise()
        downloads = [e for e in events if e[0] == 'download']
        self.assertEqual([e[1][:7] for e in downloads],
                         ['2025-12', '2026-01', '2026-02'])
        self.assertEqual([len(e[2]) for e in downloads], [2, 5, 2])
        for month in ('2025-12', '2026-01', '2026-02'):
            monthly_events = [e for e in events if e[1].startswith(month)]
            self.assertEqual(monthly_events[0][0], 'download')
            self.assertEqual(monthly_events[-1][0], 'process')

    def test_uncropped_tiles(self):
        events = self.exercise(crop=False)
        self.assertEqual(sum(e[0] == 'process' for e in events), 9)

    def test_mixed_stac_and_raw(self):
        self.exercise(source='stac-api')

    def test_download_only_visits_every_date(self):
        self.exercise(crop=False, download_only=True)

    def test_cached_products_are_still_processed(self):
        self.exercise(cached=True)

    def test_invalid_download_does_not_block_next_date(self):
        self.exercise(invalid=True)

    def test_local_products(self):
        self.exercise(local=True)

    def test_empty_queries(self):
        self.exercise(empty=True)


if __name__ == '__main__':
    unittest.main()
