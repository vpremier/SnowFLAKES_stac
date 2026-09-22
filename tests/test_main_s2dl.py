import json
import os

import pandas as pd

import main_s2dl


SCENE = 'S2A_MSIL1C_20240601T102601_N0510_R108_T32TPS_20240601T123456.SAFE'


def test_reads_query_and_output_directory_from_config(tmp_path, monkeypatch):
    query_path = tmp_path / 'query.csv'
    output_path = tmp_path / 'output'
    config_path = tmp_path / 'config.json'

    pd.DataFrame({'Name': [SCENE]}).to_csv(query_path, index=False)
    with open(config_path, 'w') as file:
        json.dump(
            {
                'satellite': 'Sentinel-2',
                'query_sentinel2': False,
                's2List_path': str(query_path),
                'output_directory': str(output_path),
            },
            file,
        )

    received = {}

    def fake_download(products, outdir):
        received['names'] = products['Name'].tolist()
        received['outdir'] = outdir
        return ['downloaded-scene']

    monkeypatch.setattr(main_s2dl, 'download_s2dl', fake_download)

    scenes = main_s2dl.run_s2dl(config_path)

    assert scenes == ['downloaded-scene']
    assert received['names'] == [SCENE]
    assert received['outdir'] == str(output_path)
    assert os.path.isdir(output_path)
