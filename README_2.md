# SnowFLAKES query and loading workflow

The workflow is started with one JSON configuration:

```bash
./main.sh path/to/config.json
```

The Bash script calls the query stage and then the loading stage:

```bash
python3 "${SCRIPT_DIR}/data_download/query_available.py" \
    --config "${CONFIG_PATH}"
python3 "${SCRIPT_DIR}/loading/main_load.py" "${CONFIG_PATH}"
```

## Query configuration

The query configuration should contain:

```json
{
  "study_area": "Mendoza",
  "working_directory": "/data/SnowFLAKES",
  "shapefile": "/data/AOI/Mendoza.geojson",
  "date_start": "2023-01-01",
  "date_end": "2023-06-01",
  "max_cloudcover": 80,
  "satellite": "both",
  "sentinel2_source": "google",
  "skip_sentinel2_tiles": ["T19HDE"],
  "skip_landsat_pathrows": ["232084"]
}
```

The mandatory inputs are the AOI (`shapefile`), `date_start`, `date_end`,
`working_directory`, and `satellite`. `study_area` defaults to the AOI filename
stem. `satellite` can be `Sentinel-2`, a Landsat name such as `Landsat-8`, or
`both`.

The query creates:

```text
<working_directory>/<study_area>/QUERY/
```

with files such as:

```text
Sentinel2_2023-01-01_2023-02-01.csv
Landsat_2023-01-01_2023-02-01.csv
```

`date_end` is exclusive. For intervals longer than one month, the query is
split into consecutive periods of at most one month. Existing CSV files are
reused. OData CSV files are reused only when they contain the CDSE `Id` column.

Sentinel-2 can be queried with:

```json
"sentinel2_source": "google"
```

for S2DL-compatible Google products, or:

```json
"sentinel2_source": "odata"
```

for CSVs compatible with `download_cdse()`.

OData and Landsat queries require these variables in `.env` (repository root or
beside the configuration):

```dotenv
CDSE_USERNAME=your_cdse_username
CDSE_PASSWORD=your_cdse_password
ERS_USERNAME=your_usgs_username
ERS_TOKEN=your_usgs_token
```

## Loading configuration

The loading stage uses:

```json
{
  "DOWNLOAD_MODE": "STAC-API",
  "CROP": true,
  "SAVE": true,
  "run_snowflakes": true
}
```

`DOWNLOAD_MODE` accepts `STAC-API` or `RAW`:

- `STAC-API` loads Sentinel-2 through CDSE STAC and Landsat through USGS STAC;
- `RAW` downloads and locally loads the archives listed in the query CSVs.

`CROP` must be `true` for `STAC-API`. In `RAW` mode, `CROP=true` uses the
configured target extent, while `CROP=false` mosaics the full union of the
downloaded tiles.

When `CROP=true`, `resampling_params.extent_target`,
`resampling_params.resolution`, and `resampling_params.epsg_target` are
required. In full-tile raw mode, these crop parameters are optional; native
source CRS information is used when possible and the default resolution is
10 m for Sentinel-2 and 30 m for Landsat.

`SAVE=true` writes merged GeoTIFF bands below `MERGED`. If `run_snowflakes` is
true, each resulting data array is passed to SnowFLAKES.

Before a raw download is started, dates already represented by scene folders in
`MERGED` are removed from the download list. This prevents re-downloading raw
archives for dates that have already been processed.

Raw archives are organized as:

```text
<working_directory>/<study_area>/RAW/
├── Sentinel2/
│   └── TxxYYY/
│       └── S2*_MSIL*.SAFE or .zip
└── Landsat/
    └── Landsat-X/
        └── pathrow/
            └── L*_L1*.tar
```

Existing archives are skipped. OData Sentinel-2 ZIP archives are retained after
extraction for subsequent reuse.
