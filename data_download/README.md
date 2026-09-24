# SnowFLAKES query and download preparation

This document describes the first step of the structured workflow: querying
available Sentinel-2 and Landsat scenes and writing CSV files for the later
download step.

## Query entry point

The recommended entry point is the Bash workflow:

```bash
./main.sh path/to/config.json
```

The query stage called by `main.sh` is:

```bash
python3 "${SCRIPT_DIR}/data_download/query_available.py" \
    --config "${CONFIG_PATH}"
```

The Python script can also be called directly:

```bash
python3 data_download/query_available.py --config config/config_mendoza.json
```

## Configuration

The following fields should be defined in the JSON configuration:

```json
{
  "study_area": "Mendoza",
  "working_directory": "/data/SnowFLAKES",
  "shapefile": "/data/AOI/Mendoza.geojson",
  "date_start": "2023-01-01",
  "date_end": "2023-06-01",
  "max_cloudcover": 80,
  "satellite": "both",
  "DOWNLOAD_SENTINEL": "Google",
  "skip_sentinel2_tiles": ["T19HDE"],
  "skip_landsat_pathrows": ["232084"]
}
```

Mandatory query inputs are:

- `shapefile`: path to a shapefile or GeoJSON defining the AOI;
- `date_start`: first date of the query, formatted as `YYYY-MM-DD`;
- `date_end`: final exclusive date, formatted as `YYYY-MM-DD`;
- `working_directory`: root directory for the study-area output;
- `satellite`: `"Sentinel-2"`, a Landsat name such as `"Landsat-8"`, or
  `"both"`.

`study_area` is recommended. If it is omitted, the AOI filename stem is used.
Existing configurations containing only `output_directory` are also supported:
the parent of `output_directory` is used as the working directory.

`max_cloudcover` defaults to `90`. Tile exclusion lists are optional:

- Sentinel-2 values are MGRS tiles, for example `T19HDE`;
- Landsat values are WRS-2 path/rows, for example `232084`.

## Query intervals and date boundaries

The query uses half-open intervals:

```text
date_start <= acquisition date < date_end
```

Therefore, `date_end` is not included in the query.

If the requested interval is longer than one month, it is split into monthly
periods. For example:

```text
2023-01-01 to 2023-02-01
2023-02-01 to 2023-03-01
2023-03-01 to 2023-04-01
```

The query code compensates for the different date-boundary conventions of the
backends. Landsat M2M uses the previous day internally for the inclusive end
boundary. CDSE OData is queried directly from `date_start` because its strict
`ContentDate/Start gt` filter already excludes scenes before the requested
start date.

If a query CSV already exists, it is reused. In OData mode, an existing CSV is
only reused when it contains the required `Id` column.

## Output directory and CSV files

The query creates a `QUERY` directory beneath the working directory and study
area:

```text
<working_directory>/
└── <study_area>/
    └── QUERY/
        ├── Sentinel2_2023-01-01_2023-02-01.csv
        ├── Sentinel2_2023-02-01_2023-03-01.csv
        ├── Landsat_2023-01-01_2023-02-01.csv
        └── Landsat_2023-02-01_2023-03-01.csv
```

The CSV contents are designed for the corresponding download functions:

- Sentinel-2 Google queries contain fields compatible with S2DL, including
  `Name`, `tile`, sensing time, cloud cover and the Google product URL;
- Sentinel-2 OData queries preserve the CDSE `Id` and `Name` fields required by
  `download_cdse()`; S3 queries additionally preserve `S3Path`;
- Landsat queries contain `Name` (the product/display ID) and `entityId`, which
  are required by the USGS M2M downloader.

## Sentinel-2 query source

The query source is normally derived from `DOWNLOAD_SENTINEL`. For example:

```json
"DOWNLOAD_SENTINEL": "Google"
```

or:

```json
"DOWNLOAD_SENTINEL": "OData"
```

Use `"DOWNLOAD_SENTINEL": "S3"` when the resulting query CSV will be used by
the Copernicus Data Space S3 downloader; this preserves the `S3Path` field.

### Google mode

Google mode queries product names available in the Google public Sentinel-2
bucket and produces CSVs for the S2DL downloader. This is the default and does
not require CDSE username/password credentials for the query.

### OData mode

OData mode uses the existing `query_cdse()` function and produces CSVs that can
be passed directly to `download_cdse()` in
`data_download/sentinel2_query_download.py`.

OData mode requires a `.env` file in the repository root or beside the JSON
configuration:

```dotenv
CDSE_USERNAME=your_cdse_username
CDSE_PASSWORD=your_cdse_password
```

## Landsat credentials

Landsat queries use the USGS M2M API and require:

```dotenv
ERS_USERNAME=your_usgs_username
ERS_TOKEN=your_usgs_token
```

The same repository-level `.env` file can contain both Sentinel-2 OData and
Landsat credentials. Do not commit this file to Git.
