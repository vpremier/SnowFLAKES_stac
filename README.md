# ❄️ Snow Mapping with SnowFLAKES

Classify snow using **SnowFLAKES**. See https://github.com/bare92/SnowFLAKES/tree/main for the original version. This version includes some changes. For example, training sample selection is based on rules derived from the spectral signatures.

## Workflow overview

The code takes two main inputs:

- **Area of Interest (AOI)**
- **Time range**

It supports both **Sentinel-2** and **Landsat** imagery and implements the following workflow:

1. **Data query**  
   Searches for all satellite acquisitions available for the selected AOI and time range. Optional metadata filters, such as maximum cloud cover, can also be applied.

2. **Data access**  
   Downloads or directly loads the required surface-reflectance bands from different data providers, including the **Copernicus Data Space Ecosystem (CDSE)** and **USGS**. Different access methods are supported depending on the data source.

3. **Data preparation**  
   Preprocesses and crops the spectral bands required by the snow-classification workflow.

4. **SnowFLAKES classification**  
   Applies the **SnowFLAKES algorithm** to generate snow-cover information from the prepared satellite imagery.

---

## 📦 Environment Setup

We recommend using **micromamba** for a fast and reproducible environment.

### Install micromamba

Follow the official guide:
👉 https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html


###️ Create the Environment

Example environment creation:

```bash
micromamba create -n snowmap_cdse -c conda-forge \
python=3.11 \
numpy \
spyder \
gdal \
rasterio \
pyproj \
fiona \
pandas=1.5 \
shapely \
geopandas \
stackstac \
netcdf4 \
opencv \
elevation \
pysolar \
timezonefinder \
scikit-image \
xgboost \
libgdal-jp2openjpeg \
rioxarray \
s2cloudless \
python-dotenv \
pystac-client \
dask \
odc-stac
```

---
## 🛠️ Set up your credentials

SnowFLAKES accesses:

- **Sentinel-2** data through the Copernicus Data Space Ecosystem (CDSE).
- **Landsat** data through the USGS.

The required credentials can be stored in a single file. Create an `.aws` directory in your home directory and add a file named `credentials`:

```bash
mkdir -p ~/.aws
touch ~/.aws/credentials
chmod 600 ~/.aws/credentials
```

The resulting file path is:

```text
~/.aws/credentials
```

Configure the following profiles:

- **`cdse`** for Copernicus Sentinel-2 access.
- **`usgs-landsat`** for USGS Landsat access.


Your credentials file should have the following structure:

```ini
[cdse]
CDSE_USERNAME = ********************
CDSE_PASSWORD = ********************
AWS_ACCESS_KEY_ID = ********************
AWS_SECRET_ACCESS_KEY = ********************

[usgs-landsat]
ERS_USERNAME = ********************
ERS_TOKEN = ********************
aws_access_key_id = ********************
aws_secret_access_key = ********************

```

### USGS Landsat credentials

The credentials required for Landsat depend on the selected data-access method. The conventional Landsat download mode requires:

- A [USGS EarthExplorer account](https://earthexplorer.usgs.gov/).
- An M2M Application Token (`ERS_TOKEN`) (follow the [USGS M2M Application Token documentation](https://www.usgs.gov/media/files/m2m-application-token-documentation)). 

Access through the USGS STAC API requires an AWS account configured for **Requester Pays**. Follow the [USGS Landsat Direct Access tutorial](https://code.usgs.gov/eros-user-services/accessing_landsat_data/tutorials/introduction-to-landsat-cloud-access-direct-requester-pays/-/blob/main/Intro_to_Landsat_Direct_Requester_Pays_v2.ipynb) and add `aws_access_key_id` and `aws_secret_access_key`.

### CDSE Sentinel-2 credentials

Sentinel-2 access requires a CDSE account. Add your account credentials to the `cdse` profile. When using STAC API–based access to data stored on the CDSE S3 service, S3 credentials are also required. Generate them through the [CDSE S3 Key Manager](https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials) and add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to the profile.


---
## Run SnowFLAKES

To run SnowFLAKES, open a terminal and execute:

```bash
./main.sh path/to/config.json
```

The workflow requires a JSON configuration file containing the following parameters:

### General settings

| Parameter | Description |
|---|---|
| **`study_area`** | Name of the study area. |
| **`working_directory`** | Path to the main working directory. |


The script automatically creates the following directory:

```text
<working_directory>/<study_area>/
```

### Query settings
These parameters define the spatial, temporal, and satellite-data filters used in the query.

| Parameter | Description | Format / accepted values |
|---|---|---|
| **`shapefile`** | Path to the vector file defining the Area of Interest (AOI). Supported formats include Shapefile and GeoJSON. Any coordinate reference system (CRS) is accepted. | File path |
| **`date_start`** | Start date of the query period. | `YYYY-MM-DD` |
| **`date_end`** | End date of the query period. | `YYYY-MM-DD` |
| **`satellite`** | Satellite mission to query. | `"Sentinel-2"`, `"Landsat"` or`"both"`|
| **`max_cloudcover`** | Maximum cloud cover allowed for each scene, based on the scene metadata. | Percentage value |


The query creates the following directory:

```text
<working_directory>/<study_area>/QUERY/
```

The query results are stored as CSV files following this naming convention:

```text
<satellite>_<date_start>_<date_end>.csv
```

For example:

```text
Sentinel2_2023-01-01_2023-02-01.csv
Landsat_2023-01-01_2023-02-01.csv
```

The **`date_end` is exclusive**: results include acquisitions from `date_start` up to, but not including, `date_end`.

For time ranges longer than one month, the query is automatically divided into consecutive intervals of no more than one month. This ensures compliance with the maximum number of items allowed per query.

> **Note:** Existing CSV files are reused to avoid repeating completed queries. To run the queries again and retrieve updated results, delete the corresponding CSV files from the `QUERY` directory.


### Download settings
"DOWNLOAD_SENTINEL": the download mode of the Sentinel-2 data. Allowed values are false (no download), "Google" (Google loud API), "S3" (CDSE S3)
and "OData" (CDSE) + STAC-API. It is suggested to use the STAC-API mode for small areas and S3 for larger areas
DOWNLOAD_LANDSAT the download mode of the Landsat data. Allowed is false, m2m-usgs and STAC-API


for S2DL-compatible Google products, or:

