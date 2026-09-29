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

## 🛠️ Set Up Your Credentials

Access and load Sentinel-2 data from the **CDSE** and Landsat data from the **USGS**. In the first case, a CDSE account is needed. Furthermore, S3 CDSE credentials also need to be set up (see https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials) when using the CDSE STAC-API based data access. In the second case, for the conventional download mode, the user needs an Earth Explorer (https://earthexplorer.usgs.gov/) account together with a M2M Application Token (please follow the instruction  herehttps://www.usgs.gov/media/files/m2m-application-token-documentation). If the USGS STAC-API based access is used, the USGS STAC catalogue is accessed and an AWS Requester Pays account is needed. Prepare your AWS configuration and credentials before running the workflow.

### Install AWS CLI

```bash
sudo apt update
sudo apt install awscli
```

### Configure AWS Credentials

Run:

```bash
aws configure
```

You will be prompted to enter your AWS Requester Pays credentials:

```text
AWS Access Key ID [None]: ****************
AWS Secret Access Key [None]: ********************
Default region name [eu-central-1]:
Default output format [text]:
```

For more information, see the USGS tutorial:

https://code.usgs.gov/eros-user-services/accessing_landsat_data/tutorials/introduction-to-landsat-cloud-access-direct-requester-pays/-/blob/main/Intro_to_Landsat_Direct_Requester_Pays_v2.ipynb

### Configure Both USGS and CDSE Credentials

Set up credentials for:

- **USGS account** (default profile)
- **Copernicus Data Space Ecosystem (CDSE)** account (`cdse` profile)

Your `~/.aws/credentials` file should look like:

```ini
[default]
aws_access_key_id = ********************
aws_secret_access_key = ********************

[cdse]
CDSE_USERNAME = ********************
CDSE_PASSWORD = ********************
AWS_ACCESS_KEY_ID = ********************
AWS_SECRET_ACCESS_KEY = ********************
```


The `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` values for the `cdse` profile can be generated from:

https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials


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
| **`satellite`** | Satellite mission to query. | `"Sentinel-2"` or `"Landsat"` |
| **`max_cloudcover`** | Maximum cloud cover allowed for each scene, based on the scene metadata. | Percentage value |




