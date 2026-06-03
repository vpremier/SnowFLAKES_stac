# ❄️ Snow Mapping Workflow with Sentinel-2 (CDSE STAC + SnowFLAKES)

This guide describes how to:
1. Access and load Sentinel-2 data from the **CDSE STAC catalogue** and Landsat data from the **USGS STAC catalogue**. In the first case, a CDSE account is needed. Furthermore, S3 CDSE credentials also need to be set up (see https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials). In the second case, the USGS STAC catalogue is accessed and an AWS Requester Pays account is needed.
2. Classify snow using **SnowFLAKES**. See https://github.com/bare92/SnowFLAKES/tree/main for the original version.

---

## 📦 Environment Setup

We recommend using **micromamba** for a fast and reproducible environment.

### Install micromamba

Follow the official guide:
👉 https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

---

## 🛠️ Create the Environment

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

---

## 🛠️ Set Up Your Credentials

Prepare your AWS configuration and credentials before running the workflow.

### Install AWS CLI

```bash
sudo apt update
sudo apt install awscli
```

### 2. Configure AWS Credentials

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