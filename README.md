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