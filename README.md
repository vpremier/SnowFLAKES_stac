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
| **`landsat_satellite`** |  Optional filter for one or more specific Landsat satellites. Use an empty list to query all supported Landsat missions. | Empty list (`[]`) or a list of satellite identifiers, e.g. [`"LC08"`]
| **`sentinel_tile_list`** | Optional filter for one or more Sentinel-2 tiles. Use an empty list to query all tiles intersecting the AOI. | Empty list (`[]`) or a list of tile identifiers, e.g. `["T19HCD"]` |
| **`landsat_tile_list`** | Optional filter for one or more Landsat path/row identifiers. Use an empty list to query all tiles intersecting the AOI. | Empty list (`[]`) or a list of path/row identifiers, e.g. `["233082"]` |

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


The **`date_end` is exclusive**: results include acquisitions from `date_start` up to, but not including, `date_end`.

For time ranges longer than one month, the query is automatically divided into consecutive intervals of no more than one month. This ensures compliance with the maximum number of items allowed per query.

> **Note:** Existing CSV files are reused to avoid repeating completed queries. To run the queries again and retrieve updated results, delete the corresponding CSV files from the `QUERY` directory.


### Download settings
The following parameters control how Sentinel-2 and Landsat products are accessed.

| Parameter | Accepted values |
|---|---|
| **`DOWNLOAD_SENTINEL`** | `"STAC-API"`, `"OData"`, `"S3"`, `"Google"`, or `false` |
| **`DOWNLOAD_LANDSAT`** | `"STAC-API"`, `"USGS-M2M"`, or `false` |

> In JSON, `false` is a Boolean value and must not be enclosed in quotation marks.

### Access modes

| Mode | Sensor | Behaviour |
|---|---|---|
| **`STAC-API`** | Sentinel-2 or Landsat | Queries the relevant STAC catalogue and loads the required bands directly from remote object storage. |
| **`OData`** | Sentinel-2 | Queries and downloads complete products from the Copernicus Data Space Ecosystem using its OData service. |
| **`S3`** | Sentinel-2 | Downloads products directly from the CDSE S3 object-storage service. |
| **`Google`** | Sentinel-2 | Searches for S2DL-compatible products in the public Google Cloud Sentinel-2 archive and downloads them locally. No credentials are required. |
| **`USGS-M2M`** | Landsat | Downloads complete Landsat products using the USGS Machine-to-Machine API. |
| **`false`** | Sentinel-2 or Landsat | Disables downloading for the corresponding sensor. Products already available locally can still be processed. |

Products retrieved using **`OData`**, **`S3`**, **`Google`**, or **`USGS-M2M`** are saved in the `RAW` directory and subsequently loaded from the local filesystem. The directory is created at:

```text
<working_directory>/RAW/
```
Because the `RAW` directory is shared across study areas, downloaded products can be reused for different AOIs without being downloaded again.

### Input-band preparation

The preparation of the input bands is controlled by two Boolean parameters:

| Parameter | Description | Accepted values |
|---|---|---|
| **`CROP`** | Controls whether the bands are cropped to a common target extent. | `true` or `false` |
| **`SAVE`** | Controls whether the prepared bands are saved to disk. | `true` or `false` |

#### Cropping and spatial parameters

- **`CROP=true`** is required when using **`STAC-API`** because a target extent must be defined.
- With locally downloaded products:
  - **`CROP=true`** crops all products to the configured target extent.
  - **`CROP=false`** processes each tile separately and preserves its reprojected extent.

When **`CROP=true`**, the following parameters are required:

```json
"resampling_params": {
  "extent_target": [xmin, ymin, xmax, ymax],
  "resolution": 20,
  "epsg_target": 32632,
  "no_data_value": "nan"
}
```

When **`CROP=false`**, only `resolution` and `epsg_target` are required.

#### Saving the prepared bands

- **`SAVE=true`** writes the prepared GeoTIFF bands to disk:
  - Cropped products are saved under `MERGED`.
  - Products processed separately are saved under `TILES`.
- **`SAVE=false`** does not write the prepared bands to disk. Instead, the resulting `DataArray` is passed directly to the SnowFLAKES algorithm, reducing disk-space usage.

> **Note:** When `SAVE=false`, **`run_snowflakes` must be `true`**. Otherwise, the prepared data would neither be processed nor saved, and the configuration is therefore rejected.



### SnowFLAKES outputs

The generation of RGB and false-color composites is controlled by the  Boolean settings `save_rgb` and `save_false_color`.
Both parameters accept `true` or `false`. When enabled, the corresponding composites are saved in each scene’s output directory:

- **`rgb_10m.tif`** or **`rgb_30m.tif`**:
  - Sentinel-2: true-color composite using bands B04/B03/B02 at the native 10 m resolution.
  - Landsat: true-color composite using the red/green/blue bands at the native 30 m resolution.

- **`fcc.tif`**: false-color composite generated at the configured target resolution using:
  - Sentinel-2: B11/B8A/B03.
  - Landsat: `swir16`/`nir08`/`green`.



The other parameters are described in the following table:

| Parameter | Description | Accepted values |
| --- | --- | --- |
| **`run_snowflakes`** | Controls whether the SnowFLAKES classification is performed and the classification output is generated. | `true` or `false` |
| **`uncertainty`** | Controls whether an additional GeoTIFF containing the classification uncertainty estimate is generated. | `true` or `false` |
| **`remove_auxiliary`** | Controls whether the auxiliary directory within each SnowFLAKES scene output directory is removed after processing. | `true` or `false` |
| **`classify_glaciers`** | Controls whether glacier-covered areas are included in the classification. | `true` or `false` |
| **`external_glacier_mask_path`** | Path to an external glacier-mask shapefile, such as one derived from the Randolph Glacier Inventory (RGI). | File path or `null` |
| **`find_closest_model`** | Controls the handling of dates for which one of the training classes—snow or snow-free—is missing. When enabled, the workflow selects and applies the temporally closest available classification model. | `true` or `false` |



The **`overwrite`** parameter controls which existing products are regenerated and which are reused.

| Value | Behaviour |
| --- | --- |
| **`false`** | Reuses all existing products and generates only missing outputs. |
| **`"QUERY"`** | Regenerates the query CSV files and all downstream products. |
| **`"CROP"`** | Regenerates the prepared bands (merged or cropped), RGB and false-color composites, and SnowFLAKES outputs. Existing query results are preserved. |
| **`"RGB"`** | Regenerates the RGB and false-color composites and all SnowFLAKES outputs. Existing query results and prepared bands are preserved. |
| **`"SnowFLAKES"`** | Regenerates only the SnowFLAKES classification outputs. All upstream products are preserved. |

The values define successive processing stages: selecting an earlier stage also regenerates all products produced by the downstream stages.









