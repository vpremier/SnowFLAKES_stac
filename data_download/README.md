# 📦 **data-download**

A Python script to **query and download Sentinel-2 and Landsat data**.

- **Sentinel-2 data**: Downloaded from the [Copernicus Dataspace Ecosystem (CDSE)](https://dataspace.copernicus.eu/).
- **Landsat data**: Downloaded from the [USGS Earth Explorer](https://earthexplorer.usgs.gov/).

---

## 🔐 **Credentials Setup**

To run this code, you **must provide valid credentials** for both **CDSE** and **USGS Earth Explorer**.

**Store your credentials securely** in a `.env` file (recommended) to avoid exposing sensitive information in your source code or version control.

Create a `.env` file in the **root** of your project with the following content:

```env
# .env
ERS_USERNAME=your_ers_username
ERS_TOKEN=your_ers_token
CDSE_USERNAME=your_cdse_username
CDSE_PASSWORD=your_cdse_password
```

> 📌 **Get your ERS token:** Follow the [USGS M2M Application Token Documentation](https://www.usgs.gov/media/files/m2m-application-token-documentation).

---

## 📥 **Clone the Repository**

```bash
git clone git@github.com:vpremier/data-download.git
cd data-download/
```

---

## 🐍 **Set Up the Conda Environment**

To set up the recommended Conda environment:

```bash
# From the project root
chmod +x setup_download_env.sh
./setup_download_env.sh
```

Follow any on-screen instructions that the script prints (e.g., activating a conda/virtualenv).
Alternatively, you can create your own Conda environment and install the required dependencies manually. Add any extra libraries if needed.

---

## ⚙️ **Run the Code**

### ✅ **Run the Main Script**

To run the full processing pipeline, use the provided bash script:

```bash
bash ./run_download.sh
```

Before running this script, you **must configure** your `config.json` file.

---

### 🗂️ ``** — Required Parameters**

Your `config.json` should include:


| Parameter            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `shapefile`          | Path to a shapefile containing your Area of Interest (AOI).                 |
| `output_directory`   | Main output directory.                                                      |
| `date_start` / `date_end` | Date range for querying data (format: YYYY-MM-DD).                   |
| `query_landsat` / `query_sentinel2` | Enable/disable querying for Landsat or Sentinel-2 products. |
| `download_landsat` / `download_sentinel2` | Enable/disable downloading Landsat or Sentinel-2 products. |
| `max_cloudcover`     | Maximum allowed cloud cover percentage (int).                                     |
| `landsat_satellite`  | List of Landsat sensors to include (`LT05`, `LE07`, `LC08`, `LC09`).        |
| `s2_tile_list`       | List of specific Sentinel-2 tiles to download (`[]`=all tiles).                    |
| `landsat_tile_list`  | List of specific Landsat path/row IDs to download (`[]`=all tiles).               |

For more details and additional filter options, check the functions in `landsat_query_download.py` and `sentinel2_query_download.py`.

---

### ▶️ **Run Individual Scripts**

You can also run the individual query/download scripts directly:

```bash
python landsat_query_download.py
```

or

```bash
python sentinel2_query_download.py
```

---

## 📂 **Output Folder Structure**

The downloaded files will be saved in the following structure:

```
output_directory/
├── Sentinel-2/
│   ├── <tile_id>/
│   └── ...
└── Landsat/
    ├── LT05/
    │   ├── <path_row>/
    │   └── ...
    ├── LE07/
    ├── LC08/
    ├── LC09/
    └── ...
```

- **Sentinel-2:** Subfolders by tile ID.
- **Landsat:** Subfolders by sensor (e.g., `LT05`, `LC08`), then by path/row.



