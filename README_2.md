

  
  "skip_sentinel2_tiles": ["T19HDE"],
  "skip_landsat_pathrows": ["232084"]





  "CROP": true,
  "SAVE": true,
  "run_snowflakes": true






`CROP` must be `true` for `STAC-API`. In `RAW` mode, `CROP=true` uses the
configured target extent, while `CROP=false` processes every downloaded tile
separately and retains that tile's reprojected extent.

When `CROP=true`, `resampling_params.extent_target`,
`resampling_params.resolution`, and `resampling_params.epsg_target` are
required. With `RAW` and `CROP=false`, only
`resampling_params.resolution` and `resampling_params.epsg_target` are
required.

`SAVE=true` writes cropped prepared GeoTIFF bands below `MERGED`, or per-tile
prepared bands below `TILES`. SnowFLAKES products and composites use the same
sensor/mission/tile structure below `SnowFLAKES`, without repeating the
`MERGED` or `TILES` directory names. With `SAVE=false`,
the prepared data remains in memory. If
`run_snowflakes` is true, each DataArray is passed to SnowFLAKES immediately
and released before the next date or tile is loaded. Consequently,
`SAVE=false` requires `run_snowflakes=true`; otherwise the configuration is
rejected because a standalone loading process would discard the only copy of
the prepared DataArray.

On later runs, a complete set of saved `*_toa.tif` bands is treated as a
prepared-data cache. The loader rebuilds the georeferenced DataArray directly
from those files and does not reopen the raw archive or repeat STAC loading,
calibration, reprojection, cropping, or mosaicking. An incomplete saved scene
or one whose CRS, resolution, or cropped extent differs from the current
configuration is ignored and regenerated from its original source.

## RGB and false-color composites

Composites are written inside each scene output directory from the prepared
DataArray, including when that array was restored from the GeoTIFF cache:

- `rgb_10m.tif`: Sentinel-2 B04/B03/B02 loaded on demand at native 10 m;
  Landsat uses red/green/blue on the configured analysis grid;
- `fcc.tif`: Sentinel-2 B11/B8A/B03 or Landsat
  swir16/nir08/green.

Use these settings:

```json
"save_rgb": true,
"save_false_color": true
```

False color defaults to `true`. RGB defaults to `false`; the existing
`save_rgb_10m` setting is accepted as a backward-compatible alias for
`save_rgb`. The false-color composite uses the DataArray's configured grid and
resolution. For Sentinel-2, RGB uses the configured EPSG and either the
cropped extent or complete tile extent, while resampling B04/B03/B02 at 10 m.
Existing files are retained unless `overwrite` is `true`.

RAW and STAC preparation use bilinear reprojection, resolution-aligned pixel
edges, and pixel-center coordinates. RAW bounds are snapped outward to the
same whole-pixel grid used by stackstac. When multiple same-day scenes overlap,
RAW uses the same reducers as the STAC loaders: maximum for Sentinel-2 and
mean for Landsat.

Uncropped prepared outputs are grouped by tile:

```text
<working_directory>/<study_area>/TILES/
├── Sentinel-2/
│   └── TxxYYY/
│       └── <scene outputs>
└── Landsat/
    ├── Landsat-5/
    │   └── pathrow/
    ├── Landsat-7/
    │   └── pathrow/
    └── Landsat-8/
        └── pathrow/
```

Merged prepared outputs use `MERGED/Sentinel-2/<scene>` and
`MERGED/Landsat/Landsat-X/<scene>` (without tile/path-row directories).
SnowFLAKES outputs mirror those structures directly below `SnowFLAKES`:
`SnowFLAKES/Sentinel-2/<scene>` and
`SnowFLAKES/Landsat/Landsat-X/<scene>` for merged scenes, or add the tile/path
row below those folders for tile scenes.

Before a raw download is started, dates already represented by scene folders in
`MERGED` are removed from the download list. This prevents re-downloading raw
archives for dates that have already been processed.

Raw archives are organized as:

```text
<working_directory>/RAW/
├── Sentinel-2/
│   └── TxxYYY/
│       └── S2*_MSIL*.SAFE or .zip
└── Landsat/
    └── Landsat-X/
        └── pathrow/
            └── L*_L1*.tar
```

Existing archives are skipped. OData Sentinel-2 ZIP archives are retained after
extraction for subsequent reuse. If a downloaded ZIP/TAR is missing, empty, or
fails archive integrity validation, the scene is recorded in
`<working_directory>/RAW/download_errors.log` and processing continues.
