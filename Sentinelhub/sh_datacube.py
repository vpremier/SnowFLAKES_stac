"""Lazy, dask-backed xarray datacubes built on top of sentinelhub-py.

The goal is odc-stac / stackstac-like ergonomics, but using the Sentinel Hub
Process API as the data source instead of direct COG access. The heavy lifting
(authentication, request building, downloading, decoding) is left to
``sentinelhub``; this module only orchestrates:

1. ask the SH Catalog for the acquisition timestamps inside the query,
2. build a single multi-band evalscript (one request == one timestamp == one
   dask chunk),
3. wrap those requests in a lazy ``dask.array`` and hand back an
   ``xarray.DataArray`` with ``(time, band, y, x)`` dimensions.

Why bands are a *dimension* and not separate variables
------------------------------------------------------
One Process API request returns *all* requested bands as a single
(height, width, n_bands) array. To honour "1 request per timestamp" the natural
unit of computation is therefore a ``(1, n_bands, y, x)`` block, which maps
cleanly onto a dask chunk of an array with a ``band`` dimension. Splitting bands
into separate DataArrays would either duplicate requests (one per band) or force
a shared compute that xarray cannot express across variables. A single request
also means a single output ``sampleType``, so all bands necessarily share one
dtype -- we pick the smallest dtype that can hold every requested band (see
``_choose_units``). If you want a ``Dataset`` with one variable per band, call
``.to_dataset(dim="band")`` on the result; the dtype stays uniform.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence

import dask
import dask.array as da
import numpy as np
import pyproj
import xarray as xr
from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
)
from sentinelhub.data_collections_bands import Band

# Mapping from a numpy-ish "rank" to the Sentinel Hub output ``sampleType`` and
# the numpy dtype the decoded array will have. Ordered from smallest to largest;
# the rank is used purely to pick the cheapest representation that fits.
_RANK_TO_SAMPLE_TYPE = {1: "UINT8", 2: "UINT16", 3: "FLOAT32"}
_RANK_TO_DTYPE = {1: np.uint8, 2: np.uint16, 3: np.float32}


def _dtype_rank(output_type: type) -> int:
    """Smallest SH sampleType bucket that can represent ``output_type``."""
    dtype = np.dtype(output_type)
    if dtype == np.dtype(bool) or dtype.itemsize == 1:
        return 1  # bool / (u)int8 -> UINT8
    if dtype == np.dtype(np.uint16) or dtype == np.dtype(np.int16):
        return 2  # 16-bit integers -> UINT16
    return 3  # float / 32-bit+ integers -> FLOAT32


def _resolve_bands(data_collection: DataCollection, band_names: Sequence[str]) -> list[Band]:
    """Look up requested band names in a collection's bands *and* metabands."""
    available: dict[str, Band] = {}
    for band in (data_collection.bands or ()):
        available[band.name] = band
    for band in (data_collection.metabands or ()):
        available[band.name] = band

    resolved = []
    for name in band_names:
        if name not in available:
            raise ValueError(
                f"Band {name!r} is not available in {data_collection.name}. "
                f"Available bands: {sorted(available)}"
            )
        resolved.append(available[name])
    return resolved


def _choose_units(bands: Sequence[Band]) -> tuple[list[str], str, np.dtype]:
    """Pick, per band, the unit whose output dtype is the smallest.

    Each band exposes one or more ``(unit, output_type)`` options. Since one
    request shares a single ``sampleType``, the response dtype is the *largest*
    chosen per-band dtype. Choosing the smallest option for every band provably
    minimises that maximum, so the whole cube uses the cheapest dtype that still
    represents every band losslessly (e.g. all-DN Sentinel-2 -> UINT16, but add
    ``sunAzimuthAngles`` and everything is promoted to FLOAT32).
    """
    units: list[str] = []
    common_rank = 1
    for band in bands:
        ranks = [_dtype_rank(ot) for ot in band.output_types]
        best = int(np.argmin(ranks))
        units.append(band.units[best].value)
        common_rank = max(common_rank, ranks[best])
    return units, _RANK_TO_SAMPLE_TYPE[common_rank], np.dtype(_RANK_TO_DTYPE[common_rank])


def _build_evalscript(band_names: Sequence[str], units: Sequence[str], sample_type: str) -> str:
    """Assemble a //VERSION=3 evalscript returning every band in order."""
    pixels = ", ".join(f"samples[{json.dumps(name)}]" for name in band_names)
    return f"""//VERSION=3
function setup() {{
    return {{
        input: [{{
            bands: {json.dumps(list(band_names))},
            units: {json.dumps(list(units))}
        }}],
        output: {{
            bands: {len(band_names)},
            sampleType: "{sample_type}"
        }},
        mosaicking: "SIMPLE"
    }};
}}

function evaluatePixel(samples) {{
    return [{pixels}];
}}
"""


def _query_timestamps(
    catalog: SentinelHubCatalog,
    data_collection: DataCollection,
    bbox: BBox,
    time: tuple,
    query_filter: str | dict | None,
    filter_lang: str,
) -> list[dt.date]:
    """Return the sorted, de-duplicated solar (UTC) days that have data."""
    search = catalog.search(
        data_collection,
        bbox=bbox,
        time=time,
        filter=query_filter,
        filter_lang=filter_lang,
        fields={"include": ["properties.datetime"], "exclude": []},
    )
    days = {
        dt.datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()
        for item in search
    }
    return sorted(days)


def _dimensions(
    bbox: BBox,
    resolution: float | tuple[float, float] | None,
    size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Resolve (width, height) in pixels from a resolution or explicit size.

    Resolution is interpreted in the BBox CRS units (metres for projected CRSs
    such as UTM or EPSG:3035), matching odc-stac semantics. We compute directly
    from the bounds rather than via ``bbox_to_dimensions`` so the grid stays in
    the request CRS instead of being reprojected to UTM.
    """
    if size is not None:
        return int(size[0]), int(size[1])
    if resolution is None:
        raise ValueError("Provide either `resolution` or `size`.")
    res_x, res_y = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    width = round((bbox.max_x - bbox.min_x) / res_x)
    height = round((bbox.max_y - bbox.min_y) / res_y)
    return width, height


def _spatial_ref(epsg: int) -> tuple[xr.DataArray, pyproj.CRS]:
    """Build a CF/rioxarray-compatible ``spatial_ref`` grid-mapping coordinate.

    Uses ``pyproj`` (a sentinelhub dependency) instead of rasterio/rioxarray.
    ``pyproj.CRS.to_cf()`` returns exactly the CF grid-mapping attributes
    rioxarray writes (``grid_mapping_name``, projection params, ``crs_wkt``); we
    additionally set the GDAL-style ``spatial_ref`` WKT key so tools that look
    for either find the CRS. The scalar value is 0 by convention.
    """
    crs = pyproj.CRS.from_epsg(epsg)
    attrs = crs.to_cf()
    attrs.setdefault("crs_wkt", crs.to_wkt())
    attrs["spatial_ref"] = attrs["crs_wkt"]  # GDAL/rioxarray convention
    return xr.DataArray(np.int32(0), attrs=attrs, name="spatial_ref"), crs


def _axis_attrs(crs: pyproj.CRS) -> tuple[dict, dict]:
    """CF attributes for the x and y coordinate variables."""
    if crs.is_geographic:
        return (
            {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east", "axis": "X"},
            {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north", "axis": "Y"},
        )
    unit = crs.axis_info[0].unit_name if crs.axis_info else "metre"
    return (
        {"standard_name": "projection_x_coordinate", "long_name": "x coordinate of projection",
         "units": unit, "axis": "X"},
        {"standard_name": "projection_y_coordinate", "long_name": "y coordinate of projection",
         "units": unit, "axis": "Y"},
    )


def load(
    data_collection: DataCollection,
    bands: Sequence[str],
    bbox: BBox,
    *,
    time: tuple,
    resolution: float | tuple[float, float] | None = None,
    size: tuple[int, int] | None = None,
    filter: str | dict | None = None,  # noqa: A002 - matches SH Catalog API naming
    filter_lang: str = "cql2-text",
    config: SHConfig | None = None,
    catalog: SentinelHubCatalog | None = None,
) -> xr.DataArray:
    """Build a lazy, dask-backed datacube from the Sentinel Hub Process API.

    :param data_collection: A ``sentinelhub.DataCollection`` (e.g.
        ``DataCollection.SENTINEL2_L2A``).
    :param bands: Band names to request. May include metabands such as ``SCL``
        or ``sunAzimuthAngles``.
    :param bbox: A ``sentinelhub.BBox`` carrying the target CRS. The cube is
        produced on this CRS' grid.
    :param time: ``(start, end)`` accepted by ``SentinelHubCatalog.search``.
    :param resolution: Pixel size in BBox CRS units. Mutually exclusive with
        ``size``.
    :param size: ``(width, height)`` in pixels. Mutually exclusive with
        ``resolution``.
    :param filter: Optional CQL2 filter passed straight to the SH Catalog
        search, e.g. ``"eo:cloud_cover < 80"`` (cql2-text) or the equivalent
        cql2-json dict. Any item property the collection exposes can be used.
    :param filter_lang: ``"cql2-text"`` (default) or ``"cql2-json"``, matching
        how ``filter`` is written.
    :param config: ``SHConfig`` with credentials / service URL. Required for
        anything but the default deployment.
    :param catalog: Optional pre-built ``SentinelHubCatalog`` (otherwise one is
        created from ``config``).
    :return: An ``xarray.DataArray`` with dims ``(time, band, y, x)``. Nothing
        is downloaded until the array is computed.
    """
    config = config or SHConfig()
    catalog = catalog or SentinelHubCatalog(config=config)

    # Pin the collection to the configured deployment (e.g. CDSE) so Catalog and
    # Process API hit the same service. We derive a definition and resolve it via
    # ``DataCollection(...)`` rather than ``define_from``: the latter raises if an
    # equally-defined collection already exists under a different name (``_name``
    # is excluded from equality), which happens e.g. when the caller already ran
    # ``define_from`` themselves. ``DataCollection(...)`` instead returns the
    # existing member, or registers a new one if none matches.
    if config.sh_base_url and data_collection.service_url != config.sh_base_url:
        derived = data_collection.value.derive(
            service_url=config.sh_base_url,
            _name=f"{data_collection.name}_{config.sh_base_url}",
        )
        data_collection = DataCollection(derived)

    band_objs = _resolve_bands(data_collection, bands)
    units, sample_type, dtype = _choose_units(band_objs)
    evalscript = _build_evalscript(bands, units, sample_type)

    width, height = _dimensions(bbox, resolution, size)
    days = _query_timestamps(catalog, data_collection, bbox, time, filter, filter_lang)
    if not days:
        raise ValueError("Catalog search returned no acquisitions for this query.")

    n_bands = len(bands)

    def _fetch(day: dt.date) -> np.ndarray:
        """Download one day, returning a (1, band, y, x) block."""
        day_str = day.isoformat()
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=(day_str, day_str),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=(width, height),
            config=config,
        )
        arr = np.asarray(request.get_data(max_threads=1)[0])
        if arr.ndim == 2:  # single-band responses come back 2D
            arr = arr[..., np.newaxis]
        arr = np.moveaxis(arr, -1, 0)  # (band, y, x)
        return arr.astype(dtype, copy=False)[np.newaxis]  # (1, band, y, x)

    chunks = [
        da.from_delayed(
            dask.delayed(_fetch)(day),
            shape=(1, n_bands, height, width),
            dtype=dtype,
        )
        for day in days
    ]
    cube = da.concatenate(chunks, axis=0)

    res_x = (bbox.max_x - bbox.min_x) / width
    res_y = (bbox.max_y - bbox.min_y) / height
    xs = bbox.min_x + (np.arange(width) + 0.5) * res_x
    ys = bbox.max_y - (np.arange(height) + 0.5) * res_y

    spatial_ref, crs = _spatial_ref(bbox.crs.epsg)
    x_attrs, y_attrs = _axis_attrs(crs)

    return xr.DataArray(
        cube,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.array(days, dtype="datetime64[ns]"),
            "band": list(bands),
            "y": xr.DataArray(ys, dims="y", attrs=y_attrs),
            "x": xr.DataArray(xs, dims="x", attrs=x_attrs),
            "spatial_ref": spatial_ref,
        },
        name="data",
        attrs={
            "grid_mapping": "spatial_ref",
            "sample_type": sample_type,
            "units": dict(zip(bands, units)),
        },
    )
