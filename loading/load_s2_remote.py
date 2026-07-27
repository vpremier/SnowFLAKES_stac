"""Remote Sentinel-2 loader with native-TLM support when available.

This module is a drop-in alternative to :mod:`loading.load_stac` for the
per-date workflow.  It deliberately uses only dependencies already used by
the project (``stackstac``, ``rasterio`` and the CDSE STAC client).

Sentinel-2 products that contain native JPEG2000 TLM markers are read using
the optimized OpenJPEG path.  Older products, which do not contain TLM
markers, are still supported through ordinary remote range requests.  A
precomputed historical TLM index is not required, but older products may be
slower to read.
"""

import logging
import os

from loading.load_stac import convert_sentinel2_bands as _convert_sentinel2_bands
from loading.load_stac import setup_cdse_credentials


def convert_sentinel2_bands(*args, **kwargs):
    """Load one Sentinel-2 date through CDSE with optimized remote settings.

    The signature and return value are inherited from
    :func:`loading.load_stac.convert_sentinel2_bands`, so this function can be
    used as a drop-in replacement in the existing workflow.  It returns
    ``(data, scene_id)``.

    Native TLM handling is performed by the installed GDAL/OpenJPEG driver;
    this wrapper does not create or inject TLM indexes.
    """
    os.environ.setdefault("AWS_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
    os.environ.setdefault("AWS_HTTPS", "YES")
    os.environ.setdefault("AWS_VIRTUAL_HOSTING", "FALSE")
    os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
    os.environ.setdefault("GDAL_HTTP_TCP_KEEPALIVE", "YES")
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".jp2,.JP2,.xml,.XML")

    logging.getLogger(__name__).info(
        "Loading Sentinel-2 through CDSE remote JP2 access "
        "(native TLM when available)"
    )
    return _convert_sentinel2_bands(*args, **kwargs)


__all__ = ["convert_sentinel2_bands", "setup_cdse_credentials"]
