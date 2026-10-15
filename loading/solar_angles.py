"""Load and mosaic Sentinel-2 and Landsat solar-angle grids from STAC items."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer







def _read_metadata(href, s3_client):
    parsed = urlparse(href)
    if parsed.scheme == "s3":
        if s3_client is None:
            import boto3

            s3_client = boto3.Session(profile_name="cdse").client(
                "s3", endpoint_url="https://eodata.dataspace.copernicus.eu"
            )
        response = s3_client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        with response["Body"] as stream:
            return stream.read()
    if parsed.scheme in ("http", "https"):
        with urlopen(href, timeout=60) as stream:
            return stream.read()
    if parsed.scheme:
        raise ValueError(f"Unsupported metadata URL scheme: {parsed.scheme}")
    return Path(href).read_bytes()


def _parse_grid(xml):
    root = ET.fromstring(xml)
    # Sentinel-2 metadata uses namespace-qualified and unqualified elements.
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    geocoding = root.find(".//Tile_Geocoding")
    angles = root.find(".//Sun_Angles_Grid")
    if geocoding is None or angles is None:
        raise ValueError("Tile metadata lacks Tile_Geocoding or Sun_Angles_Grid")
    position = geocoding.find("Geoposition")
    size = next(
        entry for entry in geocoding.findall("Size")
        if entry.get("resolution") == position.get("resolution")
    )
    x0, y0 = (float(position.findtext(key)) for key in ("ULX", "ULY"))
    xend = x0 + float(position.findtext("XDIM")) * int(size.findtext("NCOLS"))
    yend = y0 + float(position.findtext("YDIM")) * int(size.findtext("NROWS"))
    grids = []
    for name in ("Zenith", "Azimuth"):
        angle = angles.find(name)
        values = np.array([
            [float(value) for value in row.text.split()]
            for row in angle.findall("Values_List/VALUES")
        ], dtype=np.float32)
        dx, dy = (float(angle.findtext(key)) for key in ("COL_STEP", "ROW_STEP"))
        if values.ndim != 2 or not values.size or dx <= 0 or dy <= 0:
            raise ValueError(f"Invalid {name} angle grid")
        grids.append((values, dx, dy))
    return geocoding.findtext("HORIZONTAL_CS_CODE"), (x0, y0, xend, yend), grids


def load_s2_solar_angle_grids(items, *, dst_crs, dst_transform, dst_shape,
                           s3_client=None):
    """Return merged ``(zenith, azimuth)`` float32 arrays in degrees.

    Parameters
    ----------
    items : iterable of dict or pystac.Item
        Items with a ``granule_metadata`` asset pointing to MTD_TL.xml.
        Local XML paths, HTTPS URLs and CDSE S3 URLs are supported.
    dst_crs : CRS-like
        Target grid CRS, for example ``"EPSG:32619"``.
    dst_transform : affine.Affine
        Target raster transform (pixel corners, not pixel centres).
    dst_shape : tuple[int, int]
        Target raster height and width.
    s3_client : optional
        Authenticated boto3 S3 client. Defaults to the ``cdse`` profile and
        CDSE endpoint when an S3 asset is encountered.

    Angle nodes are sampled by nearest neighbour at target pixel centres,
    with coverage restricted to each XML tile's rectangular raster extent.
    Both angles must be finite to contribute. The first valid item wins in
    overlaps; uncovered pixels are NaN. An empty item list yields NaN arrays.
    Missing or malformed metadata raises an error identifying the item.

    This merges solar geometry only: it does not reproduce band-wise maximum
    compositing or cloud masks applied to reflectance imagery. Pass items in
    the desired priority order. Native grids are coarse (normally 5 km);
    resampling does not add spatial detail.
    """
    height, width = dst_shape
    if height <= 0 or width <= 0:
        raise ValueError("dst_shape must contain positive height and width")
    zenith = np.full((height, width), np.nan, dtype=np.float32)
    azimuth = np.full_like(zenith, np.nan)
    columns = np.arange(width, dtype=float)[None, :] + 0.5
    for item in items:
        item = item.to_dict() if hasattr(item, "to_dict") else item
        try:
            href = item["assets"]["granule_metadata"]["href"]
            crs, (x0, y0, xend, yend), grids = _parse_grid(
                _read_metadata(href, s3_client)
            )
            transformer = Transformer.from_crs(dst_crs, crs, always_xy=True)
            # Work in strips to avoid full-scene coordinate arrays.
            for start in range(0, height, 256):
                stop = min(start + 256, height)
                rows = np.arange(start, stop, dtype=float)[:, None] + 0.5
                x = dst_transform.a * columns + dst_transform.b * rows + dst_transform.c
                y = dst_transform.d * columns + dst_transform.e * rows + dst_transform.f
                x, y = transformer.transform(x, y)
                valid = (np.isfinite(x) & np.isfinite(y)
                         & (x >= min(x0, xend)) & (x < max(x0, xend))
                         & (y > min(y0, yend)) & (y <= max(y0, yend))
                         & np.isnan(zenith[start:stop]))
                samples = []
                for values, dx, dy in grids:
                    # Round half up; the first node is exactly at ULX/ULY.
                    col = np.floor(np.where(valid, (x - x0) / dx, 0) + 0.5).astype(int)
                    row = np.floor(np.where(valid, (y0 - y) / dy, 0) + 0.5).astype(int)
                    valid &= ((row >= 0) & (row < values.shape[0])
                              & (col >= 0) & (col < values.shape[1]))
                    sampled = values[np.clip(row, 0, values.shape[0] - 1),
                                     np.clip(col, 0, values.shape[1] - 1)]
                    valid &= np.isfinite(sampled)
                    samples.append(sampled)
                zenith[start:stop][valid] = samples[0][valid]
                azimuth[start:stop][valid] = samples[1][valid]
        except Exception as error:
            raise ValueError(f"Cannot load solar angles for {item.get('id', '<unknown>')}: {error}") from error
    return zenith, azimuth


def load_landsat_solar_angle_grids(items, *, dst_crs, dst_transform, dst_shape,
                                   aws_session=None):
    """Return merged Landsat C2 L1 ``(zenith, azimuth)`` float32 degree arrays.

    Uses the ready-made ``SZA`` and ``SAA`` GeoTIFF assets (band-4 solar
    geometry), not the scalar SUN_ELEVATION/SUN_AZIMUTH in MTL metadata.
    ANG.txt-only items require the USGS angle-generation tools and are not
    supported here. Missing assets raise an error rather than substituting
    scene-centre angles.

    Target arguments and first-valid-item overlap priority match
    :func:`load_s2_solar_angle_grids`. Nearest-neighbour resampling preserves
    angle discontinuities. Stored hundredths of degrees are converted to
    degrees; azimuth is normalized to [0, 360). Fill (-32768), raster nodata,
    and uncovered pixels become NaN. Both angles must be valid in an item
    before that item contributes a pixel. This is not a temporal average.

    Items may be dictionaries or pystac.Item objects. Local files and HTTPS
    are supported. USGS S3 alternates are preferred when present, using the
    ``usgs-landsat`` profile, us-west-2, and requester-pays access. Supply a
    rasterio.session.AWSSession through ``aws_session`` to override credentials.
    """
    from contextlib import ExitStack

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    height, width = dst_shape
    if height <= 0 or width <= 0:
        raise ValueError("dst_shape must contain positive height and width")
    zenith = np.full((height, width), np.nan, dtype=np.float32)
    azimuth = np.full_like(zenith, np.nan)
    for item in items:
        item = item.to_dict() if hasattr(item, "to_dict") else item
        try:
            hrefs = []
            for key in ("SZA", "SAA"):
                if key not in item["assets"]:
                    raise ValueError(f"Missing {key} raster asset; ANG.txt coefficients "
                                     "must first be processed with USGS angle-generation tools")
                asset = item["assets"][key]
                hrefs.append(asset.get("alternate", {}).get("s3", {}).get("href")
                             or asset["href"])
            with ExitStack() as stack:
                if any(href.startswith("s3://") for href in hrefs):
                    if aws_session is None:
                        import boto3
                        from rasterio.session import AWSSession

                        aws_session = AWSSession(
                            boto3.Session(profile_name="usgs-landsat", region_name="us-west-2"),
                            requester_pays=True,
                        )
                    stack.enter_context(rasterio.Env(
                        aws_session, AWS_S3_ENDPOINT="s3.us-west-2.amazonaws.com",
                        AWS_VIRTUAL_HOSTING=True, AWS_HTTPS=True,
                    ))
                rasters = []
                for href in hrefs:
                    source = stack.enter_context(rasterio.open(href))
                    rasters.append(stack.enter_context(WarpedVRT(
                        source, crs=dst_crs, transform=dst_transform,
                        height=height, width=width, resampling=Resampling.nearest,
                        src_nodata=source.nodata if source.nodata is not None else -32768,
                        nodata=np.nan, dtype="float32",
                    )))
                for start in range(0, height, 256):
                    stop = min(start + 256, height)
                    window = Window(0, start, width, stop - start)
                    values = [raster.read(1, window=window, masked=True).filled(np.nan)
                              for raster in rasters]
                    valid = (np.isnan(zenith[start:stop])
                             & np.isfinite(values[0]) & np.isfinite(values[1])
                             & (values[0] != -32768) & (values[1] != -32768))
                    zenith[start:stop][valid] = values[0][valid] * 0.01
                    azimuth[start:stop][valid] = (values[1][valid] * 0.01) % 360
        except Exception as error:
            raise ValueError(f"Cannot load Landsat solar angles for {item.get('id', '<unknown>')}: {error}") from error
    return zenith, azimuth


def attach_solar_angles(data, zenith, azimuth):
    """Attach degree arrays as spatial coordinates without changing image bands.

    Coordinates have dimensions (y, x); they are available through
    data['solar_zenith'] and data['solar_azimuth']. Spectral data.values and
    band selection retain their existing shape and values. The STAC loaders
    also export these coordinates via save_solar_angles when save=True.
    The input must represent one merged acquisition day.
    """
    if data.sizes.get("day", 1) != 1 or data.sizes.get("time", 1) != 1:
        raise ValueError("Solar-angle coordinates require a single acquisition day")
    shape = (data.sizes["y"], data.sizes["x"])
    if np.shape(zenith) != shape or np.shape(azimuth) != shape:
        raise ValueError(f"Solar-angle arrays must match the image grid {shape}")
    metadata = {"units": "degrees", "resampling": "nearest",
                "overlap_policy": "first valid item"}
    return data.assign_coords(
        solar_zenith=(("y", "x"), np.asarray(zenith, dtype=np.float32),
                      dict(metadata, long_name="Solar zenith angle")),
        solar_azimuth=(("y", "x"), np.asarray(azimuth, dtype=np.float32),
                       dict(metadata, long_name="Solar azimuth angle", convention="clockwise from north")),
    )


def save_solar_angles(data, outdir, scene_id, transform, crs, ow=False):
    """Write spatial angle coordinates as float32 GeoTIFFs in degrees.

    Files are named <scene_id>_solar_zenith.tif and
    <scene_id>_solar_azimuth.tif, beside the spectral band files.
    Missing coverage is stored as NaN nodata. Existing files are skipped
    unless ow=True. Angles have no reflectance (toa) suffix.
    """
    import rasterio

    output_dir = Path(outdir) / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("solar_zenith", "solar_azimuth"):
        path = output_dir / f"{scene_id}_{name}.tif"
        if path.exists() and not ow:
            print(f"Skipping {path} (already exists)")
            continue
        angle = data[name].transpose("y", "x")
        with rasterio.open(
            path, "w", driver="GTiff", height=data.sizes["y"], width=data.sizes["x"],
            count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
        ) as destination:
            destination.write(np.asarray(angle.values, dtype=np.float32), 1)
            destination.set_band_description(1, name)
            destination.set_band_unit(1, "degrees")
            destination.update_tags(**angle.attrs)
        print(f"Saved {path}")


def load_landsat_archive_solar_angle_grids(archives, *, dst_crs, dst_transform,
                                           dst_shape):
    """Read local USGS-M2M TAR SZA/SAA members without extracting them.

    Archive order determines first-valid overlap priority. The returned
    arrays follow load_landsat_solar_angle_grids conventions. Both angle
    rasters must be present; scalar MTL angles are not used as a fallback.
    """
    import tarfile

    items = []
    for archive in archives:
        archive = Path(archive).resolve()
        assets = {}
        with tarfile.open(archive) as source:
            members = [member.name for member in source.getmembers() if member.isfile()]
        for key in ("SZA", "SAA"):
            matches = [name for name in members if name.upper().endswith(f"_{key}.TIF")]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {key} raster in Landsat archive {archive}; "
                    f"found {len(matches)}. A complete Collection 2 Level-1 product is required."
                )
            assets[key] = {"href": f"/vsitar/{archive}/{matches[0]}"}
        items.append({"id": archive.stem, "assets": assets})
    return load_landsat_solar_angle_grids(
        items, dst_crs=dst_crs, dst_transform=dst_transform, dst_shape=dst_shape,
    )


def load_s2_local_solar_angle_grids(scene_dirs, *, dst_crs, dst_transform,
                                    dst_shape):
    """Read MTD_TL.xml in local Sentinel-2 product/SAFE directories.

    Requires exactly one tile metadata file per supplied product directory.
    Uses the same nearest-neighbour sampling and first-valid overlap rule as
    STAC. Missing metadata raises an error; no remote download is attempted.
    """
    items = []
    for scene_dir in scene_dirs:
        scene_dir = Path(scene_dir)
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Sentinel-2 product directory is missing: {scene_dir}")
        matches = sorted(path for path in scene_dir.rglob("*")
                         if path.is_file() and path.name.upper() == "MTD_TL.XML")
        if len(matches) != 1:
            raise ValueError(
                f"Expected one MTD_TL.xml in Sentinel-2 product {scene_dir}; "
                f"found {len(matches)}. Local tile metadata is required for solar angles."
            )
        items.append({"id": scene_dir.name, "assets": {
            "granule_metadata": {"href": str(matches[0])},
        }})
    return load_s2_solar_angle_grids(
        items, dst_crs=dst_crs, dst_transform=dst_transform, dst_shape=dst_shape,
    )
