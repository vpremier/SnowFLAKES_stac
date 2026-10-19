"""Lightweight virtual band stacks for saved prepared imagery."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import rasterio
from rasterio.shutil import copy as copy_raster
from rasterio.io import MemoryFile


def write_scene_vrt(output_dir, scene_id):
    """Stack saved spectral bands without copying pixels; use portable relative paths."""
    scene_dir = Path(output_dir) / scene_id
    # Exclude solar angles and prefer the current filename if both exist.
    band_paths = {}
    for path in sorted(scene_dir.glob(f'{scene_id}_*.tif'),
                       key=lambda path: path.stem.endswith('_toa'), reverse=True):
        token = path.stem[len(scene_id) + 1:].removesuffix('_toa')
        if re.fullmatch(r'B\d+(?:A|_VCID_[12])?|coastal|blue|green|red|nir08|swir16|swir22|lwir(?:11)?', token):
            band_paths[token] = path
    bands = sorted(
        band_paths.values(),
        key=lambda path: [int(part) if part.isdigit() else part
                          for part in re.split(r'(\d+)', path.name)],
    )
    if not bands:
        return None
    root = None
    grid = None
    for index, path in enumerate(bands, 1):
        with rasterio.open(path) as source:
            current_grid = (source.width, source.height, source.crs, source.transform)
            if source.count != 1 or (grid is not None and grid != current_grid):
                raise ValueError(f'Prepared bands must be single-band rasters on the same grid: {path}')
            grid = current_grid
            # Let GDAL preserve each source's datatype and nodata metadata.
            with MemoryFile() as memory:
                copy_raster(source, memory.name, driver='VRT')
                tree = ET.fromstring(memory.read())
        band = tree.find('VRTRasterBand')
        band.set('band', str(index))
        description = band.find('Description')
        if description is None:
            description = ET.SubElement(band, 'Description')
        description.text = path.stem[len(scene_id) + 1:].removesuffix('_toa')
        for filename in band.iter('SourceFilename'):
            filename.text = path.name
            filename.set('relativeToVRT', '1')
        if root is None:
            root = tree
            root.remove(band)
        root.append(band)
    destination = scene_dir / f'{scene_id}.vrt'
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return destination
