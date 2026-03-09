#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:49:05 2026

@author: vpremier
"""

import geopandas as gpd
from pyproj import CRS
import numpy as np
import rasterio as rio
from shapely.geometry import box
from osgeo import gdal, osr, ogr
import os
import netCDF4



def get_shape_extent(shape_name, epsg=3035, outres=500, merge=True, row=None):
    # Read shapefile
    gdf = gpd.read_file(shape_name)

    # Get current CRS
    crs_shp = gdf.crs
    target_crs = CRS.from_epsg(epsg)

    # Reproject if needed
    if crs_shp != target_crs:
        print('The input shapefile is in another reference system. Reprojecting...')
        gdf = gdf.to_crs(target_crs)

    if merge:
        if row is not None:
            raise ValueError("`row` must be None when `merge=True`.")
        xmin, ymin, xmax, ymax = gdf.total_bounds
    else:
        if row is None:
            raise ValueError("You must provide a `row` index when `merge=False`.")
        geom = gdf.iloc[row].geometry  # safer than gdf[row]
        xmin, ymin, xmax, ymax = geom.bounds

    # Round to outres
    xMin = round(int(xmin / outres) * outres, 5)
    yMin = round(int(ymin / outres) * outres, 5)
    xMax = round(int(np.ceil(xmax / outres)) * outres, 5)
    yMax = round(int(np.ceil(ymax / outres)) * outres, 5)

    return xMin, yMin, xMax, yMax





def get_bbox_wgs84(img4ext=None, extent_target=None, epsg_target=None, buffer_m=0):
    """
    Returns bbox [xmin, ymin, xmax, ymax] in WGS84.
    
    Parameters
    ----------
    img4ext : str, optional
        Path to reference raster.
    extent_target : tuple, optional
        Target extent (xmin, ymin, xmax, ymax).
    epsg_target : int or str, optional
        EPSG code of extent_target.
    buffer_m : float, optional
        Buffer in CRS units (e.g., meters if projected).
    """

    if img4ext:
        with rio.open(img4ext) as src:
            bounds = src.bounds
            crs = src.crs
        geom = gpd.GeoSeries([box(*bounds)], crs=crs)

    elif extent_target and epsg_target:
        crs = CRS.from_user_input(f"EPSG:{epsg_target}")
        geom = gpd.GeoSeries([box(*extent_target)], crs=crs)

    else:
        raise ValueError("Must provide either img4ext OR (extent_target + epsg_target)")

    # Apply buffer if requested
    if buffer_m > 0:
        geom = geom.buffer(buffer_m)

    # Convert to WGS84 and extract bbox
    geom_wgs84 = geom.to_crs(4326)
    return geom_wgs84.total_bounds.tolist()





def reproj_point(x, y, srIn, srOut):
    """Reproject a point into a defined coordinate system.
    
    Parameters
    ----------
    x : float
        x-coordinate
    y : float
        y-coordinate
    srIn : osgeo.osr.SpatialReference
        spatial reference of the input coordinate system
    srOut : osgeo.osr.SpatialReference
        spatial reference of the output coordinate system
        
    Returns
    -------
    (x, y) : tuple
        the transformed coordinates
    """
    
    epsgList = ['4326', '31287']
    # create a geometry from coordinates
    point = ogr.Geometry(ogr.wkbPoint)
    
    if int(gdal.__version__[0]) >= 3 and srIn.GetAttrValue("AUTHORITY", 1) in epsgList:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        point.AddPoint(y, x)
    else:
        point.AddPoint(x, y)
    
    coordTransform = osr.CoordinateTransformation(srIn,srOut)
    
    # transform point
    point.Transform(coordTransform)
    
    if int(gdal.__version__[0]) >= 3 and srOut.GetAttrValue("AUTHORITY", 1) in epsgList:
        (y, x) = point.GetX(), point.GetY()
    else:
        (x, y) = point.GetX(), point.GetY()
        
    return (x,y)





def get_closest_extent(info, epsg_target, res, prec =5):
    """Given two images with two different crs, gets the array indices (of the 
    top left and right bottom corners) of the second (target) image in the first 
    (source) image and the new geotransform (in the source crs).
    
    Parameters
    ----------
    info_target : dict
        dictionary containing the target image metadata (see the function open_image)
        
    info_src : dict
        dictionary containing the source image metadata    
        
    buffer : int
        number of pixels in the buffer area
    
    Returns
    -------
    x_tl : float
        the x-coordinate of the scene's top left corner point
    y_br : float
        the y-coordinate of the scene's bottom right corner point
    x_br : float
        the x-coordinate of the scene's bottom right corner point
    y_tl : float
        the y-coordinate of the scene's top left corner point
    geotransform : tuple
        the geotransform information used to georeference the image
    """
    
    srOut = osr.SpatialReference()
    srOut.ImportFromEPSG(int(epsg_target))
    
    # reference points to project: top left, bottom right
    xtl = info['geotransform'][0]
    ytl = info['geotransform'][3]
    xbr = (info['geotransform'][0] +
              info['X_Y_raster_size'][0] * 
              info['geotransform'][1])
    ybr = (info['geotransform'][3] +
              info['X_Y_raster_size'][1] *
              info['geotransform'][5])
    

    srIn = osr.SpatialReference()
    srIn.ImportFromWkt(info['projection'])
    
    # reproject reference points
    (xtl_r, ytl_r) = reproj_point(xtl, ytl, srIn, srOut)
    (xtr_r, ytr_r) = reproj_point(xbr, ytl, srIn, srOut)
    (xbr_r, ybr_r) = reproj_point(xbr, ybr, srIn, srOut)
    (xbl_r, ybl_r) = reproj_point(xtl, ybr, srIn, srOut)
    
    # extent of the image
    xMin = min(xtl_r, xbl_r)
    xMax = max(xtr_r, xbr_r)
    yMin = min(ybr_r, ybl_r)
    yMax = max(ytl_r, ytr_r)
    
    
    # find extent in the new reference system
    xMin = round(int(xMin / res) * res, prec)
    yMin = round(int(yMin / res) * res, prec)
    xMax = round(np.ceil(xMax / res) * res, prec)
    yMax = round(np.ceil(yMax / res) * res, prec)
    
    # check if the extent exceeds 180 degrees longitude
    # if yes, correct reset to -180 degrees by adding 360 degrees
    if xMax < xMin and epsg_target == '4326':
        xMax = xMax + 360
        
    return xMin, yMin, xMax, yMax





def open_image(image_path):
    """Opens an image and reads its metadata.
    
    Parameters
    ----------
    image_path : str
        path to an image
    
    Returns
    -------
    image : osgeo.gdal.Dataset
        the opened image
    information : dict
        dictionary containing image metadata    
    """
    
    ext = os.path.basename(image_path).split('.')[-1]
    
    if ext == 'nc':
        nc_data = netCDF4.Dataset(image_path,'r')
        vars_nc = list(nc_data.variables)
        scf_name = list(filter(lambda x: x.startswith('scf'), vars_nc))[0]
        
        image = gdal.Open("NETCDF:{0}:{1}".format(image_path, scf_name))

            

    else:
        image = gdal.Open(image_path)
    
    if image is None:
        print('could not open ' + image_path)
        return
        
    cols = image.RasterXSize
    rows = image.RasterYSize
    geotransform = image.GetGeoTransform()
    proj = image.GetProjection()
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows
    X_Y_raster_size = [cols, rows]
    extent = [minx, miny, maxx, maxy]
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj

    if ext == 'nc':
        information['geotransform'] = tuple(map(lambda x: round(x, 2) or x, information['geotransform']))
        information['extent'] = tuple(map(lambda x: round(x, 2) or x, information['extent']))

    return image, information




def define_info(info_src, extent_target = None, epsg_target = None, 
                resolution = None, img4ext = None):
    """
    Get new information dictionary. Extent, epsg and resolution
    can be either user defined or the raw ones (of the original image), if 
    no parameter is specified. 

    Parameters
    ----------
    info_src : dict
        dictionary with the information of the source image.
    extent_target : tuple, optional
        (xmin, ymin, xmax, ymax). The default is None.
    epsg_target : str, optional
        EPSG of the target reference system. The default is None.
    resolution : int, optional
        Target resolution. The default is None.
    img4ext : str, optional
        Path of an image to bet used to set the extent/epsg. The default is False.


    Returns
    -------
    info_target : dict
        Target informations (extent, resolution, epsg) stored as dictionary.

    """
    
    if extent_target is None and img4ext is None:
        
        # keep the original extent. It is possible to define the target crs
        # and/or the target resolution
        info_target = info_src.copy()
        print('Keeping the extent of the source image..')

        if epsg_target is not None and resolution is not None:
            print('Setting EPSG:%s and resolution %i m' %(epsg_target, resolution))
            
            # reference system
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(epsg_target))
            
            
            xMin, yMin, xMax, yMax = get_closest_extent(info_src, epsg_target, 
                                                        resolution, prec =2)
            
            x_size = int(np.round((xMax - xMin)/resolution))
            y_size = int(np.round((yMax - yMin)/resolution))
        
            info_target = {'geotransform':(xMin, resolution, 0, 
                                            yMax, 0, -resolution),
                            'extent': [xMin, yMin, xMax, yMax],
                            'X_Y_raster_size': [x_size, y_size],
                            'projection': srs.ExportToWkt()}
        
        elif resolution is not None:
            print('Keeping native crs and setting resolution %i m' %resolution)
            
            x_size = int(np.round((info_src['extent'][2] - \
                                   info_src['extent'][0])/resolution))
            y_size = int(np.round((info_src['extent'][3] - \
                                   info_src['extent'][1])/resolution))
        
            info_target['geotransform'] = (info_src['geotransform'][0], resolution,
                                           0, info_src['geotransform'][3], 0, 
                                           -resolution)
            info_target['X_Y_raster_size'] = [x_size, y_size]
            
            
    else:
        
        print('User defined extent, epsg and resolution..')
        # user defined extent and epsg.
            
        # Read the target extent and crs from another image or 
        #from the user specified parameters
        
        if img4ext is not None:
            print('Reading extent and epsg from another image')
            ds, info_target = open_image(img4ext)
            extent_target = info_target['extent']
            
        else:  
            
            assert epsg_target, "Please specify the target EPSG or enter the path to a target image"

            # reference system
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(int(epsg_target))
            
            info_target = {}
            info_target['projection'] = srs.ExportToWkt()
           
          
        assert resolution, "Please specify the target resolution"
        
        x_rest = (extent_target[2] - extent_target[0])%resolution
        y_rest = (extent_target[3] - extent_target[1])%resolution
            
        #assert (x_rest ==0 and y_rest==0),'The extent is not multiple of the assigned resolution'

        
        # get the number of pixels
        x_size = int(np.round((extent_target[2] - extent_target[0])/resolution))
        y_size = int(np.round((extent_target[3] - extent_target[1])/resolution))
        
        
        # store information in a dictionary
        info_target['geotransform'] = (extent_target[0], resolution, 0, 
                                       extent_target[-1], 0, -resolution)
        info_target['extent'] = list(extent_target)
        info_target['X_Y_raster_size'] = [x_size, y_size]

        
    return info_target