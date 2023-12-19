#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:27:22 2020

@author: jamada
"""


import numpy as np
from skimage.morphology import watershed,dilation,square,erosion
from skimage.measure import label
from PIL import Image,ImageDraw
from Augmentation.coloring import colorize
from rasterio.features import shapes
import pandas as pd
from shapely.geometry import shape
from shapely.wkt import dumps
from shapely.ops import cascaded_union
import geopandas as gpd

def post_process(pred,thresh = 0.5,thresh_b = 0.6,mina=100,mina_b=50):
    if len(pred.shape) < 2:
        return None
    if len(pred.shape) == 2:
        pred = pred[...,np.newaxis]
    
    ch = pred.shape[2]
    buildings = pred[...,0]
    if ch > 1:
        borders = pred[...,1]
        nuclei = buildings * (1.0 - borders)

        if ch == 3:
            spacing = pred[...,2]
            nuclei *= (1.0 - spacing)

        basins = label(nuclei>thresh_b,background = 0, connectivity = 2)
        if mina_b > 0:
            basins = noise_filter(basins, mina = mina_b)
            basins = label(basins,background = 0, connectivity = 2)

        washed = watershed(image = -buildings,
                           markers = basins,
                           mask = buildings>thresh,
                           watershed_line=False)

    elif(ch == 1):
        washed  = buildings > thresh 


    washed = label(washed,background = 0, connectivity = 2)
    washed = noise_filter(washed, mina=mina)
    washed = label(washed,background = 0, connectivity = 2)
        
    return washed

def noise_filter(washed,mina):
    values = np.unique(washed)
    for val in values[1:]:
        area = (washed[washed == val]>0).sum()
        if(area<=mina):  
            washed[washed == val] = 0
    return washed

def extract_poly(mask):
    shps = shapes(mask.astype(np.int16),mask>0)
    polys =[]
    
    for shp,value in shps:
        
        p = shape(shp).buffer(0.0)
        
        typ = p.geom_type    
        if(typ == 'Polygon' or typ == 'MultiPolygon'):  
            polys.append(p.simplify(0.01))
        else:
            continue
    if(len(polys) == 0):
        return None
    else:
        return cascaded_union(polys)
        #break
    

def instance_mask_to_gdf(
        instance_mask,
        transform = None,
        crs=None
        ):
    """
    Input:
        - instance_mask : np.array of shape (H,W), where each instance is labeled by a unique id/number
        - transform : geospatial transform of the raster - default is None
        - crs : crs of the raster - default is None
    Output:
        - GeoDataFrame of the shapes projected to the specified crs using the transform
    """

    #transform should be Identity if None is provided
    transform = rio.transform.IDENTITY if transform is None else transform
    
    all_shapes = shapes(instance_mask,mask=None,transform=transform)
    data = [
            {'properties' : {'id' : v} , 'geometry' : s} for i,(s,v) in enumerate(all_shapes) if v!=0
    ]
    
    if len(data) == 0:
        ##return empty dataframe
        return gpd.GeoDataFrame(columns=['id','geometry'], geometry='geometry',crs=crs)
    
    gdf = gpd.GeoDataFrame.from_features(data,crs=crs)
    gdf = gdf.dissolve(by='id')
    
    return gdf