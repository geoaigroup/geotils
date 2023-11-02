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
from Data.utils import colorize
from rasterio.features import shapes
import pandas as pd
from shapely.geometry import shape
from shapely.wkt import dumps
from shapely.ops import cascaded_union
import geopandas as gpd
def post_process(raw,thresh = 0.5,mina=40,save=None):
    
    try:
        ch = raw.shape[2]
    except:
        ch=1
    if(ch == 2):
        rraw = ranger(raw)
        
        rbuilds = raw[...,0]
        rborders = raw[...,1]
        
        nuclei = rbuilds * (1 - rborders)
        
        builds = raw[...,0]
        
        basins = label(nuclei>0.1,background = 0, connectivity = 2)
        #Image.fromarray(basins>0).show()
        #basins = noise_filter(basins, mina = 2 )
        basins = label(basins,background = 0, connectivity = 2)
        washed = watershed(image = -builds,
                           markers = basins,
                           mask = builds>thresh,
                           watershed_line=False)
        washed = label(washed,background = 0, connectivity = 2)
        washed = noise_filter(washed, mina=thresh)
        washed = label(washed,background = 0, connectivity = 2)
        #col = colorize(washed)
        #Image.fromarray(col).show()
        
    elif(ch == 1):
        builds = raw[...,0]
        washed  = label(builds > thresh,background = 0, connectivity = 2)
        washed = noise_filter(washed, mina=thresh)
        washed = label(washed,background = 0, connectivity = 2)
        #col = colorize(washed)
        #Image.fromarray(col).show()
        
    else:
        raise NotImplementedError(
            )
        
    return washed

def noise_filter(washed,mina):
    values = np.unique(washed)
    #a =0
    #print(values)
    for val in values[1:]:
        #a+=1
        area = (washed[washed == val]>0).sum()
        if(area<=mina):  
            washed[washed == val] = 0
    #print(a)
    return washed

def ranger(x):
    x1 = x.copy()
    return np.tanh((x1 - 0.5)/0.1) * (0.5)+0.5

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
    

def mask_to_polys(iid,mask,mina = 4):
    vals = sorted(np.unique(mask))
    polys = []
    areas = []
    for i in vals[1:]:
        poly = extract_poly(mask == i)
        
        if(poly is not None):
            if(poly.area > mina):
                polys.append(poly)
                areas.append(poly.area)
    gdf = gpd.GeoDataFrame(
                            {'Id' : list(range(1,len(polys)+1)),
                            'geometry'    : polys,
                             'area'       : areas
                                })
    return gdf
    
        
        
        
    
    