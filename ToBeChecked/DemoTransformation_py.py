#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:31:21 2021

@author: hasan
"""
import geopandas as gpd
import os
from shapely.geometry import Polygon,MultiPolygon
from utils import pix_to_utm, utm_to_pix, _tf, parse_tfw,tf_upper,tf_utm
import matplotlib.pyplot as plt
from skimage.io import imread

tfw_map = {
    'LEBANON_2013_50CM_RGB2_7' : 'LEBANON_2013_50CM_NRGB2_7.tfw',
    'LEBANON_2013_50CM_RGB3_5' : 'LEBANON_2013_50CM_NRGB3_5.tfw'
    }

tfws_path = '../tfw_files'
shape_path = '../pred_shapefile/pred_csv.shp'
imgs_path = '../visualized_results_inference'

Image_ID = 'LEBANON_2013_50CM_RGB2_7_31744_27648'
img_path = os.path.join(imgs_path,Image_ID,f'{Image_ID}_img.png')
img = imread(img_path)

print('----------------------------------------------------------------------')
#load the ShapeFile as a geo-DataFrame
gdf = gpd.read_file(shape_path)

#get rows corresponding to building polygon of the selected Image_ID
gdf_sample = gdf.loc[gdf['ImageId'] == Image_ID]
gdf_sample.reset_index(inplace = True,drop = True)

pixel_polys = gdf_sample['geometry']
n_polys = len(pixel_polys)
print(gdf_sample)
print(f'There are {n_polys} polygons(building footprints) in Image {Image_ID}')
print('----------------------------------------------------------------------')

# Now Get the x_offset and y_offset from the Image_ID
split_iid = Image_ID.split('_')
x_offset , y_offset = int(split_iid[-1]), int(split_iid[-2])

print(f'For Image {Image_ID} : \nX_offset : {x_offset}\nY_offset : {y_offset}')
print('----------------------------------------------------------------------')

#Now get the corresponf .TFW file path 
Image_Tiff_ID = '_'.join(split_iid[:-2])
TFW_ID = tfw_map[Image_Tiff_ID]
tfw_path = os.path.join(tfws_path,TFW_ID)

print(f'The Corresponding .tfw File for {Image_ID} is : {TFW_ID}')
# parse the Georeferencing Parameters
A,B,C,D,E,F = parse_tfw(tfw_path)

#Add Offset to pixel_polys
pixel_poly_coordinates = []
for p in pixel_polys:
    if(p.geom_type == 'Polygon'):
        coords = list(p.exterior.coords)
        pixel_poly_coordinates.append(coords)
    else:
        #ignore multipolygons for now
        pass
n_polys = len(pixel_poly_coordinates)
xs = [x_offset] * n_polys
ys = [y_offset] * n_polys

offset_polys = tf_upper(pixel_poly_coordinates,xs,ys)

#Now Transform To GeoReferenced Coordinates
georeferenced_polys = tf_utm(offset_polys,pix_to_utm,A,B,C,D,E,F)
#And Put them In a DataFrame
gdf_referenced = gpd.GeoDataFrame({'geometry' : georeferenced_polys})


fig,axs = plt.subplots(1,2,figsize = (10,10))
axs[0].imshow(img)
gdf_referenced.plot(ax = axs[1])


