# Standard libraries
import matplotlib.pyplot as plt

import numpy as np
import os
from time import time

# ML libraries
# import tensorflow as tf
# import keras
# from keras.layers.core import *
# from keras.models import Sequential, Model, load_model
# from keras.layers import Dense
# from keras.layers import concatenate
# from keras.layers import Input
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import Conv1D, Conv2D
# from keras.layers import MaxPool2D
# from keras.layers import Dropout
# from keras.utils import to_categorical
# from keras.optimizers import Adam
from sklearn import metrics

import glob
import tqdm
import geopandas as gpd
import torch
import gc
import cv2
from shapely.geometry import shape
# from ..ToBeChecked.utils import utils
import rasterio
from rasterio.enums import ColorInterp
from rasterio.plot import show
from rasterio.windows import Window
import sys
# Define some constants


# Fix random seed for reproducibility
np.random.seed(42)


class LargeTiffLoader:
   
    def __init__(self,image_directory,image_suffix='.tif'):
        self.image_directory=image_directory
        self.image_suffix=image_suffix


        
    def load_index(self,save_path,col_off, row_off, width, height):
        for file in os.listdir(self.image_directory):
             if file.endswith(self.image_suffix):
                name = file.split('.')[0]
                with rasterio.open(f"{self.image_directory}/{file}") as src:
                    original_profile = src.profile
                    window = Window(col_off, row_off, width, height)
                    # fragment = src.read(1, window=window)
                    
                    # Lookup table for the color space in the source file
                    source_colorinterp = dict(zip(src.colorinterp, src.indexes))

                    # Read the image in the proper order so the numpy array will have the colors in the
                    # order expected by matplotlib (RGB)
                    rgb_indexes = [
                        source_colorinterp[ci]
                        for ci in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                    ]
                    data = src.read(rgb_indexes, window=window)

                    # Also pass in the affine transform corresponding to the window in order to
                    # display the correct coordinates and possibly orientation
                    show(data, transform=src.window_transform(window))
                    #Save the fragment as a new georeferenced image
                   
                    fragment_profile = original_profile.copy()
                    fragment_profile['width'] = data.shape[1]
                    fragment_profile['height'] = data.shape[2]
             
                    # Update the transformation to match the fragment's position
                    fragment_profile['transform'] = src.window_transform(window)
                 
                
                    fragment_output_path = os.path.join(save_path, f'{name}.tif')
                    with rasterio.open(fragment_output_path, 'w',**fragment_profile) as dst:
                        dst.write(data)



    def pre_load(self,mask_directory,mask_suffix='.tif', fragment_size=1024, PATCH_SIZE=1024, STRIDE_SIZE=512, CROP_SIZE =768, DOWN_SAMPLING=1,transform=None):
        loaded_images=[]
        loaded_masks=[]
        for file in os.listdir(self.image_directory):
            filename = os.fsdecode(file)
            if filename.endswith(self.image_suffix):
                name=filename.split('.')[0]
                raster_file = rasterio.open(f'{self.image_directory}/{filename}')
                full_img = raster_file.read([1,2,3]).transpose(1,2,0)
                
                HEIGHT_orig, WIDTH_orig = full_img.shape[:2]
                # if self.mask_suffix==".shp":    
                #     with rio.open(f"{self.image_directory}/{filename}") as src:
                #         transform=src.transform

                #     mask_shp = gpd.read_file(f'{self.mask_directory}/{name}{self.mask_suffix}')
                    
                #     mask=poly_conv.convert_polygon_to_mask(mask_shp['geometry'],(HEIGHT_orig,WIDTH_orig),transform=transform)
                   
                # elif self.mask_suffix==".tiff":
                #     mask = rio.open(glob.glob(f'{self.mask_directory}/{name}{self.mask_suffix}'))
                #     mask = mask.read()[0]#.transpose(1,2,0)
                # else:
                #     print("provide .tiff or .shp mask file")  
                #     return
                
                mask = rasterio.open(glob.glob(f'{mask_directory}/{name}{mask_suffix}'))
                mask = mask.read()[0]#.transpose(1,2,0)
                

                full_img = cv2.resize(full_img, (WIDTH_orig//DOWN_SAMPLING, HEIGHT_orig//DOWN_SAMPLING))

                #Use below for gray images only
                #full_img = raster_file.read().transpose(1,2,0)[:,:,0]
                #full_img = cv2.cvtColor(full_img,cv2.COLOR_GRAY2RGB)

                full_img, rrp_info = utils.ratio_resize_pad(full_img, ratio = None, div=fragment_size)
                full_mask, mask_rrp_info = utils.ratio_resize_pad(mask, ratio = None, div=fragment_size)


                HEIGHT, WIDTH = full_img.shape[:2]


                full_mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                full_mask[...] = np.nan

                a = 0
                M = 0
                patched_image=[]
                patched_mask=[]
                for hs in range(a,HEIGHT,STRIDE_SIZE):

                    for ws in range(a,WIDTH,STRIDE_SIZE):

                        he = hs+PATCH_SIZE
                        we = ws+PATCH_SIZE
                        patch = full_img[hs:he,ws:we,:]
                        patch_mask = full_mask[hs:he,ws:we]
                      
                        shapes = rasterio.features.shapes(patch_mask)
                        # read the shapes as separate lists
                        geometry = []
                        for shapedict, value in shapes:
                            if value == 0:
                                continue
                            geometry.append(shape(shapedict))

                        # build the gdf object over the two lists
                        patch_gdf = gpd.GeoDataFrame({'geometry': geometry})
                       
                        if len(patch_gdf) == 0:
                            full_mask[hs:he,ws:we] = 0
                        else:
                            patched_image.append(patch)
                            patched_mask.append(patch_gdf)


                            # y_pred = y_pred.detach().cpu().long().numpy()[:,0,:,:].astype(np.int16)

                            # n_patch,_,_ = y_pred.shape
                            # b_ids = np.arange(n_patch) + 1
                            # b_ids = b_ids[:,np.newaxis,np.newaxis]

                            # y_pred_mask = (y_pred.copy().sum(axis=0) > 0).astype(np.int16)
                            # y_pred *= b_ids
                            # y_pred = y_pred.max(axis=0) + M*y_pred_mask
                            # M = y_pred.max()
                            # full_mask[hs:he,ws:we] = y_pred

              
                loaded_images.append(patched_image)
                loaded_masks.append(patched_mask)
                
                return loaded_images,loaded_masks