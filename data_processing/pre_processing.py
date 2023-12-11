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
import rasterio as rio
import glob
import tqdm
import geopandas as gpd
import torch
import gc
import cv2
from shapely.geometry import shape
from utils import poly_conv,utils
# from old_utils import * 
import sys
# Define some constants
CORN = 0
SOYBEAN = 1
OTHER = 2
N_CLASSES = 3
N_BANDS = 6
N_TIMESTEPS = 3
BATCH_SIZE = 4096
N_EPOCHS = 25
HEIGHT = 3660
WIDTH = 3660

# Fix random seed for reproducibility
np.random.seed(42)


class LargeTiffLoader:
   
    def __init__(self,input_image_directory,input_mask_directory,image_suffix='.tif',mask_suffix='.tif'):
        self.image_directory=input_image_directory
        self.mask_directory=input_mask_directory
        self.image_suffix=image_suffix
        self.mask_suffix=mask_suffix

    def pre_load(self, fragment_size=1024, PATCH_SIZE=1024, STRIDE_SIZE=512, CROP_SIZE =768, DOWN_SAMPLING=1,transform=None):
        loaded_images=[]
        loaded_masks=[]
        for file in os.listdir(self.image_directory):
            filename = os.fsdecode(file)
            if filename.endswith(self.image_suffix):
                name=filename.split('.')[0]
                raster_file = rio.open(f'{self.image_directory}/{filename}')
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
                
                mask = rio.open(glob.glob(f'{self.mask_directory}/{name}{self.mask_suffix}'))
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
                      
                        shapes = rio.features.shapes(patch_mask)
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