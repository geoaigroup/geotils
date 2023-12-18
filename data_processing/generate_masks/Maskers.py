#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:28:43 2020

@author: jamada
"""


 
import rasterio as rs
import numpy as np
import json
from skimage.morphology import erosion, square, binary_erosion
from skimage.draw import polygon
from tqdm import tqdm
from cv2 import fillPoly,copyMakeBorder
import cv2
from math import ceil
from shapely.geometry import Polygon
from utililities import colorize
from PIL import Image
class masker():
    def __init__(self,
                 out_size = (1024,1024),
                 erosion_kernel ='cross',
                 iterator_verbose = True):
        self.sz = out_size
        self.pd_sz = (1044,1044)
        self.x_off = ceil((self.pd_sz[1] - self.sz[1])/2)
        self.y_off = ceil((self.pd_sz[0] - self.sz[0])/2)
        assert self.x_off >= 0 and self.y_off >= 0, f'out size {self.sz} should be less than padded size {self.pd_sz}'
        assert erosion_kernel.lower() in ['square','cross'], f'erosion kernel type : [ {erosion_kernel} ] is not valid'
        self.ek_type = erosion_kernel.lower()
        self.itr_vrbs = iterator_verbose

    def load_labels(self,json_path):
        jfile=open(json_path,'r')
        f=json.load(jfile)
        return f
    
    def load_raster_file(self,raster_path):
        return rs.open(raster_path)
    
    def get_img(self,raster_file):
        return raster_file.read().transpose(1,2,0).astype(np.uint8)
    
    def poly_size(self,h,w):
        return Polygon([[0,0],[0,w-1],[h-1,w-1],[h-1,0],[0,0]])
    
    def get_strc(self):
        if(self.ek_type == 'square'):
            return square(3)
        else:
            return np.array([[0,1,0],
                             [1,1,1],
                             [0,1,0]],dtype=np.uint8)
    
    def project_poly(self,poly,frs,size,x_off= 0,y_off=0): 
        k = []
        for tup in poly:
            _x,_y = frs.index(*tup)
            _x+=x_off
            _y+=y_off
            k.append([_x,_y])
        
        polk= Polygon(k)
        if(not polk.is_valid):
          polk = polk.buffer(0.01)
        poll = self.poly_size(*size)
        intr = poll.intersection(polk)
        verbs = intr.geom_type == 'Polygon' 
        return list(intr.exterior.coords) if(verbs) else []
    
    def crop(self,img,y_off,x_off,h,w):
        return img[y_off:y_off + h,x_off:x_off + w]
    
    def make_mask(self,polys, size = (1024,1024)):
        builds,border = [np.zeros(size,dtype=np.uint8),np.zeros(size,dtype=np.uint8)] 
        instances = np.zeros(size,dtype=np.int32)
        strc = self.get_strc()
        itr = enumerate(polys)
        if(self.itr_vrbs):
          itr = tqdm(itr)
          itr.set_description('generating mask')
        for i,mulpol in itr:
            for j,pol in enumerate(mulpol):
                arr_pol= np.array(pol,dtype=np.int32)
                hs,ws = polygon(arr_pol[:,0],arr_pol[:,1],size)
                instances[hs,ws,...] = np.int32(i+1) if(j==0) else 0
                #fillPoly(instances, [arr_pol[:,::-1]], i+1 if(j == 0) else 0)
            instance = instances == np.int32(i+1)
            try:
                k=np.where(instance>0)
                _t = k[0].min() - 2
                _l = k[1].min() - 2
                _b = k[0].max() + 2
                _r = k[1].max() + 2
                
                crop_instance = instance[_t:_b,_l:_r]
                bld = binary_erosion(crop_instance,selem = strc)
                brdr = bld ^ crop_instance
                brdr1 = np.zeros_like(instance,dtype=brdr.dtype)
                brdr1[_t:_b,_l:_r] =brdr
                border[brdr1 == True] = np.uint8(255)
                
            except:
                bld = binary_erosion(instance,selem = strc)
                brdr = bld ^ instance
                border[brdr == True] = np.uint8(255)
                
        builds[instances > 0] = np.uint8(255)
        return instances,builds,border
        
    def mask(self,raster_path,json_path):
            raster = self.load_raster_file(raster_path)
            img = self.get_img(raster)
            js = self.load_labels(json_path)
            labels = js['features']
            polys =[]
            for label in  labels:
        
                multipoly = label['geometry']['coordinates']
                proj_multipoly = []
                for poly in multipoly:
                    mm = self.project_poly(poly, raster,self.pd_sz,self.x_off,self.y_off) 
                    if(len(mm)>0):
                        proj_multipoly.append(mm)
                polys.append(proj_multipoly)
                
            ins,b,br = self.make_mask(polys,size = self.pd_sz)
            kwargs = {'y_off' : self.y_off,
                      'x_off' : self.x_off,
                      'h'     : self.sz[0],
                      'w'     : self.sz[1]}
            ins = self.crop(ins,**kwargs)
            b = self.crop(b,**kwargs)       
            br = self.crop(br,**kwargs)         
            img = cv2.copyMakeBorder(img,
                                     top=0, 
                                     bottom = self.sz[0] - img.shape[0], 
                                     left=0,
                                     right = self.sz[1] - img.shape[1],
                                     borderType =cv2.BORDER_CONSTANT,
                                     value = 0)
            return img,ins,b,br
        

##############################################################################################################################################
###example##
#define main directory of dataset
main_dir = '/home/jamada/Desktop/Spacenet7/Dataset/train'
#define region/location id
region_id = 'L15-1690E-1211N_6763_3346_13'
#define month and year between 2018_01 and 2020_01
year_month = '2019_05'
tif_id = f'global_monthly_{year_month}_mosaic_{region_id}'
#define image path and labels_match geojson path [not labels geojson path!]
img_path = f'{main_dir}/{region_id}/images/{tif_id}.tif'
json_path =f'{main_dir}/{region_id}/labels_match/{tif_id}_Buildings.geojson' 
#create a masker object
mskr = masker()
#get the image , instances mask, buildings mask, and borders mask using .mask() function of the masker object
#you should provide the image path , followed by the geojson path
img,ins,b,br = mskr.mask(img_path,json_path)
#color the intances mask to distinguish buildings

colored = colorize(ins)
size =512,512
c = Image.fromarray(colored)
a = Image.open(f'{main_dir}/{region_id}/images/{tif_id}.tif')
#b = Image.open(f'{main_dir}/{region_id}/UDM_masks/{tif_id}_UDM.tif')
ims = [a,c]
for i in ims:
    i.thumbnail(size, Image.ANTIALIAS)
    i.show()