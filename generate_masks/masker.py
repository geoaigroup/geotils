#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:15:11 2020

@author: jamada
"""


import numpy as np
import cv2
from PIL import Image
import os
import json
from shapely.geometry import Polygon, MultiPolygon
from skimage.draw import polygon
from skimage.morphology import erosion, square, binary_erosion
from skimage.io import imread,imsave
from utililities import colorize,create_path
from tqdm import tqdm
class masker():
    def __init__(self,data,
                 out_size = (1024,1024),
                 erosion_kernel ='cross',
                 verbose = True):
        self.data = data
        self.sz = out_size
        self.ids = sorted(os.listdir(self.data))
        self.ldir = 'labels_match_pix'
        assert erosion_kernel in ['cross','square']
        self.ek_type = erosion_kernel
        self.verbose = verbose
        
    def load_labels(self,iid,extension):
        json_path = f'{self.data}/{iid}/{self.ldir}/{extension}_Buildings.geojson'
        jfile=open(json_path,'r')
        f=json.load(jfile)
        return f
    
    def _collect(self,labels):
        _meta = {}
        for label in labels['features']:
            pid = str(np.int32(label['properties']['Id']))
            _meta[pid] = {}
            _meta[pid]['area']= label['properties']['area']
            _meta[pid]['geometry'] = label['geometry']['coordinates']
            
        return _meta
    def int_coords(self,x):
        return np.array(x).round().astype(np.int32)
    def poly_size(self,w,h):
        return Polygon([[0,0],[w-1,0],[w-1,h-1],[0,h-1],[0,0]])
    def get_strc(self):
        if(self.ek_type == 'square'):
            return square(3)
        else:
            return np.array([[0,1,0],
                             [1,1,1],
                             [0,1,0]],dtype=np.uint8)
        
    def instances(self,size,labels):
        ins_mask = np.zeros(size,dtype = np.int32)
        
        for pid,d in labels.items():
            int_id = np.int32(pid) 
            polys = d['geometry']
            for i,poly in enumerate(polys):
                poly.append(poly[0])
                S = Polygon(poly)
                PS = self.poly_size(size[1],size[0])
                S = S.intersection(PS)
                Stype = S.geom_type
                
                if(Stype == 'Polygon'):    
                    arr_pol = self.int_coords(S.exterior.coords)
                    if(len(arr_pol.shape) != 2): continue
                    ws,hs = polygon(arr_pol[:,0],arr_pol[:,1],size)
                    ins_mask[hs,ws,...] = int_id if(i==0) else 0
                    
                elif(Stype == 'MultiPolygon'):
                    for s in S:
                        arr_pol = self.int_coords(s.exterior.coords)
                        if(len(arr_pol.shape) != 2): continue
                        ws,hs = polygon(arr_pol[:,0],arr_pol[:,1],size)
                        ins_mask[hs,ws,...] = int_id if(i==0) else 0
                        
                else:
                    for point in list(S.coords):
                        x,y = list(map(np.int32,point))
                        ins_mask[y,x,...] = int_id if(i==0) else 0
                        
        return ins_mask
    
    def borders(self,ins_mask):
        ins_borders = np.zeros_like(ins_mask,dtype = np.int32)
        ids = sorted(np.unique(ins_mask))[1:]
        strc = self.get_strc()
        for iid in ids:
            instance = ins_mask == iid
            try:
                k=np.where(instance>0)
                _t = k[0].min() - 3
                _l = k[1].min() - 3
                _b = k[0].max() + 3
                _r = k[1].max() + 3
                
                crop_instance = instance[_t:_b,_l:_r]
                bld = binary_erosion(crop_instance,selem = strc)
                brdr = bld ^ crop_instance
                brdr1 = np.zeros_like(instance,dtype=brdr.dtype)
                brdr1[_t:_b,_l:_r] =brdr
                ins_borders[brdr1 == True] = iid
                
            except:
                bld = binary_erosion(instance,selem = strc)
                brdr = bld ^ instance
                ins_borders[brdr == True] = iid
        return ins_borders
    
    def to_rgb(self,img):
        rgb = np.ascontiguousarray(img[...,:3],dtype = np.uint8)
        return rgb
    def to_gray(self,mask):
        return (mask>0).astype(np.uint8) * 255
    def generate_dataset(self,save_path):
        create_path(save_path)
        
        for iid in self.ids[42:]:
            imgs_save = f'{save_path}/{iid}'
            imgs_path = f'{self.data}/{iid}/images_masked'
            labels_path = f'{self.data}/{iid}/{self.ldir}'
            lod = os.listdir(imgs_path)
            loader = tqdm(lod) if(self.verbose) else lod
            loader.set_description(f'{iid}')
            create_path(imgs_save)
            for exten in loader:
                e = exten.split('.tif')[0]
                img_save = f'{imgs_save}/{e}'
                create_path(img_save)
                
                img_rgba = imread(f'{imgs_path}/{e}.tif')
                img_rgb = self.to_rgb(img_rgba)
                
                lf = self.load_labels(iid, e)
                labels = self._collect(lf)
                
                size = img_rgb.shape[:2]
                ins_mask = self.instances(size, labels)
                ins_borders = self.borders(ins_mask)
                
                imsave(f'{img_save}/image.png',img_rgb)
                imsave(f'{img_save}/buildings.png',self.to_gray(ins_mask))
                imsave(f'{img_save}/borders.png',self.to_gray(ins_borders))
                np.save(f'{img_save}/buildings.npy',ins_mask)
                np.save(f'{img_save}/borders.npy',ins_borders)
                