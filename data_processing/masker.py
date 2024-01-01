import numpy as np
import cv2
from PIL import Image
import os
import json
from shapely.geometry import Polygon, MultiPolygon
from skimage.draw import polygon
from skimage.morphology import erosion, square, binary_erosion
from skimage.io import imread,imsave
from utility import colorize,create_path
from tqdm import tqdm

import rasterio as rs
from cv2 import fillPoly,copyMakeBorder
from math import ceil


class Masker():
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
        self.ldir = 'labels_match_pix'

   
    def load_labels(self,json_path):
        jfile=open(json_path,'r')
        f=json.load(jfile)
        return f
    
    def poly_size(self,w,h):
        return Polygon([[0,0],[w-1,0],[w-1,h-1],[0,h-1],[0,0]])
    
    def get_strc(self):
        if(self.ek_type == 'square'):
            return square(3)
        else:
            return np.array([[0,1,0],
                             [1,1,1],
                             [0,1,0]],dtype=np.uint8)
    



    def load_raster_file(self,raster_path):
        return rs.open(raster_path)
    
    def get_img(self,raster_file):
        return raster_file.read().transpose(1,2,0).astype(np.uint8)
    
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




    def _collect(self,labels):
        _meta = {}
        for label in labels['features']:
            pid = str(np.int32(label['properties']['Id']))
            _meta[pid] = {}
            _meta[pid]['area']= label['properties']['area']
            _meta[pid]['geometry'] = label['geometry']['coordinates']

    def int_coords(self,x):
        return np.array(x).round().astype(np.int32)

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
    
    def generate_dataset(self,save_path, data):
        create_path(save_path)
        ids = sorted(os.listdir(data))
        for iid in ids[42:]:
            imgs_save = f'{save_path}/{iid}'
            imgs_path = f'{self.data}/{iid}/images_masked'
            labels_path = f'{self.data}/{iid}/{self.ldir}'
            lod = os.listdir(imgs_path)
            loader = tqdm(lod) if(self.itr_vrbs) else lod
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
                
    
