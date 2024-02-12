#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:30:12 2020

@author: jamada
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:17:07 2020

@author: hasan
"""
""" at some point i tested a 2 classes segmentation with classes=['buildings','spacing']
    so i made this script to modify the spacing....i tried at first the spacing of really close buildings only labelled, 
    but then we decided to test a bigger spacing width...which means to increase the range to label the spacing of buildings,
    u can see the changes in the make_separation function between here and make_dataset.py script

most of these functions were commented before in make_dataset.py
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from skimage.morphology import dilation, square
from skimage.segmentation import watershed
import cv2
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image,ImageDraw
from imantics import Mask
from simplification.cutil import simplify_coords_vwp
def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
def create_separation(labels):
    tmp = dilation(labels > 0, square(23))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(11))
    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 17
            else:
                sz = 23
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1

def shape_polys(polyg):
    all_polys = []
    for poly in polyg:
        if len(poly) >= 3:
            f = poly.reshape(-1, 2)
            simplified_vw = simplify_coords_vwp(f, .3)
            if len(simplified_vw) > 2:
                mpoly = []
                # Rebuilding the polygon in the way that PIL expects the values [(x1,y1),(x2,y2)]
                for i in simplified_vw:
                    mpoly.append((i[0], i[1]))
                    # Adding the first point to the last to close the polygon
                mpoly.append((simplified_vw[0][0], simplified_vw[0][1]))
                all_polys.append(mpoly)
    return all_polys

def make_instance_mask(all_polys,size):
    bg=np.zeros((size,size)).astype(np.uint8)
    bg=Image.fromarray(bg).convert('L')
    shift=255-len(all_polys)
    for i,poly in enumerate(all_polys):
        ImageDraw.Draw(bg).polygon(poly,outline=shift+i,fill=shift+i)
    return bg

def new_spacing(mask):
    #get the old mask, where buildings are 64 encoded,borders are 128, and spacing is 255
    w,h=mask.shape
    #remove the old spacing, set to zero
    mask[(np.where(mask==np.uint8(255)))]=np.uint8(0)
    #get the polygons of the buildings (64)
    polyg = Mask(mask==np.uint8(64)).polygons()
    #rest is explained in make_dataset.py
    extracted=shape_polys(polyg)
    labels=np.array(make_instance_mask(extracted,w))
    spacing=np.array(create_separation(labels)).astype(np.uint8)*255
    spacing[(np.where(mask==np.uint8(128)))] = np.uint8(128)
    spacing[(np.where(mask==np.uint8(64)))] = np.uint8(64)
    return spacing
def main(old,new):
    paths=[f for f in os.listdir(old)]
    #iterate over the old dataset masks and make new masks with modified spacing
    for pth in tqdm(paths):
        mask=cv2.imread(f'{old}/{pth}',0)
        new_mask=new_spacing(mask=mask)
        Image.fromarray(new_mask).save(f'{new}/{pth}')
    print('DOne!')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--old',type=str)
    parser.add_argument('--new',type=str)
    args=parser.parse_args()
    create_dir(args.new)
    main(old=args.old,new=args.new)
    

      










