
import torch
from torch.nn.functional import interpolate

import os
import cv2
import random

import numpy as np

from math import ceil
from skimage.segmentation import watershed
from skimage.measure import label


def make_dir(path):
    os.makedirs(path,exist_ok=True)

#load model util
def load_model(model,path):
    checkpoint = torch.load(f'{path}/best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

#simple send to tensor
def totensor(x):
    return torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float()

def normalize(x):
    x = x.astype(np.float64)
    x -= (0.5 * 255.0)
    x /= (0.5 * 255.0)
    return x
    
#padding - unpadding utils
def get_pads(L,div=32):
    if L % div == 0:
        x1,x2 = 0,0
    else:
        L_pad = ceil(L / div) * div
        dL = L_pad - L
        x1 = max(1,dL//2)
        x2 = dL - x1
    return x1,x2

def ratio_resize_pad(img,ratio=None,div=32):

    h_rsz,w_rsz = h_orig,w_orig = img.shape[:2]

    if ratio is not None:
        if ratio != 1.0:
            h_rsz = ceil(h_orig * ratio)
            w_rsz = ceil(w_orig * ratio)
            img = cv2.resize(img,(w_rsz,h_rsz))

    t,b = get_pads(h_rsz,div)
    l,r = get_pads(w_rsz,div)
    img = cv2.copyMakeBorder(img,t,b,l,r,borderType=cv2.BORDER_CONSTANT,value=0.0)

    info = {'orig_size' : (h_orig,w_orig),'pads':(t,b,l,r)}
    return img,info
    
def unpad_resize(img,info):
    h,w = img.shape[2:]

    t,b,l,r = info['pads']
    orig_size = info['orig_size']

    img = img[:,:,t:h-b,l:w-r]
    if h != orig_size[0] or w != orig_size[1]:
        #img = cv2.resize(img,orig_size).astype(np.uint8)
        img = interpolate(img,size=orig_size)
    return img