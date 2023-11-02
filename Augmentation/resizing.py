import torch
from torch.nn.functional import interpolate,pad
import numpy as np
from math import ceil

def remove_boundary_positives(img,pixels=20):
    H,W = img.shape[-2:]
    bg = torch.zeros_like(img,dtype=img.dtype,device=img.device)

    s1 = min(pixels,H-1)
    e1 = max(s1+1,H-pixels) 
    s2 = min(pixels,W-1)
    e2 = max(s2+1,W-pixels) 
    
    bg[...,s1:e1,s2:e2] = img[...,s1:e1,s2:e2]
    return bg

def remove_boundary_positives_np(img,pixels=20):
    H,W = img.shape[:2]
    bg = np.zeros_like(img,dtype=img.dtype)

    s1 = min(pixels,H-1)
    e1 = max(s1+1,H-pixels) 
    s2 = min(pixels,W-1)
    e2 = max(s2+1,W-pixels) 
    bg[s1:e1,s2:e2,...] = img[s1:e1,s2:e2,...]
    return bg

def resize_pad(x,padsize=None,resize=None,pad_value = -1):
  
    if padsize is None and resize is None:
        return x
    
    input_shape = x.shape
    if len(input_shape) == 5:
        B,T,C,H,W = input_shape
        x = x.view(B*T,C,H,W)
        
    if resize is not None:
        x = interpolate(x,size=(resize,resize),mode='bilinear')

    if padsize is not None:
        if resize is not None:
            ppix = padsize - resize 
        else:
            ppix = padsize - 256
        s = ppix // 2
        e = ppix - s
        x = pad(x, (s,e,s,e), mode='constant', value=pad_value)

    
    if len(input_shape) == 5:
        H,W = x.shape[-2:]
        x = x.view(B,T,C,H,W)
    
    return x

def unpad_resize(x,padsize=None,resize=None):
    if padsize is None and resize is None:
        return x
    if padsize is not None:
        if resize is not None:
            ppix = padsize - resize 
        else:
            ppix = padsize - 256
        
        s = ppix // 2
        e = ppix - s
        H,W = x.shape[-2:]
        x = x[...,s:H-e,s:W-e]
        
    if resize is not None:
        x = interpolate(x,size=(256,256),mode='bilinear')

    return x

def tta(data,i):
    if i == 0:
        x = data
    elif i == 1:
        x = torch.flip(data.clone(),dims=(-1,))
    elif i == 2:
        x = torch.flip(data.clone(),dims=(-2,))
    elif i == 3:
        x = torch.flip(data.clone(),dims=(-2,-1))
    
    return x