import torch
from torch.nn.functional import interpolate,pad
import numpy as np
from math import ceil
from cv2 import copyMakeBorder,BORDER_CONSTANT


def remove_boundary_positives(img: torch.Tensor, pixels: int = 20) -> torch.Tensor:
    """Remove boundary pixels from input image tensor."""

    H,W = img.shape[-2:]
    bg = torch.zeros_like(img,dtype=img.dtype,device=img.device)

    s1 = min(pixels,H-1)
    e1 = max(s1+1,H-pixels) 
    s2 = min(pixels,W-1)
    e2 = max(s2+1,W-pixels) 
    
    bg[...,s1:e1,s2:e2] = img[...,s1:e1,s2:e2]
    return bg


def remove_boundary_positives_np(img: np.ndarray, pixels: int = 20) -> np.ndarray:
    """Remove boundary pixels from input image numpy array."""

    H,W = img.shape[:2]
    bg = np.zeros_like(img,dtype=img.dtype)

    s1 = min(pixels,H-1)
    e1 = max(s1+1,H-pixels) 
    s2 = min(pixels,W-1)
    e2 = max(s2+1,W-pixels) 
    bg[s1:e1,s2:e2,...] = img[s1:e1,s2:e2,...]
    return bg


def resize_pad(x: torch.Tensor, padsize: [int] = None, resize: [int] = None, pad_value: int = -1) -> torch.Tensor:
    """Resize and/or pad input tensor."""

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


def unpad_resize(x: torch.Tensor, padsize: [int] = None, resize: [int] = None) -> torch.Tensor:
    """Unpad and/or resize input tensor."""

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
