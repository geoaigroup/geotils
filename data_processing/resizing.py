import torch
from torch.nn.functional import interpolate, pad
import numpy as np
from typing import List


def remove_boundary_positives(img: torch.Tensor, pixels: int = 20) -> torch.Tensor:
    r"""Remove boundary pixels from input image tensor.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor with shape (B, C, H, W), where B is the batch size,
        C is the number of channels, H is the height, and W is the width.
    pixels : int, optional
        Number of pixels to remove from each boundary of the image. Default is 20.

    Returns
    -------
    torch.Tensor
        Output tensor with the same shape as the input image tensor,
        but with boundary pixels set to zero.
    """
    H, W = img.shape[-2:]
    bg = torch.zeros_like(img, dtype=img.dtype, device=img.device)

    s1 = min(pixels, H - 1)
    e1 = max(s1 + 1, H - pixels)
    s2 = min(pixels, W - 1)
    e2 = max(s2 + 1, W - pixels)

    bg[..., s1:e1, s2:e2] = img[..., s1:e1, s2:e2]
    return bg


def remove_boundary_positives_np(img: np.ndarray, pixels: int = 20) -> np.ndarray:
    r"""Remove boundary pixels from input image numpy array.

    Parameters
    ----------
    img : np.ndarray
        Input image numpy array with shape (H, W, C), where H is the height,
        W is the width, and C is the number of channels.
    pixels : int, optional
        Number of pixels to remove from each boundary of the image. Default is 20.

    Returns
    -------
    np.ndarray
        Output numpy array with the same shape as the input image array,
        but with boundary pixels set to zero.
    """
    H, W = img.shape[:2]
    bg = np.zeros_like(img, dtype=img.dtype)

    s1 = min(pixels, H - 1)
    e1 = max(s1 + 1, H - pixels)
    s2 = min(pixels, W - 1)
    e2 = max(s2 + 1, W - pixels)

    bg[s1:e1, s2:e2, ...] = img[s1:e1, s2:e2, ...]
    return bg


def resize_pad(x: torch.Tensor, padsize: int = None, resize: int = None, pad_value: int = -1) -> torch.Tensor:
    r"""Resize and/or pad input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor to be resized and/or padded.
    padsize : int, optional
        int specifying the number of pixels to pad on each side of the input tensor. Default is None.
    resize : int , optional
        int specifying the target size of the output tensor. Default is None.
    pad_value : int, optional
        Value to use for padding. Default is -1.

    Returns
    -------
    torch.Tensor
        Output tensor with the specified resizing and/or padding applied.
    """
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


def unpad_resize(x: torch.Tensor, padsize: int = None, resize: int = None) -> torch.Tensor:
    r"""Unpad and/or resize input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (B, C, H, W), where B is the batch size,
        C is the number of channels, H is the height, and W is the width.
    padsize : int
        List specifying the pad size for each dimension [pad_top, pad_bottom, pad_left, pad_right].
    resize : int
        List specifying the target size for resizing [new_height, new_width].

    Returns
    -------
    torch.Tensor
        Output tensor with the specified unpadded and/or resized dimensions.
    """
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

