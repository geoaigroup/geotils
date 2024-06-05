
import cv2
from math import ceil


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
    r"""Rescales the image to a given height/width ratio if specified, and then 
        pads the image so that the height and width are divisible by div if 
        specificed, this function is for numpy arrays and I mainly used it for 
        inference since the model might except a certain input shape (HxWxC)
    """
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
