import numpy as np
from color_map import get_cm_data
import random 

#visualization utils
def mask2rgb(mask,max_value=1.0):
    shape = mask.shape
    if len(shape) == 2:
        mask = mask[:,:,np.newaxis]
    h,w,c = mask.shape
    if c == 3:
        return mask
    if c == 4:
        return mask[:,:,:3]
    
    if c > 4:
        raise ValueError
    
    padded = np.zeros((h,w,3),dtype=mask.dtype)
    padded[:,:,:c] = mask
    padded = (padded * max_value).astype(np.uint8)
    
    return padded


def make_rgb_mask(mask,color=(255,0,0)):
    h,w = mask.shape[:2]
    rgb = np.zeros((h,w,3),dtype=np.uint8)
    rgb[mask == 1.0,:] = color
    return rgb

def overlay_rgb_mask(img,mask,sel,alpha):

    sel = sel == 1.0
    img[sel,:] = img[sel,:] * (1.0 - alpha) + mask[sel,:] * alpha
    return img

def overlay_instances_mask(img,instances,cmap=get_cm_data(),alpha=0.9):
    h,w = img.shape[:2]
    overlay = np.zeros((h,w,3),dtype=np.float32)

    _max = instances.max()
    _cmax = cmap.shape[0]
    

    if _max == 0:
        return img
    elif _max > _cmax:
        indexes = [(i % _cmax) for i in range(_max)]    
    else:
        indexes = random.sample(range(0,_cmax),_max)
    
    for i,idx in enumerate(indexes):
        overlay[instances == i+1,:] = cmap[idx,:]
    
    overlay = (overlay * 255.0).astype(np.uint8)
    viz = overlay_rgb_mask(img,overlay,instances>0,alpha=alpha)
    return viz