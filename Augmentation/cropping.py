import numpy as np
import random
from skimage.measure import label as label_fn


def random_crop(image_stack, mask, image_size):
    '''
    THIS FUNCTION DEFINES RANDOM IMAGE CROPPING.
     :param image_stack: input image in size [Time Stamp, Image Dimension (Channel), Height, Width]
    :param mask: input mask of the image, to filter out uninterested areas [Height, Width]
    :param image_size: It determine how the data is partitioned into the NxN windows
    :return: image_stack, mask
    '''

    H, W = image_stack.shape[2:]

    # skip random crop is image smaller than crop size
    if H - image_size // 2 <= image_size:
        return image_stack, mask
    if W - image_size // 2 <= image_size:
        return image_stack, mask
    flag = True
    for i in range(0,100):
        h = np.random.randint(image_size, H - image_size // 2)
        w = np.random.randint(image_size, W - image_size // 2)

        image_stack = image_stack[:, :, h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
                    w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]
        mask = mask[h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
            w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]
        if 1 in mask:
            break
    return image_stack, mask

def random_crop_around_aoi(img,mask,size = 32,min_area=0):
    h,w = img.shape[2:]
    mask_original = mask.copy()
    size_h,size_w = size,size
    
    if h <= size and w <= size:
        return img,mask
    if h < size:
        size_h = h
    if w < size:
        size_w = w
        
    if mask.max() == 0:
        t,b,l,r = 0,h-1,0,w-1
    else:
        mask = label_fn(mask,connectivity=2)
        values = [value for value in np.unique(mask)[1:] if mask[mask==value].sum()/value >= min_area]
        
        if len(values) == 0:
            t,b,l,r = 0,h-1,0,w-1
        else:
            sval = values[random.randint(0,len(values)-1)]
            mask[mask!=sval] = 0
            mask = ((mask / sval) * 255.0).astype(np.uint8)
            pos = np.nonzero(mask)
            t, b, l, r = pos[0].min(),pos[0].max(),pos[1].min(),pos[1].max()
        
    h_aoi,w_aoi = b-t,r-l
    pt = random.randint(t+h_aoi//2, b-h_aoi//2),random.randint(l+w_aoi//2, r-w_aoi//2)
    
    max_up = pt[0]
    max_left = pt[1]
    min_up = max(0,size_h - (h - pt[0]))
    min_left = max(0,size_w - (w - pt[1]))
    
    t_crop = pt[0] - min(max_up, random.randint(min_up, size_h-1))
    l_crop = pt[1] - min(max_left, random.randint(min_left, size_w-1))

    cropped_img = img[:,:,t_crop:t_crop+size_h,l_crop:l_crop+size_w]
    cropped_mask = mask_original[t_crop:t_crop+size_h,l_crop:l_crop+size_w]

    return cropped_img,cropped_mask



