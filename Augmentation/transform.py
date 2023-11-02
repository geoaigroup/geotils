import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import rotate

from torchvision.transforms import InterpolationMode
import cv2
from albumentations import PadIfNeeded,HorizontalFlip,Crop,CenterCrop,Compose,Resize,RandomCrop,VerticalFlip,OneOf
import numpy as np



class TorchRandomRotate(nn.Module):
    def __init__(self, degrees, probability=1.0,interpolation=InterpolationMode.BILINEAR, center=None, fill=0,mask_fill=0):
        super().__init__()
        if not isinstance(degrees,(list,tuple)):
            degrees = (-abs(degrees),abs(degrees))

        self.degrees = degrees
        self.interpolation = interpolation
        self.center = center
        self.fill_value = fill
        self.mask_fill_value = mask_fill
        self.proba = probability

    @staticmethod
    def get_params(degrees) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle
    
    @torch.no_grad()
    def __call__(self,img,mask=None):

        batch_size = img.shape[0]

        for i in range(batch_size):
            
            if random.random() > self.proba:
                continue

            angle = self.get_params(self.degrees)
            img[i,...] = rotate(img[i,...], angle, self.interpolation, False, self.center, self.fill_value)
            #mask = mask.long()
            if mask is not None:
                mask[i,...] =  rotate(mask[i,...], angle, self.interpolation, False, self.center, self.mask_fill_value)
            mask = mask.float()
        if mask is not None:
            mask[mask<0] = self.mask_fill_value
            return img,mask
        return 


class RandomMaskIgnore(nn.Module):

    def __init__(self,min_length=50,max_length=10,proba=0.5,ignore_index=-10):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.proba = proba
        self.ignore_index = ignore_index
    

    def generate_random_bbox(self,shape):
        H,W = shape
        L = random.randint(self.min_length,self.max_length)

        t = random.randint(0,H-L)
        b = t + L

        l = random.randint(0,W-L)
        r = l + L

        return (t,l,b,r)
    
    def mask_channel(self,bbox,channel):
        (t,l,b,r) = bbox
        channel[:,t:b,l:r] = self.ignore_index
        return channel
    
    @torch.no_grad()
    def __call__(self,mask):
        
        B,C,H,W = mask.shape
        for i in range(B):
            if random.random() > self.proba:
                continue
            bbox = self.generate_random_bbox((H,W))
            mask[i,...] = self.mask_channel(bbox,mask[i,...])
        
        return mask

class MaskPixelDrop(nn.Module):

    def __init__(self,neg_drop=50,pos_drop=50,ignore_index=-10):
        super().__init__()

        if not isinstance(neg_drop,tuple):
            neg_drop = (0,neg_drop)
        if not isinstance(pos_drop,tuple):
            pos_drop = (0,pos_drop)
        
        self.neg_drop = neg_drop
        self.pos_drop = pos_drop

        self.ignore_index = ignore_index
    
    @staticmethod
    def get_drop_proba(_range):
        return random.randint(_range[0],_range[1]) / 100
    
    def random_pixel_drop(self,gt,mask,_range):
        Cs,Hs,Ws = mask.nonzero(as_tuple=True)
        proba = self.get_drop_proba(_range)
        max_num = Cs.shape[0]
        drop_count = min(max_num,int(proba * max_num))
        
        if drop_count == 0 or max_num == 0:
            return gt

        indexes = random.sample(range(0, max_num), drop_count)
        Cs,Hs,Ws = Cs[indexes].tolist(),Hs[indexes].tolist(),Ws[indexes].tolist()
        gt[Cs,Hs,Ws] = self.ignore_index
        return gt

    @torch.no_grad()
    def __call__(self,mask):
        B,C,H,W = mask.shape
        pos_mask = mask.gt(0)
        neg_mask = mask.eq(0)
        for i in range(B):
            mask[i] = self.random_pixel_drop(mask[i],pos_mask[i],self.pos_drop)
            mask[i] = self.random_pixel_drop(mask[i],neg_mask[i],self.neg_drop)
        return mask
    




def cnd(a,b):
    return a==b

def get_crop(a):
    a=a-1
    s,k = int(a%2),int(a/2)
    return s,k

def Augs512(img,mask,fold):
    
    data = {'image': img,
            'mask': mask}
    
    st = 512
    s,k = get_crop(fold)
    s*=st
    k*=st
    c = cnd(fold,4)
    Transform = Compose(
        [ PadIfNeeded(min_height=1024,
                      min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=0, 
                      mask_value=0, always_apply=True, 
                      p=1.0),
         Crop(x_min=s,
              y_min=k,
              x_max=s+st,
              y_max=k+st, 
               p= not c),
         
         CenterCrop(height=st, 
                    width=st,
                    p = c),
         HorizontalFlip(p=0.5)
         
            ])
    out = Transform(**data)
    return out['image'],out['mask']

def Augs512Resize(img,mask):
    data = {'image': img,
            'mask': mask}
    Transform = Compose(
        [PadIfNeeded(min_height=1024,
                      min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=0, 
                      mask_value=0, always_apply=True, 
                      p=1.0),
         Resize(height = 2048,
                width = 2048,
                interpolation =1,
                p = 1.0),
         RandomCrop(height = 512,
                    width = 512,
                    p = 1.0),
         OneOf(
             [ 
                 HorizontalFlip(p=1),
                 VerticalFlip(p=1),
                 Compose(
                     [
                         HorizontalFlip(p=1.0),
                         VerticalFlip(p=1.0)],
                     p=1.0)
             ],p=0.6)
         
            ],p=1.0)
    out = Transform(**data)
    return out['image'],out['mask']
def Augs1024test(img,mask,nomask = False):
    
    r1,c1,ch1 = img.shape
    
    #print(mask.shape)
    img = cv2.copyMakeBorder(img,
                             top = 0,
                             bottom = 1024 - r1,
                             left = 0,
                             right = 1024 - c1,
                             borderType = cv2.BORDER_CONSTANT,
                             value = 0)
    if(not nomask):
        r2,c2,ch2 = mask.shape
        mask= cv2.copyMakeBorder(mask,
                                 top = 0,
                                 bottom = 1024 - r2,
                                 left = 0,
                                 right = 1024 - c2,
                                 borderType = cv2.BORDER_CONSTANT,
                                 value = 0)
        if(ch2 == 1):
            mask = np.stack([mask],axis = -1).astype(np.uint8)
    else:
        mask = None
    #print(mask.shape)
    return img,mask

def Augs1024train(img,mask):
    
    data = {'image': img,
            'mask': mask}
    Transform = Compose(
                    [ PadIfNeeded(min_height=1024,
                      min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=0, 
                      mask_value=0, always_apply=True, 
                      p=1.0)])
    out = Transform(**data)
    return out['image'],out['mask']

def Augs2048train(img,mask):
    
    data = {'image': img,
            'mask': mask}
    Transform = Compose(
                    [ PadIfNeeded(min_height=1024,
                      min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, 
                      value=0, 
                      mask_value=0, always_apply=True, 
                      p=1.0),
                      Resize(height = 2048,
                             width = 2048,
                             interpolation =1,
                             p = 1.0),])
    out = Transform(**data)
    return out['image'],out['mask']
    
    
    
    """
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines a sample Data Transformer for augmentation
"""

import numpy as np
import torch
import random
from skimage.measure import label as label_fn


class EOTransformer():
    """
    THIS CLASS DEFINE A SAMPLE TRANSFORMER FOR DATA AUGMENTATION IN THE TRAINING, VALIDATION, AND TEST DATA LOADING
    """
    def __init__(self,spatial_encoder=True, normalize=True, image_size=32, use_random_crop=True):
        '''
        THIS FUNCTION INITIALIZES THE DATA TRANSFORMER.
        :param spatial_encoder: It determine if spatial information will be exploited or not. It should be determined in line with the training model.
        :param normalize: It determine if the data to be normalized or not. Default is TRUE
        :param image_size: It determine how the data is partitioned into the NxN windows. Default is 32x32
        :return: None
        '''
        self.spatial_encoder = spatial_encoder
        self.image_size=image_size
        self.normalize=normalize
        self.use_random_crop=use_random_crop

    def transform(self,image_stack, mask=None):
        '''
        THIS FUNCTION INITIALIZES THE DATA TRANSFORMER.
        :param image_stack: If it is spatial data, it is in size [Time Stamp, Image Dimension (Channel), Height, Width],
                            If it is not spatial data, it is in size [Time Stamp, Image Dimension (Channel)]
        :param mask: It is spatial mask of the image, to filter out uninterested areas. It is not required in case of having non-spatial data
        :return: image_stack, mask
                '''
        # mean_clp = np.mean(image_stack[:,12,:,:], axis=(1,2))<51
        # image_stack = image_stack[mean_clp,:,:,:] #removed cloudy timesteps
        # SELECTED_INDICES=[ 2,  3,  4,  5,  7, 10, 12, 16, 18, 19, 22, 23, 25, 28, 29, 32, 33,
        # 35, 40, 42, 43, 44, 45, 47, 51, 52, 53, 55, 56, 58, 59, 60, 62, 63,
        # 65, 68, 70, 71, 74, 75]
        # SELECTED_INDICES=[7, 33, 47]
        # image_stack = image_stack[SELECTED_INDICES,:,:,:]
        if self.spatial_encoder == False:  # average over field mask: T, D = image_stack.shape
            image_stack = image_stack[:, :, mask > 0].mean(2)
            mask = -1  # mask is meaningless now but needs to be constant size for batching
        else:  # crop/pad image to fixed size + augmentations: T, D, H, W = image_stack.shape
            if image_stack.shape[2] >= self.image_size and image_stack.shape[3] >= self.image_size and self.use_random_crop:
                image_stack, mask = random_crop(image_stack, mask, self.image_size)


            image_stack, mask = crop_or_pad_to_size(image_stack, mask, self.image_size)

            # rotations
            rot = np.random.choice([0, 1, 2, 3])
            image_stack = np.rot90(image_stack, rot, [2, 3])
            mask = np.rot90(mask, rot)

            # flip up down
            if np.random.rand() < 0.5:
                image_stack = np.flipud(image_stack)
                mask = np.flipud(mask)

            # flip left right
            if np.random.rand() < 0.5:
                image_stack = np.fliplr(image_stack)
                mask = np.fliplr(mask)

        image_stack = image_stack * 1e-4


        # z-normalize
        if self.normalize:
            image_stack -= 0.1014 + np.random.normal(scale=0.01)
            image_stack /= 0.1171 + np.random.normal(scale=0.01)

        #the following's added by Ali
        new_mask = np.broadcast_to(mask,(3,1,mask.shape[0],mask.shape[1]))
        green = image_stack[:,2,:,:]
        swir1 = image_stack[:,11,:,:]
        ndwi_map = (green - swir1)/(green + swir1)
        ndwi_map = np.nan_to_num(ndwi_map, nan=0.0, posinf=1, neginf=-1)
        ndwi_map = ndwi_map[:, np.newaxis, :, :]

        red = image_stack[:,3,:,:]
        nir = image_stack[:,7,:,:]
        ndvi_map = (nir - red)/(nir + red)
        ndvi_map = np.nan_to_num(ndvi_map, nan=0.0, posinf=1, neginf=-1)
        ndvi_map = ndvi_map[:,np.newaxis,:,:]

        image_stack = np.delete(image_stack, [0,12], axis=1) #drop band 1 and clp
        image_stack = np.concatenate((image_stack,ndvi_map), axis = 1) #added NDVI band
        #image_stack = np.delete(image_stack, [12], axis=1) #drop clp
        image_stack = np.concatenate((image_stack,ndwi_map), axis = 1)
        image_stack = np.concatenate((image_stack,new_mask), axis = 1)
        return torch.from_numpy(np.ascontiguousarray(image_stack)).float(), torch.from_numpy(np.ascontiguousarray(mask)) #Edited By Ali image*mask

class PlanetTransform(EOTransformer):
    """
    THIS CLASS INHERITS EOTRANSFORMER FOR DATA AUGMENTATION IN THE PLANET DATA
    """
    pass #TODO: some advanced approach special to Planet Data might be implemented

class Sentinel1Transform(EOTransformer):
    """
    THIS CLASS INHERITS EOTRANSFORMER FOR DATA AUGMENTATION IN THE SENTINEL-1 DATA
    """
    pass #TODO: some advanced approach special to Planet Data might be implemented

class Sentinel2Transform(EOTransformer):
    """
    THIS CLASS INHERITS EOTRANSFORMER FOR DATA AUGMENTATION IN THE SENTINEL-2 DATA
    """
    pass #TODO: some advanced approach special to Planet Data might be implemented

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
            break;
    return image_stack, mask

# def random_crop(img,mask,size = 32,min_area=0):
#     h,w = img.shape[2:]
#     mask_original = mask.copy()
#     size_h,size_w = size,size
    
#     if h <= size and w <= size:
#         return img,mask
#     if h < size:
#         size_h = h
#     if w < size:
#         size_w = w
        
#     if mask.max() == 0:
#         t,b,l,r = 0,h-1,0,w-1
#     else:
#         mask = label_fn(mask,connectivity=2)
#         values = [value for value in np.unique(mask)[1:] if mask[mask==value].sum()/value >= min_area]
        
#         if len(values) == 0:
#             t,b,l,r = 0,h-1,0,w-1
#         else:
#             sval = values[random.randint(0,len(values)-1)]
#             mask[mask!=sval] = 0
#             mask = ((mask / sval) * 255.0).astype(np.uint8)
#             pos = np.nonzero(mask)
#             t, b, l, r = pos[0].min(),pos[0].max(),pos[1].min(),pos[1].max()
        
#     h_aoi,w_aoi = b-t,r-l
#     pt = random.randint(t+h_aoi//2, b-h_aoi//2),random.randint(l+w_aoi//2, r-w_aoi//2)
    
#     max_up = pt[0]
#     max_left = pt[1]
#     min_up = max(0,size_h - (h - pt[0]))
#     min_left = max(0,size_w - (w - pt[1]))
    
#     t_crop = pt[0] - min(max_up, random.randint(min_up, size_h-1))
#     l_crop = pt[1] - min(max_left, random.randint(min_left, size_w-1))

#     cropped_img = img[:,:,t_crop:t_crop+size_h,l_crop:l_crop+size_w]
#     cropped_mask = mask_original[t_crop:t_crop+size_h,l_crop:l_crop+size_w]

#     return cropped_img,cropped_mask

def crop_or_pad_to_size(image_stack,  mask, image_size):
    '''
    THIS FUNCTION DETERMINES IF IMAGE TO BE CROPPED OR PADDED TO THE GIVEN SIZE.
     :param image_stack: input image in size [Time Stamp, Image Dimension (Channel), Height, Width]
    :param mask: input mask of the image, to filter out uninterested areas [Height, Width]
    :param image_size: It determine how the data is cropped or padded into the NxN windows.
                       If the size of input image is larger than the given image size, it will be cropped, otherwise padded.
    :return: image_stack, mask
    '''
    T, D, H, W = image_stack.shape
    hpad = image_size - H
    wpad = image_size - W

    # local flooring and ceiling helper functions to save some space
    def f(x):
        return int(np.floor(x))
    def c(x):
        return int(np.ceil(x))

    # crop image if image_size < H,W
    if hpad < 0:
        image_stack = image_stack[:, :, -c(hpad) // 2:f(hpad) // 2, :]
        mask = mask[-c(hpad) // 2:f(hpad) // 2, :]
    if wpad < 0:
        image_stack = image_stack[:, :, :, -c(wpad) // 2:f(wpad) // 2]
        mask = mask[:, -c(wpad) // 2:f(wpad) // 2]
    # pad image if image_size > H, W
    if hpad > 0:
        padding = (f(hpad / 2), c(hpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), padding, (0, 0)))
        mask = np.pad(mask, (padding, (0, 0)))
    if wpad > 0:
        padding = (f(wpad / 2), c(wpad / 2))
        image_stack = np.pad(image_stack, ((0, 0), (0, 0), (0, 0), padding))
        mask = np.pad(mask, ((0, 0), padding))
    return image_stack, mask