import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import rotate

from torchvision.transforms import InterpolationMode
import cv2
from albumentations import PadIfNeeded,HorizontalFlip,Crop,CenterCrop,Compose,Resize,RandomCrop,VerticalFlip,OneOf
import numpy as np
from skimage.measure import label as label_fn
from Augmentation.transform import TorchRandomRotate




if __name__=="__main__":
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    # img=cv2.imread("test_img/2_1229.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=Image.open("test_img/2_1229.png")
    print(img)
    randrotate=TorchRandomRotate(degrees=30)
    new_img,mask=randrotate(img)
    plt.imshow(new_img)
    
    # procimage,procmask = random_crop_around_aoi(random_image_stack,random_mask,size = 32,min_area=0)
    # procimage,procmask = random_crop(random_image_stack,random_mask,32)