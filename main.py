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
# from Augmentation.transform import TorchRandomRotate
import os
import tqdm
import geopandas as gpd
# from Metrics.metrics import cal_scores
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utililities.poly_conv import convert_polygon_to_mask_batch
from data_processing.preprocessing import loading_large_tile
width=512
height=512



if __name__=="__main__":
    
    # orig_shp="data/orig_shp"
    # pred = "data/pred_shapefile"
    images="data/images"
    output_im="data/output_image"
    output_mask="data/output_mask"
    
    mask="data/n1"
    # os.makedirs(output,exist_ok=True)
    ids = [f for f in os.listdir(mask)]
    # ff = gpd.read_file(pred)


    ll=loading_large_tile(images,mask,image_suffix='.tif',mask_suffix='.shp')
    loaded_images,loaded_masks=ll.pre_load()

    os.makedirs(output_im,exist_ok=True)
    os.makedirs(output_mask,exist_ok=True)
    for im,m in zip(loaded_images,loaded_masks):
        for index,(image,mask) in enumerate(zip(im,m)):
            cv2.imwrite(f"{output_im}/n1_{index}.png",image)
            os.makedirs(f"{output_mask}/n1_{index}",exist_ok=True)
            mask.to_file(f"{output_mask}/n1_{index}/n1_{index}.shp")
   
   
   
   
    # if name in os.listdir(orig_shp):
    #     gt = gpd.read_file(orig_shp + "/" + name)
        
    # predic = ff.loc[ff["ImageId"] == name]
    # geo = predic["geometry"]
    
    # image = cv2.imread(images + "/" + name +'.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # pred_mask=convert_polygon_to_mask_batch(geo,image.shape[:2])

    # score=cal_scores(output,output)
    # print(score.micro_match_iou(pred_mask,name,gt,image))


    
    
    
    
    
    
    
    # img=cv2.imread("test_img/2_1229.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img=Image.open("test_img/2_1229.png")
    # print(img)
    # randrotate=TorchRandomRotate(degrees=30)
    # new_img,mask=randrotate(img)
    # plt.imshow(new_img)
    
    # procimage,procmask = random_crop_around_aoi(random_image_stack,random_mask,size = 32,min_area=0)
    # procimage,procmask = random_crop(random_image_stack,random_mask,32)