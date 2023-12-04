import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import geopandas as gpd
import os
import json
import glob
from tqdm import tqdm
import shapely.geometry as sg
from shapely import affinity
from shapely.geometry import Point, Polygon
import random
from PIL import Image, ImageDraw
from skimage import measure
import rasterio
from rasterio.features import geometry_mask
#from metrics import DiceScore,IoUScore
import pandas as pd
import torch.nn as nn


def binary_mask_to_polygon(binary_mask):
    # Find contours in the binary mask
    contours = measure.find_contours(binary_mask, 0.5)
    # Get the largest contour (in case there are multiple objects)
    max_contour = max(contours, key=len)

    # Convert the contour points to a polygon (list of (x, y) coordinates)
    polygon = Polygon([(int(point[1]), int(point[0])) for point in max_contour])

    return polygon

def convert_polygon_to_mask(geo,shape,transform=None):
    gtmask=np.zeros(shape)
    if transform:
       for orig_row in geo:
          polygon=[]
          if orig_row.geom_type=="Polygon":
              binary_array = geometry_mask([orig_row], out_shape=shape, transform=transform, invert=True)
              ba=binary_array*1
              gtmask=gtmask+ba
          else:
              for x in orig_row.geoms:
                binary_array = geometry_mask([x], out_shape=shape, transform=transform, invert=True)
                ba=binary_array*1
                gtmask=gtmask+ba
    else:
      for orig_row in geo:
            polygon=[]
            if orig_row.geom_type=="Polygon":
                for point in orig_row.exterior.coords:
                    polygon.append(point)
                img = Image.new('L', shape, 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                gt_mask_building = np.array(img)
                gtmask=gtmask+gt_mask_building
            else:
                for x in orig_row.geoms:
                  for point in x.exterior.coords:
                    polygon.append(point)

                img = Image.new('L', shape, 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                gt_mask_building = np.array(img)
                gtmask=gtmask+gt_mask_building
    return gtmask


def convert_polygon_to_mask_batch(geo,shape,transform):
  gtmask=[]
  if transform:
     for row in geo:
      if row.geom_type=="Polygon":
        binary_array = geometry_mask([row], out_shape=shape, transform=transform, invert=True)
        ba=binary_array*1
        gtmask.append(ba)

      else:
        for x in row.geoms:
          binary_array = geometry_mask([x], out_shape=shape, transform=transform, invert=True)
          ba=binary_array*1
          gtmask.append(ba)
  else:   
    for orig_row in geo:
      polygon=[]
      if orig_row.geom_type=="Polygon":
          for point in orig_row.exterior.coords:
            polygon.append(point)
          img = Image.new('L', shape,0)
          ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
          img=np.array(img)
          gtmask.append(img)
      else:
          for x in orig_row.geoms:
            for point in x.exterior.coords:
              polygon.append(point)
          img = Image.new('L', shape,0)
          ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
          img=np.array(img)
          gtmask.append(img)
  
  return gtmask



class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)