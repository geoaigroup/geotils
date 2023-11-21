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

width=512
height=512


def binary_mask_to_polygon(binary_mask):
    # Find contours in the binary mask
    contours = measure.find_contours(binary_mask, 0.5)
    # Get the largest contour (in case there are multiple objects)
    max_contour = max(contours, key=len)

    # Convert the contour points to a polygon (list of (x, y) coordinates)
    polygon = Polygon([(int(point[1]), int(point[0])) for point in max_contour])

    return polygon

def convert_polygon_to_mask_batch(geo,width,height):
  gtmask=[]
  for orig_row in geo:
    polygon=[]
    if orig_row.geom_type=="Polygon":
        for point in orig_row.exterior.coords:
          polygon.append(point)
        img = Image.new('L', (width, height),0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        img=np.array(img)
        gtmask.append(img)
    else:
        for x in orig_row.geoms:
          for point in x.exterior.coords:
            polygon.append(point)
        img = Image.new('L', (width, height),0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        img=np.array(img)
        gtmask.append(img)
  return gtmask