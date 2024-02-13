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
import argparse
from tqdm import trange,tqdm
import geopandas as gp
from shapely.geometry import Polygon
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio.transform import from_bounds
from PIL import Image,ImageDraw
from skimage.morphology import dilation, square
from skimage.segmentation import watershed
from simplification.cutil import simplify_coords_vwp
from imantics import Mask
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.windows import Window
from typing import List, Tuple


def binary_mask_to_polygon(binary_mask):
    """
    @param binary_mask = binary mask to be converted in to polygon
    @type binary_mask = numpy array

    Returns:
    polygon = converted polygon
    """
  
    contours = measure.find_contours(binary_mask, 0.5)
    max_contour = max(contours, key=len)

    polygon = Polygon([(int(point[1]), int(point[0])) for point in max_contour])

    return polygon

def convert_polygon_to_mask(geo,shape,transform=None):
    """
    @param geo = polygons' geometry to be converted
    @param shape = shape of the mask to be generated
    @param transform = flag param if polygons are georeferenced
    @type geo = geopandas ['geometry']

    Returns:
    gtmask = mask of type array
    """

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
  """
    @param geo = polygons' geometry to be converted
    @param shape = shape of the mask to be generated
    @param transform = transformation information for polygons
    @type geo = geopandas ['geometry']

    Returns:
    gtmask = list of numpy array masks, each contain a polygon
    """ 
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


def generate_polygon(bbox: List[float]) -> List[List[float]]:
    """
    Generates a list of coordinates forming a polygon.

    Parameters
    ----------
    bbox : List[float]
        A list representing the bounding box coordinates [xmin, ymin, xmax, ymax].

    Returns
    -------
    List[List[float]]
        A list of coordinates representing the polygon.
    """

    return [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]]
    ]

def shape_polys(polyg: List[Polygon]) -> List[List[Tuple[float, float]]]:
    """Shapes the building polygons as a list of polygon lists.

    Parameters
    ----------
    polyg : List[Polygon]
        List of building polygons.

    Returns
    -------
    List[List[Tuple[float, float]]]
        List of shaped polygons.
    """

    all_polys = []
    for poly in polyg:
        if len(poly) >= 3:
            f = poly.reshape(-1, 2)
            simplified_vw = simplify_coords_vwp(f, .3)
            if len(simplified_vw) > 2:
                mpoly = []  
                for i in simplified_vw:
                    mpoly.append((i[0], i[1]))  
                mpoly.append((simplified_vw[0][0], simplified_vw[0][1]))
                all_polys.append(mpoly)
    return all_polys

def pol_to_np(pol: List[List[float]]) -> np.ndarray:
    """Converts a list of coordinates to a NumPy array.

    Parameters
    ----------
    pol : List[List[float]]
        List of coordinates: [[x1, y1], [x2, y2], ..., [xN, yN]].

    Returns
    -------
    np.ndarray
        NumPy array of coordinates.
    """
    
    return np.array([list(l) for l in pol])

def pol_to_bounding_box(pol: List[List[float]]) -> BoundingBox:
    """Converts a list of coordinates to a bounding box.

    Parameters
    ----------
    pol : List[List[float]]
        List of coordinates: [[x1, y1], [x2, y2], ..., [xN, yN]].

    Returns
    -------
    BoundingBox
        Bounding box of the coordinates.
    """

    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:, 0]),
                       np.min(arr[:, 1]),
                       np.max(arr[:, 0]),
                       np.max(arr[:, 1]))



def reverse_coordinates(pol: List) -> List:
    """
    Reverse the coordinates in a polygon.

    Parameters
    ----------
    pol : list of list
        List of coordinates: [[x1, y1], [x2, y2], ..., [xN, yN]].

    Returns
    -------
    list of list
        Reversed coordinates: [[y1, x1], [y2, x2], ..., [yN, xN]].
    """

    return [list(f[-1::-1]) for f in pol]

class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)
    
