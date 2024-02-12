#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:23:54 2020

@author: hasan
"""
# from pystac import (Catalog)
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


def shape_polys(polyg):
    """Shapes the building polygons as a list of polygon lists.

    Parameters
    ----------
    polyg : List[Polygon]
        List of building polygons in numpy arrays.

    Returns
    -------
    List[List[Tuple(float, float)]]
        List of shaped polygons.
    """

    all_polys = []
    for poly in polyg:
        if len(poly) >= 3:
            f = poly.reshape(-1, 2)
            simplified_vw = simplify_coords_vwp(f, .3)
            if len(simplified_vw) > 2:
                mpoly = []
                # Rebuilding the polygon in the way that PIL expects the values [(x1,y1),(x2,y2)]
                for i in simplified_vw:
                    mpoly.append((i[0], i[1]))
                    # Adding the first point to the last to close the polygon
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


def get_areas(cols: dict) -> List:
    """
    Get areas from a collection of columns.

    Parameters
    ----------
    cols : dict
        A dictionary representing columns.

    Returns
    -------
    list
        A list of tuples containing area information. Each tuple has the format (iid, id1, id2).
    """

    areas = []
    for iid in cols:
        items = [x for x in cols[iid].get_all_items()]
        for i, id in enumerate(items):
            if i % 2 == 0 and i + 1 < len(items):
                areas.append((iid, items[i].id, items[i + 1].id))
    return areas


def to_index(wind_: Window) -> List[List[int]]:
    """
    Generates a list of index coordinates (row, col) for a given Window.

    Parameters
    ----------
    wind_ : Window
        The rasterio Window object specifying the region of interest.

    Returns
    -------
    List[List[int]]
        A list of index coordinates representing the corners of the Window.
    """

    return [
        [wind_.row_off, wind_.col_off],
        [wind_.row_off, wind_.col_off + wind_.width],
        [wind_.row_off + wind_.height, wind_.col_off + wind_.width],
        [wind_.row_off + wind_.height, wind_.col_off],
        [wind_.row_off, wind_.col_off]
    ]


def create_separation(labels: np.ndarray) -> np.ndarray:
    """Creates a mask for building spacing of close buildings.

    Parameters
    ----------
    labels : np.ndarray
        Numpy array where each building's pixels are encoded with a certain value.

    Returns
    -------
    np.ndarray
        Mask for building spacing.
    """
    
    tmp = dilation(labels > 0, square(20))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(5))
    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 15
            else:
                sz = 20
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1


def make_instance_mask(all_polys: List[List[Tuple[float, float]]], size: int) -> Image:
    """Encodes each building polygon with a value in the mask.

    Parameters
    ----------
    all_polys : List[List[Tuple[float, float]]]
        List of building polygons.
    size : int
        Width and height of the square mask.

    Returns
    -------
    Image
        Instance mask.
    """

    bg=np.zeros((size,size)).astype(np.uint8)
    bg=Image.fromarray(bg).convert('L')
    shift=255-len(all_polys)
    for i,poly in enumerate(all_polys):
        ImageDraw.Draw(bg).polygon(poly,outline=shift+i,fill=shift+i)
    return bg
