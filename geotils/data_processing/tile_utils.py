#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:56:35 2021

@author: hasan
"""
#useful functions :D
import geopandas as gpd
import rasterio as rio
from shapely.geometry import Polygon,box
from mercantile import bounds
from supermercado.burntiles import burn
from tqdm import tqdm,trange
from math import ceil
from pystac import (Catalog)
import geopandas as gp
from shapely.geometry import Polygon
from rasterio import windows
from rasterio.windows import Window
from rasterio.transform import from_bounds
from PIL import Image,ImageDraw
from skimage.morphology import dilation, square
from skimage.segmentation import watershed
# import solaris as sol
from imantics import Mask
import numpy as np
import rasterio
from typing import List, Tuple

global our_crs
our_crs = 'WGS84'

def get_fitting_box_from_box(bbox,width,height):
    minx,miny,maxx,maxy = bbox.bounds
    cx = (minx+maxx) // 2
    cy = (miny+maxy) // 2
    
    box_w = maxx - minx 
    box_h = maxy - miny
    
    fbox_w = ceil(box_w / 512) * 512
    fbox_h = ceil(box_h / 512) * 512

    gap_left = fbox_w // 2 - (cx - max(0,cx - fbox_w // 2))
    gap_right = fbox_w // 2 - (min(width,cx + fbox_w // 2) - cx)
    gap_up = fbox_h // 2 - (cy - max(0,cy - fbox_h // 2))
    gap_down = fbox_h // 2 - (min(height,cy + fbox_h // 2) - cy)

    fb_minx = cx - (fbox_w // 2 + gap_right - gap_left) 
    fb_maxx = cx + (fbox_w // 2 + gap_left - gap_right)
    fb_miny = cy - (fbox_h // 2 + gap_down - gap_up)
    fb_maxy = cy + (fbox_h // 2 + gap_up - gap_down)

    fbox = box(fb_minx,fb_miny,fb_maxx,fb_maxy)
    return fbox

def poly_lonlat2pix(poly,bbox_bounds,img,width = None,height = None):
  if(img is not None):
    h,w = img.shape[:2]
  elif(None not in set([width,height])):
    h,w = height,width
  else:
    raise ValueError('Either Image or height and width should not be None')
  transform = rio.transform.from_bounds(*bbox_bounds,w,h)
  xs,ys = poly.exterior.xy
  rows,cols = rio.transform.rowcol(transform,xs,ys)
  coords = list(zip(cols,rows))
  return Polygon(coords)

def get_tiles_xyz(gdf,zoom_level):
  gdf_geo = gdf.__geo_interface__
  features  = gdf_geo['features']
  tiles_xyz = burn(features,zoom_level)
  return tiles_xyz

def get_tiles_xyz_fast(gdf,zoom_level):
    part =100000
    l = len(gdf)
    all_tiles = set()
    if(l<part):
        part = l 
    for i in trange(0,l,part):
        c = min(part,l-i)
        tiles_xyz = get_tiles_xyz(gdf[i:i+c],zoom_level)
        tiles_xyz = list(map(tuple,tiles_xyz))
        all_tiles.update(tiles_xyz)
    return list(all_tiles)

def get_covering_tiles(gdf,zoom_level):
  tiles_xyz = get_tiles_xyz_fast(gdf,zoom_level)
  tiles_bboxs = []
  for xyz in tiles_xyz:
    _b = bounds(xyz)
    tiles_bboxs.append(box(*_b))
  return tiles_xyz,tiles_bboxs

def poly2gdf(poly,crs):
  return gpd.GeoDataFrame({'geometry':[poly]},crs=our_crs)

def encode_tile_id(tile_xyz):
  return 't_{}_{}_{}'.format(*tile_xyz)

def encode_tiles_ids(tiles_xyz):
  ids = [encode_tile_id(tile_xyz) for tile_xyz in tiles_xyz]
  return ids

def encode_multi_id(ids):
  return '&'.join(ids)

def decode_id(tile_id):
  return [*map(int,tile_id.lstrip('t_').split('_'))]

def decode_multi_id(multi_tile_id):
  return [decode_id(tid) for tid in multi_tile_id.split('&')]

#very specific functions
def _compress(gdf,_keys,id_key = 'FM_RE_ID'):
  keys = gdf.keys()
  _keys.extend(['tile_id'])
  assert(False not in [(k in keys ) for k in _keys]),'Missing keys from {}'.format(_keys)
  _keys[-1] = 'covering_tiles_ids'
  #print(_keys)
  dic = {k : [] for k in _keys}
  uniq_fm_ids = gdf[id_key].unique()
  for i,fm_id in enumerate(uniq_fm_ids):
    query = gdf.loc[gdf[id_key] == fm_id] 
    values = [query[_k][query.index[0]] if(_k != id_key) else fm_id for _k in _keys[:-1] ]
    values.append(encode_multi_id(list(query['tile_id'])))
    for k,v in [*zip(_keys,values)]:dic[k].append(v)
  return gpd.GeoDataFrame(dic,crs=gdf.crs)

def get_covering_tiles_perloc(gdf,zoom_level,id_key = 'FM_RE_ID'):
  crs = gdf.crs
  tiles_xyz,tiles_bboxs = get_covering_tiles(gdf,zoom_level)
  tiles_ids = encode_tiles_ids(tiles_xyz)
  tiles_gdf = gpd.GeoDataFrame({'tile_id' : tiles_ids,
                                'geometry': tiles_bboxs,
                                },
                               crs = crs)
  joined = gpd.sjoin(gdf,tiles_gdf, how='left', op='intersects')
  del joined['index_right']
  joined.reset_index(inplace = True,drop = True) 
  return _compress(joined,[*gdf.keys()],id_key),tiles_gdf

def get_sample(fm_id,gdf,col = 'FMID_RE_ID'):
  query = gdf.loc[gdf[col]==fm_id]
  return query.loc[query.index[0]]

def get_tiles_bboxs(tiles_ids,gdf,col = 'tile_id'):
  tiles_bboxs=[]
  for tile_id in tiles_ids:
    query = gdf.loc[gdf[col] == tile_id]
    tiles_bboxs.append(query['geometry'][query.index[0]])
  return tiles_bboxs



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


def tile_area(area: str, iid: str, lid: str, tile_size: int, ipth: str, mpth: str, cols: Catalog) -> None:
    """Tiles a certain area (tif image) and generates the corresponding tile mask.
    Parameters
    ----------
    area : str
        Area ID (example: acc-665946).
    iid : str
        Image ID in the pystac.
    lid : str
        Label ID in the pystac.
    tile_size : int
        Size of width and height of the tile (use only square).
    ipth : str
        Directory to save image tiles.
    mpth : str
        Directory to save the masks.
    cols: Catalog
        Get all the columns in the catalog(main_children), it is how the pystac is organized
    """
    item=cols[area].get_item(id=iid)
    label=cols[area].get_item(id=lid)
    lbl_gfd=gp.read_file(label.make_asset_hrefs_absolute().assets['labels'].href)
    polys=lbl_gfd.geometry
    tif_url = item.assets['image'].href
    if tif_url.startswith("./"):
        tif_url = '/'.join(item.to_dict()['links'][1]['href'].split("/")[:-1])+tif_url[1:]
    rst = rasterio.open(tif_url)
    wid,hei=rst.width,rst.height
    for i in trange(0,tile_size*int(hei/tile_size),tile_size):
        for j in range(0,tile_size*int(wid/tile_size),tile_size):
                aoo,mask=get_winNmask(rst,j,i,tile_size,tile_size,polys)
                uniq=np.unique(aoo)
                if(np.all(uniq==0)):continue
                uniq=np.unique(mask[:,:,0])
                if(np.all(uniq==0)):
                    spacing=np.zeros((tile_size,tile_size)).astype(np.uint8)
                else:
                    getit=mask[:,:,0]^mask[:,:,1]
                    polyg = Mask(getit).polygons()
                    extracted=shape_polys(polyg)
                    labels=np.array(make_instance_mask(extracted,tile_size))
                    spacing=np.array(create_separation(labels)).astype(np.uint8)*255
                aoo=np.moveaxis(aoo,0,2)
                final_mask=spacing
                final_mask[(np.where(mask[:,:,0]>0))]=np.uint8(64)
                final_mask[(np.where(mask[:,:,1]>0))]=np.uint8(128)
                img=Image.fromarray(aoo).convert('RGB')
                fmsk=Image.fromarray(final_mask)
                img.save('/'.join([ipth,f'{area}_{iid}_{i}_{j}.png']))
                fmsk.save('/'.join([mpth,f'{area}_{iid}_{i}_{j}_mask.png']))