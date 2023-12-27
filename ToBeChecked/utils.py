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