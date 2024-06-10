# Standard libraries
import matplotlib.pyplot as plt

import numpy as np
import os
from time import time

from sklearn import metrics
from pathlib import Path
import glob
import tqdm
import geopandas as gpd
import torch
import gc
import cv2
import math
from shapely.geometry import MultiPolygon , Polygon, Point, shape
from fiona.crs import CRS 
from pyproj import Transformer

from data_processing.data_processing_utils import ratio_resize_pad
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.enums import ColorInterp
from rasterio.plot import show
from rasterio.windows import Window

# Define some constants


# Fix random seed for reproducibility
np.random.seed(42)


class LargeTiffLoader:
    """
    The primary purpose of this class is to load and present a sizable Tiff image as a 
    collection of smaller, consistently sized images. The `pre_load` method is specifically 
    designed to carry out this operation. Additionally, the `load_index` function facilitates
    the creation and batching of images from a larger image by specifying the pixel positions to be cropped.
    """
   
    def __init__(self,image_directory,image_suffix='.tif'):
        """
        @param image_directory = path of directory containing images to be loaded
        @param image_suffix = type of images entered, default='.tif'
        """
        
        self.image_directory=image_directory
        self.image_suffix=image_suffix


        
    def load_index(self,save_path,col_off, row_off, width, height):

        """
        @param save_path = path of the directory to save cropped image
        @param col_off = indicates the starting column position from which the batch image should begin.
        @param row_off = specifies the starting row position from which the batch image should start.
        @param width = width of the batch image
        @param height = height of the batch image
        """
        for file in os.listdir(self.image_directory):
             if file.endswith(self.image_suffix):
                name = file.split('.')[0]
                with rasterio.open(f"{self.image_directory}/{file}") as src:
                    original_profile = src.profile
                    window = Window(col_off, row_off, width, height)
                    # fragment = src.read(1, window=window)
                    
                    source_colorinterp = dict(zip(src.colorinterp, src.indexes))

                    rgb_indexes = [
                        source_colorinterp[ci]
                        for ci in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                    ]
                    data = src.read(rgb_indexes, window=window)

                    show(data, transform=src.window_transform(window))
                   
                    fragment_profile = original_profile.copy()
                    fragment_profile['width'] = data.shape[1]
                    fragment_profile['height'] = data.shape[2]
             
                    fragment_profile['transform'] = src.window_transform(window)
                 
                
                    fragment_output_path = os.path.join(save_path, f'{name}.tif')
                    with rasterio.open(fragment_output_path, 'w',**fragment_profile) as dst:
                        dst.write(data)



    def pre_load(self,mask_directory,mask_suffix='.tif', fragment_size=1024, PATCH_SIZE=1024, STRIDE_SIZE=512, DOWN_SAMPLING=1,transform=None):
        """
        @param mask_directory = path of directory contains masks
        @param mask_suffix = type of mask file, default='.tif'
        @param fragment_size = size of the copped batch image
        @param PATCH_SIZE = patch size 
        @param STRIDE_SIZE = stride size 
        @param DOWN_SAMPLING = downsampling factor
        @param transform = tansformation information of the images or the masks

        Return:
        loaded_images = list of loaded images as numpy array each
        loaded_masks = list of loaded masks as numpy array each
        """
        
        loaded_images=[]
        loaded_masks=[]
        for file in os.listdir(self.image_directory):
            filename = os.fsdecode(file)
            if filename.endswith(self.image_suffix):
                name=filename.split('.')[0]
                
                raster_file = rasterio.open(f'{self.image_directory}/{filename}')
               
                full_img = raster_file.read([1,2,3]).transpose(1,2,0)
                
                HEIGHT_orig, WIDTH_orig = full_img.shape[:2]
                # if self.mask_suffix==".shp":    
                #     with rio.open(f"{self.image_directory}/{filename}") as src:
                #         transform=src.transform

                #     mask_shp = gpd.read_file(f'{self.mask_directory}/{name}{self.mask_suffix}')
                    
                #     mask=poly_conv.convert_polygon_to_mask(mask_shp['geometry'],(HEIGHT_orig,WIDTH_orig),transform=transform)
                   
                # elif self.mask_suffix==".tiff":
                #     mask = rio.open(glob.glob(f'{self.mask_directory}/{name}{self.mask_suffix}'))
                #     mask = mask.read()[0]#.transpose(1,2,0)
                # else:
                #     print("provide .tiff or .shp mask file")  
                #     return
              
                mask = rasterio.open(glob.glob(f'{mask_directory}/{name}{mask_suffix}')[0])
                mask = mask.read()[0]#.transpose(1,2,0)
                

                full_img = cv2.resize(full_img, (WIDTH_orig//DOWN_SAMPLING, HEIGHT_orig//DOWN_SAMPLING))

                #Use below for gray images only
                #full_img = raster_file.read().transpose(1,2,0)[:,:,0]
                #full_img = cv2.cvtColor(full_img,cv2.COLOR_GRAY2RGB)

                full_img, rrp_info = ratio_resize_pad(full_img, ratio = None, div=fragment_size)
                full_mask, mask_rrp_info = ratio_resize_pad(mask, ratio = None, div=fragment_size)


                HEIGHT, WIDTH = full_img.shape[:2]


                full_mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                full_mask[...] = np.nan

                a = 0
                M = 0
                patched_image=[]
                patched_mask=[]
                for hs in range(a,HEIGHT,STRIDE_SIZE):

                    for ws in range(a,WIDTH,STRIDE_SIZE):

                        he = hs+PATCH_SIZE
                        we = ws+PATCH_SIZE
                        patch = full_img[hs:he,ws:we,:]
                        patch_mask = full_mask[hs:he,ws:we]
                      
                        shapes = rio_shapes(patch_mask)
                        geometry = []
                        for shapedict, value in shapes:
                            if value == 0:
                                continue
                            geometry.append(shape(shapedict))

                        patch_gdf = gpd.GeoDataFrame({'geometry': geometry})
                       
                        if len(patch_gdf) == 0:
                            full_mask[hs:he,ws:we] = 0
                        else:
                            patched_image.append(patch)
                            patched_mask.append(patch_gdf)


                            # y_pred = y_pred.detach().cpu().long().numpy()[:,0,:,:].astype(np.int16)

                            # n_patch,_,_ = y_pred.shape
                            # b_ids = np.arange(n_patch) + 1
                            # b_ids = b_ids[:,np.newaxis,np.newaxis]

                            # y_pred_mask = (y_pred.copy().sum(axis=0) > 0).astype(np.int16)
                            # y_pred *= b_ids
                            # y_pred = y_pred.max(axis=0) + M*y_pred_mask
                            # M = y_pred.max()
                            # full_mask[hs:he,ws:we] = y_pred

              
                loaded_images.append(patched_image)
                loaded_masks.append(patched_mask)
                
                return loaded_images,loaded_masks
            

def GetFeatureName(s):
    """
    Extracts the feature name from a path string.

    Parameters
    ----------
    s : str
        The path string from which to extract the feature name.

    Returns
    -------
    str
        The last segment of the path, typically representing the feature name.
    """
    l = s.split('\\')
    n = len(l)
    return l[n-1]


def GetFeaturePath(s):
    """
    Extracts the path without the feature name from a full path string.

    Parameters
    ----------
    s : str
        The full path string from which to extract the path without the feature name.

    Returns
    -------
    str
        The path string excluding the feature name.
    """
    l = s.split('\\')
    n = len(l)
    path = ''
    for i in range(0, n-1):
        path += l[i] + "\\"
    return path

def eccentricity(flattening: float):
    """
    Calculate the eccentricity of an ellipse or ellipsoid from its flattening.

    Parameters
    ----------
    flattening : float
        The flattening factor of the ellipse or ellipsoid 
    Returns
    -------
    float
        The eccentricity, calculated using the formula sqrt(2f - f^2) where f is the flattening.
    """

    inv_flatten = 1 / flattening
    eccentricity = math.sqrt(inv_flatten*(2 - inv_flatten))
    return eccentricity


def SterioToGeo(point, p0, a, f, m, x0, y0):
    """
    Converts a point from stereographic projection coordinates to geographic coordinates.

    Parameters
    ----------
    point : Point
        The point in stereographic projection coordinates.
    p0 : Point
        The reference point for the projection in geographic coordinates.
    a : float
        Semi-major axis of the ellipsoid.
    f : float
        Flattening of the ellipsoid.
    m : float
        Scale factor.
    x0 : float
        False Easting.
    y0 : float
        False Northing.

    Returns
    -------
    Point
        The geographic coordinates of the point.
    """
    x=point.x - x0
    y=point.y - y0
    long0=p0.x*math.pi/180 # num5
    lat0=p0.y*math.pi/180 # num6
    e = eccentricity(f)
    e2=math.pow(e,2) # num7 
    e1=e2/(1-e2) #num9
    q = math.sqrt(1 + e1 * math.pow(math.cos(lat0), 4)) #num10
    s=a * m * math.sqrt(1 - e2) / (1 - e2 * math.pow(math.sin(lat0), 2)) #num11
    w=math.asin(math.sin(lat0)/q) #num13
    w2=math.log(math.tan((math.pi/4)+w/2)) #num14
    t=math.log(math.tan((math.pi/4)+lat0/2) * math.pow((1 - e * math.sin(lat0)) / (1 + e * math.sin(lat0)), e / 2)) #num15
    z=w2-q*t #num16
    r=math.sqrt(math.pow(x,2)+math.pow(y,2)) #num17
    if r >= math.pow(10,-13):
        l=2 * math.atan(x / (r - y)) #num1
    else:
        l=0 #num1
    i=(math.pi/2)-2*math.atan(r/(2*s)) #num18
    w3=w-(math.pi/2) #num19
    u1=math.cos(i) * math.cos(l) * math.cos(w3) - math.sin(i) * math.sin(w3) #num20
    u2=math.cos(i) * math.sin(l) #num21
    u3=math.sin(i) * math.cos(w3) + math.cos(i) * math.cos(l) * math.sin(w3) #num22
    if math.sqrt(math.pow(u1,2)+math.pow(u3,2)) >= math.pow(10,-13) :
        h1=math.atan(u3 / math.sqrt(math.pow(u1, 2) + math.pow(u2, 2))) #num3
        h2=2*math.atan(u2 / (math.sqrt(math.pow(u1, 2) + math.pow(u2, 2))+u1)) #num
    else:
        h1=u3/abs(u3) * math.pi /2 #num3
        h2=0 #num
    w2=math.log(math.tan((math.pi/4)+h1/2)) #num14
    lat0=-(math.pi/2 ) + 2 * math.atan(math.exp(w2 - z) / q) #num6
    while True:
        g=-(math.pi/2) + 2 * math.atan(math.pow((1+e*math.sin(lat0))/(1-e*math.sin(lat0)),e/2)*math.exp((w2 - z) / q))  #num2
        g2=lat0 #num4
        lat0=g #num6
        if abs(g2-g) <= math.pow(10,-12):
            break
    lat=g * 180 / math.pi
    lon=(long0+h2/q) * 180 / math.pi
    point2=Point(lon,lat)
    return point2


def GeographicToGeocentric(point, a, f):
    """
    Converts geographic coordinates to geocentric coordinates.

    Parameters
    ----------
    point : Point
        The point in geographic coordinates (longitude, latitude).
    a : float
        Semi-major axis of the ellipsoid.
    f : float
        Flattening of the ellipsoid.

    Returns
    -------
    Point
        The point in geocentric coordinates.
    """
    e = eccentricity(f)
    lon=point.x
    lat=point.y
    alt=0.0
    n=a/math.sqrt(1-(e*e*math.pow(math.sin(lat*math.pi/180),2)))
    x=(n+alt)*math.cos(lat*math.pi/180)*math.cos(lon*math.pi/180)
    y=(n+alt)*math.cos(lat*math.pi/180)*math.sin(lon*math.pi/180)
    z=(n*(1-math.pow(e,2))+alt)*math.sin(lat*math.pi/180)
    p=Point(x,y,z)
    return p


def GeocentricToGeographic(point, a, f):
    """
    Converts geocentric coordinates to geographic coordinates.

    Parameters
    ----------
    point : Point
        The point in geocentric coordinates.
    a : float
        Semi-major axis of the ellipsoid.
    f : float
        Flattening of the ellipsoid.

    Returns
    -------
    Point
        The geographic coordinates of the point.
    """
    e = eccentricity(f)
    e2=math.pow(e,2) # num1
    r=math.sqrt(math.pow(point.x,2)+math.pow(point.y,2)) # num2
    phi0=math.atan(point.z/r) # num3
    phi=phi0 # num4
    while True:
        phi0=phi
        c=math.sqrt(1-e2*math.pow(math.sin(phi0),2)) # num5
        o=a/c # num6
        k=r*math.cos(phi0)+point.z*math.sin(phi0)-a*c # num
        phi=math.atan(point.z/r*math.pow(1-o*e2/(o+k),-1))
        if abs(phi-phi0) <= math.pow(10,-13):
            break
    phi=phi*180/math.pi
    lamda=math.atan(point.y/point.x)*180/math.pi
    p=Point(lamda,phi,0)
    return p


def SterioToWgs84(point):
    """
    Converts a point from stereographic projection to WGS 84 geographic coordinates through
    a series of transformations.

    Parameters
    ----------
    point : Point
        The point in stereographic projection coordinates.

    Returns
    -------
    Point
        The geographic coordinates of the point in the WGS 84 system.
    """
    p0=Point(39.15,34.2)
    print("Original Point:", point)
    p1 = SterioToGeo(point , p0,6378249.2,293.46602,0.999534104,0,0) 
    print("After SterioToGeo:", p1)
    p2 = GeographicToGeocentric(p1 , 6378249.2,293.46602)
    print("After GeographicToGeocentric:", p2)
    p3 = DatumToGRS80(p2, 175.993534,125.164623,-244.865805,17.315446,12.135795,10.542653,-6.123214) 
    print("After DatumToGRS80:", p3)
    p4 = GeocentricToGeographic(p3 , 6378137.0,298.257223563)
    print("Final Output (GeocentricToGeographic):", p4)
    return p4


def DatumToGRS80(point, tx, ty, tz, rx, ry, rz, f):
    """
    Applies a datum transformation from GRS 80 using provided shift and rotation parameters.

    Parameters
    ----------
    point : Point
        The point in geocentric coordinates.
    tx, ty, tz : float
        Translation parameters for the x, y, and z coordinates.
    rx, ry, rz : float
        Rotation parameters (in arc seconds) for the x, y, and z axes.
    f : float
        Scale factor in parts per million.

    Returns
    -------
    Point
        The transformed point in geocentric coordinates.
    """
    
    vx = point.x - tx
    vy = point.y - ty
    vz = point.z - tz
    rx = rx * math.pi / 648000
    ry = ry * math.pi / 648000
    rz = rz * math.pi / 648000
    e = 1 + (f / 1000000)
    det = e * (math.pow(e, 2) + math.pow(rx, 2) + math.pow(ry, 2) + math.pow(rz, 2))
    x2 = ((math.pow(e, 2) + math.pow(rx, 2)) * vx + (e * rz + rx * ry) * vy + (rx * rz - e * ry) * vz) / det
    y2 = ((-e * rz + rx * ry) * vx + (e * e + math.pow(ry, 2)) * vy + (rx * e + rz * ry) * vz) / det
    z2 = ((e * ry + rx * rz) * vx + (-e * rx + rz * ry) * vy + (e * e + math.pow(rz, 2)) * vz) / det
    p = Point(x2, y2, z2)
    return p



def ProjectLayer_sterioToWgs84(Layer, output):
    """
    Projects a geographic layer from stereographic to WGS 84 coordinates, supporting
    transformations into both UTM and geographic coordinate systems.

    Parameters
    ----------
    Layer : str
        The file path to the layer containing geographic data.
    output : str
        The output path where the transformed data will be saved.

    Notes
    -----
    The function handles both Polygon and MultiPolygon geometries, converting them
    through a series of transformations and saving the results in both UTM and
    WGS 84 formats.

    """
    # WGS 84 to UTM Zone 36N
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32636', always_xy=True)
    name = GetFeatureName(output)
    path = GetFeaturePath(output)
    sr_utm = CRS.from_epsg(32636)  # UTM Zone 36N
    sr_wgs = CRS.from_epsg(4326)   # WGS 84

    gdf = gpd.read_file(Layer)
    # gdf = gdf[:4] # for debugging 
    type = gdf.geometry.geom_type[0]
    new_gdf_utm = gpd.GeoDataFrame(columns=['geometry'], crs=sr_utm)
    new_gdf_wgs = gpd.GeoDataFrame(columns=['geometry'], crs=sr_wgs)

    if type == 'Polygon' or type == 'MultiPolygon':
        for index, row in tqdm(gdf.iterrows()):
            s = row.geometry
            newfeature_utm = []
            newfeature_wgs = []
            if s.geom_type == 'Polygon':
                polygons = [s]  
            elif s.geom_type == 'MultiPolygon':
                polygons = list(s.geoms)  
            for polygon in polygons:
                newpart_utm = []
                newpart_wgs = []
                for point in polygon.exterior.coords:
                    if point is not None:
                        point2 = SterioToWgs84(Point(point)) 
                        # Use transformer to project point to UTM
                        point_utm = transformer.transform(point2.x, point2.y)
                        point_wgs = (point2.x, point2.y)
                        newpart_utm.append(point_utm)
                        newpart_wgs.append(point_wgs)
                newfeature_utm.append(Polygon(newpart_utm))
                newfeature_wgs.append(Polygon(newpart_wgs))
            pol_utm = MultiPolygon(newfeature_utm) if len(newfeature_utm) > 1 else newfeature_utm[0]
            pol_wgs = MultiPolygon(newfeature_wgs) if len(newfeature_wgs) > 1 else newfeature_wgs[0]
            new_gdf_utm.loc[len(new_gdf_utm)] = [pol_utm]
            new_gdf_wgs.loc[len(new_gdf_wgs)] = [pol_wgs]
    new_gdf_utm.crs = "EPSG:32636"
    new_gdf_wgs.crs = "EPSG:4326"

    new_gdf_utm.to_file(path + name + '_utm.shp', driver='ESRI Shapefile')
    new_gdf_wgs.to_file(path + name + '_wgs.shp', driver='ESRI Shapefile')

