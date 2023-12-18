#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:23:54 2020

@author: hasan
"""
from pystac import (Catalog)
import argparse
from tqdm import trange,tqdm
import geopandas as gp
from shapely.geometry import Polygon
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio.transform import from_bounds
from PIL import Image,ImageDraw
from skimage.morphology import dilation, square, watershed
import solaris as sol
from simplification.cutil import simplify_coords_vwp
from imantics import Mask
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio.windows import Window
def get_areas(cols):
    areas=[]
    for iid in cols:
        items=[x for x in cols[iid].get_all_items()]
        for i,id in enumerate(items):
            if(i%2==0 and i+1<len(items)):
                areas.append((iid, items[i].id, items[i+1].id))
    return areas
def reverse_coordinates(pol):
    """
    Reverse the coordinates in pol
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    Returns [[y1,x1],[y2,x2],...,[yN,xN]]
    """
    return [list(f[-1::-1]) for f in pol]

def to_index(wind_):
    """
    Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    """
    return [[wind_.row_off,wind_.col_off],
            [wind_.row_off,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off],
            [wind_.row_off,wind_.col_off]]

def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]

def pol_to_np(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    return np.array([list(l) for l in pol])

def pol_to_bounding_box(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:,0]),
                       np.min(arr[:,1]),
                       np.max(arr[:,0]),
                       np.max(arr[:,1]))
    
def get_winNmask(rst,x,y,width,height,polys):
    '''This function crops a chip for a specified width and height and generates the mask of buildings with borders'''
    '''it requires solaris framework'''
    ''' input : rst = is a raster which is the tif image to be cropped open using rasterio.open(tif url)'''
    '''         x : the x pixel coordinate of the top left corner of the window desired 
                y: the y pixel coordinate of the top left corner of the window desired ( window =image chip)
                width : the width of the window to be chipped ,ex =512
                height : the height of the window to be chipped
                polys: the polygons of the whole raster image, in opencities ai the data was stored cleanly in a pystac and i was able
                to insert the polygons of the buildings in geo-pandas dataframe , the polys parameter was called by geo-pandas-df-polys.geometry
                so this is very specific and i didnt test other formats'''
    ''' output : aoo : np.array of RGBA channels of the image chipped(cropped)
        fbc_mask : an np array of shape [width,height,2] it contains the mask of buildings and borders'''
    #specify the width and height and top left corner coordinates of the window
    win = Window(x,y,width,height)
    #get the bounding box of the image chip in relative coordinates(this doesnt mean the pixel coordinates
    #but rather transforms the pixel coordinates into rst.transform system used )
    bbox = windows.bounds(win,rst.transform)
    #generate of list of coordinates for this bounding box [top left- bottom left -bottom right -top right - top left ](to for a loop)
    pol = generate_polygon(bbox)
    # put the coordinates of the window bounding box into a numpy array
    # this means that u put the x pixel in [:,0] and y in[:,1]
    #so we have a numpy array of shape [5,2]
    pol_np = np.array(pol)
    #transform all the relative coordinates of the bounding box of window to long/lat coordinates
    coords_transformed = warp.transform(rst.crs,{'init': 'epsg:4326'},pol_np[:,0],pol_np[:,1])
    
    ct=coords_transformed
    #get the extrimities in long/lat (left-bottom-right-top)
    #example for left u need the minimum of the x in channel 0 while for the bottom u need the minimum of the y coordinates (speaking long/lat wise)
    l,b,r,t=min(ct[0]),min(ct[1]),max(ct[0]),max(ct[1])
    #make a list of tuples of the corners coordinates in long/lat of the window bounding box(just to match a certain shape)
    coords_transformed = [[r,c] for r,c in zip(coords_transformed[0],coords_transformed[1])]
    #from_bounds gives the affine transformation from long/lat to pixel(i only managed to use it usefully in the solaris function)
    tfm1=from_bounds(l,b,r,t,height,width)
    #polygonize the coordinates if the bounding boxes(again just to match a certain form)
    coords_transformed= Polygon(coords_transformed)
    # read the specified window from a raster
    arr_win = rst.read(window=win)
    #make a list of buildings polygons in long/lat coords from the input ''polys'' if the building polygon intersect the window
    
    all_polys = [poly for poly in polys if poly.intersects(coords_transformed)]
    # make a geo pandas dataframe and specify the system used (epsg :4326) of the building polygons in all_polys
    all_polys_gdf = gp.GeoDataFrame(geometry=all_polys,crs='epsg:4326')
    #generate the mask of buildings and border for certain width and height (ex [512,512])
    #df= geopandas data frame
    # channels are the classes(extra class spacing didnt work with me so only ['building','boundary '] masks are generated)
    fbc_mask = sol.vector.mask.df_to_px_mask(df=all_polys_gdf,
                                            channels=['footprint', 'boundary'],
                                            affine_obj=tfm1, shape=(width,height),
                                            boundary_width=4, boundary_type='inner',meters=False)
    #shape the window from rasterio.window to numpy array of the window array (aoo)
    aoo=np.array(arr_win)
    #return RGBA window image array (aoo) and mask
    return aoo,fbc_mask

def create_separation(labels):
    #create a mask for building spacing of close buildings
    '''takes as input a numpy array where each buildings pixels is encoded with a certain value'''
    '''for example building_1 pixels are encoded as 1, building_2 as 2.....building_n as n'''
    '''note that i encoded the array as np.uint8 so on 8 bits, which means that the highest pixel value is 255'''
    '''so if u have more the 255 buildings in the image/mask u should consider a np.uint16 or np.uint32 for example to be in the safe side'''
    #perform a dilation on the image , where square(20) is the kernel( u can change the size of the kernel 
    #or the shape to a rectange if u want or whatever , but dont guarantee the result
    tmp = dilation(labels > 0, square(20))
    #apply the watershed algorithm where the basins are the original encoded labels to the dilated above
    #this generates the dams/line of separation
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    #XOR operation to remove external lines
    #u can visualize all these if u want to understand more
    tmp = tmp ^ tmp2
    #dilate the separation to get a sufficient size
    tmp = dilation(tmp, square(5))
    msk1 = np.zeros_like(labels, dtype='bool')
    #this part simply removes unwanted separation pixels by checking horizontaly and vertically the area around the pixel
    #again visualize to gain better understanding of this operation
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
    #return the separation/spacing masks
    return msk1

def shape_polys(polyg):
    #this function just shape the building polygons as a list of polygon lists
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

def make_instance_mask(all_polys,size):
    #this function encodes each building polygon with a value in the mask
    #used in the make_separation function above
    #size is the width and height of the square maks
    #zeros array of specified shape
    bg=np.zeros((size,size)).astype(np.uint8)
    bg=Image.fromarray(bg).convert('L')
    #get starting encoding value (here max is 255 since i used 8 bit encoding)
    shift=255-len(all_polys)
    #draw each polygon with a certain different encoded value
    for i,poly in enumerate(all_polys):
        ImageDraw.Draw(bg).polygon(poly,outline=shift+i,fill=shift+i)
    #return the instance mask
    #instance here refers to the fact that now we can differentiate each building from another in the image
    return bg

def tile_area(area,iid,lid,tile_size,ipth,mpth):
    '''this function is the heart of all this script'''
    '''it tiles a certain area(tif image) and generates the correspond tile mask
       input:
           area : area id (example : acc-665946)
           iid  : the image id in the pystac
           lid  :the label id in the pystac
           tile_size : size of width and height of the tile(use only square, i dont guarantee rectangles)
           ipth    : directory to save image tiles
           mpth    : directory to save the masks
           '''
    #get the item u want to tile in the pystac
    item=cols[area].get_item(id=iid)
    #get the labels of the item u picked in the pystac
    label=cols[area].get_item(id=lid)
    # make a geopandas dataframe of the labels
    #most of these i just copied from dave luo script
    lbl_gfd=gp.read_file(label.make_asset_hrefs_absolute().assets['labels'].href)
    #get the geometry of the label geo pandas dataframe
    polys=lbl_gfd.geometry
    #get the tif url
    tif_url = item.assets['image'].href
    if tif_url.startswith("./"):
        tif_url = '/'.join(item.to_dict()['links'][1]['href'].split("/")[:-1])+tif_url[1:]
    #open the tif image as a raster
    rst = rasterio.open(tif_url)
    #get the width and height of the raster
    wid,hei=rst.width,rst.height
    #move along the raster with a stride = tile_size 
    # this result in ignoring the extra left rectangles at the boundaries
    #u can manage to change it and simply pad the rectangle with zeros to get a square or something else
    
    for i in trange(0,tile_size*int(hei/tile_size),tile_size):
        for j in range(0,tile_size*int(wid/tile_size),tile_size):
                # get window array and mask of buildings and borders
                aoo,mask=get_winNmask(rst,j,i,tile_size,tile_size,polys)
                #check the unique values in the RGBA image
                
                uniq=np.unique(aoo)
                #if they are all zeros then the image contains nothing but black and u dont need it, go to the loop start
                if(np.all(uniq==0)):continue
                #check the buildins mask if any buildings exist
                uniq=np.unique(mask[:,:,0])
                #if no then the mask is zeros and u dont need to perform the create_spacing operation
                if(np.all(uniq==0)):
                    spacing=np.zeros((tile_size,tile_size)).astype(np.uint8)
                else:
                    #XOR the buildings mask with borders
                    #to get the mask of buildings minus boundaries
                    getit=mask[:,:,0]^mask[:,:,1]
                    #get all polygons in the mask
                    polyg = Mask(getit).polygons()
                    #shape the polys 
                    extracted=shape_polys(polyg)
                    #encode the buildinfs
                    labels=np.array(make_instance_mask(extracted,tile_size))
                    #get the spacing mask
                    spacing=np.array(create_separation(labels)).astype(np.uint8)*255
                    #print(np.unique(spacing))
                #encode the buildings to 64, boundaries to 128 and spacing to 255, to differentiate and for visualization
                aoo=np.moveaxis(aoo,0,2)
                final_mask=spacing
                final_mask[(np.where(mask[:,:,0]>0))]=np.uint8(64)
                final_mask[(np.where(mask[:,:,1]>0))]=np.uint8(128)
                #convert the RGBA to RGB image
                img=Image.fromarray(aoo).convert('RGB')
                fmsk=Image.fromarray(final_mask)
                #save each in the correct directory
                img.save('/'.join([ipth,f'{area}_{iid}_{i}_{j}.png']))
                fmsk.save('/'.join([mpth,f'{area}_{iid}_{i}_{j}_mask.png']))

if __name__=='__main__':
    #parse the arguments from terminal
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help='path to train_tier')
    parser.add_argument('--out',type=str,help='output path')
    parser.add_argument('--tsize',type=int,help='tile size in pixels')
    parser.add_argument('--bw',type=int,help='building contours width in pixels')
    args=parser.parse_args()
    #path to the main catalog of the pystac
    #here the path is to tier1 catalog
    path='/media/hasan/DATA/OpenCitiesAI/train_tier_1/catalog.json'
    #path to the images and masks directory to be saved
    #if parsed please create these directories
    if(args.data):path=args.data
    img_path='/media/hasan/DATA/OpenCitiesAI/Dataset/images'
    mask_path='/media/hasan/DATA/OpenCitiesAI/Dataset/masks'
    if(args.out):
        img_path='/'.join([args.out,'images'])
        mask_path='/'.join([args.out,'masks'])
    #specify tile size - default is 512
    tile_size=512
    if(args.tsize):tile_size=args.tsize
    # this doesnt work 
    bw=3
    if(args.bw):bw=args.bw
    #open the main catalog of the pystac
    main_cat=Catalog.from_file(path)
    #get all the columns in the catalog(main_children) , it is how the pystac is organized
    #i just copied this
    cols = {cols.id:cols for cols in main_cat.get_children()}
    print(cols)
    # get the list of all the areas and their relative image id and label id
    a=get_areas(cols)
    #tile each area in the dataset
    for si in tqdm(a):
        area=si[0]
        iid=si[1]
        lid=si[2]
        print(area)
        print(iid)
        tile_area(area,iid,lid,512,img_path,mask_path)   
        