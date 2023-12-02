import numpy as np
import random
from skimage.measure import label as label_fn


def random_crop(image_stack, mask, image_size):
    '''
    THIS FUNCTION DEFINES RANDOM IMAGE CROPPING.
     :param image_stack: input image in size [Time Stamp, Image Dimension (Channel), Height, Width]
    :param mask: input mask of the image, to filter out uninterested areas [Height, Width]
    :param image_size: It determine how the data is partitioned into the NxN windows
    :return: image_stack, mask
    '''

    H, W = image_stack.shape[2:]

    # skip random crop is image smaller than crop size
    if H - image_size // 2 <= image_size:
        return image_stack, mask
    if W - image_size // 2 <= image_size:
        return image_stack, mask
    flag = True
    for i in range(0,100):
        h = np.random.randint(image_size, H - image_size // 2)
        w = np.random.randint(image_size, W - image_size // 2)

        image_stack = image_stack[:, :, h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
                    w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]
        mask = mask[h - int(np.floor(image_size // 2)):int(np.ceil(h + image_size // 2)),
            w - int(np.floor(image_size // 2)):int(np.ceil(w + image_size // 2))]
        if 1 in mask:
            break
    return image_stack, mask

def random_crop_around_aoi(img,mask,size = 32,min_area=0):
    h,w = img.shape[2:]
    mask_original = mask.copy()
    size_h,size_w = size,size
    
    if h <= size and w <= size:
        return img,mask
    if h < size:
        size_h = h
    if w < size:
        size_w = w
        
    if mask.max() == 0:
        t,b,l,r = 0,h-1,0,w-1
    else:
        mask = label_fn(mask,connectivity=2)
        values = [value for value in np.unique(mask)[1:] if mask[mask==value].sum()/value >= min_area]
        
        if len(values) == 0:
            t,b,l,r = 0,h-1,0,w-1
        else:
            sval = values[random.randint(0,len(values)-1)]
            mask[mask!=sval] = 0
            mask = ((mask / sval) * 255.0).astype(np.uint8)
            pos = np.nonzero(mask)
            t, b, l, r = pos[0].min(),pos[0].max(),pos[1].min(),pos[1].max()
        
    h_aoi,w_aoi = b-t,r-l
    pt = random.randint(t+h_aoi//2, b-h_aoi//2),random.randint(l+w_aoi//2, r-w_aoi//2)
    
    max_up = pt[0]
    max_left = pt[1]
    min_up = max(0,size_h - (h - pt[0]))
    min_left = max(0,size_w - (w - pt[1]))
    
    t_crop = pt[0] - min(max_up, random.randint(min_up, size_h-1))
    l_crop = pt[1] - min(max_left, random.randint(min_left, size_w-1))

    cropped_img = img[:,:,t_crop:t_crop+size_h,l_crop:l_crop+size_w]
    cropped_mask = mask_original[t_crop:t_crop+size_h,l_crop:l_crop+size_w]

    return cropped_img,cropped_mask



###Crop images keep georefrenced

import rasterio
import os
from rasterio.windows import Window

# Define the input and output directories
input_dir = "data/images"
output_dir = "data/images_fragmented"
fragment_size = (1024, 1024)  # Set the size of the fragments

# Iterate through each image in the input directory
for i in os.listdir(input_dir):
    name = i.split('.')[0]
    input_path = os.path.join(input_dir, i)

    with rasterio.open(input_path) as src:
        original_profile = src.profile

        # Calculate the number of rows and columns of fragments
        num_rows = src.height // fragment_size[0]
        num_cols = src.width // fragment_size[1]
        ind = 0

        # Loop through rows and columns to create equal-sized fragments
        for row in range(num_rows):
            for col in range(num_cols):
                # Define the window for the fragment
                window = Window(col * fragment_size[1], row * fragment_size[0], fragment_size[1], fragment_size[0])

                # Read the fragment from the original image
                fragment = src.read(window=window)

                # Create a new georeferenced image profile for the fragment
                fragment_profile = original_profile.copy()
                fragment_profile['width'] = fragment_size[1]
                fragment_profile['height'] = fragment_size[0]

                # Update the transformation to match the fragment's position
                fragment_profile['transform'] = rasterio.windows.transform(window, src.transform)

                # Create a subdirectory for each image
                save_subdir = os.path.join(output_dir, name)
                os.makedirs(save_subdir, exist_ok=True)

                # Save the fragment as a new georeferenced image
                fragment_output_path = os.path.join(save_subdir, f'{name}_{ind}.tif')
                with rasterio.open(fragment_output_path, 'w', **fragment_profile) as dst:
                    dst.write(fragment)
                ind += 1



##cropping shapefile in to multiple shapefiles for each image

import geopandas as gpd
from shapely.geometry import Polygon
import os
import rasterio


# Load the original shapefile
shapefile_path = "data/n1"
gdf = gpd.read_file(shapefile_path)
images="data/images_fragmented"
for dirr in os.listdir(images): 
    for i in os.listdir(os.path.join(images,dirr)):
        name=dirr
        n=i.split('.')[0]
        with rasterio.open(images+'/'+name+'/'+i) as src:
            image_bounds = src.bounds
           
           
        # Define the bounding box coordinates (minx, miny, maxx, maxy)
        bbox = (image_bounds.left, image_bounds.bottom, image_bounds.right, image_bounds.top)
        cropped_gdf = gpd.clip(gdf, mask=bbox)
    
        # # Define the output directory for the cropped shapefiles
        output_dir = f"data/fragmented_shapefiles_n1_1024/{name}/{n}/"   
        os.makedirs(output_dir,exist_ok=True)
        output_filename = f"{n}.shp"
        output_path = output_dir + output_filename
        cropped_gdf.to_file(output_path)
        

##Crop images in to fragments without considering georeferencing using PIL
from PIL import Image
import os

# Define the input and output directories
input_dir = "data/images"
output_image_dir = "data/images_fragmented"
shapefile_path = "data/n1"
gdf = gpd.read_file(shapefile_path)
fragment_size = (1024, 1024)  # Set the size of the fragments

# Iterate through each image in the input directory
for i in os.listdir(input_dir):
    name = i.split('.')[0]
    input_path = os.path.join(input_dir, i)

    # Open the image using PIL
    image = Image.open(input_path)
    width, height = image.size

    # Calculate the number of rows and columns of fragments
    num_rows = height // fragment_size[1]
    num_cols = width // fragment_size[0]
    ind = 0

    # Loop through rows and columns to create equal-sized fragments
    for row in range(num_rows):
        for col in range(num_cols):
            # Define the box for the fragment
            left = col * fragment_size[0]
            upper = row * fragment_size[1]
            right = left + fragment_size[0]
            lower = upper + fragment_size[1]

            # Crop the fragment
            fragment = image.crop((left, upper, right, lower))
            bbox = box(left, upper, right, lower)

            # Clip the shapefile using the bounding box
            cropped_gdf = gdf[gdf.intersects(bbox)]
            
            # Create a subdirectory for each image
            save_subdir = os.path.join(output_image_dir, name)
            os.makedirs(save_subdir, exist_ok=True)

            # Save the fragment as a new image
            fragment_output_path = os.path.join(save_subdir, f'{name}_{ind}.png')
            fragment.save(fragment_output_path)
            output_shp_dir = f"data/fragmented_shapefiles_n1_1024_no_georef/{name}/{name}_{ind}/"
            os.makedirs(output_shp_dir, exist_ok=True)
            output_filename = f"{name}_{ind}.shp"
            output_path = os.path.join(output_shp_dir, output_filename)
            cropped_gdf.to_file(output_path)
        
            ind += 1
