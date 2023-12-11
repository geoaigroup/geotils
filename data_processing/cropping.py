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
