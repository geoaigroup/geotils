import rasterio
import os
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Polygon,box
from PIL import Image


class CropGeoTiff:

    """
    The CropGeoTiff class is designed to efficiently crop large TIFF images 
    along with their corresponding shapefiles into smaller, uniform-sized images.
    This class provides two essential functions: `crop_images` for cropping georeferenced 
    images and `crop_shapefiles` for cropping georeferenced shapefiles. Additionally, 
    the `crop_nongeoref_images_shapefile` function is specifically crafted to handle the cropping 
    of non-georeferenced images and shapefiles.
    """
    def __init__(self,fragment_size=(1024, 1024)):
        """
         @param fragment_size = fragment size cropped
        """

        self.fragment_size = fragment_size 


    def crop_images(self,input_dir,output_dir):
        """
        @param input_dir = path of the directory containing images to be cropped
        @param output_dir = path of directory for saving cropped images
        """

        for i in os.listdir(input_dir):
            name = i.split('.')[0]
            input_path = os.path.join(input_dir, i)

            with rasterio.open(input_path) as src:
                original_profile = src.profile

                num_rows = src.height // self.fragment_size[0]
                num_cols = src.width // self.fragment_size[1]
                ind = 0

                for row in range(num_rows):
                    for col in range(num_cols):
                   
                        window = Window(col * self.fragment_size[1], row * self.fragment_size[0], self.fragment_size[1], self.fragment_size[0])

                      
                        fragment = src.read(window=window)

                
                        fragment_profile = original_profile.copy()
                        fragment_profile['width'] = self.fragment_size[1]
                        fragment_profile['height'] = self.fragment_size[0]

                        
                        fragment_profile['transform'] = rasterio.windows.transform(window, src.transform)

            
                        save_subdir = os.path.join(output_dir, name)
                        os.makedirs(save_subdir, exist_ok=True)

                      
                        fragment_output_path = os.path.join(save_subdir, f'{name}_{ind}.tif')
                        with rasterio.open(fragment_output_path, 'w', **fragment_profile) as dst:
                            dst.write(fragment)
                        ind += 1




    def crop_shapefiles(self,images,shapefile_path,output_dir):
        """
        @param images = path of the diectory containing cropped images
        @param shapefile_path = path of the directory containing shapefiles to be cropped
        @param output_dir = path of the directory for saving cropped shapefiles
        """
        gdf = gpd.read_file(shapefile_path)
        for dirr in os.listdir(images): 
            for i in os.listdir(os.path.join(images,dirr)):
                name=dirr
                n=i.split('.')[0]
                with rasterio.open(images+'/'+name+'/'+i) as src:
                    image_bounds = src.bounds
                
                
                
                bbox = (image_bounds.left, image_bounds.bottom, image_bounds.right, image_bounds.top)
                cropped_gdf = gpd.clip(gdf, mask=bbox)
            
               
                output = output_dir+f"/{name}/{n}/"   
                os.makedirs(output,exist_ok=True)
                output_filename = f"{n}.shp"
                output_path = output + output_filename
                cropped_gdf.to_file(output_path)
                





    def crop_nongeoref_images_shapefile(self,input_dir,shapefile_path,output_image_dir,output_shp_dir):
        """
        @param input_dir = path of the directory containing images to be cropped
        @param shapefile_path = path of the directory containing shapefiles to be cropped
        @param output_image_dir = path of the directory to save cropped images
        @param output_shp_dir = path of the directory to save cropped shapefiles
        """
       
        gdf = gpd.read_file(shapefile_path)
        for i in os.listdir(input_dir):
            name = i.split('.')[0]
            input_path = os.path.join(input_dir, i)

            image = Image.open(input_path)
            width, height = image.size

            num_rows = height // self.fragment_size[1]
            num_cols = width // self.fragment_size[0]
            ind = 0

            for row in range(num_rows):
                for col in range(num_cols):
                    left = col * self.fragment_size[0]
                    upper = row * self.fragment_size[1]
                    right = left + self.fragment_size[0]
                    lower = upper + self.fragment_size[1]

                    fragment = image.crop((left, upper, right, lower))
                    bbox = box(left, upper, right, lower)

                    cropped_gdf = gdf[gdf.intersects(bbox)]
                    
                    save_subdir = os.path.join(output_image_dir, name)
                    os.makedirs(save_subdir, exist_ok=True)

                    fragment_output_path = os.path.join(save_subdir, f'{name}_{ind}.png')
                    fragment.save(fragment_output_path)
                    output_shp = output_shp_dir+f"/{name}/{name}_{ind}/"
                    os.makedirs(output_shp, exist_ok=True)
                    output_filename = f"{name}_{ind}.shp"
                    output_path = os.path.join(output_shp, output_filename)
                    cropped_gdf.to_file(output_path)
                
                    ind += 1
