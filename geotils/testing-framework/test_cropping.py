import unittest, os, sys
import shutil
import geopandas as gpd
from shapely.geometry import Point
from tempfile import TemporaryDirectory
from PIL import Image
sys.path.append('../')
from data_processing.cropping import CropGeoTiff  

class TestCropGeoTiff(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "input")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

        # Create a dummy georeferenced image for testing
        self.create_dummy_image()

        # Instantiate the CropGeoTiff class for testing
        self.crop_geo_tiff = CropGeoTiff()

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def create_dummy_image(self):
        # Create a dummy georeferenced image for testing
        dummy_image_path = os.path.join(self.input_dir, "dummy_image.tif")
        image = Image.new("RGB", (2048, 2048), color="white")
        image.save(dummy_image_path)

    def test_crop_images(self):
        # Test the crop_images function
        output_path = os.path.join(self.output_dir, "cropped_images")
        self.crop_geo_tiff.crop_images(self.input_dir, output_path)

        # Assert that the output directory is created
        self.assertTrue(os.path.exists(output_path))

        # Assert that some files are created in the output directory
        files = os.listdir(output_path)
        self.assertGreater(len(files), 0)

    def test_crop_shapefiles(self):
        # Test the crop_shapefiles function
        output_images_path = os.path.join(self.output_dir, "cropped_images")
        os.makedirs(output_images_path, exist_ok=True)

        # Create a dummy shapefile for testing
        dummy_shapefile_path = os.path.join(self.temp_dir.name, "dummy_shapefile.shp")
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
        gdf.to_file(dummy_shapefile_path)

        # Crop shapefiles
        self.crop_geo_tiff.crop_shapefiles(output_images_path, dummy_shapefile_path, self.output_dir)

        # Assert that some files are created in the output directory
        files = os.listdir(self.output_dir)
        self.assertGreater(len(files), 0)

    def test_crop_nongeoref_images_shapefile(self):
        # Test the crop_nongeoref_images_shapefile function
        shapefile_path = os.path.join(self.temp_dir.name, "dummy_shapefile.shp")

        # Create a dummy shapefile for testing
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
        gdf.to_file(shapefile_path)

        # Crop non-georeferenced images and shapefiles
        output_image_path = os.path.join(self.output_dir, "cropped_images")
        output_shapefile_path = os.path.join(self.output_dir, "cropped_shapefiles")
        self.crop_geo_tiff.crop_nongeoref_images_shapefile(
            self.input_dir, shapefile_path, output_image_path, output_shapefile_path
        )

        # Assert that some files are created in the output directories
        image_files = os.listdir(output_image_path)
        shapefile_files = os.listdir(output_shapefile_path)
        self.assertGreater(len(image_files), 0)
        self.assertGreater(len(shapefile_files), 0)

if __name__ == "__main__":
    unittest.main()
