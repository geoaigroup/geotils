import unittest, sys, os
import numpy as np
import torch
import geopandas as gpd
import cv2

sys.path.append('../')
from data_processing.pre_processing import LargeTiffLoader
from PIL import Image
import shutil

class TestLargeTiffLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_test")
        self.temp1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp1_test")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.temp1_dir, exist_ok=True)
        self.create_dummy_image()
        self.create_dummy_mask()
        # Instantiate the LargeTiffLoader class for testing

        self.large_tiff_loader = LargeTiffLoader(image_directory=self.temp_dir)

    def tearDown(self):
        # Clean up the temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.temp1_dir):
            shutil.rmtree(self.temp1_dir)


    def create_dummy_image(self):
         # Create a dummy georeferenced image for testing
         dummy_image_path = os.path.join(self.temp_dir, "dummy_image.tif")
         image = Image.new("RGB", (2048, 2048), color="white")
         image.save(dummy_image_path)

    def create_dummy_mask(self):
         # Create a dummy georeferenced image for testing
         dummy_mask_path = os.path.join(self.temp1_dir, "dummy_image.tif")
         image = Image.new("1", (2048, 2048))
         image.save(dummy_mask_path)

    # def test_load_index(self):
    #     # Test load_index function
    #     save_path = os.path.join(self.temp_dir, "test_output")
    #     os.makedirs(save_path, exist_ok=True)
    #     print("load index")
    #     col_off, row_off, width, height = 0, 0, 512, 512
    #     self.large_tiff_loader.load_index(save_path, col_off, row_off, width, height)

    #     # Assert that some files are created in the output directory
    #     files = os.listdir(save_path)
    #     self.assertGreater(len(files), 0)

    def test_pre_load(self):
        # Test pre_load function
        print("pre load")
        mask_directory = self.temp1_dir
        fragment_size = 1024
        PATCH_SIZE = 1024
        STRIDE_SIZE = 512
        DOWN_SAMPLING = 1
        transform = None

        loaded_images, loaded_masks = self.large_tiff_loader.pre_load(
            mask_directory, fragment_size=fragment_size, PATCH_SIZE=PATCH_SIZE,
            STRIDE_SIZE=STRIDE_SIZE, DOWN_SAMPLING=DOWN_SAMPLING, transform=transform
        )

        # Assert that the loaded_images and loaded_masks are lists
        self.assertIsInstance(loaded_images, list)
        self.assertIsInstance(loaded_masks, list)

        # Assert that the lengths of loaded_images and loaded_masks are greater than 0
        self.assertGreater(len(loaded_images), 0)
        self.assertGreater(len(loaded_masks), 0)

if __name__ == "__main__":
    unittest.main()
