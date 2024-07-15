import unittest, sys
import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Polygon
sys.path.append('../')
from data_processing.polygon_utility import binary_mask_to_polygon, convert_polygon_to_mask, convert_polygon_to_mask_batch, ArgMax

class TestFunctions(unittest.TestCase):

    def test_binary_mask_to_polygon(self):
        # Test binary_mask_to_polygon function
        binary_mask = np.zeros((10, 10))
        binary_mask[2:8, 2:8] = 1
        polygon = binary_mask_to_polygon(binary_mask)

        # Assert the type of the result
        self.assertIsInstance(polygon, Polygon)

    def test_convert_polygon_to_mask(self):
        # Test convert_polygon_to_mask function
        shape = (10, 10)
        geo = gpd.GeoSeries([Polygon([(1, 1), (1, 5), (5, 5), (5, 1)])])
        gtmask = convert_polygon_to_mask(geo, shape)

        # Assert the type of the result and its shape
        self.assertIsInstance(gtmask, np.ndarray)
        self.assertEqual(gtmask.shape, shape)

    def test_convert_polygon_to_mask_batch(self):
        # Test convert_polygon_to_mask_batch function
        shape = (10, 10)
        geo = gpd.GeoSeries([Polygon([(1, 1), (1, 5), (5, 5), (5, 1)]), Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])])
        transform = None  # Assuming non-georeferenced polygons for simplicity
        gtmask = convert_polygon_to_mask_batch(geo, shape, transform)

        # Assert the type of the result and its length
        self.assertIsInstance(gtmask, list)
        self.assertEqual(len(gtmask), len(geo))

        # Assert the shape of individual masks in the list
        for mask in gtmask:
            self.assertIsInstance(mask, np.ndarray)
            self.assertEqual(mask.shape, shape)

    def test_ArgMax(self):
        # Test ArgMax module
        dim = 1
        argmax_module = ArgMax(dim=dim)
        input_tensor = torch.randn(3, 4, 5)

        # Assert the output shape after applying ArgMax
        output_tensor = argmax_module(input_tensor)
        self.assertEqual(output_tensor.shape, (3, 5))

if __name__ == "__main__":
    unittest.main()
