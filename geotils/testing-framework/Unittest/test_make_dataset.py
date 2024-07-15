import unittest, sys
import numpy as np
from shapely.geometry import Polygon
from rasterio.coords import BoundingBox
sys.path.append('../')
from data_processing.make_dataset import generate_polygon, shape_polys, pol_to_np, pol_to_bounding_box, reverse_coordinates, get_areas, to_index, create_separation, make_instance_mask
from PIL import Image

class TestMakeDatasetFunctions(unittest.TestCase):

    def test_generate_polygon(self):
        bbox = [1.0, 2.0, 4.0, 6.0]
        result = generate_polygon(bbox)
        expected = [
            [1.0, 2.0],
            [4.0, 2.0],
            [4.0, 6.0],
            [1.0, 6.0],
            [1.0, 2.0]
        ]
        self.assertEqual(result, expected)


    def test_shape_polys(self):
        polyg = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # Square
            np.array([[2, 2], [3, 2], [3, 3], [2, 3]]),  # Another square
            np.array([[4, 4], [5, 4], [5, 5], [4, 5]]),  # Yet another square
        ]
        result = shape_polys(polyg)
        expected = [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
            [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0), (2.0, 2.0)],
            [(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0), (4.0, 4.0)],
        ]
        self.assertEqual(result, expected)


    def test_shape_polys_empty_list(self):
        result = shape_polys([])
        self.assertEqual(result, [])


    def test_pol_to_np(self):
        poly = [[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 6.0]]
        result = pol_to_np(poly)
        expected = np.array([[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 6.0]])
        np.testing.assert_array_equal(result, expected)


    def test_pol_to_bounding_box(self):
        poly = [[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 6.0]]
        result = pol_to_bounding_box(poly)
        expected = BoundingBox(1.0, 2.0, 4.0, 6.0)
        self.assertEqual(result, expected)


    def test_reverse_coordinates(self):
        input_polygon = [[1.0, 2.0], [4.0, 2.0], [4.0, 6.0], [1.0, 6.0]]
        result = reverse_coordinates(input_polygon)
        expected = [[2.0, 1.0], [2.0, 4.0], [6.0, 4.0], [6.0, 1.0]]
        self.assertEqual(result, expected)

        input_polygon2 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        result2 = reverse_coordinates(input_polygon2)
        expected2 = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        self.assertEqual(result2, expected2)

        input_polygon_empty = []
        result_empty = reverse_coordinates(input_polygon_empty)
        expected_empty = []
        self.assertEqual(result_empty, expected_empty)


    def test_get_areas_empty_columns(self):
        cols = {}
        result = get_areas(cols)
        self.assertEqual(result, [])


    def test_get_areas_multiple_column(self):
        class Item:
            def __init__(self, id):
                self.id = id
        class Group:
            def __init__(self, items):
                self.items = items
            def get_all_items(self):
                return self.items

        group1_items = [Item(id=1), Item(id=2), Item(id=3), Item(id=4)]
        group2_items = [Item(id=5), Item(id=6), Item(id=7), Item(id=8)]

        cols = {
            'group1': Group(items=group1_items),
            'group2': Group(items=group2_items),
        }
        result = get_areas(cols)

        expected = [('group1', 1, 2), ('group1', 3, 4), ('group2', 5, 6), ('group2', 7, 8)]
        self.assertEqual(result, expected)


    def test_to_index(self):
        class MockWindow:
            def __init__(self, row_off, col_off, height, width):
                self.row_off = row_off
                self.col_off = col_off
                self.height = height
                self.width = width
        wind = MockWindow(2,3,4,5)
        result = to_index(wind)
        expected_result = [[2, 3], [2, 8], [6, 8], [6, 3], [2, 3], ]
        self.assertEqual(result, expected_result)


    def test_create_separation(self):
        labels = np.array([[0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0]])

        result = create_separation(labels)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, labels.shape)


    def test_make_instance_mask(self):
        all_polys = [[(0, 0), (0, 1), (1, 1), (1, 0)]]

        result = make_instance_mask(all_polys, size=100)

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (100, 100))

    


if __name__ == '__main__':
    unittest.main()