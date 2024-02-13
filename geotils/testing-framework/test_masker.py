import unittest, sys
import json
import numpy as np
from shapely.geometry import Polygon, polygon
from skimage.morphology import square
import rasterio as rs

sys.path.append('../')
from data_processing.masker import Masker 

class TestMasker(unittest.TestCase):

    def test_load_labels(self):
        masker = Masker()
        json_path = "assets/labels.json"  
        labels = masker.load_labels(json_path)
        self.assertIsInstance(labels, dict)


    def test_poly_size(self):        
        masker = Masker()
        w, h = 500, 700
        result = masker.poly_size(w, h)
        shapely_coords = [(int(x), int(y)) for x, y in zip(result.exterior.coords.xy[0], result.exterior.coords.xy[1])]
        expected_coords = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1), (0, 0)]
        self.assertEqual(shapely_coords, expected_coords)


    def test_get_strc(self):
        masker = Masker()
        result = masker.get_strc()
        if masker.ek_type == 'square':
            expected_result = square(3)
        else:
            expected_result = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)


    def test_load_raster_file(self):
        masker = Masker()
        raster_path = "assets/file.tif"
        raster_reader = masker.load_raster_file(raster_path)
        self.assertIsInstance(raster_reader, rs.DatasetReader)


    def test_get_img(self):
        masker = Masker()  
        raster_path = "assets/file.tif"
        raster_file = rs.open(raster_path)
        result = masker.get_img(raster_file)
        expected = np.load('assets/test_get_img.npy')
        np.testing.assert_array_equal(result,expected)


    def test_project_poly(self):
        masker = Masker()
        poly = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        frs = rs.open("assets/file.tif")
        size = (1024, 1024)
        result = masker.project_poly(poly, frs, size)
        
        self.assertIsInstance(result, list)


    def test_crop(self):        
        masker = Masker()
        img = np.random.rand(1024, 1024, 3).astype(np.uint8)
        y_off, x_off, h, w = 100, 200, 300, 400
        result = masker.crop(img, y_off, x_off, h, w)
        self.assertEqual(result.shape, (h, w, 3))
        cropped_values = img[y_off : y_off + h, x_off : x_off + w]
        np.testing.assert_array_equal(result, cropped_values)


    # def test_make_mask_with_borders(self):
    #     masker = Masker()
    #     size = (1024, 1024)
    #     polys = [[[(0.0, 0.0), (100, 0.0), (100, 100), (0.0, 100), (0.0, 0.0)]]]
        
    #     result_instances, result_builds, result_border = masker.make_mask_with_borders(polys, size)
        
    #     expected_instances = np.zeros(size, dtype=np.int32)
    #     expected_builds = np.zeros(size, dtype=np.uint8)
    #     expected_border = np.zeros(size, dtype=np.uint8)
        
    #     expected_instances[0:100, 0:100] = 1
    #     expected_builds[0:100, 0:100] = 255
    #     expected_border[99:100, 99:100] = 255

    #     np.testing.assert_array_equal(result_instances, expected_instances)
    #     np.testing.assert_array_equal(result_builds, expected_builds)
    #     np.testing.assert_array_equal(result_border, expected_border)

        
    def test_mask(self):        
        masker = Masker() 
        raster_path = "assets/file.tif"
        json_path = "assets/labels.json"
        result = masker.mask(raster_path, json_path)
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)


    def test_collect(self):        
        masker = Masker()
        labels = {"features": [{"properties": {"Id": 1, "area": 100}, "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]}
        result = masker._collect(labels)
        expected_result = {"1": {"area": 100, "geometry": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}
        self.assertDictEqual(result, expected_result)


    def test_int_coords(self):        
        masker = Masker()
        result = masker.int_coords(5)
        expected_result = np.int32(5)
        np.testing.assert_array_equal(result, expected_result)


    def test_instances(self):        
        masker = Masker()
        size = (1024, 1024)
        labels = {"1": {"area": 100, "geometry": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}
    
        result_instances = masker.instances(size, labels)
        expected_instances = np.zeros(size, dtype=np.int32)
        arr_pol = np.array([[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]], dtype=np.int32)
        
        x_coords = arr_pol[:, :, 0].flatten()
        y_coords = arr_pol[:, :, 1].flatten()
        
        poly = Polygon(zip(x_coords, y_coords)) 
        hs, ws = poly.exterior.coords.xy  
        
        hs = list(map(int, hs))
        ws = list(map(int, ws))
        
        expected_instances[hs, ws, ...] = 1
        np.testing.assert_array_equal(result_instances, expected_instances)


    # def test_borders(self):
    #     masker = Masker()
    #     size = (1024, 1024)
    #     labels = {"1": {"area": 100, "geometry": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}
    #     ins_mask = masker.instances(size, labels)
    #     result = masker.borders(ins_mask)

    #     expected_result = np.zeros(size, dtype=np.int32)
    #     expected_result[0:5, 0:5] = 1  
    #     np.testing.assert_array_equal(result, expected_result)


    def test_to_rgb(self):
        masker = Masker()
        img = np.random.rand(1024, 1024, 3).astype(np.uint8)
        result = masker.to_rgb(img)
        expected_result = img[..., :3] 
        np.testing.assert_array_equal(result, expected_result)


    def test_to_gray(self):
        masker = Masker()
        mask = np.random.randint(0, 2, (1024, 1024)).astype(np.uint8)
        result = masker.to_gray(mask)
        expected_result = (mask > 0).astype(np.uint8) * 255
        np.testing.assert_array_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()


