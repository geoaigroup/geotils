import unittest, sys
from shapely.geometry import Polygon, MultiPolygon
from affine import Affine

sys.path.append('../')
from data_processing.post_process import *

class TestPostProcess(unittest.TestCase):
    def test_single_channel(self):
        # Test case with a single-channel prediction
        pred = np.ones(256) * 0.7
        ob = PostProcessing()
        result = ob.post_process(pred)
        self.assertIsNone(result)

    def test_multi_channel(self):
        # Test case with multi-channel prediction
        pred = np.zeros((256, 256, 3))
        pred[..., 0] = 0.8  # Building predictions
        pred[..., 1] = 0.3  # Border predictions
        pred[..., 2] = 0.2  # Spacing predictions
        ob = PostProcessing()
        result = ob.post_process(pred)
        self.assertIsNotNone(result)

    def test_single_channel_threshold(self):
        # Test case with a single-channel prediction and custom threshold
        pred = np.ones((256, 256)) * 0.7
        ob = PostProcessing()
        result = ob.post_process(pred, thresh=0.8)
        self.assertTrue(np.all(result == 0))

    def test_multi_channel_min_area(self):
        # Test case with multi-channel prediction and minimum area threshold
        pred = np.zeros((256, 256, 3))
        pred[..., 0] = 0.8  # Building predictions
        pred[..., 1] = 0.3  # Border predictions
        pred[..., 2] = 0.2  # Spacing predictions
        ob = PostProcessing()
        result = ob.post_process(pred, mina=200)
        self.assertIsNotNone(result)
        
        
class TestNoiseFilter(unittest.TestCase):
    def test_filter_small_regions(self):
        # Test case where small regions are filtered
        washed = np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]])
        mina = 2
        result = PostProcessing.noise_filter(washed, mina)
        expected_result = np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]])
        np.testing.assert_array_equal(result, expected_result)

    def test_no_filtering(self):
        # Test case where no regions are filtered
        washed = np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]])
        mina = 0
        result = PostProcessing.noise_filter(washed, mina)
        np.testing.assert_array_equal(result, washed)

    def test_empty_mask(self):
        # Test case with an empty mask
        washed = np.array([])
        mina = 0
        result = PostProcessing.noise_filter(washed, mina)
        np.testing.assert_array_equal(result, washed)

    def test_mina_thresholding(self):
        # Test case with minimum area thresholding
        washed = np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]])
        mina = 3
        result = PostProcessing.noise_filter(washed, mina)
        expected_result = np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]])
        np.testing.assert_array_equal(result, expected_result)


class TestExtractPoly(unittest.TestCase):
    def test_empty_mask(self):
        # Test case with an empty mask
        mask = np.array([], dtype=np.int16)
        result = PostProcessing.extract_poly(mask)
        self.assertIsNone(result)
        
    def test_extract_polygons(self):
        # Test case where polygons are successfully extracted
        mask = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [1, 1, 0]], dtype=np.int16)
        result = PostProcessing.extract_poly(mask)
        self.assertIsInstance(result, (Polygon, MultiPolygon))
        self.assertTrue(result.is_valid)

    def test_no_polygons(self):
        # Test case where no polygons are found
        mask = np.array([[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]], dtype=np.int16)
        result = PostProcessing.extract_poly(mask)
        self.assertIsNone(result)


class TestInstanceMaskToGDF(unittest.TestCase):
    def setUp(self):
        # Create a sample instance mask
        self.instance_mask = np.array([[1, 1, 2, 0],
                                       [0, 2, 2, 0],
                                       [3, 3, 0, 0]])
        self.instance_mask = self.instance_mask.astype(np.int16)

        # Create a sample transform
        self.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)

        # Set a sample CRS
        self.crs = 'EPSG:4326'

    def test_instance_mask_to_gdf(self):
        # Test case where instances are present in the mask
        result = PostProcessing.instance_mask_to_gdf(self.instance_mask, transform=self.transform, crs=self.crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 3)  # Assuming 3 instances in the sample mask
        self.assertFalse('id' in result.columns)
        self.assertTrue('geometry' in result.columns)
        self.assertTrue(result.crs.equals(self.crs))

    def test_instance_mask_to_gdf_no_instances(self):
        # Test case where no instances are present in the mask
        empty_mask = np.zeros_like(self.instance_mask)
        result = PostProcessing.instance_mask_to_gdf(empty_mask, transform=self.transform, crs=self.crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertTrue(result.empty)
        self.assertTrue(result.crs.equals(self.crs))

    def test_instance_mask_to_gdf_no_transform(self):
        # Test case where transform is not provided
        result = PostProcessing.instance_mask_to_gdf(self.instance_mask, crs=self.crs)
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue(result.crs.equals(self.crs))
