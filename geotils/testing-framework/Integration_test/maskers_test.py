import sys
import geotils.data_processing.masker as ms
from rasterio import windows
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import rasterio as rs
from matplotlib.patches import Circle, Rectangle
import unittest

masker = ms.Masker()
sys.path.append("../")


class TestMasker(unittest.TestCase):
    global raster
    raster = masker.load_raster_file(
        r"assets\0.tif"
    )

    def test_load_labels(self):
        labels = masker.load_labels(
            r"assets\all_answers.json"
        )
        self.assertEqual(
            labels["answers"][0],
            {
                "id": 0,
                "date_added": 1543570371.6343634,
                "question_id": 0,
                "people_id": 0,
                "answer": "urban",
                "active": True,
            },
        )
        print()

    def test_poly_size(self):
        poly = masker.poly_size(100, 99)
        self.assertEqual(Polygon([[0, 0], [99, 0], [99, 98], [0, 98], [0, 0]]), poly)
        print()

    ##########################
    def test_get_strc(self):
        self.assertEqual(
            masker.get_strc().all(),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8).all(),
        )

    def test_load_raster_file(self):
        raster = masker.load_raster_file(
            r"assets\0.tif"
        )
        print()

    ##########################
    def test_get_img(self):
        img = masker.get_img(raster)
        print()

    ##########################

    """

    Wierd things happen when using 0.tif check
    Try to project using QGIS

    """

    def test_project_poly(self):
        poly = [[6.0, 0.0], [6.0, 20.0], [8.0, 2.0], [8.0, 0.0], [7.0, -1.0]]

        frs = rs.open(r"assets\file2.tif")
        size = (255.5, 255.5)
        self.assertEqual(
            masker.project_poly(poly, frs, size),
            [(0.0, 8.0), (2.0, 8.0), (20.0, 6.0), (0.0, 6.0), (0.0, 8.0)],
        )
        print()

    ##########################
    def test_crop(self):
        img = masker.get_img(raster)
        img2 = masker.crop(img, 10, 10, 246, 246)
        self.assertEqual(img2.shape, (246, 246, 3))

    ##########################

    """
    failing check notes
    """
    # poly2 = [[60, 0], [60, 200], [80, 20], [80, 0], [70, -10]]
    # mask = masker.make_mask_with_borders([poll, Polygon(poly2)])
    # plt.imshow(mask[0], cmap="viridis", interpolation="none")
    # plt.show(self)

    ##########################

    """
    mask skipped until data is given
    """

    ##########################
    def test_int_coords(self):
        x = 3.13
        self.assertEqual(masker.int_coords(x), 3)

        ##########################

    def test_instances(self):
        test = masker.instances(
            size=(1024, 1024),
            labels={
                "1": {
                    "area": 100,
                    "geometry": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
                },
                "2": {
                    "area": 100,
                    "geometry": [[[100, 40], [70, 60], [1000, 50], [0, 100], [0, 0]]],
                },
            },
        )

        self.assertEqual(
            np.genfromtxt(
                r"\assets\table.csv",
                delimiter=",",
            ).all(),
            test.all(),
        )

    ##########################
    def test_to_rgb(self):
        H, W, C = 100, 100, 4

        image_array = np.zeros((H, W, C), dtype=np.uint8)

        box_color = [254, 0, 0, 0]

        box_top_left = (25, 25)
        box_bottom_right = (75, 75)

        image_array[
            box_top_left[1] : box_bottom_right[1], box_top_left[0] : box_bottom_right[0]
        ] = box_color

        mask = masker.to_rgb(image_array)
        print()

    ##########################

    """check problem mentioned in notes """

    def test_to_gray(self):
        H, W = 100, 100

        image_array = np.zeros((H, W), dtype=np.uint8)

        box_color = [254]

        box_top_left = (25, 25)
        box_bottom_right = (75, 75)

        image_array[
            box_top_left[1] : box_bottom_right[1], box_top_left[0] : box_bottom_right[0]
        ] = box_color

        mask = masker.to_gray(image_array)
        self.assertTrue(mask.max() == 255)

    ##########################
    """want sample dataset"""


if __name__ == "__main__":
    unittest.main()
