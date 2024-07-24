import unittest
import geotils.data_processing.make_dataset as m
from rasterio.coords import BoundingBox
from rasterio import windows
from PIL import Image, ImageDraw
import numpy as np


class TestMakeDataset(unittest.TestCase):
    building_polygons = [
        np.array([[0, 0], [0, 100], [100, 100], [100, 0]]),
        np.array([[200, 100], [200, 400], [300, 400], [300, 100]]),
        np.array([[400, 200], [400, 300], [500, 300], [500, 200], [450, 100]]),
        np.array([[600, 0], [600, 200], [800, 200], [800, 0], [700, -100]]),
        np.array(
            [[900, 100], [900, 300], [1000, 400], [1100, 300], [1100, 100], [1000, 0]]
        ),
    ]

    def test_generate_polygon(self):
        b = BoundingBox(150, 150, 400, 450)
        polygon = m.generate_polygon(b)
        pol2 = [[150, 150], [400, 150], [400, 450], [150, 450], [150, 150]]
        self.assertEqual(polygon, pol2)

    def test_shape_polys(self):
        global pl
        pl = m.shape_polys(self.building_polygons)
        for i in range(len(pl)):
            for j in range(len(pl[i]) - 1):
                for k in range(2):
                    self.assertEqual(pl[i][j][k], self.building_polygons[i][j][k])
            self.assertEqual(pl[i][0], pl[i][len(pl[i]) - 1])

    def test_pol_to_np(self):
        pl = m.shape_polys(self.building_polygons)
        for i in range(len(self.building_polygons)):
            narr = m.pol_to_np(pl[i])
            self.assertEqual(np.array(pl[i]).all(), narr.all())

    def test_pol_to_bounding_box(self):
        pl = m.shape_polys(self.building_polygons)
        bbox = m.pol_to_bounding_box(pl[0])
        polygon = m.generate_polygon(bbox)
        self.assertEqual(
            polygon,
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]],
        )

    def test_reverse_coordinates(self):
        pl = m.shape_polys(self.building_polygons)
        reversed_coords = m.reverse_coordinates(pl[0])
        for i in range(len(reversed_coords)):
            self.assertTrue(reversed_coords[i][0] == pl[0][i][1])
            self.assertTrue(reversed_coords[i][1] == pl[0][i][0])

    def test_to_index(self):
        win = windows.Window(100, 200, 300, 400)
        idx = m.to_index(win)

        self.assertEqual(
            idx, [[200, 100], [200, 400], [600, 400], [600, 100], [200, 100]]
        )

    def test_create_separation(self):
        labels = np.zeros((100, 100), dtype=int)
        labels[20:30, 20:30] = 1
        labels[30:50, 20:50] = 2

        separation_mask = m.create_separation(labels)
        expected = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        self.assertEqual(
            separation_mask[27][0:59].all(),
            expected.all(),
        )

    def images_are_equal(self, img1, img2, tolerance=0):
        """
        Compare two images using numpy arrays.
        An optional tolerance can be provided to allow for small differences.
        """
        np_img1 = np.array(img1)
        np_img2 = np.array(img2)

        if tolerance == 0:
            return np.array_equal(np_img1, np_img2)

        # If tolerance is specified, check the maximum difference
        diff = np.abs(np_img1 - np_img2)
        return np.all(diff <= tolerance)

    def test_make_instance_mask(self):
        self.polygons = [
            [(0, 0), (0, 10), (10, 10), (10, 0)],
            [(20, 10), (20, 40), (30, 40), (30, 10)],
        ]
        self.size = 50

        # Create the expected image
        self.expected_image = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(self.expected_image)
        draw.polygon(self.polygons[0], outline=253, fill=253)
        draw.polygon(self.polygons[1], outline=254, fill=254)
        mask = m.make_instance_mask(self.polygons, self.size)
        self.assertTrue(
            self.images_are_equal(mask, self.expected_image),
            "The generated image does not match the expected image.",
        )


if __name__ == "__main__":
    unittest.main()
