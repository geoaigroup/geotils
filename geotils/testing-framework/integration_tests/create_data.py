import geotils.data_processing.make_dataset as m
from rasterio.coords import BoundingBox
from rasterio import windows
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw


class make_data_test:
    building_polygons = [
        np.array([[0, 0], [0, 100], [100, 100], [100, 0]]),
        np.array([[200, 100], [200, 400], [300, 400], [300, 100]]),
        np.array([[400, 200], [400, 300], [500, 300], [500, 200], [450, 100]]),
        np.array([[600, 0], [600, 200], [800, 200], [800, 0], [700, -100]]),
        np.array(
            [[900, 100], [900, 300], [1000, 400], [1100, 300], [1100, 100], [1000, 0]]
        ),
    ]

    def test_generate_polygon():
        b = BoundingBox(150, 150, 400, 450)
        polygon = m.generate_polygon(b)
        polygon1 = Polygon(polygon)

        p = gpd.GeoSeries(polygon1)

    ##################################################
    ##################################################
    def test_shape_polys(self):
        width, height = 1200, 500
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        global pl
        pl = m.shape_polys(self.building_polygons)

    ##################################################
    ##################################################
    def test_pol_to_np():
        narr = m.pol_to_np(pl[0])

    ##################################################
    ##################################################
    def test_pol_to_bounding_box():
        bbox = m.pol_to_bounding_box(pl[0])
        polygon = m.generate_polygon(bbox)
        polygon1 = Polygon(polygon)
        p = gpd.GeoSeries(polygon1)

    ##################################################
    ##################################################
    def test_reverse_coordinates():
        print(m.reverse_coordinates(pl[0]))

    ##################################################
    ##################################################

    # skipped col areas ask abt col class

    ##################################################
    ##################################################

    def test_to_index():
        win = windows.Window(100, 200, 300, 400)
        print(m.to_index(win))
        polygon = m.generate_polygon(m.pol_to_bounding_box(m.to_index(win)))
        polygon1 = Polygon(polygon)
        p = gpd.GeoSeries(polygon1)

    ##################################################
    ##################################################


def test_create_separation():
    labels = np.zeros((100, 100), dtype=int)
    labels[20:30, 20:30] = 1
    labels[30:50, 20:50] = 2

    separation_mask = m.create_separation(labels)

    ##################################################
    ##################################################


def test_make_instance_mask(self):
    # should come out in diff colors?s
    mask = m.make_instance_mask(m.shape_polys(self.building_polygons), 1000)
    mask.show()
