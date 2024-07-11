import geotils.data_processing.masker as ms
from rasterio import windows
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import rasterio as rs
from matplotlib.patches import Circle, Rectangle

masker = ms.Masker()


def test_load_labels():
    masker.load_labels(
        r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotils_testing\crop\vlm\all_answers.json"
    )


def test_poly_size():
    masker.poly_size(100, 100)


##########################
def test_get_strc():
    masker.get_strc()


def test_load_raster_file():
    global raster
    raster = masker.load_raster_file(
        r"C:\Users\abbas\OneDrive\Desktop\CNRS\geotils_testing\crop\vlm\Images_LR\0.tif"
    )


##########################
def test_get_img():
    img = masker.get_img(raster)


##########################

"""

Wierd things happen when using 0.tif check
Try to project using QGIS

"""


def test_project_poly():
    poly = [[6.0, 0.0], [6.0, 20.0], [8.0, 2.0], [8.0, 0.0], [7.0, -1.0]]
    poll = Polygon(poly)
    gpd.GeoSeries(poll).plot()
    plt.show()
    frs = rs.open(r"C:\Users\abbas\OneDrive\Desktop\CNRS\file.tif")
    size = (255.5, 255.5)
    result = masker.project_poly(poly, frs, size)


##########################


img = masker.get_img(raster)
img2 = masker.crop(img, 10, 10, 246, 246)


##########################

"""
failing check notes
"""
# poly2 = [[60, 0], [60, 200], [80, 20], [80, 0], [70, -10]]
# mask = masker.make_mask_with_borders([poll, Polygon(poly2)])
# plt.imshow(mask[0], cmap="viridis", interpolation="none")
# plt.show()

##########################

"""
mask skipped until data is given
"""


##########################
def test_int_coords():
    x = 3.13
    integer = masker.int_coords(x)

    ##########################


def test_instances():
    masker.instances(
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
    ),


##########################
def test_to_rgb():
    H, W, C = 100, 100, 4
    global image_array
    image_array = np.zeros((H, W, C), dtype=np.uint8)

    box_color = [254, 0, 0, 0]

    box_top_left = (25, 25)
    box_bottom_right = (75, 75)

    image_array[
        box_top_left[1] : box_bottom_right[1], box_top_left[0] : box_bottom_right[0]
    ] = box_color

    masker.to_rgb(image_array)


##########################

"""check problem mentioned in notes """


def test_to_gray():
    image_array = masker.to_rgb(image_array)
    image_array = masker.to_gray(image_array)
    masker.to_gray(masker.to_rgb(image_array))


##########################
"""want sample dataset"""
