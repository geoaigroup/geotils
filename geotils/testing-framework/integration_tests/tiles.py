import geotils.data_processing.tile_utils as tile
from shapely.geometry import box

# Define an initial bounding box
initial_bbox = box(100, 150, 400, 450)

# Define the width and height constraints
width = 800
height = 600
print(initial_bbox.bounds)
# Get the fitting box
fitting_bbox = tile.get_fitting_box_from_box(initial_bbox, width, height)
print(fitting_bbox.bounds)
