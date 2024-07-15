import numpy as np
import matplotlib.pyplot as plt
from geotils.visualization.masking import Visualization


class visualization_test:
    H, W, C = 100, 100, 4

    image_array = np.zeros((H, W, C), dtype=np.uint8)

    box_color = [254, 0, 0, 0]
    box_top_left = (25, 25)
    box_bottom_right = (75, 75)

    image_array[
        box_top_left[1] : box_bottom_right[1], box_top_left[0] : box_bottom_right[0]
    ] = box_color

    viz = Visualization()
    image_array = viz.mask2rgb(image_array[:, :, :4])
