from geotils.data_processing.resizing import *
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


class test_resizeing:
    width, height = 200, 200
    color = (255, 0, 0)  # Red color
    image = Image.new("RGB", (width, height), color)

    np_array = np.array(image)

    def test_remove_boundary_positives_np():
        np_array = remove_boundary_positives_np(np_array, 1)

    def test_remove_boundary_positives():
        height = 200
        width = 200
        global image_tensor

        image_tensor = torch.zeros(3, height, width, dtype=torch.uint8)

        image_tensor[0, :, :] = 255  # Red channel
        image_tensor[1, :, :] = 0  # Green channel
        image_tensor[2, :, :] = 0  # Blue channel
        image_tensor = remove_boundary_positives(image_tensor, 1)

    def test_resize_pad():
        image_np = image_tensor.permute(1, 2, 0).numpy()

        batch_size = 2
        channels = 3
        height = 200
        width = 200

        tensor = torch.zeros(batch_size, channels, height, width, dtype=torch.uint8)

        tensor[:, 0, :, :] = 255
        tensor[:, 1, :, :] = 0
        tensor[:, 2, :, :] = 0

        image_tensor = tensor[0]
        image_np = image_tensor.permute(1, 2, 0).numpy()

        tensor = resize_pad(tensor, pad_value=20, padsize=300)

    def test_unpad_resize():
        image_tensor = unpad_resize(image_tensor, padsize=50, resize=1024)
