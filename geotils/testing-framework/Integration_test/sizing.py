from geotils.data_processing.resizing import *
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

import unittest
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class TestResizing(unittest.TestCase):
    def are_boundary_pixels_zero(self, image_tensor):
        """
        Check if boundary pixels (outermost rows and columns) are set to zero in the image tensor.

        Args:
        - image_tensor (torch.Tensor): Input image tensor of shape (channels, height, width).

        Returns:
        - bool: True if all boundary pixels are zero, False otherwise.
        """
        _, height, width = image_tensor.shape

        for y in [0, height - 1]:
            if not torch.all(image_tensor[:, y, :] == 0):
                return False

        for x in [0, width - 1]:
            if not torch.all(image_tensor[:, :, x] == 0):
                return False

        return True

    def are_boundary_pixels_zero_np(self, np_array):
        """
        Check if boundary pixels (outermost rows and columns) are set to zero in the NumPy array.

        Args:
        - np_array (np.ndarray): Input NumPy array of shape (height, width).

        Returns:
        - bool: True if all boundary pixels are zero, False otherwise.
        """
        height, width, _ = np_array.shape

        for y in [0, height - 1]:
            if not np.all(np_array[y, :, :] == 0):
                return False

        for x in [0, width - 1]:
            if not np.all(np_array[:, x, :] == 0):
                return False

        return True

    @classmethod
    def setUpClass(cls):
        cls.width, cls.height = 200, 200
        cls.color = (255, 0, 0)
        cls.image = Image.new("RGB", (cls.width, cls.height), cls.color)
        cls.np_array = np.array(cls.image)

        cls.image_tensor = torch.zeros(3, cls.height, cls.width, dtype=torch.uint8)
        cls.image_tensor[0, :, :] = 255
        cls.image_tensor[1, :, :] = 0
        cls.image_tensor[2, :, :] = 0
        cls.batch_size = 2
        cls.channels = 3
        cls.tensor = torch.zeros(
            cls.batch_size, cls.channels, cls.height, cls.width, dtype=torch.uint8
        )
        cls.tensor[:, 0, :, :] = 255
        cls.tensor[:, 1, :, :] = 0
        cls.tensor[:, 2, :, :] = 0

    def test_remove_boundary_positives_np(self):
        modified_np_array = remove_boundary_positives_np(self.np_array, 1)
        self.assertTrue(self.are_boundary_pixels_zero_np(modified_np_array))

    def test_remove_boundary_positives(self):
        modified_tensor = remove_boundary_positives(self.image_tensor, 1)
        self.assertTrue(self.are_boundary_pixels_zero(modified_tensor))

    def test_resize_pad(self):
        image_tensor = self.tensor[0]
        image_np = image_tensor.permute(1, 2, 0).numpy()
        resized_tensor = resize_pad(self.tensor, pad_value=20, padsize=300)
        print()

    def test_unpad_resize(self):
        resized_tensor = unpad_resize(self.image_tensor, padsize=50, resize=1024)
        print()


if __name__ == "__main__":
    unittest.main()
