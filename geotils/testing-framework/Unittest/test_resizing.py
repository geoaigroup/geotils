import unittest, sys 
import torch
import numpy as np
sys.path.append('../')
from data_processing.resizing import remove_boundary_positives, remove_boundary_positives_np, resize_pad, unpad_resize

class TestResizing(unittest.TestCase):
    
    def setUp(self):
        self.tensor_shape = (2, 3, 256, 256)
        self.numpy_array_shape = (256, 256, 3)
        self.tensor = torch.randn(self.tensor_shape)
        self.numpy_array = np.random.randn(*self.numpy_array_shape)


    def test_remove_boundary_positives_tensor(self):
        pixels = 20
        result = remove_boundary_positives(self.tensor, pixels)
        self.assertEqual(result.shape, self.tensor_shape)
        self.assertTrue(torch.all(result[:, :, :pixels, :] == 0))
        self.assertTrue(torch.all(result[:, :, -pixels:, :] == 0))
        self.assertTrue(torch.all(result[:, :, :, :pixels] == 0))
        self.assertTrue(torch.all(result[:, :, :, -pixels:] == 0))


    def test_remove_boundary_positives_np(self):
        pixels = 20
        result = remove_boundary_positives_np(self.numpy_array, pixels)
        self.assertEqual(result.shape, self.numpy_array_shape)
        self.assertTrue(np.all(result[:pixels, :, :] == 0))
        self.assertTrue(np.all(result[-pixels:, :, :] == 0))
        self.assertTrue(np.all(result[:, :pixels, :] == 0))
        self.assertTrue(np.all(result[:, -pixels:, :] == 0))


    def test_resize_pad(self):    
        resize =  28
        padsize =  100
        result = resize_pad(self.tensor, padsize=padsize, resize=resize)
        self.assertEqual(result.shape, torch.Size([2, 3, 100, 100]))


    def test_unpad_resize(self):
        resize =  256
        padsize = 20
        padded_tensor = resize_pad(self.tensor, padsize=padsize)
        result = unpad_resize(padded_tensor, padsize=padsize, resize=resize)
        self.assertEqual(result.shape, self.tensor_shape)


if __name__ == '__main__':
    unittest.main()
