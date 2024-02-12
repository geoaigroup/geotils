import random, sys, unittest
import torch
import torch.nn as nn
sys.path.append('../')
from data_processing.augmentation import *
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

class TestInferenceTimeAugmentation(unittest.TestCase):

    def test_no_flip(self):
        torch.manual_seed(42)
        data = torch.rand((3, 256, 256))  # Example input data
        expected = data
        actual = inference_time_augmentation(data, 0)
        self.assertTrue(torch.equal(expected, actual))

    def test_flip_along_last_dimension(self):
        torch.manual_seed(42)
        data = torch.rand((3, 256, 256))  # Example input data
        expected = data.flip(-1)
        actual = inference_time_augmentation(data, 1)
        self.assertTrue(torch.equal(expected, actual))

    def test_flip_along_second_to_last_dimension(self):
        torch.manual_seed(42)
        data = torch.rand((3, 256, 256))  # Example input data
        expected = data.flip(-2)
        actual = inference_time_augmentation(data, 2)
        self.assertTrue(torch.equal(expected, actual))

    def test_flip_along_both_dimensions(self):
        torch.manual_seed(42)
        data = torch.rand((3, 256, 256))  # Example input data
        expected = data.flip((-2, -1))
        actual = inference_time_augmentation(data, 3)
        self.assertTrue(torch.equal(expected, actual))
        

class TestTorchRandomRotate(unittest.TestCase):
    #We inforced a specific degree range for the testing: angle 35 for testing with rotation, angle 0 for without
    
    def test_no_rotation(self):
        # Test case: No rotation
        torch.manual_seed(42)
        data_img = torch.rand((3, 3, 256, 256))  # Example input image
        data_mask = torch.rand((3, 3, 256, 256))  # Example input mask
        expected_img = data_img
        expected_mask = data_mask
        #Instance with 0 degree angle <=> no rotation
        actual_img, actual_mask = TorchRandomRotate((0, 0))(data_img, data_mask)
        
        self.assertTrue(torch.equal(expected_img, actual_img))
        self.assertTrue(torch.equal(expected_mask, actual_mask))

    def test_no_rotation_with_mask_fill(self):
        # Test case: No rotation with mask and mask_fill specified
        torch.manual_seed(42)
        data_img = torch.rand((3, 3, 256, 256))  # Example input image
        data_mask = torch.rand((3, 3, 256, 256))  # Example input mask
        mask_fill_value = -1
        expected_img = data_img
        expected_mask = data_mask

        actual_img, actual_mask = TorchRandomRotate((0, 0), mask_fill=mask_fill_value)(
        data_img, data_mask
        )

        self.assertTrue(torch.equal(expected_img, actual_img))
        self.assertTrue(torch.equal(expected_mask, actual_mask))

    def test_random_rotation(self):
        # Test case: Random rotation
        torch.manual_seed(42)
        data_img = torch.rand((3, 3, 256, 256))  # Example input image
        data_mask = torch.rand((3, 3, 256, 256))  # Example input mask
        degrees_range = (35, 35)
        probability = 1.0
        interpolation = InterpolationMode.BILINEAR
        center = None
        fill_value = 0
        mask_fill_value = 0

        # Apply rotation using the TorchRandomRotate class
        actual_img, actual_mask = TorchRandomRotate(degrees_range, probability, interpolation, center, fill_value, mask_fill_value)(data_img.clone(), data_mask.clone())

        #Asserting that changes were done to the data
        self.assertFalse(torch.equal(data_img, actual_img))
        self.assertFalse(torch.equal(data_mask, actual_mask))

    def test_random_rotation_with_mask(self):
        # Test case: Random rotation with mask
        torch.manual_seed(42)
        data_img = torch.rand((3, 3, 256, 256))  # Example input image
        data_mask = torch.rand((3, 3, 256, 256))  # Example input mask
        degrees_range = (35, 35)
        probability = 1.0
        interpolation = InterpolationMode.BILINEAR
        center = None
        fill_value = 0
        mask_fill_value = 0

        # Apply rotation using the TorchRandomRotate class
        actual_img, actual_mask = TorchRandomRotate(degrees_range, probability, interpolation, center, fill_value, mask_fill_value)(data_img.clone(), data_mask.clone())

        #Asserting that changes were done to the data
        self.assertFalse(torch.equal(data_img, actual_img))
        self.assertFalse(torch.equal(data_mask, actual_mask))

    def test_random_rotation_with_center(self):
        # Test case: Random rotation with specified center
        torch.manual_seed(42)
        data_img = torch.rand((3, 3, 256, 256))  # Example input image
        data_mask = torch.rand((3, 3, 256, 256))  # Example input mask
        degrees_range = (35, 35)
        probability = 1.0
        interpolation = InterpolationMode.BILINEAR
        center = (128, 128)
        fill_value = 0
        mask_fill_value = 0

        # Apply rotation using the TorchRandomRotate class
        actual_img, actual_mask = TorchRandomRotate(degrees_range, probability, interpolation, center, fill_value, mask_fill_value)(data_img.clone(), data_mask.clone())

        #Asserting that changes were done to the data
        self.assertFalse(torch.equal(data_img, actual_img))
        self.assertFalse(torch.equal(data_mask, actual_mask))

        
    def test_random_rotation_with_mask_fill(self):
        # Test case: Random rotation with mask fill
        torch.manual_seed(42)
        data_img = torch.rand((3, 3, 256, 256))  # Example input image
        data_mask = torch.rand((3, 3, 256, 256))  # Example input mask
        degrees_range = (35, 35)
        probability = 1.0
        interpolation = InterpolationMode.BILINEAR
        center = None
        fill_value = 0
        mask_fill_value = 1

        # Apply rotation using the TorchRandomRotate class
        actual_img, actual_mask = TorchRandomRotate(degrees_range, probability, interpolation, center, fill_value, mask_fill_value)(data_img.clone(), data_mask.clone())

        #Asserting that changes were done to the data
        self.assertFalse(torch.equal(data_img, actual_img))
        self.assertFalse(torch.equal(data_mask, actual_mask))
        

class TestRandomMaskIgnore(unittest.TestCase):

    def test_no_mask_modification(self):
        """
        Test case where no mask modification should occur.
        """
        mask_tensor = torch.ones((1, 1, 256, 256))  # Create a dummy mask tensor
        modified_mask = RandomMaskIgnore(proba=0)(mask_tensor.clone())
        self.assertTrue(torch.equal(mask_tensor, modified_mask), "Mask should remain unchanged")

    def test_mask_modification(self):
        """
        Test case where mask modification is expected to occur.
        """
        mask_tensor = torch.ones((1, 1, 256, 256))  # Create a dummy mask tensor
        modified_mask = RandomMaskIgnore(proba=1)(mask_tensor.clone())
        self.assertFalse(torch.equal(mask_tensor, modified_mask), "Mask should be modified")

    def test_mask_modification_shape(self):
        """
        Test if the modified mask has the same shape as the original mask.
        """
        mask_tensor = torch.ones((1, 1, 256, 256))  # Create a dummy mask tensor
        modified_mask = RandomMaskIgnore()(mask_tensor.clone())
        self.assertEqual(mask_tensor.shape, modified_mask.shape, "Modified mask should have the same shape as the original mask")


    def test_batch_mask_modification(self):
        """
        Test case for batched mask modification.
        """
        batch_size = 5
        mask_tensor = torch.ones((batch_size, 1, 256, 256))  # Create a dummy batched mask tensor
        modified_mask = RandomMaskIgnore(proba=1)(mask_tensor.clone())
        self.assertFalse(torch.equal(mask_tensor, modified_mask), "Batched mask should be modified")
        
        
class TestMaskPixelDrop(unittest.TestCase):
    
    def test_no_drop(self):
        # Test case with no pixel drops
        data_mask = torch.ones((1, 1, 256, 256))  # Initial mask tensor
        mask_dropper = MaskPixelDrop(neg_drop=0, pos_drop=0, ignore_index=-1)
        expected = data_mask.clone()
        actual = mask_dropper(data_mask.clone())
        self.assertTrue(torch.equal(expected, actual))

    def test_positive_pixel_drop(self):
        # Test case with positive pixel drops
        data_mask = torch.ones((1, 1, 256, 256))  # Initial mask tensor
        mask_dropper = MaskPixelDrop(neg_drop=0, pos_drop=(20, 40), ignore_index=-1)
        expected = data_mask.clone()
        actual = mask_dropper(data_mask.clone())
        self.assertFalse(torch.equal(expected, actual))  # Expecting modifications

    def test_both_positive_and_negative_pixel_drop(self):
        # Test case with both positive and negative pixel drops
        data_mask = torch.ones((1, 1, 256, 256))  # Initial mask tensor
        mask_dropper = MaskPixelDrop(neg_drop=(30, 60), pos_drop=(20, 40), ignore_index=-1)
        expected = data_mask.clone()
        actual = mask_dropper(data_mask.clone())
        self.assertFalse(torch.equal(expected, actual))  # Expecting modifications

    def test_batch_processing(self):
        # Test case with multiple samples in a batch
        data_mask = torch.ones((3, 1, 256, 256))  # Initial mask tensor with batch size 3
        mask_dropper = MaskPixelDrop(neg_drop=50, pos_drop=50, ignore_index=-1)
        expected = data_mask.clone()
        actual = mask_dropper(data_mask.clone())
        self.assertFalse(torch.equal(expected, actual))  # Expecting modifications in at least one sample
        
