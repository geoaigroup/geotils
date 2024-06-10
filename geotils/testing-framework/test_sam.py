# import unittest
# import torch, sys 

# sys.path.append('../')
# from models.pred_SAM import SAM

# class TestSAM(unittest.TestCase):
#     def setUp(self):
#         self.checkpoint = "your_checkpoint_path"
#         self.sam = SAM(self.checkpoint)

#     def test_predictSAM_with_point(self):
#         # Define input tensors
#         x = torch.randn(1, 3, 224, 224)
#         image = torch.randn(1, 224, 224)
#         input_point = torch.tensor([[0, 0], [10, 10]])  # Example input point tensor
#         input_label = torch.tensor([0, 1])  # Example input label tensor
#         input_boxes = torch.tensor([[0, 0, 10, 10]])  # Example input box tensor

#         # Call the predictSAM method
#         pred_mask = self.sam.predictSAM(x, image, input_point=input_point, input_label=input_label, input_boxes=input_boxes)

#         # Check if the output is a tensor
#         self.assertIsInstance(pred_mask, torch.Tensor)

#     def test_predictSAM_with_image(self):
#         # Define input tensors
#         x = torch.randn(1, 3, 224, 224)
#         image = torch.randn(1, 224, 224)
#         input_point = None
#         input_label = None
#         input_boxes = torch.tensor([[0, 0, 10, 10]])  # Example input box tensor

#         # Call the predictSAM method
#         pred_mask = self.sam.predictSAM(x, image, input_point=input_point, input_label=input_label, input_boxes=input_boxes)

#         # Check if the output is a tensor
#         self.assertIsInstance(pred_mask, torch.Tensor)

# if __name__ == '__main__':
#     unittest.main()
