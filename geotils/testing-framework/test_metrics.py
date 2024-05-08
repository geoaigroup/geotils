import torch
import unittest
from unittest import TestCase
import sys 
sys.path.append('../')
from evaluation.metrics import DiceScore, IoUScore, _threshold

class TestMetrics(TestCase):
    def test_dice_score(self):
        dice_score = DiceScore()

        # Define sample tensors
        y_pred = torch.tensor([[[[0.6, 0.4], [0.5, 0.2]], [[0.3, 0.8], [0.6, 0.9]]]])
        y_gt = torch.tensor([[[[1, 0], [1, 0]], [[0, 1], [0, 1]]]])

        # Compute Dice score
        dice_score_value = dice_score(y_pred, y_gt)
        expected_dice_score = torch.tensor(0.9)

        self.assertTrue(torch.allclose(dice_score_value, expected_dice_score, atol=1e-4))

    def test_iou_score(self):
        iou_score = IoUScore()

        y_pred = torch.tensor([[[[0.6, 0.4], [0.5, 0.2]], [[0.3, 0.8], [0.6, 0.9]]]])
        y_gt = torch.tensor([[[[1, 0], [1, 0]], [[0, 1], [0, 1]]]])

        iou_score_value = iou_score(y_pred, y_gt)
        expected_iou_score = torch.tensor(0.83)

        self.assertTrue(torch.allclose(iou_score_value, expected_iou_score, atol=1e-2))
        
    def test_threshold(self):
        input_tensor = torch.tensor([[0.6, 0.4], [0.3, 0.8]])
        threshold = 0.5

        # Apply thresholding
        thresholded_tensor = _threshold(input_tensor, threshold)
        expected_tensor = torch.tensor([[1, 0], [0, 1]])

        # Check if the actual output matches the expected output
        self.assertTrue(torch.all(torch.eq(thresholded_tensor, expected_tensor)))
