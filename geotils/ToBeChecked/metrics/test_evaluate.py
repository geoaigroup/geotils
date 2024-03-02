import unittest
from unittest.mock import patch
import os
import json
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon

class TestIoUNumpy(unittest.TestCase):
    def test_iou_numpy_perfect_match(self):
        outputs = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        labels = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        result = iou_numpy(outputs, labels)
        self.assertEqual(result, 1.0, "IoU should be 1.0 for perfect match")

    def test_iou_numpy_no_overlap(self):
        outputs = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        labels = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        result = iou_numpy(outputs, labels)
        self.assertEqual(result, 0.0, "IoU should be 0.0 for no overlap")

    def test_iou_numpy_empty_input(self):
        outputs = torch.tensor([])
        labels = torch.tensor([])
        result = iou_numpy(outputs, labels)
        self.assertTrue(torch.isnan(result), "IoU should be NaN for empty input")
        
class TestMatchingAlgorithm(unittest.TestCase):
    def setUp(self):
        # Sample ground truth and predicted bounding boxes for testing
        self.gt_bbox = [np.array([[1, 1, 3, 3], [4, 4, 6, 6]]),
                        np.array([[7, 7, 9, 9], [10, 10, 12, 12]])]
        self.pred_bbox = [np.array([[0, 0, 2, 2], [2, 2, 5, 5]]),
                          np.array([[7, 7, 9, 9], [9, 9, 12, 12]])]

    def test_matching_method(self):
        matcher = MatchingAlgorithm(self.gt_bbox, self.pred_bbox)
        iou_list, f1_scores, tp_pred_indices, tp_gt_indices, fp_indices, fn_indices, _, precision, recall = matcher.matching()

        self.assertEqual(len(iou_list), 2, "Incorrect number of IoU scores")
        self.assertEqual(len(f1_scores), 2, "Incorrect number of F1 scores")
        self.assertEqual(len(tp_pred_indices), 2, "Incorrect number of true positive predicted indices")
        self.assertEqual(len(tp_gt_indices), 2, "Incorrect number of true positive ground truth indices")
        self.assertEqual(len(fp_indices), 0, "Incorrect number of false positive indices")
        self.assertEqual(len(fn_indices), 0, "Incorrect number of false negative indices")
        self.assertAlmostEqual(precision, 1.0, places=2, msg="Incorrect precision value")
        self.assertAlmostEqual(recall, 1.0, places=2, msg="Incorrect recall value")

    def test_tp_iou_method(self):
        matcher = MatchingAlgorithm(self.gt_bbox, self.pred_bbox)
        _, _, tp_pred_indices, tp_gt_indices, _, _, _, _, _ = matcher.matching()
        tp_iou_list, avg_tp_iou = matcher.tp_iou(tp_pred_indices, tp_gt_indices)
        self.assertEqual(len(tp_iou_list), 2, "Incorrect number of true positive IoU scores")
        self.assertAlmostEqual(avg_tp_iou, 0.875, places=2, msg="Incorrect average true positive IoU value")

class TestCalScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = 'test_output'
        cls.score_dir = 'test_scores'
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.score_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir)
        shutil.rmtree(cls.score_dir)

    def test_initialization(self):
        scores = CalScores(self.output_dir, self.score_dir)
        self.assertEqual(scores.output_dir, self.output_dir)
        self.assertEqual(scores.score_dir, self.score_dir)
        
    @patch("builtins.open")
    def test_micro_match_iou(self, mock_open):
        # Test inputs
        output_dir = "output"
        score_dir = "scores"
        pred_mask = torch.zeros(2, 1, 10, 10)
        name = "test_image"
        gt = {"geometry": []}
        image = np.zeros((10, 10, 3))
        input_point = None
        input_label = None
        tile_boxes = [(1, 1, 5, 5)]

        # Initialize CalScores object
        cal_scores = CalScores(output_dir, score_dir)
        cal_scores.micro_match_iou(pred_mask, name, gt, image, input_point, input_label, tile_boxes, save=None, visualize=True)


    def test_macro_score(self):
        output_dir = "output"
        score_dir = "scores"
        cal_scores = CalScores(output_dir, score_dir)

        with patch("os.path.join"), patch("os.listdir"), patch("glob.glob"), patch("pandas.DataFrame.to_csv"), patch("builtins.open", create=True) as mocked_open, patch("json.dump"):
            cal_scores.macro_score()
