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
import torch
import sys 
import shutil
sys.path.append('../../')
from evaluation.evaluate import iou_numpy, MatchingAlgorithm

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
