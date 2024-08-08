import numpy as np
import torch
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import json
import glob
from .tmp import *
import pandas as pd

def iou_numpy(outputs, labels):
    """
    Calculate the Intersection over Union (IoU) score between binary segmentation outputs and ground truth labels.

    Args:
        outputs (torch.Tensor): Binary segmentation outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: IoU score.
    """
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score


class MatchingAlgorithm():
    """
    Initialize the MatchingAlgorithm.
    This class is designed to perform matching between ground truth bounding boxes
    and predicted bounding boxes based on Intersection over Union (IoU) scores.
    It takes lists of ground truth (GT) and predicted bounding boxes along with
    an optional IoU threshold for matching.

    Args:
        gt_bbox (List[np.ndarray]): List of ground truth bounding boxes.
        pred_bbox (List[np.ndarray]): List of predicted bounding boxes.
        iou_threshold (float): Intersection over Union (IoU) threshold for matching. Default is 0.5.
    """
    def __init__(self, gt_bbox, pred_bbox, iou_threshold=0.5):
        self.gt_bboxes = gt_bbox
        self.pred_bboxes = pred_bbox
        self.iou_threshold = iou_threshold

    def matching(self):
        """ 
        This method calculates the IoU scores between all pairs of ground truth
        and predicted bounding boxes. It then identifies true positives, false positives,
        and false negatives based on the IoU threshold. The method returns various metrics
        including IoU list, F1 scores, indices of true positives, false positives, and false negatives.

        Returns:
            Tuple of results including IoU list, F1 scores, indices of true positives,
            indices of false positives, indices of false negatives, matching scores,
            precision, and recall.
        """
        if len(self.pred_bboxes) == 0 or len(self.gt_bboxes) == 0:
            print("Both predicted and ground truth bounding boxes are empty.")
            return [], [], [], [], [], [], [], [], []

        iou_matrix = np.zeros((len(self.pred_bboxes), len(self.gt_bboxes)))

        for i in range(len(self.pred_bboxes)):
            for j in range(len(self.gt_bboxes)):
                iou_matrix[i, j] = iou_numpy(torch.from_numpy(self.pred_bboxes[i]), torch.from_numpy(self.gt_bboxes[j]))

        iou_list = []
        f1_scores = []
        pred_matched = set()
        gt_matched = set()
        tp_pred_indices = []
        tp_gt_indices = []
        m_score=[]
        mscores=[]

        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            max_index = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape)
            iou_list.append(max_iou)
            pred_matched.add(max_index[0])
            gt_matched.add(max_index[1])

            tp_pred_indices.append(max_index[0])
            tp_gt_indices.append(max_index[1])

            f1_score = 2 * max_iou / (max_iou + 1)
            f1_scores.append(f1_score)

            print(
                f"Matched predicted box {max_index[0]} with GT box {max_index[1]}, IoU = {max_iou}, F1 = {f1_score}")
            m_score={
                'pred_box':int(max_index[0]),
                'GT_box':int(max_index[1]),
                'iou':float(max_iou),
                'f1':float(f1_score)
            }
            mscores.append(m_score)
            iou_matrix[max_index[0], :] = 0
            iou_matrix[:, max_index[1]] = 0

        for i in set(range(len(self.pred_bboxes))) - pred_matched:
            iou_list.append(0)
            f1_scores.append(0)
            print(f"Unmatched predicted box {i} has no match, IoU = 0, F1 = 0")

        for i in set(range(len(self.gt_bboxes))) - gt_matched:
            iou_list.append(0)
            f1_scores.append(0)
            print(f"Unmatched GT box {i} has no match, IoU = 0, F1 = 0")

        print("number of GT boxes:", len(self.gt_bboxes))
        print("number of predicted boxes:", len(self.pred_bboxes))

        fp_indices = list(set(range(len(self.pred_bboxes))) - pred_matched)
        fn_indices = list(set(range(len(self.gt_bboxes))) - gt_matched)

        true_positives = len(tp_pred_indices)
        false_positives = len(fp_indices)
        false_negatives = len(fn_indices)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        return iou_list, f1_scores, tp_pred_indices, tp_gt_indices, fp_indices, fn_indices, mscores, precision, recall

    def tp_iou(self, tp_pred_indices, tp_gt_indices):
        """
        Calculate IoU for true positive matches by taking the indices of true positive predicted and 
        ground truth boxes and calculating the Intersection over Union (IoU) scores for each true positive match.
        It returns a tuple containing the list of IoU scores for true positive matches
        and the average IoU.

        Args:
            tp_pred_indices (List[int]): Indices of true positive predicted boxes.
            tp_gt_indices (List[int]): Indices of true positive ground truth boxes.

        Returns:
            Tuple containing the list of IoU scores for true positive matches and the average IoU.
        """
        tp_iou_list = []
        for i, j in zip(tp_pred_indices, tp_gt_indices):
            iou = iou_numpy(torch.from_numpy(self.pred_bboxes[i]), torch.from_numpy(self.gt_bboxes[j]))
            tp_iou_list.append(float(np.nan_to_num(iou)))

        if len(tp_iou_list) > 0:
            avg_tp_iou = float(np.mean(tp_iou_list))
        else:
            avg_tp_iou = None
        return tp_iou_list, avg_tp_iou


class SemanticScorer:
    def __init__(self,threshold=0):

        self.tp_list = []
        self.fp_list = []
        self.fn_list = []
        self.n_samples = 0
    
    def update(self,gt,pr):

        tp_mask = gt * pr
        fp_mask = pr - tp_mask
        fn_mask = gt - tp_mask
        
        self.tp_list.append(tp_mask.sum())
        self.fp_list.append(fp_mask.sum())
        self.fn_list.append(fn_mask.sum())
        self.n_samples += 1
        ###return np.concatenate([tp_mask[...,None],fp_mask[...,None],fn_mask[...,None]],axis=-1)###
        #pass
    
    def calculate_macro_scores(self):
        score_dict = {}
        
    
        tps = np.asarray(self.tp_list)
        fps = np.asarray(self.fp_list)
        fns = np.asarray(self.fn_list)

        zero_preds = ((tps == 0) * (fps == 0) * (fns == 0)).astype(tps.dtype)
        precs = tps / (tps + fps).clip(min=1)
        recs = tps / (tps + fns).clip(min=1)
        ious = tps / (tps + fns + fps).clip(min=1)
        f1s = (2 * precs * recs) / (precs + recs).clip(min=1)

        ####keep zero preds
        precs[zero_preds > 0] = 1.0
        recs[zero_preds > 0] = 1.0
        ious[zero_preds > 0] = 1.0
        f1s[zero_preds > 0] = 1.0

        prec = precs.mean()
        rec = recs.mean()
        iou = ious.mean()
        f1 = f1s.mean()
        score_dict['with_zero_preds'] = {'Precision' : prec,'Recall':rec,'IoU':iou,'F1':f1}
        
        ####remove zero preds
        precs[zero_preds > 0] = np.nan
        recs[zero_preds > 0] = np.nan
        ious[zero_preds > 0] = np.nan
        f1s[zero_preds > 0] = np.nan

        prec = np.nanmean(precs)
        rec = np.nanmean(recs)
        iou = np.nanmean(ious)
        f1 = np.nanmean(f1s)
        score_dict['without_zero_preds'] = {'Precision' : prec,'Recall':rec,'IoU':iou,'F1':f1}
        return score_dict
    
    def calculate_micro_scores(self):
        score_dict = {}
        
    
        tps = np.asarray(self.tp_list)
        fps = np.asarray(self.fp_list)
        fns = np.asarray(self.fn_list)

        zero_preds = ((tps == 0) * (fps == 0) * (fns == 0)).astype(tps.dtype)
        tp = tps.sum()
        fp = fps.sum()
        fn = fns.sum()
        prec = tp / (tp + fp).clip(min=1)
        rec = tp / (tp + fn).clip(min=1)
        iou = tp / (tp + fn + fp).clip(min=1)
        f1 = (2 * prec * rec) / (prec + rec).clip(min=1)
        score_dict = {'Precision' : prec,'Recall':rec,'IoU':iou,'F1':f1}
        return score_dict
