import torch
import torch.nn as nn
#import pandas as pd
from icecream import ic

def _threshold(x, threshold=0.5):
    if threshold is not None:
        return (x >= threshold).type(x.dtype)
    else:
        return x


class MicroScores:
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-8
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = 0.
        self.fp = 0.
        self.fn = 0.
        self.tn = 0.

    def get_scores(self):
        eps = self.eps
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        recall =(tp + eps) / (tp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        
        iou = (tp + eps) / (tp + fp + fn + eps)
        f1_score = ((2 * recall * precision) + eps) / (recall + precision + eps)

        return {
            'recall' : recall.mean().item(),
            'precision' : precision.mean().item(),
            'iou' : iou.mean().item(),
            'f1' : f1_score.mean().item()
        }

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):
        
        b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)

        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW

        tp = torch.sum(y_pr * y_gt,dim=-1)
        tn = torch.sum((1.0 - y_pr) * (1.0 - y_gt),dim=-1)
        fp = torch.sum(y_pr,dim=-1) - tp
        fn = torch.sum(y_gt,dim=-1) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        return self.get_scores()

class DiceScore:
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-6
        self.dice_score_sum = 0.0
        self.weights_sum = 0.0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):

        b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)
        
        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW


        intersection = torch.sum(y_pr * y_gt, dim=-1)
        cardinality = torch.sum(y_pr + y_gt, dim=-1)

        dice_score = (2.0 * intersection)/ cardinality.clamp_min(self.eps)
        
        weights = 1.0 - (cardinality == 0.0).float()
        weights = weights.sum(dim=-1)

        dice_score = dice_score.sum(dim=-1) / weights.clamp_min(1.0)


        self.dice_score_sum += dice_score.sum()
        self.weights_sum += (weights > 0.0).float().sum()

        dice_score = self.dice_score_sum / self.weights_sum.clamp_min(1.0)


        return dice_score

class IoUScore:
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-6
        self.iou_score_sum = 0.0
        self.weights_sum = 0.0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):


        b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)
        
        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW


        intersection = torch.sum(y_pr * y_gt, dim=-1)
        union = torch.sum((y_pr + y_gt).clamp_max(1.0), dim=-1)

        iou_score =  intersection / union.clamp_min(self.eps)
        
        weights = 1.0 - (union == 0.0).float()
        weights = weights.sum(dim=-1)

        iou_score = iou_score.sum(dim=-1) / weights.clamp_min(1.0)


        self.iou_score_sum += iou_score.sum()
        self.weights_sum += (weights > 0.0).float().sum()

        iou_score = self.iou_score_sum / self.weights_sum.clamp_min(1.0)


        return iou_score






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:37:05 2020

@author: hasan
"""

import keras.backend as K
import numpy as np


def hard_dice_coef_mask(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_jacard_coef_mask(y_true, y_pred, smooth=1e-3):
    # K.flatten(K.round(y_true[..., 0]))
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f =K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def Jaccard_micro_building(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f =K.flatten(K.round(y_pred[..., 0]))
    tp=K.sum(y_true_f*y_pred_f)
    fn=K.sum(y_true_f*(1.- y_pred_f))
    fp=K.sum((1. - y_true_f)*y_pred_f)
    return (tp+1e-3)/(tp+fn+fp+1e-3)
    

def hard_dice_coef_border(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_jacard_coef_border(y_true, y_pred, smooth=1e-3):
    # K.flatten(K.round(y_true[..., 0]))
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f =K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100.0 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def hard_dice_coef_spacing(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 2]))
    y_pred_f = K.flatten(K.round(y_pred[..., 2]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_jacard_coef_spacing(y_true, y_pred, smooth=1e-3):
    # K.flatten(K.round(y_true[..., 0]))
    y_true_f = K.flatten(K.round(y_true[..., 2]))
    y_pred_f =K.flatten(K.round(y_pred[..., 2]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100.0 * (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def calc_iou(gt_masks, predicted_masks, height=768, width=768):
    true_objects = gt_masks.shape[2]
    pred_objects = predicted_masks.shape[2]
    labels = np.zeros((height, width), np.uint16)
    
    for index in range(0, true_objects):
        labels[gt_masks[:, :, index] > 0] = index + 1
    y_true = labels.flatten()
    labels_pred = np.zeros((height, width), np.uint16)
    
    for index in range(0, pred_objects):
        if sum(predicted_masks[:, :, index].shape) == height + width:
            labels_pred[predicted_masks[:, :, index] > 0] = index + 1
    y_pred = labels_pred.flatten()
    
    intersection = np.histogram2d(y_true, y_pred, bins=(true_objects + 1, pred_objects + 1))[0]
    
    area_true = np.histogram(labels, bins=true_objects + 1)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects + 1)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    
    # Compute union
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    # print(union)
    # print(intersection)
    iou = intersection / union
    return iou


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn




# if __name__ == '__main__':

    
#     y_pred = torch.tensor(
        
#         [[
#             [[0,1,0],
#              [0,1,0],
#              [0,0,0]],
#             [[0,1,0],
#              [0,0,0],
#              [0,0,0]],
#             [[0,0,0],
#              [0,0,0],
#              [0,0,0]]
#         ],
#          [
#             [[0,1,0],
#              [0,0,0],
#              [0,0,0]],
#             [[0,0,0],
#              [0,0,0],
#              [0,0,0]],
#             [[0,1,0],
#              [0,0,1],
#              [1,0,0]]
#         ],
#         ]
        
#     ).float()

#     y_gt = torch.tensor(
        
#         [[
#             [[0,1,0],
#              [0,1,0],
#              [0,0,0]],
#             [[0,0,0],
#              [0,0,0],
#              [0,0,0]],
#             [[0,0,0],
#              [0,0,0],
#              [0,0,0]]
#         ],
#          [
#             [[0,1,0],
#              [0,0,0],
#              [0,0,0]],
#             [[0,0,0],
#              [0,0,0],
#              [1,0,0]],
#             [[0,0,0],
#              [0,0,0],
#              [0,0,0]]
#         ],
#         ]
        
#     ).float()
    
#     scorer = DiceScore()
#     dice = scorer(y_pred,y_gt)
#     ic(dice)
#     # y_pred *=0
#     # y_gt = y_pred
#     # dice = scorer(y_pred,y_gt)
#     # ic(dice)
    
#     score=MicroScores()
#     micro=score(y_pred,y_gt)
#     ic(micro)