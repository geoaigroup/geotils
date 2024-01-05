import torch
import torch.nn as nn
from icecream import ic
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import geopandas as gpd
import os
import json
import glob
import keras.backend as K
from utililities.poly_conv import convert_polygon_to_mask_batch,binary_mask_to_polygon
import pandas as pd

from segmentation_models_pytorch.utils.metrics import Accuracy,Recall
from segmentation_models_pytorch.utils.base import Metric


def _threshold(x, threshold=0.5):
    if threshold is not None:
        return (x >= threshold).type(x.dtype)
    else:
        return x


# class MicroScores:
#     def __init__(self,threshold = 0.5,ignore_index=-100):
#         self.thresh = threshold
#         self.eps = 1e-8
#         self.ignore_index = ignore_index
#         self.reset()

#     def reset(self):
#         self.tp = 0.
#         self.fp = 0.
#         self.fn = 0.
#         self.tn = 0.

#     def get_scores(self):
#         eps = self.eps
#         tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

#         recall =(tp + eps) / (tp + fn + eps)
#         precision = (tp + eps) / (tp + fp + eps)
        
#         iou = (tp + eps) / (tp + fp + fn + eps)
#         f1_score = ((2 * recall * precision) + eps) / (recall + precision + eps)

#         return {
#             'recall' : recall.mean().item(),
#             'precision' : precision.mean().item(),
#             'iou' : iou.mean().item(),
#             'f1' : f1_score.mean().item()
#         }

#     @torch.no_grad()
#     def __call__(self, y_pr,y_gt,thresh_gt=False):
        
#         b,c,h,w = y_pr.shape
#         y_pr = _threshold(y_pr,self.thresh)
#         if thresh_gt:
#             y_gt = _threshold(y_gt,self.thresh)

#         if self.ignore_index is not None:
#             Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
#             y_gt = y_gt * Y_nindex
#             y_pr = y_pr * Y_nindex

#         y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
#         y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW

#         tp = torch.sum(y_pr * y_gt,dim=-1)
#         tn = torch.sum((1.0 - y_pr) * (1.0 - y_gt),dim=-1)
#         fp = torch.sum(y_pr,dim=-1) - tp
#         fn = torch.sum(y_gt,dim=-1) - tp

#         self.tp += tp
#         self.fp += fp
#         self.fn += fn
#         self.tn += tn

#         tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

#         return self.get_scores()

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

        print(y_pr.shape)
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


class Fscore(Metric):
    __name__ = 'fscore'
    def __init__(self,threshold = 0.5, eps=1e-5,beta = 1):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.beta = 1
    def reset(self):
        pass
    @torch.no_grad()
    def __call__(self,prediction,target):
        prediction = (prediction > self.threshold).float()
        tp = (prediction * target).sum()
        fp = prediction.sum() - tp
        fn = target.sum() -  tp
        score = ((1 + self.beta ** 2) * tp + self.eps) \
            / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.eps)

        return score







# def calc_iou(gt_masks, predicted_masks, height=768, width=768):
#     true_objects = gt_masks.shape[2]
#     pred_objects = predicted_masks.shape[2]
#     labels = np.zeros((height, width), np.uint16)
    
#     for index in range(0, true_objects):
#         labels[gt_masks[:, :, index] > 0] = index + 1
#     y_true = labels.flatten()
#     labels_pred = np.zeros((height, width), np.uint16)
    
#     for index in range(0, pred_objects):
#         if sum(predicted_masks[:, :, index].shape) == height + width:
#             labels_pred[predicted_masks[:, :, index] > 0] = index + 1
#     y_pred = labels_pred.flatten()
    
#     intersection = np.histogram2d(y_true, y_pred, bins=(true_objects + 1, pred_objects + 1))[0]
    
#     area_true = np.histogram(labels, bins=true_objects + 1)[0]
#     area_pred = np.histogram(y_pred, bins=pred_objects + 1)[0]
#     area_true = np.expand_dims(area_true, -1)
#     area_pred = np.expand_dims(area_pred, 0)
    
#     # Compute union
#     union = area_true + area_pred - intersection
    
#     # Exclude background from the analysis
#     intersection = intersection[1:, 1:]
#     union = union[1:, 1:]
#     union[union == 0] = 1e-9
#     # print(union)
#     # print(intersection)
#     iou = intersection / union
#     return iou


# def precision_at(threshold, iou):
#     matches = iou > threshold
#     true_positives = np.sum(matches, axis=1) == 1  # Correct objects
#     false_positives = np.sum(matches, axis=0) == 0  # Missed objects
#     false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
#     tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
#     return tp, fp, fn






def iou_numpy(outputs, labels):
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

class matching_algorithm:
    def __init__(self, gt_bbox, pred_bbox, iou_threshold=0.5):
        self.gt_bboxes = gt_bbox
        self.pred_bboxes = pred_bbox
        self.iou_threshold = iou_threshold

    def matching(self):
        if len(self.pred_bboxes) == 0 or len(self.gt_bboxes) == 0:
            print("Both predicted and ground truth bounding boxes are empty.")
            return [], [], [], [], [], []
      
    
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
        tp_iou_list = []
        for i, j in zip(tp_pred_indices, tp_gt_indices):
            iou = iou_numpy(torch.from_numpy(self.pred_bboxes[i]), torch.from_numpy(self.gt_bboxes[j]))
            tp_iou_list.append(float(np.nan_to_num(iou)))

        if len(tp_iou_list) > 0:
            avg_tp_iou = float(np.mean(tp_iou_list))
        else:
            avg_tp_iou = None
        return tp_iou_list, avg_tp_iou


class cal_scores:
    def __init__(self,output_dir,score_dir):
        self.output_dir=output_dir
        self.score_dir=score_dir

    def micro_match_iou(self,pred_mask, name, gt, image):
        pred_tile = []
        gt_tile = []
        # msk = pred_mask.int()
        # msk = msk.cpu().numpy()
        scores_b = []
        score = {}
        # mask_tile = np.zeros(image.shape[:2])

        # for i in range(msk.shape[0]):
        #     batch = msk[i]
        #     for b in range(batch.shape[0]):
        #         mask_tile = mask_tile + batch[b]
        #         pred_tile.append(batch[b])


        gt_tile=convert_polygon_to_mask_batch(gt['geometry'],image.shape[:2])
        
        pred_tile=np.expand_dims(np.array(pred_mask),axis=1)
        gt_tile=np.expand_dims(np.array(gt_tile),axis=1)

        matcher = matching_algorithm(gt_tile, pred_tile)
        iou_list, f1_scores, tp_pred_indices, tp_gt_indices, fp_indices, fn_indices, mscores, precision,recall = matcher.matching()
        tp_iou_list, avg_tp_iou = matcher.tp_iou(tp_pred_indices, tp_gt_indices)

        score['iou_list'] = iou_list
        score['f1_scores'] = f1_scores
        score['tp_iou_list'] = tp_iou_list
        score['fp_indices'] = fp_indices
        score['fn_indices'] = fn_indices
        score['Mean_iou'] = np.mean(iou_list, dtype=float)
        score['Mean_f1'] = np.mean(f1_scores, dtype=float)
        score['avg_tp_iou'] = float(avg_tp_iou) if avg_tp_iou != None else 0.0
        score['precision'] = precision
        score['recall'] = recall

        for s in mscores:
            scores_b.append(s)
        scores_b.append(score)

        with open(self.score_dir + f'/{name}_score.json', 'w') as f1:
            json.dump(scores_b, f1)

        polys=[]
        for k in pred_tile:
            if not np.any(k):
                continue
            polys.append(binary_mask_to_polygon(k))

        gdf = gpd.GeoDataFrame({
                            'ImageId':name,
                            'geometry':polys
                            })
        gdf.to_file(f"{self.output_dir}/{name}/{name}.shp")

   

    def macro_score(self):
        score_list = []
        all_iou_scores = []
        all_f1_scores = []
        all_precision=[]
        all_recall=[]
        for i in glob.glob(os.path.join(self.score_dir, "*.json")):
            name = i.split("/")[-1]
            name = name.split("_score")[0]

            f = open(i)
            file_data = json.load(f)
            ds = {}
            iou = file_data[len(file_data) - 1]["Mean_iou"]
            f1 = file_data[len(file_data) - 1]["Mean_f1"]
            avg_tp_iou = file_data[len(file_data) - 1]["avg_tp_iou"]
            precision = file_data[len(file_data) - 1]["precision"]
            recall = file_data[len(file_data) - 1]["recall"]

            all_precision.append(precision)
            all_recall.append(recall)

            all_iou_scores.append(file_data[len(file_data) - 1]["iou_list"])
            all_f1_scores.append(file_data[len(file_data) - 1]["f1_scores"])


            ds["name"] = name
            ds["iou"] = iou
            ds["f1"] = f1
            ds["avg_tp_iou"] = avg_tp_iou
            ds["precision"] = precision
            ds["recall"] = recall
            score_list.append(ds)

        df = pd.DataFrame(score_list)
        df.to_csv(self.score_dir + "/scores.csv", index=False)

        all_i = []
        all_f = []
        all_tpi = []
        all_tpf = []

        for i1, f11 in zip(all_iou_scores, all_f1_scores):
            for i2, f12 in zip(i1, f11):
                all_i.append(i2)
                all_f.append(f12)
                if i2 > 0 and f12 > 0:
                    all_tpi.append(i2)
                    all_tpf.append(f12)

        total_iou = np.nanmean(np.array(all_i))
        total_f1 = np.nanmean(np.array(all_f))
        total_tpiou = np.mean(np.array(all_tpi))
        total_tpf1 = np.mean(np.array(all_tpf))
        total_precision = np.mean(np.array(all_precision))
        total_recall = np.mean(np.array(all_recall))

        print("Mean iou score of all buildings in all tiles:", total_iou)
        print("Mean F1 score of all buildings in all tiles:", total_f1)
        print("Mean tp iou score of all buildings in all tiles:", total_tpiou)
        print("Mean tp f1 score of all buildings in all tiles:", total_tpf1)
 





# def iou(pr, gt, eps=1e-7, threshold=0.5, activation='sigmoid'):
#     """
#     Source:
#         https://github.com/catalyst-team/catalyst/
#     Args:
#         pr (torch.Tensor): A list of predicted elements
#         gt (torch.Tensor):  A list of elements that are to be predicted
#         eps (float): epsilon to avoid zero division
#         threshold: threshold for outputs binarization
#     Returns:
#         float: IoU (Jaccard) score
#     """

#     if activation is None or activation == "none":
#         activation_fn = lambda x: x
#     elif activation == "sigmoid":
#         activation_fn = torch.nn.Sigmoid()
#     elif activation == "softmax2d":
#         activation_fn = torch.nn.Softmax2d()
#     else:
#         raise NotImplementedError(
#             "Activation implemented for sigmoid and softmax2d"
#         )

#     pr = activation_fn(pr)

#     if threshold is not None:
#         pr = (pr > threshold).float()

#     intersection = torch.sum(gt * pr)
#     union = torch.sum(gt) + torch.sum(pr) - intersection + eps
#     return (intersection + eps) / union


# jaccard = iou


# def f_score(pr, gt, beta=1, eps=1e-7, threshold=0.5, activation='sigmoid'):
#     """
#     Args:
#         pr (torch.Tensor): A list of predicted elements
#         gt (torch.Tensor):  A list of elements that are to be predicted
#         beta (float): positive constant
#         eps (float): epsilon to avoid zero division
#         threshold: threshold for outputs binarization
#     Returns:
#         float: F score
#     """

#     if activation is None or activation == "none":
#         activation_fn = lambda x: x
#     elif activation == "sigmoid":
#         activation_fn = torch.nn.Sigmoid()
#     elif activation == "softmax2d":
#         activation_fn = torch.nn.Softmax2d()
#     else:
#         raise NotImplementedError(
#             "Activation implemented for sigmoid and softmax2d"
#         )

#     pr = activation_fn(pr)

#     if threshold is not None:
#         pr = (pr > threshold).type(pr.dtype)

#     tp = torch.sum(gt * pr)
#     fp = torch.sum(pr) - tp
#     fn = torch.sum(gt) - tp

#     score = ((1 + beta ** 2) * tp + eps) \
#             / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

#     return score






# class MicroIoU(Metric):
#     __name__ = "micro_iou"

#     def __init__(self, threshold=0.5):
#         super().__init__()
#         self.eps = 1e-5
#         self.intersection = 0.
#         self.union = 0.
#         self.threshold = threshold

#     def reset(self):
#         self.intersection = 0.
#         self.union = 0.

#     @torch.no_grad()
#     def __call__(self, prediction, target):
#         prediction = (prediction > self.threshold).float()

#         intersection = (prediction * target).sum()
#         union = (prediction + target).sum() - intersection

#         self.intersection += intersection.detach()
#         self.union += union.detach()

#         score = (self.intersection + self.eps) / (self.union + self.eps)
#         return score
    
# class IoU(Metric):
#     __name__ = "iou_score"

#     def __init__(self, threshold=0.5):
#         super().__init__()
#         self.eps = 1e-5
#         self.threshold = threshold

#     def reset(self):
#         pass

#     @torch.no_grad()
#     def __call__(self, prediction, target):
#         prediction = (prediction > self.threshold).float()

#         intersection = (prediction * target).sum()
#         union = (prediction + target).sum() - intersection

#         score = (intersection + self.eps) / (union + self.eps)
#         return score
    


# class MicroFscore(Metric):
#     __name__ = 'micro_fscore'
#     def __init__(self,threshold = 0.5, eps=1e-5,beta = 1):
#         super().__init__()
#         self.eps = eps
#         self.threshold = threshold
#         self.beta = 1
#         self.tp,self.fp,self.fn = 0,0,0
#     def reset(self):
#         self.tp,self.fp,self.fn = 0,0,0
#     @torch.no_grad()
#     def __call__(self,prediction,target):
#         prediction = (prediction > self.threshold).float()
#         tp = (prediction * target).sum()
#         self.tp += tp
#         self.fp += (prediction.sum() - tp)
#         self.fn += (target.sum() -  tp)
#         score = ((1 + self.beta ** 2) * self.tp + self.eps) \
#             / ((1 + self.beta ** 2) * self.tp + self.beta ** 2 * self.fn + self.fp + self.eps)

#         return score
        
    

# def get_metrics(metrics = ['iou','fscore'],thresh=0.5):
#     metric_funcs =[]
#     if (metrics == []):
#         raise ValueError
#     else:
#         for metric in metrics:
            
#             if(metric =='iou_score' or metric == 'iou'):
#                 #IoU.__name__='iou'
#                 metric_funcs.append(IoU(threshold=thresh))
#             elif(metric =='fscore'):
#                 Fscore.__name__='fscore'
#                 metric_funcs.append(Fscore(threshold=thresh))
#             elif(metric =='accuracy'):
#                 Accuracy.__name__='accuracy'
#                 metric_funcs.append(Accuracy)
#             elif(metric  == 'recall'):
#                 Recall.__name__='recall'
#                 metric_funcs.append(Recall)
#             elif(metric == "micro_iou"):
#                 metric_funcs.append(MicroIoU(threshold = thresh))
#             elif(metric == "micro_fscore"):
#                 metric_funcs.append(MicroFscore(threshold = thresh))
#             else:
#                 raise NotImplementedError(f'the metric {metric} is not implemented')
#     return metric_funcs

# class Micro_Metric(nn.Module):
    
#     def __init__(self,metric,channel = None,name = ''):
#         super(Micro_Metric,self).__init__()
#         self.metric = metric
#         self.ch = channel
#         self.__name__ =  name +'_'+ self.metric.__name__
        
#     def reset(self):
#         self.metric.reset()
        
#     def forward(self,y_pred,y_target):
        
#         if(self.ch is None):
#             return self.metric(y_pred,y_target)
#         else:
#             return self.metric(y_pred[:,self.ch,...],y_target[:,self.ch,...])

# def get_micro_metrics(metrics = [['iou','fscore']]*4,
#                       threshs = [0.5]*4,
#                       channels = [None,0,1,2],
#                       names = ['','Buildings','Borders','Spacing'],
#                       num_cls = 1
#                       ):
#     micro_metrics_funcs = []
    
#     for i in range(num_cls):
#         metric_funcs = get_metrics(metrics[i],thresh = threshs[i])
        
#         for metric_func in metric_funcs:
#             micro_metrics_funcs.append(Micro_Metric(metric_func,
#                                                     channels[i],
#                                                     names[i]))
#     return micro_metrics_funcs
     




















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