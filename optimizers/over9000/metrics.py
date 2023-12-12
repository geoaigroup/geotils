import torch
import torch.nn as nn
#import pandas as pd


def _threshold(x, threshold=0.5):
    if threshold is not None:
        return (x >= threshold).type(x.dtype)
    else:
        return x

class RMSE_METER(nn.Module):
    def __init__(self):
        self.rmse_values = None

    @torch.no_grad()
    def __call__(self,y_pred,y_gt):
        rmse_values = ((y_pred - y_gt) ** 2).view(-1)
        if self.rmse_values is None:
            self.rmse_values = rmse_values
        else:
            self.rmse_values = torch.cat([self.rmse_values,rmse_values],dim=0)
        
        self.rmse_avg = torch.sqrt(self.rmse_values.mean()).item()

        return self.rmse_avg
    
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
        
        #b,c,h,w = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)

        if self.ignore_index is not None:
            Y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * Y_nindex
            y_pr = y_pr * Y_nindex

        #y_gt = y_gt.view(b,c,-1).float() # >> B,C,HxW
        #y_pr = y_pr.view(b,c,-1).float() # >> B,C,HxW

        tp = torch.sum(y_pr * y_gt)
        tn = torch.sum((1.0 - y_pr) * (1.0 - y_gt))
        fp = torch.sum(y_pr) - tp
        fn = torch.sum(y_gt) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        return self.get_scores()



if __name__ == '__main__':
    pass
    