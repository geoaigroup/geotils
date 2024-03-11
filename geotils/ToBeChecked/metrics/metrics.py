import torch
from icecream import ic
import geopandas as gpd
from tmp import convert_polygon_to_mask_batch,binary_mask_to_polygon
from segmentation_models_pytorch.utils.metrics import Accuracy,Recall
from segmentation_models_pytorch.utils.base import Metric


def _threshold(x, threshold=0.5):
    """
    Thresholds the input tensor based on a specified threshold value.

    Args:
        x (torch.Tensor): Input tensor to be thresholded.
        threshold (float, optional): The threshold value. If None, no threshold is applied.
                                     Default is 0.5.

    Returns:
        torch.Tensor: Thresholded tensor where values greater than or equal to the threshold
                      are set to 1, and values below the threshold remain unchanged.

    Note:
        - The input tensor `x` and the returned tensor will have the same dtype.
        - If `threshold` is None, the input tensor `x` is returned unchanged.
    """
    if threshold is not None:
        return (x >= threshold).type(x.dtype)
    else:
        return x

class DiceScore:
    """
    Initializes an instance of the DiceScore class.

    Args:
        threshold (float, optional): Threshold value for thresholding predictions. Default is 0.5.
        ignore_index (int, optional): Index to ignore in ground truth. Default is -100.
    """
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-6
        self.dice_score_sum = 0.0
        self.weights_sum = 0.0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):
        """
        Calculates the Dice Score between predicted and ground truth tensors.

        Args:
            y_pr (torch.Tensor): Predicted tensor.
            y_gt (torch.Tensor): Ground truth tensor.
            thresh_gt (bool, optional): Whether to threshold the ground truth tensor. Default is False.

        Returns:
            torch.Tensor: Dice score between the predicted and ground truth tensors.

        Note:
            - This function accumulates dice scores and weights over multiple calls.
            - The final dice score is computed as the sum of dice scores divided by the sum of weights.
        """
        b,c,_,_ = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)
        
        if self.ignore_index is not None:
            y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * y_nindex
            y_pr = y_pr * y_nindex

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
    """
    Initializes an instance of the IoUScore class.

    Args:
        threshold (float, optional): Threshold value for thresholding predictions. Default is 0.5.
        ignore_index (int, optional): Index to ignore in ground truth. Default is -100.
    """
    def __init__(self,threshold = 0.5,ignore_index=-100):
        self.thresh = threshold
        self.eps = 1e-6
        self.iou_score_sum = 0.0
        self.weights_sum = 0.0
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, y_pr,y_gt,thresh_gt=False):
        """
        Calculates the Intersection over Union (IoU) score between predicted and ground truth tensors.

        Args:
            y_pr (torch.Tensor): Predicted tensor.
            y_gt (torch.Tensor): Ground truth tensor.
            thresh_gt (bool, optional): Whether to threshold the ground truth tensor. Default is False.

        Returns:
            torch.Tensor: IoU score between the predicted and ground truth tensors.

        Note:
            - This function accumulates IoU scores and weights over multiple calls.
            - The final IoU score is computed as the sum of IoU scores divided by the sum of weights.
        """
        b,c,_,_ = y_pr.shape
        y_pr = _threshold(y_pr,self.thresh)
        if thresh_gt:
            y_gt = _threshold(y_gt,self.thresh)
        
        if self.ignore_index is not None:
            y_nindex = (y_gt != self.ignore_index).type(y_gt.dtype)
            y_gt = y_gt * y_nindex
            y_pr = y_pr * y_nindex

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
