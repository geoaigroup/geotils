import numpy as np
import geopandas as gpd
import shapely.geometry as sg
from shapely import affinity
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw
from skimage import measure
import rasterio
from skimage.io import imread,imsave
from segmentation_models_pytorch.utils.metrics import Accuracy,Recall
from segmentation_models_pytorch.utils.base import Metric
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def save_shp(pred_mask,name,output_dir,image_shape):
    pred_tile = []
    mask_tile = np.zeros(image_shape)
    msk = pred_mask.int()
    msk = msk.cpu().numpy()
    for i in range(msk.shape[0]):
        batch = msk[i]
        for b in range(batch.shape[0]):
            mask_tile = mask_tile + batch[b]
            pred_tile.append(batch[b])

    polys=[]
    for k in pred_tile:
        if not np.any(k):
            continue
        polys.append(binary_mask_to_polygon(k))

    gdf = gpd.GeoDataFrame({
                        'ImageId':name,
                        'geometry':polys
                        })
    gdf.to_file(f"{output_dir}/{name}/{name}.shp")
    
def convert_polygon_to_mask_batch(geo):
    gtmask=[]
    width = 512
    height = 512
    for orig_row in geo:
        polygon=[]
        if orig_row.geom_type=="Polygon":
            for point in orig_row.exterior.coords:
                polygon.append(point)
            img = Image.new('L', (width, height),0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            img=np.array(img)
            gtmask.append(img)
        else:
            for x in orig_row.geoms:
                for point in x.exterior.coords:
                    polygon.append(point)
            img = Image.new('L', (width, height),0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            img=np.array(img)
            gtmask.append(img)
    return gtmask

def binary_mask_to_polygon(binary_mask):
    # Find contours in the binary mask
    contours = measure.find_contours(binary_mask, 0.5)
    # Get the largest contour (in case there are multiple objects)
    max_contour = max(contours, key=len)

    # Convert the contour points to a polygon (list of (x, y) coordinates)
    polygon = Polygon([(int(point[1]), int(point[0])) for point in max_contour])

    return polygon

def get_micro_metrics(metrics = [['iou','fscore']]*4,
                      threshs = [0.5]*4,
                      channels = [None,0,1,2],
                      names = ['','Buildings','Borders','Spacing'],
                      num_cls = 1
                      ):
    micro_metrics_funcs = []
    
    for i in range(num_cls):
        metric_funcs = get_metrics(metrics[i],thresh = threshs[i])
        
        for metric_func in metric_funcs:
            micro_metrics_funcs.append(Micro_Metric(metric_func,
                                                    channels[i],
                                                    names[i]))
    return micro_metrics_funcs

def get_metrics(metrics = ['iou','fscore'],thresh=0.5):
    metric_funcs =[]
    if (metrics == []):
        raise ValueError
    else:
        for metric in metrics:
            
            if(metric =='iou_score' or metric == 'iou'):
                #IoU.__name__='iou'
                metric_funcs.append(IoU(threshold=thresh))
            elif(metric =='fscore'):
                Fscore.__name__='fscore'
                metric_funcs.append(Fscore(threshold=thresh))
            elif(metric =='accuracy'):
                Accuracy.__name__='accuracy'
                metric_funcs.append(Accuracy)
            elif(metric  == 'recall'):
                Recall.__name__='recall'
                metric_funcs.append(Recall)
            elif(metric == "micro_iou"):
                metric_funcs.append(MicroIoU(threshold = thresh))
            elif(metric == "micro_fscore"):
                metric_funcs.append(MicroFscore(threshold = thresh))
            else:
                raise NotImplementedError(f'the metric {metric} is not implemented')
    return metric_funcs

class Micro_Metric(nn.Module):
    
    def __init__(self,metric,channel = None,name = ''):
        super(Micro_Metric,self).__init__()
        self.metric = metric
        self.ch = channel
        self.__name__ =  name +'_'+ self.metric.__name__
        
    def reset(self):
        self.metric.reset()
        
    def forward(self,y_pred,y_target):
        
        if(self.ch is None):
            return self.metric(y_pred,y_target)
        else:
            return self.metric(y_pred[:,self.ch,...],y_target[:,self.ch,...])

class MicroIoU(Metric):
    __name__ = "micro_iou"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.intersection = 0.
        self.union = 0.
        self.threshold = threshold

    def reset(self):
        self.intersection = 0.
        self.union = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()

        intersection = (prediction * target).sum()
        union = (prediction + target).sum() - intersection

        self.intersection += intersection.detach()
        self.union += union.detach()

        score = (self.intersection + self.eps) / (self.union + self.eps)
        return score
    
class IoU(Metric):
    __name__ = "iou_score"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.threshold = threshold

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()

        intersection = (prediction * target).sum()
        union = (prediction + target).sum() - intersection

        score = (intersection + self.eps) / (union + self.eps)
        return score
    
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

class MicroFscore(Metric):
    __name__ = 'micro_fscore'
    def __init__(self,threshold = 0.5, eps=1e-5,beta = 1):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.beta = 1
        self.tp,self.fp,self.fn = 0,0,0
    def reset(self):
        self.tp,self.fp,self.fn = 0,0,0
    @torch.no_grad()
    def __call__(self,prediction,target):
        prediction = (prediction > self.threshold).float()
        tp = (prediction * target).sum()
        self.tp += tp
        self.fp += (prediction.sum() - tp)
        self.fn += (target.sum() -  tp)
        score = ((1 + self.beta ** 2) * self.tp + self.eps) \
            / ((1 + self.beta ** 2) * self.tp + self.beta ** 2 * self.fn + self.fp + self.eps)

        return score
    
def show_mask(mask,ax,random_color=False,s=""):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            if s=="gt":
                color = np.array([30/255, 144/255, 255/255, 0.5])
            elif s=="whu":
               color = np.array([0/255, 255/255, 0/255, 0.4])
            elif s=="pred":
                color = np.array([255/255, 0/255, 0/255, 0.5])
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
            

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))