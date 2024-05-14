
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.utils import functional as F_
#from lovasz import LovaszLoss,BinaryLovaszLoss

from segmentation_models_pytorch.utils.losses import JaccardLoss,DiceLoss


from segmentation_models_pytorch.losses import DiceLoss,SoftBCEWithLogitsLoss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss









def _soft_neg_loss(pred, gt,eps=0.2):
  ''' Modified focal loss to soft focal loss for noisy labels :)
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  mask = gt.ge(0).float() * gt.le(1).float()

  pos_inds = gt.ge(1 - eps).float() * mask
  neg_inds = gt.lt(1 - eps).float() * mask

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss










class KeypointFocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  __name__ = 'keypoint_focal_loss'
  def __init__(self):
    super(KeypointFocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    out = out.clamp_max(0.999).clamp_min(0.0001)
    return self.neg_loss(out, target)







class SmoothKeypointFocalLoss(nn.Module):
  '''nn.Module warpper for smooth focal loss'''
  __name__ = 'keypoint_smooth_focal_loss'
  def __init__(self,eps):
    super(SmoothKeypointFocalLoss, self).__init__()
    self.eps = eps
    self.soft_neg_loss = _soft_neg_loss

  def forward(self, out, target):
    out = out.clamp_max(0.999).clamp_min(0.0001)
    return self.soft_neg_loss(out, target,self.eps)


class DiceBCELoss(nn.Module):

  def __init__(self,alpha_beta=(1.0,1.0),mode='binary',bce_pos_weight=1.0):
    super().__init__()
    
    self.dice = DiceLoss(mode=mode,from_logits=True,smooth=0,eps=1e-7)
    self.bce = SoftBCEWithLogitsLoss(pos_weight=bce_pos_weight)

    self.alpha ,self.beta = alpha_beta
  
  def forward(self,pred,gt):

    dice_loss = self.dice(pred,gt)
    bce_loss = self.bce(pred,gt)

    loss = self.alpha * dice_loss + self.beta * bce_loss

    return loss
 


class MaskedBCELoss(nn.Module):

  def __init__(self,device='cpu'):
    super().__init__()
    self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
    self.device = device 
  
  def forward(self,pred,gt,mask):
    B,C,H,W = pred.shape
    loss = torch.zeros_like(pred, requires_grad=True).to(device=self.device)

    pred = torch.masked_select(pred,mask)
    gt = torch.masked_select(gt,mask)

    if pred.shape[0] != 0:
      bce_loss = self.bce_criterion(pred,gt)
      loss = loss.masked_scatter(mask,bce_loss)

    loss = loss.view(B,C,-1)
    mask = mask.view(B,C,-1)
    counts = mask.float().sum(dim=-1).clamp_min(1.0)

    loss = loss.sum(dim=-1) / counts
    loss = loss.mean()

    return loss















    



class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
    



class FocalLoss(nn.Module):
    __name__='focal_loss'
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



class BCELoss(nn.Module):
    __name__='bce_loss'
    def __init__(self,logits=True):
        super().__init__()
        self.logits = logits
        self.bce = nn.BCELoss(reduction='mean') if(not(logits)) else nn.BCEWithLogitsLoss(reduction='mean')
    def forward(self,y_pr,y_gt):
        bce = self.bce(y_pr,y_gt)
        return bce


class CELoss(nn.Module):
    __name__='ce_loss'
    def __init__(self,weights=[1.,5.,5.,1.]):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean',weight=weights)
    def forward(self,y_pr,y_gt):
        ce = self.ce(y_pr,y_gt)
        return ce
    



class DiceBCELoss(nn.Module):
    __name__='dice_bce_loss'
    def __init__(self,weights=[1.,1.],activation='sigmoid'):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss(activation=activation)
        self.w1,self.w2=weights[0],weights[1]
    def forward(self,y_pr,y_gt):
        return self.w2*self.bce(y_pr,y_gt) + self.w1*self.dice(y_pr, y_gt)
    



class JaccardBCELoss(nn.Module):
    __name__='jaccard_bce_loss'
    def __init__(self,weights=[1.,1.],activation='sigmoid'):
        super().__init__()
        self.bce = BCELoss()
        self.jaccard = JaccardLoss(activation=activation)
        self.w1,self.w2=weights[0],weights[1]
    def forward(self,y_pr,y_gt):
        return self.w2*self.bce(y_pr,y_gt) + self.w1*self.jaccard(y_pr, y_gt)
    



class DiceFocalLoss(nn.Module):
    __name__='dice_focal_loss'
    def __init__(self,weights=[1.,1.],activation='sigmoid'):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss(activation=activation)
        self.w1,self.w2=weights[0],weights[1]
    def forward(self,y_pr,y_gt):
        return self.w2*self.focal(y_pr,y_gt) + self.w1*self.dice(y_pr, y_gt)
    


class JaccardFocalLoss(nn.Module):
    __name__='jaccard_focal_loss'
    def __init__(self,weights=[1.,1.],activation='sigmoid'):
        super().__init__()
        self.focal = FocalLoss()
        self.jaccard = JaccardLoss(activation=activation)
        self.w1,self.w2=weights[0],weights[1]
    def forward(self,y_pr,y_gt):
        return self.w2*self.focal(y_pr,y_gt) + self.w1*self.jaccard(y_pr, y_gt)



class DiceFocalBCELoss(nn.Module):
    __name__='dice_focal_bce_loss'
    def __init__(self,weights=[1.,1.,1.],activation='sigmoid'):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss(activation=activation)
        self.bce = BCELoss()
        self.w1,self.w2,self.w3=weights[0],weights[1],weights[2]
    def forward(self,y_pr,y_gt):
        return self.w1*self.dice(y_pr, y_gt) + self.w2*self.focal(y_pr, y_gt) + self.w3*self.bce(y_pr, y_gt)



class JaccardFocalBCELoss(nn.Module):
    __name__='jaccard_focal_bce_loss'
    def __init__(self,weights=[1.,1.,1.],activation='sigmoid'):
        super().__init__()
        self.focal = FocalLoss()
        self.jaccard = JaccardLoss(activation=activation)
        self.bce = BCELoss()
        self.w1,self.w2,self.w3=weights[0],weights[1],weights[2]
    def forward(self,y_pr,y_gt):
        return self.w1*self.jaccard(y_pr, y_gt) + self.w2*self.focal(y_pr, y_gt) + self.w3*self.bce(y_pr, y_gt)




class MultiClass_loss(nn.Module) :
    __name__='multiclass_loss'
    def __init__(self,sloss='Dice',focal=False,entropy=1.,balance = [0.5,0.5],weights = [1.,5.,5.,1.],activation='sigmoid',normalize=True):
        super().__init__()
        #main_loss = []
        self.sloss = sloss
        self.focal = focal
        self.entropy = entropy
        self.act = activation
        self.w1=balance
        self.w2=weights
        self.norm = normalize
        self.loss_list = [self.get() for i in range(len(weights))]
        if(self.entropy>0.):self.ce_loss = nn.BCEWithLogitsLoss(reduction='mean')                                        
    def get(self):
        lname = self.sloss.lower()
        if(self.focal):
            lname += '+focal'
        if(self.entropy):
             lname += '+bce'
        return get_criterion(name=lname,activation = self.act,w =self.w1)[0]
    def forward(self,y_pr,y_gt):
        loss = 0
        assert (len(self.w2) == y_pr.size(1) == y_gt.size(1)),'weights and channel numbers dont match!'
        for i,(wi,lf) in enumerate(list(zip(self.w2,self.loss_list))):
            #loss += wi * lf(y_pr[:,i,...].unsqueeze(1),y_gt[:,i,...].unsqueeze(1))
            loss += wi * lf(y_pr[:,i,...],y_gt[:,i,...])
        #if(self.entropy>0):
           # loss += self.entropy * self.ce_loss(y_pr,y_gt)
        if(self.norm):
                loss /= (np.sum(self.w2))#+self.entropy)
        return loss
        
    

class CombinedLoss(nn.Module):
    def __init__(self,name = 'dice+bce',
                 weights = [0.5,0.5],
                 activation = 'sigmoid',
                 channel_weights = [1.,1.,1.],
                 losses_per_channel = ['dice','bce','jaccard','focal'],
                 normalize = True):
        super().__init__() 
        self.mapping = {
            'dice' : DiceLoss(activation = None),
            'jaccard': JaccardLoss(activation = None),
            'bce'  : BCELoss(),
            'focal' : FocalLoss()
            }
        self.weights = list( weights / np.sum(weights))
        self.activation = Activation(activation)
        self.need_activation = ['bce','focal']
        self.names = name.split('+')
        self.channel_losses = losses_per_channel
        self.channel_weights = channel_weights
        self.__name__ = f'{name.lower()}_loss'
        self.norm = normalize
        
    def forward(self,y_pred,y_target):     
        loss = 0
        num_channels = y_target.size(1)  
        act_y_pred = self.activation(y_pred)     
        for w,name in enumerate(self.names):            
            criterion = self.mapping[name]
            pred_tensor = y_pred if(name in self.need_activation) else act_y_pred          
            if(name in self.channel_losses):            
                for i in range(num_channels):  
                    loss += self.channel_weights[i] * self.weights[w] * criterion(pred_tensor[:,i,...],y_target[:,i,...])
            else:
                loss += self.weights[w] * criterion(pred_tensor,y_target)               
        return loss / sum(self.channel_weights) if(self.norm) else loss
    
    

class DiceLoss2(DiceLoss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F_.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=0.3,
            ignore_channels=self.ignore_channels,
        )

# def get_criterion(name='jaccard',activation='sigmoid',w=[1.,1.,1.]):
#     if(name=='dice'):
#         #from segmentation_models_pytorch.utils.losses import DiceLoss
#         return DiceLoss(activation=activation),1
#     elif(name=='focal'):
#         from Training.losses import FocalLoss
#         return FocalLoss(),1#activation=activation)
#     elif(name == 'jaccard'):
#         from segmentation_models_pytorch.utils.losses import JaccardLoss
#         return JaccardLoss(activation=activation),1
#     elif(name =='bce'):
#         from Training.losses import BCELoss
#         return BCELoss(),1#activation=activation)
#     elif(name=='blovasz'):
#         from Training.lovasz import BinaryLovaszLoss
#         return BinaryLovaszLoss(activation=activation),1
#     elif(name =='lovasz'):
#         from Training.lovasz import LovaszLoss
#         return LovaszLoss(activation=activation),1
#     elif(name == 'dice+bce'):
#         from Training.losses import DiceBCELoss
#         return DiceBCELoss(weights=w[:2]),w[0]+w[1]
#     elif(name == 'jaccard+bce'):
#         from Training.losses import JaccardBCELoss
#         return JaccardBCELoss(weights=w[:2]),w[0]+w[1]
#     elif(name =='dice+focal'):
#         from Training.losses import DiceFocalLoss
#         return DiceFocalLoss(weights=w[:2]),w[0]+w[1]
#     elif(name == 'jaccard+focal'):
#         from Training.losses import JaccardFocalLoss
#         return JaccardFocalLoss(weights=w[:2]),w[0]+w[1]
#     elif(name == 'jaccard+focal+bce'):
#         from Training.losses import JaccardFocalBCELoss
#         return JaccardFocalBCELoss(weights=w),w[0]+w[1]+w[2]
#     elif(name == 'dice+focal+bce'):
#         from Training.losses import DiceFocalBCELoss
#         return DiceFocalBCELoss(weights=w),w[0]+w[1]+w[2]
#     else:
#         raise NotImplementedError()
        
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size(0)
    index = torch.randperm(bs)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index], lam

class Mixup_Criterion(nn.Module):
    def __init__(self,criterion:nn.Module):
        super().__init__()
        self.criterion = criterion
        self.__name__ = 'mixup_' + self.criterion.__name__
    def forward(self,y_gt,y_gti,y_pr,lam):
        return lam * self.criterion(y_pr,y_gt) + (1 - lam) * self.criterion(y_pr,y_gti)
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha = 1.0):
    indices = torch.randperm(x.size(0))
    shuffled_y = y[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
    target = y
    target[:, :, bbx1:bbx2, bby1:bby2] = target[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    mixed_x = x
    return mixed_x, y , shuffled_y, lam,target
class CutMix_Criterion(nn.Module):
    def __init__(self,criterion:nn.Module):
        super().__init__()
        self.criterion = criterion
        self.__name__ = 'cutmix_' + self.criterion.__name__
    def forward(self,y_gt,y_gti,y_pr,lam):
        return lam * self.criterion(y_pr,y_gt) + (1. - lam) * self.criterion(y_pr,y_gti)
class CutMix_Criterion2(nn.Module):
    def __init__(self,criterion:nn.Module):
        super().__init__()
        self.criterion = criterion
        self.__name__ = 'cutmix_' + self.criterion.__name__
    def forward(self,y_gt_new,y_pr,lam):
        return self.criterion(y_pr,y_gt_new)
'''
class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=0.5, activation=self.activation)
class Mixed_loss(nn.Module):
    def __init__(self,losses={'jaccard_loss' : {'multiplier' : 2.,'weights':[1.,2.,3.]},
                              'dice_loss' : {'multiplier' : 1.,'weights': [1.]} },reduce = 'sum'):
        super().__init__()
        self.mapping = {'jaccard_loss' : JaccardLoss(),
                        'dice_loss' : DiceLoss(),
                        'focal_loss' : FocalLoss(),
                        'bce_loss' : BCELoss()}
        self.overall_loss = 0
        self.losses = losses
        self.need_sigmoid = {'jaccard_loss','dice_loss','bce_loss'}
        self.reduce = reduce
        self.__name__= 'loss:'
        for l in self.losses:
            k=self.losses[l]['multiplier']
            self.__name__ += f'{k}*{l}_'
    def forward(self,y_pred,y_gt):
        sigmoid_y_pred = torch.sigmoid(y_pred)
        for loss in self.losses:
            this_loss = 0
            channel_losses = []
            loss_name = loss
            loss_func = self.mapping[loss_name]
            weights = self.losses[loss_name]['weights']
            multiplier = self.losses[loss_name]['multiplier']
            channels = y_gt.size(1)
            needs_sigmoid = True if loss_name in self.need_sigmoid else False
            per_channel = True if len(weights)>1 else False
            if(per_channel == True) : 
                assert ( len(channels) == len(weights))
                for i,w in enumerate(weights):
                    channel_losses.append(w*loss_func(sigmoid_y_pred[:,i,...] if needs_sigmoid else y_pred[:,i,...],
                                                      y_gt[:,i,...]))        
                if(self.reduce == 'avg'):
                    this_loss = sum(channel_losses)/channels
                elif(self.reduce == 'sum'):
                    this_loss = sum(channel_losses)
                else:
                    raise NotImplementedError()
            else : this_loss = loss_func(sigmoid_y_pred if needs_sigmoid else y_pred,
                                                      y_gt)
            self.overall_loss += multiplier * this_loss
        return self.overall_loss
'''




# if __name__ == '__main__':

#   #"""
#   x2 = torch.tensor(
#     [
#       [
#         [[0,0.2,0.3],
#           [0.2,0.1,0.1],
#           [0.3,0.5,0.5]]
#       ]
#     ])

#   x = torch.tensor(
#     [
#       [
#         [[0,1.0,1.0],
#         [1.0,1.0,0.0],
#         [1.0,0.0,0.0]]
#       ]
#     ])
  
  
#   mask = (x != 0).bool()
#   crit = MaskedBCELoss()
#   print(x.shape)
#   crit(x2,x,mask)
#   print(x.shape)

#   #pos = x.gt(2)
#   #print(pos.shape)
#   #print(pos)
#   #"""
  