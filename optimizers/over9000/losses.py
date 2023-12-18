
import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    __name__='focal_loss'
    def __init__(self, alpha=1, gamma=2, logits=True, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        assert reduction in ['sum','mean','none']
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss

class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        if 'pos_weight' in kwargs.keys():
            kwargs['pos_weight'] = torch.tensor(kwargs['pos_weight'],requires_grad=False)
        super().__init__(**kwargs)

def get_cls_loss(**kwargs):
    cls_loss_name = kwargs['name']
    params = kwargs['params']
    if cls_loss_name == 'bce':
        loss_fnc = BCEWithLogitsLoss
    elif cls_loss_name =='focal':
        loss_fnc = FocalLoss
    else:
        raise NotImplementedError(f'Loss {cls_loss_name} not implemented!!!')
  
    return cls_loss_name,loss_fnc(**params)    

def get_reg_loss(**kwargs):
    reg_loss_name = kwargs['name']
    params = kwargs['params']

    if reg_loss_name == 'l1':
        loss_fnc = nn.L1Loss
    elif reg_loss_name in ['mse','rmse']:
        loss_fnc = nn.MSELoss
    else:
        raise NotImplementedError(f'Loss {reg_loss_name} not implemented!!!')
  
    return reg_loss_name,loss_fnc(**params)   
        
class DualClsRegLoss(nn.Module):
    
    def __init__(
            self,
            cls_loss_kwargs,
            reg_loss_kwargs,
            eliminate_neg_cls_targets=False,
            w1=1.0,
            w2=1.0
            ):
        super().__init__()

        self.cls_criterion_name,self.cls_criterion = get_cls_loss(**cls_loss_kwargs)
        self.reg_criterion_name,self.reg_criterion = get_reg_loss(**reg_loss_kwargs)
        self.enct = eliminate_neg_cls_targets
        self.register_buffer('w1',torch.tensor(w1))
        self.register_buffer('w2',torch.tensor(w2))

    def register_losses(self,reg_loss,cls_loss):
        self.registered_losses = {
            self.reg_criterion_name : reg_loss,
            self.cls_criterion_name : cls_loss
        }
        
    def get_registered_losses(self):
        return self.registered_losses 

    def forward(self,cls_gt,cls_pred,reg_gt,reg_pred):


      loss = 0.0
      
      cls_loss = self.cls_criterion(cls_pred,cls_gt)
      loss = loss + self.w1 * cls_loss

      apply_reg = True
      reg_loss = 0.0
      
      if self.enct:
          mask = reg_gt != 0
          apply_reg = mask.any().item()
          reg_pred = reg_pred[mask]
          reg_gt = reg_gt[mask]
      
      if apply_reg:
        
        reg_loss = self.reg_criterion(reg_pred,reg_gt)
        
        if self.reg_criterion_name == 'rmse':
            reg_loss = torch.sqrt(reg_loss)
        
        loss = loss + self.w2 * reg_loss

      self.register_losses(reg_loss=reg_loss.item() if apply_reg else reg_loss,cls_loss=cls_loss.item())
      return loss


class TripleClsLoss(nn.Module):
    
    def __init__(
            self,
            cls_loss_kwargs,
            base_loss_kwargs,
            shift_loss_kwargs,
            eliminate_neg_cls_targets=False,
            w1=1.0,
            w2=1.0,
            w3=1.0
            ):
        super().__init__()

        self.cls_criterion_name,self.cls_criterion = get_cls_loss(**cls_loss_kwargs)
        self.base_criterion = nn.CrossEntropyLoss(**base_loss_kwargs)
        self.shift_criterion = nn.CrossEntropyLoss(**shift_loss_kwargs)

        self.enct = eliminate_neg_cls_targets
        self.register_buffer('w1',torch.tensor(w1))
        self.register_buffer('w2',torch.tensor(w2))
        self.register_buffer('w3',torch.tensor(w3))

    def register_losses(self,base_loss,shift_loss,cls_loss):
        self.registered_losses = {
            'base_ce_loss' : base_loss,
            'shift_ce_loss' : shift_loss,
            self.cls_criterion_name : cls_loss
        }
        
    def get_registered_losses(self):
        return self.registered_losses 

    def forward(self,cls_gt,cls_pred,base_gt,base_pred,shift_gt,shift_pred):


      loss = 0.0
      
      cls_loss = self.cls_criterion(cls_pred,cls_gt)
      loss = loss + self.w1 * cls_loss

      apply_reg = True
      base_loss = 0.0
      shift_loss = 0.0
      
      if self.enct:
          mask = cls_gt != 0
          mask = mask[:,0]
          apply_reg = mask.any().item()

          base_pred = base_pred[mask,:]
          base_gt = base_gt[mask]

          shift_pred = shift_pred[mask,:]
          shift_gt = shift_gt[mask]
      
      if apply_reg:
        
        base_loss = self.base_criterion(base_pred,base_gt)
        shift_loss = self.shift_criterion(shift_pred,shift_gt)
        
        loss = loss + self.w2 * base_loss + self.w3 * shift_loss

      self.register_losses(
          base_loss=base_loss.item() if apply_reg else base_loss,
          shift_loss=shift_loss.item() if apply_reg else shift_loss,
          cls_loss=cls_loss.item()
          )
      
      return loss







if __name__ == '__main__':
    
    cls_gt = torch.tensor([1,0,1,0,1,1,0,0,1,1,1,0,0]).float().view(-1,1)
    cls_pred = torch.tensor([1.22,-5,0,4,-0.4,2,-3,1,3,4,1.5,0.5,-5]).view(-1,1)

    reg_gt = torch.tensor([0,0,1,2,3,4,5,6,7,8,9,10,11]).float().view(-1,1)
    reg_pred = torch.tensor([0,0,1,2,3,4,5,6,7,8,9,10,11]).view(-1,1) * 0.8
    print(torch.sigmoid(cls_pred))

    loss_dict = dict(
        cls = dict(
          name = 'bce',
          params = dict(
                    pos_weight=1.0,
                    reduction = 'mean')
        ),
        
        reg = dict(
          name = 'rmse',
          params = dict(reduction='mean')
        )
      )
    
    criterion = DualClsRegLoss(cls_loss_kwargs=loss_dict['cls'],reg_loss_kwargs=loss_dict['reg'],eliminate_neg_cls_targets=True)
    print(criterion)
  
    with torch.no_grad():
      loss = criterion(cls_gt,cls_pred,reg_gt,reg_pred)
      print(loss)
    print(criterion.get_registered_losses())