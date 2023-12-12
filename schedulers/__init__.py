import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, CosineAnnealingLR, 
    MultiStepLR,
    LambdaLR,
    OneCycleLR,
    ConstantLR,
    #ReduceLROnPlateau,
    #LinearLR,
    #ExponentialLR
    )
# polylr
from .polylr import PolyLR,PolyLR_WWP

needs_total_iters = {
    'cosine-anneal' : 'T_max',
    'onecyclelr' : ' total_steps',
}

scheduler_mapping = {
    'polylr' : PolyLR,
    'polylrwwp' : PolyLR_WWP,
    'multi-steplr' : MultiStepLR,
    'cosine-anneal' : CosineAnnealingLR,
    'cosine-anneal-wr' : CosineAnnealingWarmRestarts,
    'lambdalr' : LambdaLR,
    'onecyclelr' : OneCycleLR,
    'constantlr' : ConstantLR,
    #'linearlr' : LinearLR,
    #'exponentiallr' : ExponentialLR
}
# schedulers
def get_scheduler(name, optimizer, **kwargs):
    name = name.lower()
    if name not in scheduler_mapping.keys():
        raise ValueError(f'scheduler {name} is not implemented!!!')
    
    return scheduler_mapping[name](optimizer=optimizer, **kwargs)

class AutoScheduler:

    def __init__(
            self,
            name,
            optimizer,
            data_loader = None,
            total_epochs = None,
            iters_per_epoch = None,
            mode = 'per_epoch', #per_epoch #per_iter
            **kwargs
            ):
        
        assert isinstance(total_epochs,int)
        if data_loader is not None:
            self.iters_per_epoch = len(data_loader)
        elif isinstance(iters_per_epoch,int):
            self.iters_per_epoch = iters_per_epoch
        else:
            raise ValueError(f'please correctly provide data_loader or iter_per_epoch')
        
        self.total_iters = total_epochs * self.iters_per_epoch 
        self.iter_counter = 0

        self.set_mode(mode)

        if name.startswith('polylr'):
            kwargs['epochs'] = total_epochs if mode == 'per_epoch' else self.total_iters
        if name in needs_total_iters.keys():
            kwargs[needs_total_iters[name]] = self.total_iters

        self.scheduler = get_scheduler(name=name,optimizer=optimizer,**kwargs)
    
    def set_mode(self,mode):
        assert mode in ['per_epoch','per_iter']
        self.mode = mode
        self.set_stepsize()

    def set_stepsize(self):
        if self.mode == 'per_epoch':
            self.stepsize = self.iters_per_epoch
        else:
            self.stepsize = 1

    def step(self):
        self.iter_counter += 1

        if self.iter_counter > self.total_iters:
            return
        
        if self.iter_counter % self.stepsize == 0:
            self.scheduler.step()

