import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, 
    CosineAnnealingLR, 
    MultiStepLR,
    LambdaLR,
    OneCycleLR,
    ConstantLR,
    ReduceLROnPlateau,
    LinearLR,
    ExponentialLR,
    MultiplicativeLR,
    StepLR,
    PolynomialLR,
    ChainedScheduler,
    SequentialLR,
    CyclicLR,
    LRScheduler
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
    'linearlr' : LinearLR,
    'exponentiallr' : ExponentialLR,
    'reducelronplateau':ReduceLROnPlateau,
    'multiplicativelr':MultiplicativeLR,
    'steplr':StepLR,
    'polynomiallr':PolynomialLR,
    'chainedscheduler':ChainedScheduler,
    'sequentiallr':SequentialLR,
    'cycliclr':CyclicLR,
}


def get_scheduler(name :str, optimizer :torch.optim.Optimizer, **kwargs) -> LRScheduler:
    r"""This function returns the scheduler given its name

    Parameters
    ----------
    name: str
        name of the scheduler
    optimizer: torch.optim.Optimizer
        Optimizer to schedule
    args & kwargs: _
        parameters for the scheduler

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler

    """
    name = name.lower()
    if name not in scheduler_mapping.keys():
        raise ValueError(f'scheduler {name} is not implemented!!!')
    
    return scheduler_mapping[name](optimizer=optimizer, **kwargs)

class AutoScheduler:
    """
    This class automate the step of the schechduler, the user will call scheduler.step on every iteration and the scheduler once required
    """
    def __init__(
            self,
            name :str,
            optimizer :torch.optim.Optimizer,
            data_loader = None,
            total_epochs :int = None,
            iters_per_epoch :int|None = None,
            mode = 'per_epoch', 
            **kwargs
            ):
        
        """
            This function returns the scheduler given its name

            @param name: name of the scheduler
            @param params: Optimeizer to schedule
            @param data_loader: (optional) used to find the iters_per_epoch
            @param total_epochs: (optional)
            @param iters_per_epoch: (optional)
            @param kwargs:named parameters for the sheduler

            @type name:str
            @type optimizer: torch.optim.Optimizer


        """
        
        assert isinstance(total_epochs,int), "the total_epochs must be an int"
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
        """
        sets mode of the iter_counter
        @param mode: can be only 'per_epoch' or 'per_iter'
        """
        assert mode in ['per_epoch','per_iter']
        self.mode = mode
        self.set_stepsize()

    def set_stepsize(self):
        """define stepsize"""
        if self.mode == 'per_epoch':
            self.stepsize = self.iters_per_epoch
        else:
            self.stepsize = 1

    def step(self):
        """
            done in each iteration, steps only when iter_counter == self.stepsize 
        """
        self.iter_counter += 1

        if self.iter_counter > self.total_iters:
            return
        
        if self.iter_counter % self.stepsize == 0:
            self.scheduler.step()

