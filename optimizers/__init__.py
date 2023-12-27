from torch.optim import Optimizer,Adam,RMSprop,SGD

from .over9000 import *
from .adamw import AdamW

optimizer_mapping = {
    'adam' : Adam,
    'adamw' : AdamW,
    'rmsprop' : RMSprop,
    'sgd' : SGD,
    'radam' : RAdam,
    'plain_radam' : PlainRAdam,
    'novograd' : Novograd,
    'ranger' : Ranger,
    'ralamb' : Ralamb,
    ################
    'rangerlars' : RangerLars, ##Known as Over9000 optimizer
    'over9000' : RangerLars,
    ################
        # 'lookahead' : Lookahead,
    'lookahead_adam' : LookaheadAdam,
    'diffgrad' : DiffGrad,
    'adamod' : AdaMod,
    'madam' : Madam,
    'apollo' : Apollo,
    'adabelief' : AdaBelief,
    'madgrad' : MADGRAD,
    'adan' : Adan

}

def get_optimizer(name:str ,params,lookAhead=False,lookAhead_alpha=0.5, lookAhead_k=6,lookAhead_pullback_momentum="none",*args,**kwargs) -> Optimizer:
    """
    This function returns the optimizer given its name

    @param name: name of the optimzer
    @param params: parameters of the model that need to be optimzed 
    @param *args & **kwargs: parameters for the optimizer

    @type name:str
    @type params: list or dict
    
    @return: torch.optim.Optimzer

    """
    name = name.lower()
    if name not in optimizer_mapping.keys():
        raise ValueError('Optimizer {} not an option'.format(name))

    if lookAhead:
        assert name != 'adan', "lookahead adan is not supported"
        return Lookahead(optimizer_mapping[name](params,*args,**kwargs),alpha=lookAhead_alpha, k=lookAhead_k,pullback_momentum=lookAhead_pullback_momentum)
    return optimizer_mapping[name](params,*args,**kwargs)