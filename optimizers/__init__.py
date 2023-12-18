import torch
import math
from torch.optim import Adam,RMSprop,SGD

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
    'lookahead' : Lookahead,
    'lookahead_adam' : LookaheadAdam,
    'diffgrad' : DiffGrad,
    'adamod' : AdaMod,
    'madam' : Madam,
    'apollo' : Apollo,
    'adabelief' : AdaBelief,
    'madgrad' : MADGRAD,
    'adan' : Adan

}
#optimizers
def get_optimizer(name ,params,*args,**kwargs):
    name = name.lower()
    if name not in optimizer_mapping.keys():
        raise ValueError('Optimizer {} not an option'.format(name))
    return optimizer_mapping[name](params,*args,**kwargs)