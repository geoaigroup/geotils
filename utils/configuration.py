#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:50:11 2021

@author: hasan
"""
#configs
from yacs.config import CfgNode as CN
import yaml

cfg = CN()

cfg.save_path = 'content/try2'
cfg.resume = False
cfg.seed = 911
#training and validation
cfg.trainval = CN()
cfg.trainval.epochs = 300
cfg.trainval.lr = 0.0001
cfg.trainval.batch_size = 6
cfg.trainval.val_batch_size = 1
cfg.trainval.accum_steps = 1
cfg.trainval.val_period = 1
cfg.trainval.fold = 0
cfg.trainval.amp = True
cfg.trainval.max_grad_norm = 1.0
cfg.trainval.transform_number = 3
cfg.trainval.device = 'cuda'
#model
cfg.model = CN()
cfg.model.encoder = 'timm-efficientnet-b2'
cfg.model.pretrained = 'imagenet'
cfg.model.activation = 'sigmoid'
cfg.model.decoder = 'unet'
cfg.model.in_channels = 3
cfg.model.out_channels = 10
cfg.model.cls_params = CN(dict(
                            pooling='avg',                            # one of 'avg', 'max'
                            dropout=0.8,                              # dropout ratio, default is None
                            #activation='sigmoid',                     # activation function, default is None
                            classes=1           # define number of output labels
                            ))
model_kwargs = {
    'unet' : {
        'encoder_depth' : 5,
        'decoder_use_batchnorm' : True,
        'decoder_channels' : (256, 128, 64, 32, 16),
        #'decoder_attention_type' : 'scse',
        #'activation' : None,
        #'aux_params' : cfg.model.cls_params 
    },
    'fpn': {
        'encoder_depth'  : 5, 
        'decoder_pyramid_channels' : 256,
        'decoder_segmentation_channels' : 128,
        'decoder_merge_policy' : 'add',
        'decoder_dropout' : 0.2,
        #'activation' : None, 
        'upsampling' : 4,
        #'aux_params' : cfg.model.cls_params 
    },
    'unet++': {
        'encoder_depth' : 5,
        'decoder_use_batchnorm' : True,
        'decoder_channels' : (256, 128, 64, 32, 16),
        #'decoder_attention_type' : None,
        #'activation' : None,
        #'aux_params' : cfg.model.cls_params 
    },
    'manet': {
        'encoder_depth' : 5,
        'decoder_use_batchnorm' : True,
        'decoder_channels' : (256, 128, 64, 32, 16),
        #'decoder_attention_type' : None,
        'decoder_pab_channels' : 64,
        #'activation' : None,
        #'aux_params' : cfg.model.cls_params 
    },
    'deeplabv3+': {
        'encoder_depth' : 5,
        'encoder_output_stride' : 16, 
        'decoder_channels' : 256, 
        'decoder_atrous_rates' : (12, 24, 36),
        #'activation' : None,
        'upsampling' : 4, 
        #'aux_params' : cfg.model.cls_params
    },
    'pan': {
        'encoder_dilation' : True, 
        'decoder_channels' : 32,
        #'activation' : None,
        'upsampling' : 4,
        #'aux_params' : None
    },
    'pspnet' : {
        'encoder_depth' : 4, 
        'psp_out_channels' : 512,
        'psp_use_batchnorm' : True, 
        'psp_dropout' : 0.2, 
        #'activation' : None, 
        'upsampling' : 16, 
        #'aux_params' : cfg.model.cls_params 
    }
}
cfg.model.kwargs = CN(model_kwargs[cfg.model.decoder])
cfg.model.activation = 'sigmoid'
#loss
cfg.criterions = CN()
cfg.criterions.seg_criterion = CN()
cfg.criterions.seg_criterion.criterion = 'dice'
cfg.criterions.seg_criterion.weights = [1.0]
#_ch = [ 0.2, 2.5, 1., 1., 0.5, 0.5, 2., 3., 5., 1. ]
_ch = [3.0, 4.0, 2.0, 4.0, 3.0, 0.5, 0.5, 5.0, 5.0, 0.5]
#['background' , 'building_flooded' ,'building_non-flooded' , 'road_flooded' , 'road_non-flooded' , 'water' , 'tree' , 'vehicle' , 'pool' ,  'grass']
cfg.criterions.seg_criterion.channels = _ch
cfg.criterions.cls_criterion = 'bce'
#optimizer
cfg.optimizer = CN()
cfg.optimizer.optimizer = 'adam'
optimizer_kwargs = {
    'adam':
    {
         'lr' : cfg.trainval.lr,
         'betas' : (0.9, 0.999),
         'weight_decay': 1e-6,
         'amsgrad' : False
    },
    'adamw':
    {
         'lr' : cfg.trainval.lr,
         'betas' : (0.9, 0.999),
         'weight_decay' : 1e-6,     
    },
    'rmsprop':
    {
         'lr' : cfg.trainval.lr,
         'alpha' : 0.99,
         'weight_decay' : 1e-6,
         'momentum' : 0,
         'centered' : False
        
    }
}
cfg.optimizer.kwargs = CN(optimizer_kwargs[cfg.optimizer.optimizer])
#scheduler
cfg.scheduler = CN()
cfg.scheduler.scheduler = 'polylr'
scheduler_kwargs = {
    'polylr':{
        'epochs' : cfg.trainval.epochs,
        'ratio' :0.9
    },
    'multisteplr':{
        'milestones' : [4],
        'gamma' : 0.1
    },
    'cosine-anneal':{
        'T_max' : 5,
        'eta_min' : 1e-8
    },
    'cosine-anneal-wm':{
        'T_0' : 1,
        'T_mult' : 2,
        'eta_min' : 1e-8
    }
} 
cfg.scheduler.kwargs = CN(scheduler_kwargs[cfg.scheduler.scheduler])


def get_configs():
    return cfg.clone()
def convert_cfg_to_dict(cfg_node, key_list=[]):
    
    """ Convert a config node to dictionary """
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict
    
def save_cfg_as_yaml(dic,path):
    
    with open('/'.join([path,'configs.yaml']),'w+') as yamlfile:
              yaml.safe_dump(dic,yamlfile, default_flow_style=False)
    yamlfile.close()

def load_cfg_from_yaml(path):
    with open('/'.join([path,'configs.yaml']),'r') as yamlfile:
        loaded = yaml.safe_load(yamlfile)
    yamlfile.close()
    return CN(loaded)