#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:50:11 2021

@author: hasan
"""
#configs
from yacs.config import CfgNode as CN
import yaml

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