#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 23:26:55 2020
@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
My own implementation of spatial relation and channel relation modules
Paper : A Relation-Augmented Fully Convolutional Network for Semantic Segmentation in Aerial Scenes
link : https://openaccess.thecvf.com/content_CVPR_2019/papers/Mou_A_Relation-Augmented_Fully_Convolutional_Network_for_Semantic_Segmentation_in_Aerial_CVPR_2019_paper.pdf
'''
class SR_Module(nn.Module):
    def __init__(self,
                 in_channels):
        
        super(SR_Module,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 1,
                               padding = 0,
                               stride = 1)
        self.conv2 = nn.Conv2d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 1,
                               padding = 0,
                               stride = 1)
        self.relu = nn.ReLU()
        
    def forward(self,feats):
        B, C, H, W = feats.size()
        
        us = self.conv1(feats)
        vs = self.conv2(feats)
        us = us.view(B,C,-1) #(B, C, HxW)
        
        vs = vs.view(B,C,-1) #(B, C, HxW)
        us = us.permute(0,2,1) #(B, HxW, C)
        relations = torch.matmul(us,vs) #(B, HxW, HxW)
        relations = relations.view(B, H*W, H, W)
        relations = self.relu(relations)
        
        aug_feats = torch.cat((feats,relations),dim = 1)
        return aug_feats

class CR_Module(nn.Module):
    def __init__(self,
                 in_channels):
        super(CR_Module,self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 1,
                               padding = 0,
                               stride = 1)
        self.conv2 = nn.Conv2d(in_channels = in_channels,
                               out_channels = in_channels,
                               kernel_size = 1,
                               padding = 0,
                               stride = 1)
    def forward(self,feats):
        B, C, H, W = feats.size()
        descriptors = self.gap(feats)
        us = self.conv1(descriptors)
        vs = self.conv2(descriptors)
        us = us.view(B,C,-1)
        vs = vs.view(B,C,-1).permute(0,2,1)
        relations = torch.matmul(us, vs)
        relations = F.softmax(relations,dim=1)
        feats = feats.view(B,C,-1).permute(0,2,1)
        aug_feats = torch.matmul(feats, relations)
        aug_feats = aug_feats.view(B,C,H,W)
        return aug_feats
    
class S_RA_Module(nn.Sequential):
    def __init__(self,in_channels):
        
        super().__init__(CR_Module(in_channels),
                         SR_Module(in_channels))
        
class P_RA_Module(nn.Module):
    
    def __init__(self,in_channels):
        super(P_RA_Module,self).__init__()
        self.crm = CR_Module(in_channels)
        self.srm = SR_Module(in_channels)
        
    def forward(self,feats):
        rc = self.crm(feats)
        rs = self.srm(feats)
        relations = torch.cat((rs,rc),dim=1)
        return relations
    
def get_relational_module(name = 'S_RA',in_channels = 256, in_size = (16,16)):
    __mods__ = ['S_RA_M','P_RA_M','CR_M','SR_M']
    assert name in __mods__, f'{name} is not a valid relational module!'
    H,W = in_size
    if(name == 'CR_M'):
        m = CR_Module(in_channels)
        out_channels = in_channels
    elif(name == 'SR_M'):
        m = SR_Module(in_channels)
        out_channels = in_channels + H * W
    elif(name == 'S_RA_M'):
        m = S_RA_Module(in_channels)
        out_channels = in_channels + H * W
    elif(name == 'P_RA_M'):
        m = P_RA_Module(in_channels)
        out_channels = 2 * in_channels + H * W
    return m, out_channels
        
class Relational_Module(nn.Module):
    def __init__(self,
                 name,
                 in_channels,
                 in_size):
        super().__init__()
        self.mod ,self.out_channels = get_relational_module(name,in_channels,in_size)
        
    def forward(self,feats):
        aug_feats = self.mod(feats)
        return aug_feats
        
        
'''      
x= torch.rand(size=(1,128,16,16))
print(x.size())
m=P_RA_Module(x.size(1))

y = m(x)
print(y.size())
'''