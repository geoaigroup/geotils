#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:27:06 2020
@author: hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from torch.cuda.amp import autocast
#from ..base import modules as md
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock,CenterBlock
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel,SegmentationHead, ClassificationHead
from ..models.ASPP import ASPP,DenseASPP
from .RA_modules import Relational_Module
from ..models.OCR import OCR
#from torchviz import make_dot
#8import matplotlib.pyplot as plt
global mod_map,mod_types
mod_map = {
                'aspp' : ASPP,
                'sep_aspp' : ASPP,
                'dense_aspp' : DenseASPP,
                'sep_dense_aspp' : DenseASPP,
                'S_RA_M' : Relational_Module,
                'P_RA_M' : Relational_Module,
                'CR_M' : Relational_Module,
                'SR_M' : Relational_Module,
                'OCR_original' : OCR,
                'OCR_unet' : OCR
                }
mod_types = {
                'aspp' : 'aspp',
                'sep_aspp' : 'aspp',
                'dense_aspp' : 'dense_aspp',
                'sep_dense_aspp' : 'dense_aspp',
                'S_RA_M' : 'relational',
                'P_RA_M' : 'relational',
                'CR_M' : 'relational',
                'SR_M' : 'relational',
                'OCR_original'  : 'context',
                'OCR_unet' : 'context'
                }


class DecoderBlock1x(DecoderBlock):
    
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 use_batchnorm=True,
                 attention_type=None):
        
        super().__init__(in_channels,
                         skip_channels,
                         out_channels,
                         use_batchnorm,
                         attention_type)
        
    def forward(self, x, skip=None):
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
class UnetWMDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            mod = None,
            mod_configs = {},
            output_stride =32,
            in_size = (512,512)
            
    ):
        super().__init__()
        self.os=output_stride
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        #print(encoder_channels)
        
        
        
        if(mod):
            assert mod in mod_map,f'Module {mod} is not implemented!'
            mods = []
            mod_in_channels = encoder_channels[::-1]
            mod_out_channels = []
            for block in range(n_blocks):
                _b = str(block)
                if(_b in mod_configs.keys()):
                    
                    _m = mod_map[mod]
                    _mtype = mod_types[mod]
                    if(_mtype == 'relational'):
                        mod_configs[_b] = {'name' : mod,
                                        'in_channels' : mod_in_channels[block],
                                       'in_size' : self.get_resolution(in_size,block+1)}
                        mods.append(_m(**mod_configs[_b]))
                        mod_out_channels.append(mods[block].out_channels)
                        
                    elif(_mtype == 'aspp'):
                        mod_configs[_b]['separable'] = True if(mod == 'sep_aspp') else False
                        mods.append(_m(mod_in_channels[block],
                                       **mod_configs[_b]))
                        mod_out_channels.append(mod_configs[_b]['out_channels'])
                    elif(_mtype == 'context'):
                        mods.append(_m(mod_in_channels[block],
                                       mod_in_channels[block]//2,
                                       mod_in_channels[block],
                                       0,
                                       1))
                        mod_out_channels.append(mod_in_channels[block])

                    elif(_mtype == 'dense_aspp'):
                        mod_configs[_b]['separable'] = True if(mod == 'sep_dense_aspp') else False
                        mods.append(_m(mod_in_channels[block],
                                           **mod_configs[_b]))
                        #print(mods[-1].out_channels)
                        mod_out_channels.append(mods[-1].out_channels)



                else:
                    mods.append(nn.Identity())
                    mod_out_channels.append(mod_in_channels[block])
                    
            self.with_mods = True
            self.mods = nn.ModuleList(mods[::-1])
            #print(mods)
            
            mod_out_channels = mod_out_channels[::-1]
            #print(mod_out_channels)
            head_channels = mod_out_channels[0]
            in_channels = [head_channels] + list(decoder_channels[:-1])
            skip_channels = list(mod_out_channels[1:]) + [0]
            
        else:
            self.with_mods = False
            self.mods = []
            # computing blocks input and output channels
            head_channels = encoder_channels[0]
            in_channels = [head_channels] + list(decoder_channels[:-1])
            skip_channels = list(encoder_channels[1:]) + [0]
            
        out_channels = decoder_channels
        
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        if(n_blocks == 5 and self.os==16):
            blocks[0]=DecoderBlock1x(in_channels[0],skip_channels[0],out_channels[0])
        self.blocks = nn.ModuleList(blocks)
        
        
                       
                    
                
    def get_resolution(self,in_size = (512,512),stage = 1):
        h,w = in_size
        factor = stage 
        if(self.os == 16 and stage ==5):
            factor -= 1
        return (h//(2**factor),w//(2**factor))
            
        
    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        if(self.with_mods):
            features  = list(map(lambda x, y : x(y),self.mods,features))
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
class UnetWM(SegmentationModel):
 

    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_depth: int = 5,
        encoder_weights: str = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        mod = None,
        mod_configs = {},
        output_stride = 32,
        in_size = (384,384)
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        if(output_stride == 32):
            pass
        elif(output_stride ==16):
            self.encoder.make_dilated(output_stride)
        else:
            raise ValueError(f'Output stride can only be either 32 or 16 and not {output_stride}')

        self.decoder = UnetWMDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            mod = mod,
            mod_configs = mod_configs,
            output_stride = output_stride,
            in_size=in_size
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
'''       
_aspp = {
        '4':{
                 'mid_channels' : 256,
                 'inter_channels' : 128,
                 'atrous_rates' : [3,6,12,18],
                 'dropout_rate' : 0,
             }
        }
rr = {'3':{},
      '4' : {}}
print(_aspp)
x = torch.rand((1,3,384,384))
m = UnetWM(mod = 'dense_aspp',
           mod_configs =_aspp)
x = x.cuda(non_blocking = True)
m.train()
m.cuda()
<<<<<<< HEAD
y = m(x) 
=======
with autocast():y = m(x) 
>>>>>>> 9ae615f0dba35685db77246d00157d670e32ded4
print(y.size())
'''
