#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:03:13 2020

@author: jamada
"""
import numpy as np
import os
from random import randint
class Color():
    def __init__(self,shift=100):
        self.shift = shift
        self.sh = randint(1,shift)
        self.color = [120,60,40]
        self.pos = 0
        self.max = 240
        self.min = 40
        assert self.sh <= self.max - self.min ,f'choose a smaller shift than {self.max - self.min}'
    def get_value(self,val,ceil,sh,floor=0):
        shifted = val + sh
        
        what = max(0,ceil-shifted)
        if(what>0):
            return shifted
        else :
            return floor + shifted - ceil
        
    def step(self):
        self.color[self.pos] = self.get_value(self.color[self.pos],self.max,self.sh,self.min)
        self.pos = self.get_value(self.pos,3,1,0)
        self.sh = randint(1,self.shift)
    def get_color(self):
        return self.color
    
def colorize(washed):
    values = sorted(np.unique(washed))
    w,h = washed.shape
    mask = np.zeros((3,w,h),dtype =np.uint8)
    picaso = Color()
    for val in values[1:]:
        clr = picaso.get_color()
        for i in range(3):
            mask[i][washed == val] = np.uint8(clr[i])
        picaso.step()
    for i in range(3):
            mask[i][washed == 0] = np.uint8(255)
    return mask.transpose((1,2,0))
def create_path(path):
    try:
        os.makedirs(path)
    except:
        print(f'{path} already exists!!!')
    
