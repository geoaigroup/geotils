#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:16:29 2020

@author: hasan
"""
from sklearn.model_selection import StratifiedKFold
import os

from tqdm import tqdm
import pandas as pd

def main():
    path='/home/jamada/Desktop/OpenCitiesAI/Dataset/Tilestrain_tier_4/images'
    items=[]
    for image in os.listdir(path):
        if('.png' in image ):
            isplit=image.split('_')
            region_id=f'{isplit[0]}_{isplit[1]}'
            items.append({'id':image,'region_id':region_id,'x':isplit[2],'y':isplit[3].split('.')[0]})
    
    df=pd.DataFrame(items,columns=['id','region_id','x','y'])
    df['tile_id'] =df['region_id'].astype(str)+'_'+ df['x'].astype(str) + '_' + df['y'].astype(str)
    X = df.groupby('tile_id')['region_id'].first().index.values
    y = df.groupby('tile_id')['region_id'].first().values

    skf=StratifiedKFold(n_splits=5, random_state=98, shuffle=True)
    for i,(tfold,vfold) in enumerate(skf.split(X,y)):
        df.loc[df['tile_id'].isin(X[vfold]),'fold']=int(i)

    df.to_csv('folds4.csv')
    folds=[int(fold) for fold in df.groupby('fold').first().index.values]


    for fold in folds:
        print(f'fold:\t{fold}')
        print(df.loc[df['fold']==fold].set_index(['fold','region_id']).count(level='region_id'))
        

if __name__=='__main__':
    main()