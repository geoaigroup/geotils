#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:15:42 2019

@author: hasan
"""
import json
import os
import pandas as pd
import shutil
import random

def merge_lists(a,b):
    if(len(a)>=len(b)):
        for item in b:
            a.append(item)
        return a
    else:
        for item in a:
            b.append(item)
        return b
    
def get_dicts(dict_dir):
    dict_dir=os.path.expanduser((os.path.join(dict_dir,'labels_dictionary')))
    print(dict_dir)
    with open(dict_dir,'r') as dfile:
        labels=dfile.read()
    label_file=json.loads(labels)
    counter=0
    dis_dict=dict()
    reg_dict=dict()
    names_ac_dis=dict()
    names_ac_reg=dict()
    for label in label_file:
        img_name=label_file[label]['img_name']
        # and len(label_file[label]['classes'])>0
        if(img_name.split('.')[0].endswith('_pre_disaster')):
            this_region=img_name.split('_')[0]
            this_disaster=label_file[label]['disaster']
            if(this_region not in reg_dict.keys()):
                reg_dict[this_region]=1
                names_ac_reg[this_region]=[]
                names_ac_reg[this_region].append(img_name.split('.')[0])
            else:
                reg_dict[this_region]+=1
                names_ac_reg[this_region].append(img_name.split('.')[0])
            if(this_disaster not in dis_dict.keys()):
                dis_dict[this_disaster]=1
                names_ac_dis[this_disaster]=[]
                names_ac_dis[this_disaster].append(img_name.split('.')[0])
            else:
                dis_dict[this_disaster]+=1
                names_ac_dis[this_disaster].append(img_name.split('.')[0])
            counter+=1
    '''       
    df_reg=pd.DataFrame.from_dict(reg_dict,orient='index')
    df_dis=pd.DataFrame.from_dict(dis_dict,orient='index')
    df_reg.plot(kind='bar',title='Data distributions acc to regions')
    df_dis.plot(kind='bar',title='Data distributions acc to disaster types')
    mean1=int(df_reg.mean(axis=0)[0])
    mean2=int(df_dis.mean(axis=0)[0])
    print('mean1= '+str(mean1))
    print('mean2= '+str(mean2))
    print("number of images: "+str(counter))
    print(names_ac_dis['tsunami'])
    '''
    return dis_dict,reg_dict,names_ac_dis,names_ac_reg,counter

def copy_files(src,dst,file_list,extension=''):
    print('start')
    for files in file_list:
        src_file_path = src + files+extension+'.png'
        dst_file_path = dst + files+extension+'.png'
        if os.path.exists(dst_file_path):
            print(dst_file_path+" already exists")
        else:
            #print("Copying: " + dst_file_path)
            try:
                shutil.copyfile(src_file_path,dst_file_path)
                print(dst_file_path)
            except IOError:
                a=0
                print(src_file_path + " does not exist")
                #input("Please, press enter to continue.")
                
def get_split(stat,names,count,ratio=0.9,var=0.1):
    df_reg=pd.DataFrame.from_dict(stat,orient='index')
    #mean=int(df_reg.mean(axis=0)[0])+1
    #print(str(mean))
    #imprE=int(float((1-ratio)*float(mean)))
    evalstat=dict()
    for key in stat.keys():
        temp=stat[key]
        evalstat[key]={}
        if(temp>=24):
            evalstat[key]['count']=30
            evalstat[key]['keep']=False
        else:
            evalstat[key]['count']=temp
            evalstat[key]['keep']=True
        '''
        if(temp>=(mean*2)):
            evalstat[key]['count']=int(imprE*(1+var))
            evalstat[key]['keep']=False
        else:
            if(int(float(float(temp)*(1+var)))>=mean):
                evalstat[key]['count']=int(float((1-ratio)*temp))
                evalstat[key]['keep']=False
            else:
                
                if(int(float(float(temp)*(1+var)))>=int(mean/2)):
                    evalstat[key]['count']=int(float((1-0.5)*temp))
                    evalstat[key]['keep']=False
                else:
                    evalstat[key]['count']=temp
                    evalstat[key]['keep']=True
        ''' 
    print('new distribution' + str(evalstat))
    plot_dict=dict()
    counter=0
    for key in evalstat.keys():
        plot_dict[key]=evalstat[key]['count']
        counter+=evalstat[key]['count']
    print("counter is : "+str(counter))
    df_new=pd.DataFrame.from_dict(plot_dict,orient='index')
    df_reg.plot(kind='bar',title='Data distributions acc to regions')
    df_new.plot(kind='bar',title='Data distributions acc to regions in eval set')
    
    
    evallist=[]
    trainlist=[]
    for key in evalstat.keys():
        if(evalstat[key]['keep']==True):
            templist=names[key]
            random.shuffle(templist)
            evallist=merge_lists(evallist,templist)
            trainlist=merge_lists(trainlist,templist)
        else:
            templist=names[key]
            random.shuffle(templist)
            evallist=merge_lists(evallist,templist[:int(evalstat[key]['count'])])
            trainlist=merge_lists(trainlist,templist[int(evalstat[key]['count']):])
    random.shuffle(evallist)
    random.shuffle(trainlist)
    
    print("length of val set: "+str(len(evallist)))
    print("length of train set: "+str(len(trainlist)))
    x=input("please press any key")
    
    return trainlist,evallist
   
         
    
if __name__ == '__main__':
    d_pth='/usr/local/NotSynced/xView2/xview2_data'
    a,b,c,d,e = get_dicts(d_pth)
    train,eval_ = get_split(b,d,e)
    '''
    lists=[]
    for key in d.keys():
        temp=d[key]
        random.shuffle(temp)
        lists=merge_lists(lists,temp)
        random.shuffle(lists)
    '''
    main_path=os.path.expanduser(os.getcwd())
    all_mskpth=os.path.join(main_path,'masks/')  
    src='/usr/local/NotSynced/xView2/train/images/'
    destTrain=''.join([d_pth,'/train/data/'])
    destEval=''.join([d_pth,'/val/data/'])
    copy_files(src,destTrain,train)
    copy_files(src,destEval,eval_)

    destTrain=''.join([d_pth,'/train/masks/'])
    destEval=''.join([d_pth,'/val/masks/'])

    copy_files(all_mskpth,destTrain,train,extension='_mask')
    copy_files(all_mskpth,destEval,eval_,extension='_mask')
    
         
