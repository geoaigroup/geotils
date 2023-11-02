import numpy as np
import random
from PIL import Image
import Dataset
import torch


class FloodNetDataset_FS(Dataset):
  def __init__(self,data_path,classes = 11,ignore_class = 10,fold=1,train=True,transform_number=2,to_tensor=False,shuffle=False):
    self.data = []
    self.train = train
    self.fold = fold
    self.tfm = get_tfm(transform_number)
    self.tt = to_tensor
    self.shuff = shuffle
    self.classes = classes
    self.ig_class = ignore_class
    self.class_names = ['background' , 'building_flooded' ,'building_non-flooded' ,
                        'road_flooded' , 'road_non-flooded' , 'water' , 'tree' ,
                        'vehicle' , 'pool' ,  'grass']
    df = pd.read_csv(f'{data_path}/folds.csv')

    if(train):
      df = df.loc[(df['fold'] != self.fold)]
    else: 
      df = df.loc[(df['fold'] == self.fold)]

    for i,row in df.iterrows():
      iid = row['iid']
      #if(iid == 6707):print(len(self.data))
      clss = row['class']
      self.data.append({
          'iid' : row['iid'],
          'class' : 0.0 if(clss == 'non-flooded') else 1.0,
          'img_path' : f'{data_path}/labeled/images/{iid}_{clss}.png',
          'mask_path' : f'{data_path}/labeled/masks/{iid}_{clss}_mask.png',
      })

    if(self.shuff):
      random.shuffle(self.data)
    self.regen()

  def __getitem__(self,index):
    self.idx = index
    curr = self.__get_current_data__()
    img = np.array(Image.open(curr['img_path']),dtype=np.uint8)
    mask = np.array(Image.open(curr['mask_path']),dtype=np.uint8)

    aug = self.tfm(image=img,mask=mask)
    img,mask = aug['image'],aug['mask']

    masks = []
    h,w = mask.shape
    ig_mask = None

    for i in range(self.classes):
      m = np.zeros((h,w),dtype=np.uint8)
      if(i == self.ig_class):
        m[mask != i] = 1
        ig_mask = m
      else:
        m[mask == i] = 1
        masks.append(m)

    mask = np.stack(masks,axis = -1).astype(np.uint8)
    label = np.ones(1) * curr['class']
    if(self.tt):
      img = torch.from_numpy(img.transpose(2,0,1)).float()
      mask = torch.from_numpy(mask.transpose(2,0,1)).float()
      ig_mask = torch.from_numpy(ig_mask).unsqueeze(0).float()
      label = torch.from_numpy(label).float()

    return img,mask,ig_mask,label

  def regen(self):
    self.idx = 0

  def change_tfm(self,tfm):
    self.tfm = tfm

  def change_tt(self,tt):
    self.tt = tt
    
  def __get_current_data__(self):
    return self.data[self.idx]
    
  def __len__(self):
    return len(self.data)
     