import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torch

from albumentations import Resize,HorizontalFlip,VerticalFlip,Compose,OneOf,NoOp,\
                            Cutout,InvertImg,Normalize,ChannelShuffle,\
                            HueSaturationValue,GaussianBlur,GaussNoise,MultiplicativeNoise,ShiftScaleRotate,Normalize,CLAHE,ColorJitter,RandomBrightnessContrast

pos_augs = OneOf([
                  VerticalFlip(p=1.0),
                   HorizontalFlip(p=1.0),
                   Compose([
                            VerticalFlip(1.0),
                            HorizontalFlip(p=1.0),
                                
                     ],p=1.0),
                ],p=0.8)

color_augs = OneOf([
                    #ChannelShuffle(p=1.0),
                    HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15,p=1.0),
                    CLAHE(p=1.0),
                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,p=1.0) ,
                    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1.0)
                ],p=0.4)

hard_augs = OneOf([
                   GaussianBlur(blur_limit=3,p=1.0),
                   GaussNoise(var_limit=(100.0, 200.0),mean=0,p=1.0),
                   Cutout(num_holes=4, max_h_size=30, max_w_size=35, fill_value=[0],p=1.0),
                   MultiplicativeNoise(multiplier=(0.7, 1.1), per_channel=True,p=1.0)
              ],p=0.3)

norm_aug = Normalize(mean=(0.485, 0.456, 0.406), 
                     std=(0.229, 0.224, 0.225), 
                     max_pixel_value=255.0, 
                     always_apply=True,
                     p=1.0)
tfm0 = NoOp()


tfm1 = Compose([
                norm_aug
                ],p=1.0)

tfm2 = Compose([
                pos_augs,
                norm_aug
              ],p=1.0)

tfm3 = Compose([
                pos_augs,
                color_augs,
                norm_aug
              ],p=1.0)

tfm4 = Compose([
                pos_augs,
                color_augs,
                hard_augs,
                norm_aug
              ],p=1.0)

def get_tfm(number):
  if(number == 0):
    return tfm0
  elif(number == 1):
    return tfm1
  elif(number == 2):
    return tfm2
  elif(number == 3):
    return tfm3
  elif(number == 4):
    return tfm4
  else:
    raise ValueError('Transform {} is not an option!!'.format(number))
  
  

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
     