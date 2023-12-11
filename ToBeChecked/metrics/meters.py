
import numpy as np


class Meter(object):
    def reset(self):
        pass
    def update(self,value):
        pass
    def get_update(self):
        pass

class AverageMeter(Meter):
    
    def __init__(self):
        super(AverageMeter,self).__init__()
        self.reset()
    def reset(self):
        self.value = 0.
        self.average = 0.
        self.count = 0.
    def update(self,value):
        self.count += 1.
        self.value = value
        self.average = (self.average * (self.count - 1)+ self.value)/float(self.count)
    def get_update(self):
        return self.average

class ShortTermMemoryMeter(Meter):
    def __init__(self,memory_length):
        super(ShortTermMemoryMeter,self).__init__()
        self.reset()
        self.memory_length = memory_length
        assert(self.memory_length >1)
        #to atleast average last 2 batch's results or else it would be only performing additional non-neccessary operations 
    def reset(self):
        self.value = 0
        self.length = 0
        self.in_memory = []
        self.average = 0
    def update(self,value):
        self.value = value 
        if (self.length >= self.memory_length):
            self.in_memory = self.in_memory[1:]
        self.in_memory.append(self.value)
        self.average = np.average(np.array(self.in_memory))
        self.length = len(self.in_memory)
    def get_update(self):
        return self.average

class Sampler(Meter):
    def __init__(self,value_names,rate):
        super(Sampler,self).__init__()
        self.reset()
        self.v_names = value_names
        self.rate = rate
    def reset(self):
        self.history=dict()
        for name in self.v_names:
            self.history[name] = list()
    def update(self,values,i):
        if(i%self.rate == 0):
            for j,k in enumerate(values):
                name = self.v_names[j]
                self.history[name].append(k)
    def get_update(self):
        return self.history
                
            
            
classes = ['overall','background' , 'building_flooded' ,'building_non-flooded' , 'road_flooded' , 'road_non-flooded' , 'water' , 'tree' , 'vehicle' , 'pool' ,  'grass']
def register_metrics_avg_meters(names = classes,metric_names = ['iou'],thresh = 0.5):
  l = len(names)
  metrics = get_micro_metrics(
      metrics=[metric_names] * l,
      threshs=[thresh] * l,
      channels=[None,*[i for i in range(l-1)]],
      names=classes,
      num_cls=l
  )
  tups = []
  for metric in metrics:
    tups.append([metric,AverageMeter()])
  return tups

def get_meters_info(seg_loss_meter,seg_loss_name,cls_loss_meter,cls_loss_name, MnMs,cls_accuracy_meter,cls_f1score_meter):
  info = ''
  info += f'| {seg_loss_name} : {seg_loss_meter.get_update():.5} '
  info += f'| {cls_loss_name} : {cls_loss_meter.get_update():.5} '
  info += f'| cls_accuracy : {cls_accuracy_meter.get_update():.5} '
  info += f'| cls_f1score : {cls_f1score_meter.get_update():.5} '
  for metric,meter in MnMs:
    info += f'| {metric.__name__} : {meter.get_update():.5}'
  info += ' |'
  return info

def register_scores(pdict,seg_loss_meter,seg_criterion,cls_loss_meter,cls_criterion, MnMs,cls_accuracy_meter,cls_f1score_meter,prefix = 'train'):
  assert prefix in ['train','val'],'{} is not a valid prefix'.format(prefix)
  seg_loss_name = f'{prefix}_seg_' + seg_criterion.__name__
  cls_loss_name = f'{prefix}_cls_' + cls_criterion.__name__
  keys = list(pdict.keys())

  if(seg_loss_name not in keys):
    pdict[seg_loss_name] = []
    pdict[cls_loss_name] = []
    pdict['cls_accuracy'] = []
    pdict['cls_f1score'] = []
  
  pdict[seg_loss_name].append(seg_loss_meter.get_update())
  pdict[cls_loss_name].append(cls_loss_meter.get_update())
  pdict['cls_accuracy'].append(cls_accuracy_meter.get_update())
  pdict['cls_f1score'].append(cls_f1score_meter.get_update())

  for metric,meter in MnMs:
    if(f'{prefix}_' + metric.__name__ not in keys):
      pdict[f'{prefix}_' + metric.__name__] = []
    pdict[f'{prefix}_' + metric.__name__].append(meter.get_update())
  return pdict        
    
    
        
    