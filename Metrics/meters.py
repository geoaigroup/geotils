
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
                
            
            
        
    
    
        
    