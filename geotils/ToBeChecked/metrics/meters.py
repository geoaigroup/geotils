    import numpy as np
    from tmp import get_micro_metrics

    class Meter(object):
        """
        Meter is a general interface / class to use as a base for our classes
        With no implemented methods (to be implemented in children)
        """
        def reset(self):
            """
            Resets the meter to its initial state.
            """

        def update(self,value):
            """
            Updates the meter with a new value.

            Args:
                value: The value to update the meter with.
            """

        def get_update(self):
            """
            Gets the current state or value of the meter.

            Returns:
                The current state or value of the meter.
            """

    class AverageMeter(Meter):
        """
        Initializes an instance of the AverageMeter class subclass of Meter.
        """
        def __init__(self):
            super(AverageMeter,self).__init__()
            self.value=0.
            self.average=0.
            self.count=0.

        def reset(self):
            """
            Resets the AverageMeter to its initial state.
            """
            self.value = 0.
            self.average = 0.
            self.count = 0.
            
        def update(self,value):
            """
            Updates the AverageMeter with a new value.

            Args:
                value: The value to update the AverageMeter with.
            """
            self.count += 1.
            self.value = value
            self.average = (self.average * (self.count - 1)+ self.value)/float(self.count)
            
        def get_update(self):
            """
            Gets the current average value of the AverageMeter.

            Returns:
                The current average value of the AverageMeter.
            """
            return self.average

    class ShortTermMemoryMeter(Meter):
        """
        Initializes an instance of the ShortTermMemoryMeter class sublcass of Meter.

        Args:
            memory_length (int): Length of the short-term memory window.
                                Should be greater than 1 for meaningful averaging.
        """
        def __init__(self,memory_length):
            super(ShortTermMemoryMeter,self).__init__()
            self.value = 0
            self.length = 0
            self.in_memory = []
            self.average = 0
            self.memory_length = memory_length
            assert(self.memory_length >1)
            #to at least average last 2 batches results or else it would be only performing additional non-neccessary operations 
            
        def reset(self):
            """
            Resets the ShortTermMemoryMeter to its initial state.
            """
            self.value = 0
            self.length = 0
            self.in_memory = []
            self.average = 0
            
        def update(self,value):
            """
            Updates the ShortTermMemoryMeter with a new value.

            Args:
                value: The value to update the ShortTermMemoryMeter with.
            """
            self.value = value
            if (self.length >= self.memory_length):
                self.in_memory = self.in_memory[1:]
            self.in_memory.append(self.value)
            self.average = np.average(np.array(self.in_memory))
            self.length = len(self.in_memory)
            
        def get_update(self):
            """
            Gets the current average value of the ShortTermMemoryMeter.

            Returns:
                The current average value of the ShortTermMemoryMeter.
            """
            return self.average

    class Sampler(Meter):
        """
        Initializes an instance of the Sampler class.

        Args:
            value_names (list): List of names for the sampled values.
            rate (int): Sampling rate, i.e., how often to store values in the history.
        """
        def __init__(self,value_names,rate):
            super(Sampler,self).__init__()
            self.v_names = value_names
            self.rate = rate
            self.reset()
            
        def reset(self):
            """
            Resets the Sampler to its initial state.
            """
            self.history=dict()
            for name in self.v_names:
                self.history[name] = list()
        
        def update(self,values,i):
            """
            Updates the Sampler with a new set of values at a given iteration.

            Args:
                values (list): List of values to update the Sampler with.
                i (int): Current iteration or step.

            Note:
                - Values are stored in the history based on the specified sampling rate.
            """
            if(i%self.rate == 0):
                for j,k in enumerate(values):
                    name = self.v_names[j]
                    self.history[name].append(k)
                    
        def get_update(self):
            """
            Gets the current history of sampled values.

            Returns:
                dict: Dictionary containing sampled values for each value name.
            """
            return self.history

    class MetricsManager():
        """
        A class for managing and reporting various metrics during training or validation.
        """ 
        classes = ['overall','background' , 'building_flooded' ,'building_non-flooded' , 'road_flooded' ,
            'road_non-flooded' , 'water' , 'tree' , 'vehicle' , 'pool' ,  'grass']
        
        def __init__(self, classes, metric_names=['iou'], thresh=0.5):
            """
            Initializes an instance of the MetricsManager class.

            Args:
                classes (list): List of class names or labels.
                metric_names (list, optional): List of metric names. Default is ['iou'].
                thresh (float, optional): Threshold value for metrics. Default is 0.5.
            """
            self.classes = classes
            self.metric_names = metric_names
            self.thresh = thresh
            self.metrics = self.register_metrics_avg_meters()
            
        def register_metrics_avg_meters(self):
            """
            Registers metric functions with AverageMeter instances.

            Returns:
                list: List of tuples containing metric functions and corresponding AverageMeter instances.
            """
            l = len(self.classes)
            #get_micro_metrics is from the NoteBook which includes a lot of dependecies -> new file with all functions ?
            metrics = get_micro_metrics(
                metrics=[self.metric_names] * l,
                threshs=[self.thresh] * l,
                channels=[None,*[i for i in range(l-1)]],
                names=self.classes,
                num_cls=l
            )
            tups = []
            for metric in metrics:
                tups.append([metric,AverageMeter()])
            return tups

        @staticmethod
        def get_meters_info(seg_loss_meter,seg_loss_name,cls_loss_meter,cls_loss_name, mnms,cls_accuracy_meter,cls_f1score_meter):
            """
            Gets a formatted string with information about various metrics.

            Args:
                seg_loss_meter (Meter): Meter instance for segmentation loss.
                seg_loss_name (str): Name for segmentation loss.
                cls_loss_meter (Meter): Meter instance for classification loss.
                cls_loss_name (str): Name for classification loss.
                mnms (list): List of tuples containing metric functions and corresponding Meter instances.
                cls_accuracy_meter (Meter): Meter instance for classification accuracy.
                cls_f1score_meter (Meter): Meter instance for classification F1 score.

            Returns:
                str: Formatted string with information about various metrics.
            """
            info = ''
            info += f'| {seg_loss_name} : {seg_loss_meter.get_update():.5} '
            info += f'| {cls_loss_name} : {cls_loss_meter.get_update():.5} '
            info += f'| cls_accuracy : {cls_accuracy_meter.get_update():.5} '
            info += f'| cls_f1score : {cls_f1score_meter.get_update():.5} '
            for metric,meter in mnms:
                info += f'| {metric.__name__} : {meter.get_update():.5}'
            info += ' |'
            return info

        @staticmethod
        def register_scores(pdict,seg_loss_meter,seg_criterion,cls_loss_meter,cls_criterion, mnms,cls_accuracy_meter,cls_f1score_meter,prefix = 'train'):
            """
            Registers scores for different metrics in a provided dictionary.

            Args:
                pdict (dict): Dictionary to store metric scores.
                seg_loss_meter (Meter): Meter instance for segmentation loss.
                seg_criterion (torch.nn.Module): Segmentation loss criterion.
                cls_loss_meter (Meter): Meter instance for classification loss.
                cls_criterion (torch.nn.Module): Classification loss criterion.
                mnms (list): List of tuples containing custom metrics and their corresponding Meter instances.
                cls_accuracy_meter (Meter): Meter instance for classification accuracy.
                cls_f1score_meter (Meter): Meter instance for classification F1 score.
                prefix (str, optional): Prefix for metric names (e.g., 'train' or 'val'). Default is 'train'.

            Returns:
                dict: Updated dictionary with metric scores.
            """
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

            for metric,meter in mnms:
                if(f'{prefix}_' + metric.__name__ not in keys):
                    pdict[f'{prefix}_' + metric.__name__] = []
                pdict[f'{prefix}_' + metric.__name__].append(meter.get_update())
            return pdict
