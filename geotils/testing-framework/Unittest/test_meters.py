import unittest
import sys
import torch
sys.path.append('../')
from evaluation.meters import AverageMeter, ShortTermMemoryMeter, Sampler, MetricsManager

class TestMeters(unittest.TestCase):
    def test_reset(self):
        #testing if the 2 meters are actually resetting
        avg_meter = AverageMeter()
        avg_meter.value = 5.0
        avg_meter.average = 3.0
        avg_meter.count = 10
        avg_meter.reset()
        self.assertEqual(avg_meter.count, 0)
        self.assertEqual(avg_meter.value, 0.0)
        self.assertEqual(avg_meter.average, 0.0)

        short_term_meter = ShortTermMemoryMeter(memory_length=3)
        short_term_meter.value = 5.0
        short_term_meter.length = 3
        short_term_meter.in_memory = [3.0, 4.0, 5.0]
        short_term_meter.average = 4.0
        short_term_meter.reset()
        self.assertEqual(short_term_meter.length, 0)
        self.assertEqual(short_term_meter.value, 0)
        self.assertEqual(short_term_meter.average, 0)
    
    def test_update(self):
        avg_meter = AverageMeter()
        avg_meter.value = 4.0
        avg_meter.count = 1
        avg_meter.average = 4.0
        avg_meter.update(5.0)
        self.assertEqual(avg_meter.count, 2)
        self.assertEqual(avg_meter.value, 5.0)
        self.assertEqual(avg_meter.average, 4.5)

        short_term_meter = ShortTermMemoryMeter(memory_length=3)
        short_term_meter.update(2.0)
        self.assertEqual(short_term_meter.length, 1)
        self.assertEqual(short_term_meter.average, 2.0)
        
class TestSampler(unittest.TestCase):
    def test_reset(self):
        sampler = Sampler(value_names=['value1', 'value2'], rate=2)
        sampler.reset()
        self.assertEqual(sampler.history, {'value1': [], 'value2': []})

    def test_update(self):
        sampler = Sampler(value_names=['value1', 'value2'], rate=2)
        sampler.update([1, 2], 0)
        self.assertEqual(sampler.history, {'value1': [1], 'value2': [2]})
        
class DummyCriterion(torch.nn.Module):
    def __init__(self):
        super(DummyCriterion, self).__init__()
        self.__name__="Module"

    def forward(self, x, y):
        return x + y
    
class DummyMetric:
    def __init__(self):
        self.__name__=""

    def get_update(self):
        return 0.5

class TestMetricsManager(unittest.TestCase):
    def test_initialization(self):
        classes = ['class1', 'class2']
        metric_names = ['iou', 'fscore']
        thresh = 0.5

        manager = MetricsManager(classes, metric_names, thresh)

        self.assertEqual(manager.classes, classes)
        self.assertEqual(manager.metric_names, metric_names)
        self.assertEqual(manager.thresh, thresh)
        self.assertEqual(len(manager.metrics), len(classes) * len(metric_names))

    def test_register_scores(self):
        classes = ['class1', 'class2']
        metric_names = ['iou', 'fscore']
        thresh = 0.5

        manager = MetricsManager(classes, metric_names, thresh)

        pdict = {}
        seg_loss_meter = AverageMeter()
        seg_criterion = DummyCriterion()
        cls_loss_meter = AverageMeter()
        cls_criterion = DummyCriterion()
        cls_accuracy_meter = AverageMeter()
        cls_f1score_meter = AverageMeter()
        dm1 = DummyMetric()
        dm1.__name__="metric1"
        dm2 = DummyMetric()
        dm2.__name__="metric2"
        mnms = [(dm1, AverageMeter()), (dm2, AverageMeter())]

        pdict = manager.register_scores(pdict, seg_loss_meter, seg_criterion, cls_loss_meter, cls_criterion, mnms, cls_accuracy_meter, cls_f1score_meter)

        expected_length = len(classes) * len(metric_names) + 2
        self.assertEqual(len(pdict), expected_length)
        
