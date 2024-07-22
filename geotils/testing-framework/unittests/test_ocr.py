import unittest
import torch
import torch.nn.functional as F
import sys 
sys.path.append('../../')

from models.OCR import BNReLU, SpatialGather_Module, ObjectAttentionBlock2D, SpatialOCR_Module, OCR


class TestBNReLU(unittest.TestCase):
    def test_forward(self):
        model = BNReLU(in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        self.assertEqual(y.shape, (2, 3, 64, 64))

class TestSpatialGather_Module(unittest.TestCase):
    def test_forward(self):
        model = SpatialGather_Module(cls_num=2)
        feats = torch.randn(2, 256, 64, 64)
        probs = torch.randn(2, 2, 64, 64)
        y = model.forward(feats, probs)
        self.assertEqual(y.shape, (2, 256, 2, 1))

class TestObjectAttentionBlock2D(unittest.TestCase):
    def test_forward(self):
        model = ObjectAttentionBlock2D(in_channels=256, key_channels=128, cls_num=1)
        x = torch.randn(2, 256, 64, 64)
        proxy = torch.randn(2, 256, 64, 64)
        y = model.forward(x, proxy)
        self.assertEqual(y.shape, (2, 256, 64, 64))

class TestSpatialOCR_Module(unittest.TestCase):
    def test_forward(self):
        model = SpatialOCR_Module(in_channels=256, key_channels=128, out_channels=512, dropout=0.1, cls_num=1)
        feats = torch.randn(2, 256, 64, 64)
        proxy_feats = torch.randn(2, 256, 64, 64)
        y = model.forward(feats, proxy_feats)
        self.assertEqual(y.shape, (2, 512, 64, 64))

class TestOCR(unittest.TestCase):
    def test_forward(self):
        model = OCR(in_channels=256, key_channels=128, out_channels=512, dropout=0.1, num_cls=1)
        feats = torch.randn(2, 256, 64, 64)
        y = model.forward(feats)
        self.assertEqual(y.shape, (2, 512, 64, 64))


if __name__ == '__main__':
    unittest.main()
