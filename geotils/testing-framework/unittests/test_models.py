import unittest
import torch
import sys

sys.path.append('../../')

from models.ASPP import ASPPConv, SeparableASPPConv, ASPPPooling, ASPP, DenseASPPConv, DenseASPP

class TestASPPConv(unittest.TestCase):
    def test_forward(self):
        model = ASPPConv(in_channels=3, out_channels=64, kernel=3, dilation=1)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        self.assertEqual(y.shape, (2, 64, 64, 64))

class TestSeparableASPPConv(unittest.TestCase):
    def test_forward(self):
        model = SeparableASPPConv(in_channels=3, out_channels=64, kernel=3, dilation=1)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        self.assertEqual(y.shape, (2, 64, 64, 64))

class TestASPPPooling(unittest.TestCase):
    def test_forward(self):
        model = ASPPPooling(in_channels=3, out_channels=64)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        self.assertEqual(y.shape, (2, 64, 64, 64))

class TestASPP(unittest.TestCase):
    def test_forward(self):
        model = ASPP(in_channels=3, out_channels=64, atrous_rates=[6, 12, 18], dropout_rate=0.5, separable=False)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        self.assertEqual(y.shape, (2, 64, 64, 64))

class TestDenseASPPConv(unittest.TestCase):
    def test_forward(self):
        model = DenseASPPConv(in_channels=3, mid_channels=32, out_channels=64, rate=3, separable=False, dropout_rate=0.5)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        self.assertEqual(y.shape, (2, 64, 64, 64))


class TestDenseASPP(unittest.TestCase):
    def test_forward(self):
        model = DenseASPP(in_channels=3, mid_channels=32, inter_channels=64, atrous_rates=[3, 6, 12, 18], dropout_rate=0.5, separable=False)
        x = torch.randn(2, 3, 64, 64)
        y = model.forward(x)
        expected_out_channels = 3 + len([3, 6, 12, 18]) * 64
        self.assertEqual(y.shape, (2, expected_out_channels, 64, 64))


if __name__ == '__main__':
    unittest.main()
