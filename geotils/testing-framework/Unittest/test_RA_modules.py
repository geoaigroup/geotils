import sys 
sys.path.append('../')

import unittest
import torch
from models.RA_modules import Relational_Module, SR_Module, CR_Module, S_RA_Module, P_RA_Module

class TestRelationalModule(unittest.TestCase):
    def test_CR_Module_forward(self):        
        in_channels = 128
        in_size = (16, 16)
        x = torch.randn(1, in_channels, 16, 16)
        
        cr_module = Relational_Module(name='CR_M', in_channels=in_channels, in_size=in_size)
        
        y = cr_module.forward(x)
        
        self.assertEqual(y.shape, (1, in_channels, 16, 16))

    def test_SR_Module_forward(self):        
        in_channels = 128
        in_size = (16, 16)
        x = torch.randn(1, in_channels, 16, 16)
        
        sr_module = Relational_Module(name='SR_M', in_channels=in_channels, in_size=in_size)
        
        y = sr_module.forward(x)
        
        self.assertEqual(y.shape, (1, in_channels + 16 * 16, 16, 16))

    def test_S_RA_Module_forward(self):        
        in_channels = 128
        in_size = (16, 16)
        x = torch.randn(1, in_channels, 16, 16)
        
        s_ra_module = Relational_Module(name='S_RA_M', in_channels=in_channels, in_size=in_size)
        
        y = s_ra_module.forward(x)
        
        self.assertEqual(y.shape, (1, in_channels + 16 * 16, 16, 16))

    def test_P_RA_Module_forward(self):        
        in_channels = 128
        in_size = (16, 16)
        x = torch.randn(1, in_channels, 16, 16)
        
        p_ra_module = Relational_Module(name='P_RA_M', in_channels=in_channels, in_size=in_size)
        
        y = p_ra_module.forward(x)
        
        self.assertEqual(y.shape, (1, 2 * in_channels + 16 * 16, 16, 16))


class TestSRModule(unittest.TestCase):
    def test_forward(self):
        in_channels = 128
        x = torch.randn(1, in_channels, 16, 16)
        
        sr_module = SR_Module(in_channels)
        
        y = sr_module.forward(x)
        
        self.assertEqual(y.shape, (1, in_channels + 16 * 16, 16, 16))


class TestCRModule(unittest.TestCase):
    def test_forward(self):
        in_channels = 128
        x = torch.randn(1, in_channels, 16, 16)
        
        cr_module = CR_Module(in_channels)
        
        y = cr_module.forward(x)
        
        self.assertEqual(y.shape, (1, in_channels, 16, 16))


class TestSRAModule(unittest.TestCase):
    def test_forward(self):
        in_channels = 128
        x = torch.randn(1, in_channels, 16, 16)
        
        s_ra_module = S_RA_Module(in_channels)
        
        y = s_ra_module.forward(x)
        
        self.assertEqual(y.shape, (1, in_channels + 16 * 16, 16, 16))


class TestPRAModule(unittest.TestCase):
    def test_forward(self):
        in_channels = 128
        x = torch.randn(1, in_channels, 16, 16)
        
        p_ra_module = P_RA_Module(in_channels)
        
        y = p_ra_module.forward(x)
        
        self.assertEqual(y.shape, (1, 2 * in_channels + 16 * 16, 16, 16))


if __name__ == '__main__':
    unittest.main()
