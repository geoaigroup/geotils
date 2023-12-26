import unittest
import torch
from torch import nn
from optimizers import get_optimizer
from schedulers import AutoScheduler,scheduler_mapping,get_scheduler
import numpy as np


class test_optimizers(unittest.TestCase):
    def test_keys_lower_case(self):
        for schechduler in scheduler_mapping:
            self.assertEqual(schechduler,schechduler.lower())



if __name__=="__main__":
    unittest.main()
