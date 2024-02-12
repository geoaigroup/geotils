import unittest, sys, os


current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))

sys.path.append(parent_directory)

from schedulers import AutoScheduler, scheduler_mapping
import numpy as np


class testScheduler(unittest.TestCase):
    def test_keys_lower_case(self):
        for schechduler in scheduler_mapping:
            self.assertEqual(schechduler,schechduler.lower())



if __name__=="__main__":
    unittest.main()
