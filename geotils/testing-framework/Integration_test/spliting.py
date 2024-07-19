import unittest
from geotils.data_processing.splitting import *
import pandas as pd
import numpy as np


class test_Splitting(unittest.TestCase):
    def setUp(self):
        self.data = {
            "age": [25, 30, 35, 40, 45],
            "gender": ["M", "F", "M", "M", "F"],
            "height": [170, 165, 180, 175, 160],
            "weight": [70, 60, 80, 75, 55],
            "income": [50000, 60000, 70000, 80000, 90000],
            "label": ["A", "B", "A", "B", "A"],
        }

        self.df = pd.DataFrame(self.data)
        self.split = Splitting(
            self.df,
            2,
            42,
            "income",
            "label",
            ["age", "gender", "height", "weight", "income", "label"],
            False,
        )

    def test_Split_K_split_K_stratified_folds(self):
        df = self.split.split_K_stratified_folds()
        self.assertEqual(
            np.array(
                [
                    [25, "M", 170, 70, 50000, "A", 0.0],
                    [30, "F", 165, 60, 60000, "B", 1.0],
                    [35, "M", 180, 80, 70000, "A", 0.0],
                    [40, "M", 175, 75, 80000, "B", 0.0],
                    [45, "F", 160, 55, 90000, "A", 1.0],
                ],
                dtype=object,
            ).all(),
            df.values.all(),
        )


if __name__ == "__main__":
    unittest.main()
