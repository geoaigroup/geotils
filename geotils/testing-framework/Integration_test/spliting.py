from geotils.data_processing.splitting import *
import pandas as pd


class test_Splitting:
    data = {
        "age": [25, 30, 35, 40, 45],
        "gender": ["M", "F", "M", "M", "F"],
        "height": [170, 165, 180, 175, 160],
        "weight": [70, 60, 80, 75, 55],
        "income": [50000, 60000, 70000, 80000, 90000],
        "label": ["A", "B", "A", "B", "A"],
    }

    df = pd.DataFrame(data)
    split = Splitting(
        df,
        2,
        42,
        "income",
        "label",
        ["age", "gender", "height", "weight", "income", "label"],
        True,
    )

    df = split.split_K_stratified_folds()
