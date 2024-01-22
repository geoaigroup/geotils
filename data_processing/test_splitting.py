import unittest
import numpy as np

class TestSplitKFolds(unittest.TestCase):
    #Create a dataset to be used in the test cases
    def setUp(self):
        # Create a large dataset for testing
        np.random.seed(42)
        num_samples = 1000

        data = {
            'id': np.arange(num_samples),
            'split_key': np.random.choice(['A', 'B', 'C'], size=num_samples),
            'label': np.random.choice([0, 1], size=num_samples)
        }
        self.df = pd.DataFrame(data)

    def test_split_k_folds(self):
        df_result = split_K_stratified_folds(self.df, nfolds=5, seed=42, id_key='id', split_key='split_key', label_keys=['label'])
        unique_folds = df_result['fold'].unique()

        self.assertEqual(len(unique_folds), 5)  # Check if correct number of folds are created

    def test_split_k_folds_verbose(self):
        df_result = split_K_stratified_folds(self.df, nfolds=5, seed=42, id_key='id', split_key='split_key', label_keys=['label'], verbose=True)

        unique_folds = df_result['fold'].unique()

        self.assertEqual(len(unique_folds), 5)  # Check if correct number of folds are created

    def test_split_k_folds_labels(self):
        df_result = split_K_stratified_folds(self.df, nfolds=5, seed=42, id_key='id', split_key='split_key', label_keys=['label'])
        unique_labels = df_result.set_index(['fold', 'label']).groupby(level='label').count()

        self.assertTrue(unique_labels.min()['id'] > 1)  # Check if each label has samples in each fold

    def test_split_k_folds_reset_index(self):
        df_result = split_K_stratified_folds(self.df, nfolds=5, seed=42, id_key='id', split_key='split_key', label_keys=['label'])
        self.assertTrue('id' in df_result.columns)  # Check if 'id' column is present after resetting index
