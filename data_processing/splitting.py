from sklearn.model_selection import StratifiedKFold
import pandas as pd

class Splitting():
    """
    Splitting class for all split methods
 
    Args:
        df (pd.DataFrame): Input DataFrame.
        nfolds (int): Number of folds.
        seed (int): Random seed for reproducibility.
        id_key (str): Key representing the identifier for grouping.
        split_key (str): Key for stratification.
        label_keys (list): List of keys for labels.
        verbose (bool, default=False): If True, print fold statistics.
    """
    def __init__(self, df, nfolds, seed, id_key, split_key, label_keys, verbose=False):
        self.df = df
        self.nfolds = nfolds
        self.seed = seed
        self.id_key = id_key
        self.split_key = split_key
        self.label_keys = label_keys
        self.verbose = verbose
        
    def split_K_stratified_folds(self):
        """
        Split the DataFrame into K stratified folds based on specified keys.
        
        Returns:
            pd.DataFrame: DataFrame with an additional 'fold' column indicating the fold number.
        """
        X = self.df.groupby(self.id_key)[self.split_key].first().index.values
        y = self.df.groupby(self.id_key)[self.split_key].first().values
        skf = StratifiedKFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
    
        for i, (tfold, vfold) in enumerate(skf.split(X, y)):
            self.df.loc[self.df[self.id_key].isin(X[vfold]), 'fold'] = int(i)
    
        folds = [int(fold) for fold in self.df.groupby('fold').first().index.values]
        if self.verbose:
            for fold in folds:
                for label_key in self.label_keys:
                    print(f'fold:\t{fold}')
                    print(f'Label Key:{label_key}')
                    print(self.df.loc[self.df['fold'] == fold].set_index(['fold', label_key]).groupby(level=label_key).count())
        self.df.reset_index(drop=True, inplace=True)
        return self.df
    
