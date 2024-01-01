from sklearn.model_selection import StratifiedKFold

def split_K_stratified_folds(
          df,
          id_key="id",
          split_key="class",
          label_keys="label",
          nfolds=5,
          seed=313,
          verbose=False
          ):
    """
    split a given dataframe into a K startified folds (equal ditribution for classes in each split)
    
    
    @param df: dataframe to split
    @param id_key: the id column key in df
    @param split_key : the key based for the split
    @param label_keys : the label class
    @param nfolds : nunber of folds
    @param seed : random seed
    @param verbose : enable to print the procedure
    

    @type df: Dataframe
    @type id_key: str
    @type split_key : str
    @type label_keys : str
    @type nfolds : int
    @type seed : int
    @type verbose : bool
    
    This function split a dataframe using the StratifiedKFold of sci-kit learn library.
    it is used to train the model compare multiple learning techniques.
    
    Note: stratified K split is not always the optimal split, sometimes choosing the normal ksplit is better (such as with ensambles)  
    """
    X = df.groupby(id_key)[split_key].first().index.values
    y = df.groupby(id_key)[split_key].first().values
    skf = StratifiedKFold(n_splits = nfolds, random_state = seed, shuffle=True) 
    
    for i,(tfold,vfold) in enumerate(skf.split(X,y)):
        df.loc[df[id_key].isin(X[vfold]),'fold'] = int(i)

    folds=[int(fold) for fold in df.groupby('fold').first().index.values]
    if verbose:
        for fold in folds:
            for label_key in label_keys:
                print(f'fold:\t{fold}')
                print(f'Label Key:{label_key}')
                print(df.loc[df['fold']==fold].set_index(['fold',label_key]).groupby(level=label_key).count())
    df.reset_index(drop=True,inplace=True)
    return df