from sklearn.model_selection import StratifiedKFold

def split_K_stratified_folds(
          df,
          nfolds,
          seed,
          id_key,
          split_key,
          label_keys,
          verbose=False
          ):
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