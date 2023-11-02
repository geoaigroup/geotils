import numpy as np

def noise_filter(washed,mina):
    values = np.unique(washed)
    for val in values[1:]:
        area = (washed[washed == val]>0).sum()
        if(area<=mina):  
            washed[washed == val] = 0
    return washed