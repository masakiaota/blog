import numpy as np
import pandas as pd
from scipy import stats

def data_pattern1(random_state=2019):
    '''
    get artificial dataset

    sample

    data, target=data_pattern1()

    1. 同じ分布(正規分布とか)で分散が違う場合
    
    5次元、200サンプル
    
    
    '''

    np.random.seed(seed=random_state)

    norm=stats.norm
    data=np.vstack([norm.rvs(loc=0, scale=(i+1),size=(200,)) for i in range(5)]).T

    binom=stats.binom
    target=binom.rvs(n=1,p=0.5,size=(200,))


    return data, target

def data_pattern2(random_state=2019):
    '''
    get artificial dataset

    sample

    data, target=data_pattern2()

    2. 同じ分布(正規分布とか)で分散も同じだが、ある程度集約する場合(binを20個、10個、5個、2個みたいに変えてみる)
    
    5次元、200サンプル
    
    
    '''

    np.random.seed(seed=random_state)

    norm=stats.norm
    columns=[norm.rvs(loc=0, scale=(1),size=(200,)) for _ in range(5)]
    
    # bins=20の場合
    bin_edges=[]#99%入る範囲を20等分しよう
    for 
    
    binom=stats.binom
    target=binom.rvs(n=1,p=0.5,size=(200,))


    return data, target