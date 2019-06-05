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

    2. 同じ分布(正規分布とか)で分散も同じだが、ある程度集約する場合(binを50個、20個、10個、3個みたいに変えてみる)
    3シグマ区間を端として集計する
    
    5次元、200サンプル
    '''
    def _get_partations_and_representatives(n_bins,sigma):
        '''
        returns
        ----------
        partations ... 集計で区切るポイント
        representative ... その集計の代表値(長さはlen(partations)-1となる)
        '''
        #3シグマ区間を集計の端とする(正確にはそれの一つ内側)
        edge_min=-3*sigma
        edge_max=3*sigma
        width=edge_max-edge_min
        partations=[i*width/n_bins + edge_min for i in range(n_bins+1)]
        representative=[(left+right)/2 for left, right in\
                        zip(partations[:-1],partations[1:])]

        #端を都合の良いように書き換える
        partations[0],partations[-1]=-np.inf,np.inf
        
        return partations, representative
        
    np.random.seed(seed=random_state)

    norm=stats.norm
    sigma=1
    
    #正規分布のまんまの場合
    columns=[norm.rvs(loc=0, scale=sigma,size=(200,))]
    
    # bins=50の場合
    partations, representative=_get_partations_and_representatives(50,sigma)
    norm_tmp=norm.rvs(loc=0, scale=sigma,size=(200,))
    for left,right,rep in zip(partations[:-1],partations[1:],representative):
        norm_tmp[(left<=norm_tmp)&(norm_tmp<right)]=rep
    
    columns.append(norm_tmp)
    
    # bins=20の場合
    partations, representative=_get_partations_and_representatives(20,sigma)
    norm_tmp=norm.rvs(loc=0, scale=sigma,size=(200,))
    for left,right,rep in zip(partations[:-1],partations[1:],representative):
        norm_tmp[(left<=norm_tmp)&(norm_tmp<right)]=rep
    
    columns.append(norm_tmp)
    
    # bins=10の場合
    partations, representative=_get_partations_and_representatives(10,sigma)
    norm_tmp=norm.rvs(loc=0, scale=sigma,size=(200,))
    for left,right,rep in zip(partations[:-1],partations[1:],representative):
        norm_tmp[(left<=norm_tmp)&(norm_tmp<right)]=rep
    
    columns.append(norm_tmp)
    
    # bins=3の場合
    partations, representative=_get_partations_and_representatives(3,sigma)
    norm_tmp=norm.rvs(loc=0, scale=sigma,size=(200,))
    for left,right,rep in zip(partations[:-1],partations[1:],representative):
        norm_tmp[(left<=norm_tmp)&(norm_tmp<right)]=rep
    
    columns.append(norm_tmp)
    
    
    #ndarrayに変換
    data=np.array(columns).T
    
    #targetの生成
    binom=stats.binom
    target=binom.rvs(n=1,p=0.5,size=(200,))


    return data, target

def data_pattern3(random_state=2019):
    '''
    get artificial dataset

    sample

    data, target=data_pattern3()

    3. 分散を揃えて違う分布でやってみる
    正規分布、指数分布、カイ二乗分布、コーシー分布、極値分布(?)
    
    5次元、200サンプル
    
    
    '''

    np.random.seed(seed=random_state)
    columns=[]
    
    #正規分布
    norm=stats.norm
    columns.append(norm.rvs(loc=0, scale=1,size=(200,)))

    # 指数分布
    expo=stats.expon
    columns.append(expo.rvs(size=(200,)))

    # カイ二乗分布
    columns.append(stats.chi2.rvs(1,size=(200,)))

    # コーシー分布
    columns.append(stats.cauchy.rvs(size=(200,)))
    
    #ndarrayに変換
    data=np.array(columns).T
    
    
    # targetの生成
    binom=stats.binom
    target=binom.rvs(n=1,p=0.5,size=(200,))

    return data, target

def data_pattern4(random_state=2019):
    '''
    data4
    null case
    全て(5次元)独立(正規分布)でやる
    '''

    np.random.seed(seed=random_state)
    columns=[]
    
    #正規分布
    norm=stats.norm
    data=norm.rvs(loc=0, scale=1,size=(200,5))

    # targetの生成
    binom=stats.binom
    target=binom.rvs(n=1,p=0.5,size=(200,))

    return data, target

def data_pattern5(random_state=2019):
    '''
    data5
    null case
    5次元のうち、2次元だけdependancyをもたせる
    '''

    np.random.seed(seed=random_state)
    columns=[]
    
    #正規分布
    norm=stats.norm
    data=norm.rvs(loc=0, scale=1,size=(200,4))
    
    # make dependancy
    dep=np.exp(data[:,-1]) #指数をかませることで変換する
    data= np.hstack([data,dep.reshape(-1,1)])
    

    # targetの生成
    binom=stats.binom
    target=binom.rvs(n=1,p=0.5,size=(200,))

    return data, target

def data_pattern6(random_state=2019):
    '''
    data6
    power case
    
    0 -> N([-1]*5,2*np.eye(5)) (分散は少し広めに取るか)
    1 -> N([1]*5,2*np.eye(5)) (分散は少し広めに取るか)
    というルールでデータを生成
    '''

    np.random.seed(seed=random_state)
    columns=[]
    
    #正規分布
    norm=stats.norm
    positive=norm.rvs(loc=1,scale=3,size=(100,5))
    negative=norm.rvs(loc=-1,scale=3,size=(100,5))
    data=np.vstack([positive,negative])
    
    # target
    target=np.array(([1]*100)+([0]*100))

    return data, target

def data_pattern7(random_state=2019):
    '''
    data7
    power case
    
    0 -> N([-1]*5,2*np.eye(5)) (分散は少し広めに取るか)
    1 -> N([1]*5,2*np.eye(5)) (分散は少し広めに取るか)
    というルールでデータを生成
    
    ただし、最後の2つの次元には完全なdependancyをもたせる
    '''

    np.random.seed(seed=random_state)
    columns=[]
    
    #正規分布
    norm=stats.norm
    positive=norm.rvs(loc=1,scale=3,size=(100,4))
    negative=norm.rvs(loc=-1,scale=3,size=(100,4))
    data=np.vstack([positive,negative])
    dep=np.exp(data[:,-1]).reshape(-1,1)
    data=np.hstack([data,dep])
    
    # target
    target=np.array(([1]*100)+([0]*100))

    return data, target