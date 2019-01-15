import numpy as np
import matplotlib.pyplot as plt

class xy_data_generator:
    def __init__(self, mu, sigma, distribution):
        """
        args
            mu, sigma ... function
            distribution ... scipy.stats function(としようとしたんだけど今のところnormにしか対応してない)
        """
        self.mu=mu
        self.sigma=sigma
        self.distribution=distribution
        self.start=0
        self.stop=3
        print('start',self.start,'stop',self.stop, '\nyou can change these parameter')
    
    def t_quantile(self, t, x):
        """
        return tau-quantile
        args
            t ... tau-quantile's tau
            x ... input
        """
        return self.distribution.ppf(q=t, loc=self.mu(x), scale=self.sigma(x))
    
    def sample(self, x):
        #![WIP]shapeの整形
        return self.distribution.rvs(self.mu(x), self.sigma(x))
    
    def n_sample(self, n=100):
        '''
        指定の範囲からxをランダムに決め、
        そこから乱数を発生しyとする。
        x,yを返す。
        return 
        '''
        X=np.random.uniform(self.start, self.stop, size=n).reshape(n,1)
        Y=self.sample(X)
        
        return X, Y
        
    
    def show_mu(self):
        x=np.linspace(self.start, self.stop,num=int(50*(self.stop-self.start)))
        y=self.mu(x)
        plt.plot(x,y,label="mean")
        
    def show_t_quantile(self, tau=[0.05, 0.3, 0.7,0.95]):
        x=np.linspace(self.start, self.stop, num=int(50*(self.stop-self.start)))
        y=self.mu(x)
        plt.plot(x,y,label="mean")
        for t in tau:
            plt.plot(x, self.t_quantile(t,x), label=str(t)+"-quantile")
        plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=12)
        plt.show()
    
   
        
def mu(x):
    #from [Takeuchi et.al, 2003]
#     return np.sinc(x)
    return x*np.sin(x)

def sigma(x):
    return 1*np.exp(1-x)