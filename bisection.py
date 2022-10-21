"""

WIP 

Implement algorithms from 
S. Rodriguez and M. Ludkovski, ACM Trans. Model. Comput. Simul. 30, 1, Article 2. 
https://doi.org/10.1145/3355607

To do: entropy driven sampling from
S. Rodriguez and M. Ludkovski, European Journal of Operational Research 286 (2020) 588–603.
https://doi.org/10.1016/j.ejor.2020.03.049

demo: 
>>> from bisection import exper
>>> nIter = 20   # number of coordinates sampled
>>> nReps = 1    # number of iterations at the same coordinate
>>> t = exper(nIter,nReps)

""" 

import numpy as np
import random
import logging
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import binom
from scipy.interpolate import interp1d

FORMAT = '%(asctime)-15s- %(levelname)s - %(name)s -%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

def oracle(xq, noisy=True):
    """
    returns h and sigma. if no estimate for sigma, return -1
    """
    xs = 4.5
    func = np.loadtxt("input.csv",usecols=[1,2])

    # apply the desired offset along the x-axis (i.e., move the zero-crossing)
    x = func[:,0]+xs
    # invert the function so that it's monotonically decreasing
    y = -func[:,1]

    f = interp1d(x, y)

    h = f([xq])[0]

    sigma = 0.3

    #h = (xs - xq)

    if(noisy):
        Z = random.gauss(h, sigma)
    else :
        Z = h
    return Z,sigma


class State:
    
    def __init__(self, logf):
        self._logf = copy.deepcopy(logf)
        self._xq  = np.nan
        self._data = []
        self._Z = np.nan
        self._sig = np.nan
        self._P = np.nan
        self._theta = np.nan
        self._B = np.nan
        self._Bmax = np.nan
        self._K = np.nan

    def copy(self):
        copystate = State(copy.deepcopy(self._logf))
        copystate.set_xq(self._xq)
        copystate.set_data(self._data)
        copystate.set_Z(self._Z)
        copystate.set_sig(self._sig)
        copystate.set_theta(self._theta)
        copystate.set_K(self._K)
        copystate.set_B(self._B, self._Bmax)
        return copystate
        
    def get_logf(self): 
        return copy.deepcopy(self._logf)

    def get_xq(self): 
        return copy.deepcopy(self._xq)
    
    def get_Z(self): 
        return copy.deepcopy(self._Z)

    def get_sig(self): 
        return copy.deepcopy(self._sig)

    def get_theta(self):
        return copy.deepcopy(self._theta)
    
    def get_K(self): 
        return copy.deepcopy(self._K)

    def get_Bmax(self): 
        return copy.deepcopy(self._Bmax)


    
    def set_logf(self,logf): 
        self._logf = copy.deepcopy(logf)

    def set_xq(self,xq): 
        self._xq = copy.deepcopy(xq)

    def set_data(self,data): 
        self._data = copy.deepcopy(data)

    def set_Z(self,Z): 
        self._Z = copy.deepcopy(Z)

    def set_sig(self,sig): 
        self._sig = copy.deepcopy(sig)

    def set_theta(self,theta):
        """
        theta := probability of a positive answer 
        P := accuracy or specificity, i.e. probability that the answer is correct
        """ 
        self._theta = copy.deepcopy(theta)
        self._P = max(theta, 1-theta)

    def set_K(self, K):
        self._K = copy.deepcopy(K)
        
    def set_B(self, B, Bmax): 
        self._B = copy.deepcopy(B)
        self._Bmax = copy.deepcopy(Bmax)


        
class bisect:

    def __init__(self, name, oracle, xmin, xmax, xbin, prior=None):

        # this may fail to find a zero if it's located within eps/2 of xmax or min
        self.name = name
        
        steps = int(np.floor(xmax-xmin)/xbin)
        rounding = xmax - xmin - xbin*steps
        self.x = [ xmin + rounding/2 + xq*xbin for xq in np.arange(steps+1) ] # edges
        self.oracle = oracle
        self.noisy_oracle = lambda xq : self.oracle(xq, noisy = True)       
            
        if prior is not None:
            f = [ prior(xq) for xq in self.x ]
        else: 
            f = [ 1 for xq in self.x ]        # use uniform prior
        logprior = np.log(f/np.sum(f))

        loedge_state = State(logprior)
        loedge_state.set_xq(self.x[0])
        loedge_state.set_theta(1.)
        
        hiedge_state = State(logprior)
        hiedge_state.set_xq(self.x[-1])
        hiedge_state.set_theta(-1.)
        
        self.trial = [loedge_state, hiedge_state]
        
        self.get_next_x = self.sampling_DQS
        self.query = lambda xq, data : self.S(xq, data, empirical = False)  

        self.mlog = -15.+np.log(1./steps) 


        
    def bisect_ask(self):
        """
        gets the last logf from trial history
        generate the next sampling point 
        """        
        return self.get_next_x(self.trial[-1].get_logf())        

    
    def bisect_tell(self, xq, data):
        """
        pass xq as well, b/c one may actually have changed xq before this step ...
        """

        self.trial.append(
            self.update_posterior(
                self.query(xq, data)
            )
        )
        
        return self.trial[-1]

        
    def run(self, N_iter, K_batches):

        for ie in range(N_iter):

            xq = self.bisect_ask()

            data = []
            for _ in range(K_batches):
                data.append( self.noisy_oracle(xq) )

            self.bisect_tell(xq, data)
                
        return len(self.trial)
    

    def get_posterior_quantile(self, logf, q):

        f = np.exp(logf)
        alpha = np.sum(f) * q
        
        return self.x[np.argmin(np.abs(np.cumsum(f) / np.sum(f) - q))]  

    def sampling_DQS(self, logf):
        """ 

        """
        q = [0.25,0.75]
        #q = [0.5]
        index = len(self.trial)%len(q)
        return self.get_posterior_quantile( logf, q[index] )
        #return self.get_posterior_quantile( logf, random.choice(q) )
    
    def sampling_RQS(self, logf):
        """ 

        """
        q = random.uniform(0,1) 
        return self.get_posterior_quantile( logf, q )        

    
    def S(self, xq, data, empirical=False):
        
        state = self.trial[-1].copy()
        state.set_xq(xq)
        state.set_data(data)
        
        K = len(data)
        state.set_K(K)
        
        Z=0 ; sig=0 ; ZZ=0
        for h,s in data:
            Z+=h ; sig+=s ; ZZ+=h*h
        if K>0: 
            Z/=K ; ZZ/=K ; sig/=K
        if (empirical and K>1): 
            sig=np.sqrt((ZZ-Z*Z)*K/(K-1))

        state.set_Z(Z)
        state.set_sig(sig)

        Bmax = 1
        B = int(Z>0)
        state.set_B(B, Bmax)
        
        
        if (sig>0):
            theta = norm.cdf(np.sqrt(K)*Z/sig)
        else:
            theta = 0.5
        state.set_theta(theta)
        
        return state

            
    def update_posterior(self, state):

        newstate = state.copy()
        xq = newstate._xq
        P = newstate._P
        B = newstate._B
        Bmax = newstate._Bmax

        logP = max(self.mlog,np.log(P))
        logQ = max(self.mlog,np.log(1-P))

        logf = newstate.get_logf()
        logf[self.x >= xq] += ( B*logP+(Bmax-B)*logQ )
        logf[self.x < xq] +=  ( B*logQ+(Bmax-B)*logP )
    
        f = np.exp(logf)
        f /= np.sum(f)
        logf = np.log(f)
        
        newstate.set_logf(logf)
        
        return newstate

    

        
    
def exper(N_iter=10, K_batches=3):

    logger.info("IMPORTANT! To proceed with the next step, close the matplotlib window!")
    
    bis = bisect("test",oracle,-20,20,0.1)
    bis.run(N_iter,K_batches)

    alpha = (1.-0.95)/2
    
    for t in bis.trial:
        posterior_median = bis.get_posterior_quantile(t.get_logf(),0.5)
        left = bis.get_posterior_quantile(t.get_logf(),alpha)
        right = bis.get_posterior_quantile(t.get_logf(),1-alpha)
        xq = t.get_xq()
        logger.info(f"xq = {xq} : x* = {posterior_median} [{left} - {right}]")
        fig, ax = plt.subplots()
        ax.plot(bis.x, np.exp(t.get_logf()))
        ax.set(xlabel='bin', ylabel='posterior',
               title='posterior after update with xq')
        ax.grid()
        plt.show()
        
    return bisect


if __name__ == '__main__':
    exper(20,1)

