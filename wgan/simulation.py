import numpy as np
import scipy.stats as sp


def make_correlation_matrix(no_var):
    corr = np.zeros([no_var,no_var])
    corr_temp = np.random.uniform(-1,1,size=[(no_var-1)*2])
    corr[np.triu_indices(no_var, 1)] = corr_temp
    corr + corr.T + np.eye(no_var)
    return corr


def create_continuous_data(n_samples, n_var=10, n_dependent=0, pos_ratio=0, noise_ratio=0, cov=None, random_state=None):
    """
    Simulate observations from one or two groups of several independent normal distributions and/or a multinormal distribution.
    
    Parameters
    ----------
    n_samples : int
    n_var : int
    n_dependent: int
        Number of dependent variables. n_var - n_dependent determines the number of independent
        variables
    pos_ratio: float
        Part of observations from group 1. Used to simulate a classification task. Must be between 0 and 1.
    cov : matrix
        A covariance matrix can be passed to the multinormal distribution.

    Returns
    -------
    Dictionary with items:
    - X : Simulated data
    - y : Binary group indicator
    - mean0/mean1 : List of true means for group 0/1
    - cov0/cov1 : True covariance matrix for group 0/1
    """
    cov0 = None
    cov1 = None


    if random_state is not None: np.random.seed(random_state)
    # Group indicator
    #group = sp.binom.rvs(p=0.25, n=1, size=N    
    n_neg = int(n_samples*(1-pos_ratio))
    n_pos = n_samples-n_neg
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    idx_dependent = n_var - n_dependent
    
    mean = 0 #np.random.uniform(size=no_var)
    mean0 = np.random.normal(loc=mean,scale=0.5, size=n_var)
    mean1 = np.random.normal(loc=mean,scale=0.5, size=n_var)

    # # Noise are variables with same distribution in majority and minority class
    # if noise_ratio != 0:  
    #     n_noise = int(noise_ratio*n_var)
    #     noise_idx = n_var - n_noise
    #     X_noise = sp.multivariate_normal.rvs(mean=mean0[noise_idx:], cov=cov0[noise_idx:,noise_idx:], 
    #                                          size=n_samples).reshape([n_samples,-1])

    X1 = []
    X0 = []
    # Independent variables
    if n_var-n_dependent>0:
        X1.append(sp.norm.rvs(loc=mean1[:idx_dependent], scale=1, size=[n_pos, n_var-n_dependent]) )
        X0.append(sp.norm.rvs(loc=mean0[:idx_dependent], scale=1, size=[n_neg, n_var-n_dependent]) )

    # Dependent variables
    if n_dependent>0:
        if cov is None: 
            cov0 = sp.invwishart.rvs(df=n_var*2, scale=np.eye(n_dependent))
            cov1 = sp.invwishart.rvs(df=n_var*2, scale=np.eye(n_dependent))

        X1.append( sp.multivariate_normal.rvs(mean=mean1[idx_dependent:], 
                                        cov= cov1, size=n_pos)
        )
        X0.append( sp.multivariate_normal.rvs(mean=mean0[idx_dependent:], 
                                        cov= cov0, size=n_neg)
        )
    
    X0 = np.hstack([*X0])
    X1 = np.hstack([*X1])
    X = np.vstack([X0,X1])
    #X = np.hstack([X, X_noise])
    
    #return {"X":X, "y":y,"mean0":mean0,"mean1":mean1, "cov0":cov0, "cov1":cov1}
    return X, y, mean0, mean1, cov0, cov1
