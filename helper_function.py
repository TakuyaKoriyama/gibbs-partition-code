
import numpy as np
from scipy.special import polygamma, gamma, loggamma
import scipy.optimize as opt
from scipy.special import psi
import math

def simplex(C, alpha):
    k = len(C)
    n = sum(C)
    p = np.zeros(k+1)
    p[:k] = (C-alpha)/n
    p[k] = k/n * alpha
    return p

def compute_r_nk(n,k, alpha, theta, theta_0=0):
    return np.exp(
        loggamma(k+theta/alpha)-loggamma(k+theta_0/alpha)
        +loggamma(n+theta_0)-loggamma(n+theta)
        +loggamma(1+theta)-loggamma(1+theta_0)
        +loggamma(1+theta_0/alpha)-loggamma(1+theta/alpha)
    )


def p_alpha_j(alpha, j):
    return alpha * np.exp(loggamma(j - alpha) - loggamma(j + 1) - loggamma(1 - alpha))

def fisher_info(alpha, iter_num=100000):
    assert 0 < alpha and alpha < 1
    
    N = np.arange(1, iter_num + 1)
    return alpha**(-2) + p_alpha_j(alpha, N)@(1/((N-alpha)*alpha))

def grad_alpha(alpha, C):
    k = len(C)
    return (k-1)/alpha - np.sum(polygamma(0, C-alpha) - polygamma(0, 1-alpha))

def solve_pmle(C):
    k = len(C)
    n = np.sum(C)
    if k == 1:
        return 0
    elif k==n:
        return 1
    else:
        ## mle exists in (0,1)
        alpha_init=0.5
        xmin = alpha_init
        xmax = alpha_init
        while grad_alpha(alpha=xmin, C=C) <= 0:
            xmin = (xmin)/2
        while grad_alpha(alpha=xmax, C=C) >= 0:
            xmax = (1 + xmax)/2
        bracket = [xmin, xmax]
        mle = opt.root_scalar(f=grad_alpha, x0=alpha_init, method='brentq', args=(C), bracket=bracket).root
        assert 0 < mle and mle < 1
        return mle

