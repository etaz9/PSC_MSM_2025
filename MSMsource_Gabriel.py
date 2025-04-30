# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:54:49 2024

@author: brioche
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from statsmodels import regression
import statsmodels.formula.api as sm
from numba import jit, njit, prange, float64, int64
import os
import glob
from datetime import datetime, timedelta, date
from scipy import stats
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random as rd
import copy
import seaborn as sns
sns.set(style="ticks", palette="muted")
import warnings
warnings.filterwarnings("ignore")


"--------------------FONCTIONS DE MINIMISATION ----------------------"

def glo_min(kbar, data, niter, temperature, stepsize):
    """2-step basin-hopping method combines global stepping algorithm
       with local minimization at each step.
    """

    """step 1: local minimizations
    """
    theta, theta_LLs, theta_out, ierr, numfunc = loc_min(kbar, data)

    """step 2: global minimum search uses basin-hopping
       (scipy.optimize.basinhopping)
    """
    # objective function
    f = g_LLb_h
    # x0 = initial guess, being theta, from Step 1.
    # Presents as: [b, m0, gamma_kbar, sigma]
    x0 = theta

    # basinhopping arguments
    niter = niter
    T = temperature
    stepsize = stepsize
    args = (kbar, data)

    # bounds
    bounds = ((1.001,50),(1,1.99),(1e-3,0.999999),(1e-4,5))

    # minimizer_kwargs
    minimizer_kwargs = dict(method = "L-BFGS-B", bounds = bounds, args = args)

    res = opt.basinhopping(
                func = f, x0 = x0, niter = niter, T = T, stepsize = stepsize,
                minimizer_kwargs = minimizer_kwargs)

    parameters, LL, niter, output = res.x,res.fun,res.nit,res.message

    return(parameters, LL, niter, output)

def loc_min(kbar, data):
    """step 1: local minimization
       parameter estimation uses bounded optimization (scipy.optimize.fminbound)
    """

    # set up
    b = np.array([1.5, 5, 15, 30])
    lb = len(b)
    gamma_kbar = np.array([0.1, 0.5, 0.9, 0.95])
    lg = len(gamma_kbar)
    sigma = np.std(data)

    # templates
    theta_out = np.zeros(((lb*lg),3))
    theta_LLs = np.zeros((lb*lg))

    # objective function
    f = g_LL

    # bounds
    m0_l = 1
    m0_u = 2

    # Optimizaton stops when change in x between iterations is less than xtol
    xtol = 1e-05

    # display: 0, no message; 1, non-convergence; 2, convergence;
    # 3, iteration results.
    disp = 1

    idx = 0
    for i in range(lb):
        for j in range(lg):

            # args
            theta_in = [b[i], gamma_kbar[j], sigma]
            args = (kbar, data, theta_in)

            xopt, fval, ierr, numfunc = opt.fminbound(
                        func = f, x1 = m0_l, x2 = m0_u, xtol = xtol,
                        args = args, full_output = True, disp = disp)

            m0, LL = xopt, fval
            theta_out[idx,:] = b[i], m0, gamma_kbar[j]

            theta_LLs[idx] = LL
            idx +=1

    idx = np.argsort(theta_LLs)

    theta_LLs = np.sort(theta_LLs)

    theta = theta_out[idx[0],:].tolist()+[sigma]
    theta_out = theta_out[idx,:]

    return(theta, theta_LLs, theta_out, ierr, numfunc)

"-------------------DEFINITION----------------------------------------"
def g_LL(m0, kbar, data, theta_in):
    """return LL, the vector of log likelihoods
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    sigma = theta_in[2]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = sigma*g_m

    # returns
    w_t = data
    
    #w_t = pa * stats.t.pdf(w_t / s, df=2) / s
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s
    w_t = w_t + 1e-16

    # log likelihood using numba
    LL = _LL(kbar2, T, A, g_m, w_t)

    return(LL)


@jit(nopython=True)
def _LL(kbar2, T, A, g_m, w_t):
    """speed up Bayesian recursion with numba
    """

    LLs = np.zeros(T)
    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    for t in range(T):

        piA = np.dot(pi_mat[t,:],A)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)

        if abs(ft-0) <= 1e-05:
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft

        # vector of log likelihoods
        LLs[t] = np.log(np.dot(w_t[t,:],piA))

    LL = -np.sum(LLs)

    return(LL)


def g_pi_t(m0, kbar, data, theta_in):
    """return pi_t, the current distribution of states
    """

    # set up
    b = theta_in[0]
    gamma_kbar = theta_in[1]
    sigma = theta_in[2]
    kbar2 = 2**kbar
    T = len(data)
    pa = (2*np.pi)**(-0.5)
    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    # gammas and transition probabilities
    A = g_t(kbar, b, gamma_kbar)

    # switching probabilities
    g_m = s_p(kbar, m0)

    # volatility model
    s = sigma*g_m

    # returns
    w_t = data
    #w_t = pa * stats.t.pdf(w_t / s, df=2) / s
    
    w_t = pa*np.exp(-0.5*((w_t/s)**2))/s

    w_t = w_t + 1e-16

    # compute pi_t with numba acceleration
    pi_t = _t(kbar2, T, A, g_m, w_t)

    return(pi_t)


@jit(nopython=True)
def _t(kbar2, T, A, g_m, w_t):

    pi_mat = np.zeros((T+1,kbar2))
    pi_mat[0,:] = (1/kbar2)*np.ones(kbar2)

    for t in range(T):

        piA = np.dot(pi_mat[t,:],A)
        C = (w_t[t,:]*piA)
        ft = np.sum(C)
        if abs(ft-0) <= 1e-05:
            pi_mat[t+1,1] = 1
        else:
            pi_mat[t+1,:] = C/ft

    pi_t = pi_mat[-1,:]

    #return(pi_t)
    return(pi_mat)


class memoize(dict):
    """use memoize decorator to speed up compute of the
       transition probability matrix A
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

@memoize
def  g_t(kbar, b, gamma_kbar):
    """return A, the transition probability matrix
    """

    # compute gammas
    gamma = np.zeros((kbar,1))
    gamma[0,0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    for i in range(1,kbar):
        gamma[i,0] = 1-(1-gamma[0,0])**(b**(i))
    gamma = gamma*0.5
    gamma = np.c_[gamma,gamma]
    gamma[:,0] = 1 - gamma[:,0]

    # transition probabilities
    kbar2 = 2**kbar
    prob = np.ones(kbar2)

    for i in range(kbar2):
        for m in range(kbar):
            tmp = np.unpackbits(
                        np.arange(i,i+1,dtype = np.uint16).view(np.uint8))
            tmp = np.append(tmp[8:],tmp[:8])
            prob[i] =prob[i] * gamma[kbar-m-1,tmp[-(m+1)]]

    A = np.fromfunction(
        lambda i,j: prob[np.bitwise_xor(i,j)],(kbar2,kbar2),dtype = np.uint16)

    return(A)


def j_b(x, num_bits):
    """vectorize first part of computing transition probability matrix A
    """

    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2**np.arange(num_bits).reshape([1, num_bits])

    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


@jit(nopython=True)
def s_p(kbar, m0):
    """speed up computation of switching probabilities with Numba
    """

    # switching probabilities
    m1 = 2-m0
    kbar2 = 2**kbar
    g_m = np.zeros(kbar2)
    g_m1 = np.arange(kbar2)

    for i in range(kbar2):
        g = 1
        for j in range(kbar):
            if np.bitwise_and(g_m1[i],(2**j))!=0:
                g = g*m1
            else:
                g = g*m0
        g_m[i] = g

    return(np.sqrt(g_m))


def g_LLb_h(theta, kbar, data):
    """bridge global minimization to local minimization
    """

    theta_in = unpack(theta)
    m0 = theta[1]
    LL = g_LL(m0, kbar, data, theta_in)

    return(LL)


def unpack(theta):
    
    
    """unpack theta, package theta_in
    """
    b = theta[0]
    m0 = theta[1]
    gamma_kbar = theta[2]
    sigma = theta[3]

    theta_in = [b, gamma_kbar, sigma]

    return(theta_in)


# "--------------------------SIMULE UN MSM-------------------------------"


def simulatedata(b,m0,gamma_kbar,sig,kbar,T):

    m0 = m0
    m1 = 2-m0
    g_s = np.zeros(kbar)
    M_s = np.zeros((kbar,T))
    g_s[0] = 1-(1-gamma_kbar)**(1/(b**(kbar-1)))
    
    for i in range(1,kbar):
        g_s[i] = 1-(1-g_s[0])**(b**(i))
    #g_s[0]=1/1460
    for j in range(kbar):
        M_s[j,:] = np.random.binomial(1,g_s[j],T)
    dat = np.zeros(T)
    tmp = (M_s[:,0]==1)*m1+(M_s[:,0]==0)*m0
    dat[0] = np.prod(tmp)
    for k in range(1,T):
        for j in range(kbar):
            if M_s[j,k]==1:
                tmp[j] = np.random.choice([m0,m1],1,p = [0.5,0.5])
        dat[k] = np.prod(tmp)
    dat = sig*np.sqrt(dat)*np.random.normal(size = T)# VOL TIME SCALING
    dat = dat.reshape(-1,1)
    return (dat,g_s)
    
# # "--------------------------OUVERTURE DES FICHIERS----------------------"



# # "-------------------------FONCTION D'AFFICHAGE-----------------------"

# def tsdisplay(y,z,both, figsize = (16,9), title = "", color = "",color_bis=""):
#     tmp_data = pd.Series(y)
#     other_data=pd.Series(z)
    
#     fig = plt.figure(figsize = figsize)
#     # Plot time series
#     ax1 = fig.add_subplot(311)
#     tmp_data.plot(ax=ax1 , title = "$Log\ returns\ time\ series:\ " + title + "$", legend = False, color=color, linewidth=0.5, alpha=0.75)
#     if (both):
#         other_data.plot(ax=ax1,title = "$Log\ returns\ time\ series:\ " + title + "$", legend = False, color=color_bis, linewidth=0.5, alpha=0.75)
#     # Plot ACF:
#     sm.graphics.tsa.plot_acf(tmp_data, lags = 20, zero = False, color=color, ax = fig.add_subplot(323))
#     # Plot PACF:
#     sm.graphics.tsa.plot_pacf(tmp_data, lags = 20, zero = False, color=color, ax = fig.add_subplot(324))
#     # QQ plot of the data:
#     sm.qqplot(tmp_data, line='s', color=color, ax = fig.add_subplot(325)) 
#     plt.title("Q-Q Plot")
#     # Plot residual histogram:
#     ax=fig.add_subplot(326)
#     ax.hist(tmp_data, color=color, bins = 120)
#     if (both):
#             ax.hist(pd.Series(z),color=color_bis,bins=120,alpha=0.5)
#     plt.title("Histogram")
#     # Fix layout of the plots:
#     plt.tight_layout()


# # "------------------------SIMULATION----------------------------------"
def compute_MSM_vol(data_r, parameters,k_bar):
 
    # name parameters for later use:
    b = parameters[0]
    m0 = parameters[1]
    gamma_kbar = parameters[2]
    sigma = parameters[3]

    pi_mat = g_pi_t(m0,k_bar,data_r,[b, gamma_kbar, sigma])
    # plt.imshow(pi_mat, aspect='auto', cmap="viridis")
    # plt.colorbar()
    # plt.show()

    def compute_state_vols(m0, k_bar, sigma):
        m1 = 2-m0
        """Computes the volatility values for each of the 2**k_bar states in the MSM."""
        states = np.array([m0**(k_bar - np.sum(np.array(list(np.binary_repr(i, width=k_bar)), dtype=int))) *
                        m1**(np.sum(np.array(list(np.binary_repr(i, width=k_bar)), dtype=int)))
                        for i in range(2**k_bar)])
        states = sigma*np.sqrt(states)
        return states

    def msm_volatility(pi_t, m0, k_bar, sigma):
        state_vols = compute_state_vols(m0, k_bar, sigma)
        return np.dot(pi_t, state_vols)
                    

    return([np.array([msm_volatility(pi_t, m0, k_bar, sigma) for pi_t in pi_mat]),gamma_kbar])

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(volatilities, label='MSM Volatility', color='tab:blue', linewidth=2)
# ax1.set_xlabel('Time')
# ax1.set_ylabel('MSM Volatility', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# ax1.set_title('MSM Volatility Over Time')
# ax2 = ax1.twinx()  # Create a second y-axis
# ax2.plot(dat_real, label='Returns', linestyle='dashed', color='tab:orange', linewidth=2, alpha=0.7)
# ax2.set_ylabel('Return BTC', color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')
# ax1.grid(True)
# fig.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

# dat1,g_s = simulatedata(b,m0,gamma_kbar,sigma,kbar,T)
# #dat1=dat1+mu


# # "-----------------------ANALYSE DU RESULTAT------------------------------"

# j = stats.describe(dat1)
# print("Descriptive statistics for Simulated Data: ","\n"
#       "Number of observations = ",j.nobs,"\n"
#       "Minimum, Maximum = ",str(j.minmax),"\n"
#       "Mean = %.5f" %  j.mean,"\n"
#       "Variance = %.5f" %  j.variance,"\n"
#       "Standard deviation = %.5f" %  j.variance**0.5,"\n"
#       "Skewness = %.5f" %  j.skewness,"\n"
#       "Kurtosis = %.5f" %  j.kurtosis)

# k = stats.describe(dat_real)
# print("Descriptive statistics for BTC: ","\n"
#       "Number of observations = ",k.nobs,"\n"
#       "Minimum, Maximum = ",str(k.minmax),"\n"
#       "Mean = %.5f" %  k.mean,"\n"
#       "Variance = %.5f" %  k.variance,"\n"
#       "Standard deviation = %.5f" %  k.variance**0.5,"\n"
#       "Skewness = %.5f" %  k.skewness,"\n"
#       "Kurtosis = %.5f" %  k.kurtosis)

# # "-------------------------AFFICHAGE DES RESULTATS--------------------"
    
# s = np.array(dat_real).astype(float)
# s = s[s != 0].copy()
# t=s.copy()

# tsdisplay(s, t,False,title = "BTC\ daily", color='green')


# s = np.array(dat1).astype(float)
# s = s[s != 0].copy()

# tsdisplay(s,t,False,title = "MSM simulated daily returns", color='red')
# tsdisplay(s,t,True,title = "MSM simulated daily returns", color='red',color_bis='green')


# # "-------------------------DIVERGENCE DE KULLBACK LEIBLER--------------"
# def ecart_hist(s,t):
#     hist1,_=np.histogram(s,bins=120,density=True)
#     hist2,_=np.histogram(t,bins=120,density=True)
#     hist1+=1e-10
#     hist2+=1e-10
#     kl_divergence = stats.entropy(hist1, hist2)
#     print("La divergence de Kullback Leibler vaut: ",kl_divergence)

# ecart_hist(s,t)

# # "------------------------------------------------------------------------"

# niter = 1
# temperature = 1.0
# stepsize = 1.0

# parameters, LL, niter, output = glo_min(kbar, dat1, niter, temperature, stepsize)

# # name parameters for later use:
# b_sim = parameters[0]
# m_0_sim = parameters[1]
# gamma_kbar_sim = parameters[2]
# sigma_sim = parameters[3]
# LL_sim = LL

# print("Parameters from glo_min for Simulated dataset: ", "\n"
#       "kbar = ", kbar,"\n"
#       'b = %.5f' % b_sim,"\n"
#       'm_0 = %.5f' % m_0_sim,"\n"
#       'gamma_kbar = %.5f' % gamma_kbar_sim,"\n"
#       'sigma = %.5f' % (sigma_sim*np.sqrt(252)),"\n"
#       'Likelihood = %.5f' % LL_sim,"\n"
#       "niter = " , niter,"\n"
#       "output = " , output,"\n")



# print("Parameters from glo_min for BTC: ", "\n"
#       "kbar = ", kbar,"\n"
#       'b = %.5f' % b,"\n"
#       'm_0 = %.5f' % m0,"\n"
#       'gamma_kbar = %.5f' % gamma_kbar,"\n"
#       'sigma = %.5f' % (sigma*np.sqrt(252)),"\n"
#       'Likelihood = %.5f' % LL,"\n"
#       "niter = " , niter,"\n"
#       "output = " , output,"\n")

# # "--------------------------AFFICHAGE DES PRIX----------------------------"

# plt.figure()
# P_0=752
# prices = P_0 * np.exp((np.cumsum(dat_real)))      
# plt.plot(prices,'r',label="Données réelles")
# ind=np.zeros_like(prices)
# fit=np.zeros_like(prices)
# for i in range (len(ind)):
#     ind[i]=i
#     fit[i]=1.074257096378921e-05*(i+1)**2.602175793359495
# plt.plot(ind,fit)
# x=[921 for i in range (70000)]
# y=np.linspace(0,70000,70000)
# plt.plot(x,y,'g')
# x=[2324 for i in range (70000)]
# plt.plot(x,y,'g')
# x=[3762 for i in range (70000)]
# plt.plot(x,y,'g')
# plt.figure()
# prices2=P_0*np.exp((np.cumsum(dat1)))

# plt.plot(prices2,'b',label="Données simulées")
# plt.show()