import sys
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
from scipy.stats import gennorm
import matplotlib.pyplot as plt
import random as rd
import copy
import seaborn as sns
sns.set(style="ticks", palette="muted")
import warnings
warnings.filterwarnings("ignore")
plt.ion()

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

"-------------------DEFINITION DES FONCTIONS MATHEMATIQUES ----------------------------------------"
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
    #Pour changer la loi de distribution
    #w_t = pa*stats.t.pdf(w_t/s,param_student)/s
    w_t=stats.gennorm.pdf(w_t/s,beta)/s
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

    #Pour changer la loi de distribution
    #w_t = stats.t.pdf(w_t/s,param_student)/s
    w_t=stats.gennorm.pdf(w_t/s,beta)/s
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

    return(pi_t)


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


"--------------------------SIMULE UN MSM-------------------------------"
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
    #dat = sig * gennorm.rvs(beta, size=T)
    dat = dat.reshape(-1,1)
    return (dat,g_s)

def simulatedata_modif(b,m0,gamma_kbar,sig,kbar,T):

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
    #dat = sig*np.sqrt(dat)*np.random.normal(size = T)# VOL TIME SCALING
    dat = sig * gennorm.rvs(beta, size=T)
    dat = dat.reshape(-1,1)
    return (dat,g_s)


"--------------------------OUVERTURE DES FICHIERS----------------------"
T=4086
dat_real = pd.read_csv("BTC-USD-ANAS.csv")  #Etude du BTC pour commencer
dat_real.rename(columns={'Date': 'DATE', 'Price': 'BTC_'}, inplace=True)                          
dat_real = dat_real.loc[dat_real.BTC_ != ","].BTC_.astype(float)
dat_real = np.array(dat_real)
dat2_rtn = dat_real[0:-1]
dat_real = np.log(dat_real[1:])-np.log(dat_real[0:-1])
dat_real = dat_real[dat_real != 0]
dat_real = dat_real[:,np.newaxis]

T=len(dat_real)
param_student=3
niter = 1
temperature = 1.0
stepsize = 1.0
kbar=3
beta=1.5
LL_moy=[]

for i in range (10):
    parameters, LL, niter, output = glo_min(kbar, dat_real, niter, temperature, stepsize)
    LL_moy.append(LL)
# name parameters for later use:
b = parameters[0]
m0 = parameters[1]
gamma_kbar = parameters[2]
sigma = parameters[3]

mean = np.mean(LL_moy)
sem = stats.sem(LL_moy)  # standard error of the mean
margin_of_error = stats.t.ppf(0.975, len(LL_moy)-1) * sem  # 0.975 pour 95%
print(f"Intervalle de confiance à 95% pour la LL du MSM sans piecewise sur des données BTC: {mean:.4f} ± {margin_of_error:.4f}")

"""
print("Parameters from glo_min for BTC: ", "\n"
      "kbar = ", kbar,"\n"
      'b = %.5f' % b,"\n"
      'm_0 = %.5f' % m0,"\n"
      'gamma_kbar = %.5f' % gamma_kbar,"\n"
      'sigma = %.5f' % (sigma*np.sqrt(252)),"\n"
      'Likelihood = %.5f' % LL,"\n"
      "niter = " , niter,"\n"
      "output = " , output,"\n")
"""


#Fonction effectuant le découpage et l'estimation piecewise
def piecewise(dates,dat_real):
    debut=dates[0]
    vect_LL=[]
    simul_cumul = np.array([]).reshape(0, 1)
    ls_param=[]
    for i in range (1,len(dates)):
        end=dates[i]
        temp=dat_real[debut:end+1]
        niter = 1
        temperature = 1.0
        stepsize = 1.0
        parameters, LL_temp, niter, output = glo_min(kbar, temp, niter, temperature, stepsize)
        ls_param.append(parameters)
        vect_LL.append(LL_temp)
        b = parameters[0]
        m0 = parameters[1]
        gamma_kbar = parameters[2]
        sigma = parameters[3]
        dat_temp,g_s = simulatedata_modif(b,m0,gamma_kbar,sigma,kbar,len(temp)-1)
        simul_cumul=np.concatenate((simul_cumul,dat_temp))
        debut=end
    for i in range (1,len(vect_LL)):
        vect_LL[i]+=vect_LL[i-1]
    return simul_cumul,vect_LL,ls_param

#Définition des dates de découpage du jeu de données selon les différents critères haussier/baissier, volatilité ou aléatoire
dates_norm_hausse=['2014-01-01','2014-10-26','2015-10-13','2018-05-21','2019-04-29', '2019-11-20', 
            '2020-01-25','2020-03-06','2020-05-11','2021-07-04','2021-07-25', '2021-12-27'
            ,'2023-02-14','2025-03-09']
dates_hausse=[0,299,651,1602,1945,2150,2216,2257,2323,2742,2763,2918,3332,4085]
dates_norm_vol=['2014-05-28'	'2015-01-04'
'2015-01-05'	'2015-03-25'
'2015-03-26'	'2015-11-08'
'2015-11-09'	'2016-03-01'
'2016-03-02'	'2016-06-01'
'2016-06-02'	'2016-08-29'
'2016-08-30'	'2017-06-25'
'2017-06-26'	'2017-09-23'
'2017-09-24'	'2017-12-11'
'2017-12-12'	'2018-03-28'
'2018-03-29'	'2018-11-19'
'2018-11-20'	'2019-01-31'
'2019-02-01'	'2019-05-15'
'2019-05-16'	'2019-08-23'
'2019-08-24'	'2020-02-28'
'2020-02-29'	'2020-05-11'
'2020-05-12'	'2020-12-29'
'2020-12-30'	'2022-12-17'
'2022-12-18'	'2023-02-24'
'2023-02-25'	'2023-07-18'
'2023-07-19'	'2024-03-07'
'2024-03-08'	'2024-11-30']
dates_vol=[0,146,368,448,676,790,882,971,1271,1361,1440,1547,1783,1856,1961,2060,2249,2322,2554,3272,3341,3485,3718,3986,4085]
dates_rand_hausse=rd.sample(range(1, 4085), 12)
dates_rand_hausse =[0]+dates_rand_hausse+[4085]
dates_rand_hausse.sort()
dates_rand_vol=rd.sample(range(1,4085),23)
dates_rand_vol =[0]+dates_rand_vol+[4085]
dates_rand_vol.sort()
LL_moy=[]
N_simul=10

#Etude du BTC
for i in range (N_simul):
    dat1,vect_LL,ls_param=piecewise(dates_rand_vol,dat_real)
    LL_moy.append(vect_LL[-1])




"""
#Pour simuler les returns du piecewise MSM
taille_plage=[]
LL_points=[]
for i in range (1,75):
    dates_rand=[]
    if i>1:
        dates_rand=rd.sample(range(1,4085),i-1)
    dates_rand =[0]+dates_rand+[4085]
    dates_rand.sort()
    dat1,vect_LL,ls_param=piecewise(dates_rand,dat_real)
    LL_points.append(vect_LL[-1])
    taille_plage.append(i)
    print(i)
"""
"-------------------------FONCTION D'AFFICHAGE-----------------------"

def tsdisplay(y,z,both, figsize = (16,9), title = "", color = "",color_bis=""):
    tmp_data = pd.Series(y)
    other_data=pd.Series(z)
    
    fig = plt.figure(figsize = figsize)
    # Plot time series
    ax1 = fig.add_subplot(311)
    tmp_data.plot(ax=ax1 , title = "$Log\ returns\ time\ series:\ " + title + "$", legend = False, color=color, linewidth=0.5, alpha=0.75)
    if (both):
        other_data.plot(ax=ax1,title = "$Log\ returns\ time\ series:\ " + title + "$", legend = False, color=color_bis, linewidth=0.5, alpha=0.75)
    # Plot ACF:
    sm.graphics.tsa.plot_acf(tmp_data, lags = 20, zero = False, color=color, ax = fig.add_subplot(323))
    # Plot PACF:
    sm.graphics.tsa.plot_pacf(tmp_data, lags = 20, zero = False, color=color, ax = fig.add_subplot(324))
    # QQ plot of the data:
    sm.qqplot(tmp_data, line='s', color=color, ax = fig.add_subplot(325)) 
    plt.title("Q-Q Plot")
    # Plot residual histogram:
    ax=fig.add_subplot(326)
    ax.hist(tmp_data, color=color, bins = 120,density=True)
    if (both):
            ax.hist(pd.Series(z),color=color_bis,bins=120,density=True,alpha=0.5)
    plt.title("Histogram")
    # Fix layout of the plots:
    plt.tight_layout()

def tsdisplay_simple(y, z, both, figsize=(16, 9), title="", color="", color_bis=""):
    tmp_data = pd.Series(y)
    other_data = pd.Series(z)

    plt.figure()
    tmp_data.plot(title=r"$Log\ returns\ time\ series:\ " + title + "$", legend=False, color=color, linewidth=0.5, alpha=0.75)
    if both:
        other_data.plot(title=r"$Log\ returns\ time\ series:\ " + title + "$", legend=False, color=color_bis, linewidth=0.5, alpha=0.75)
    plt.show()

    # Histogram part
    plt.figure()

    # Définir les bords communs pour les bins
    all_data = np.concatenate([y, z]) if both else y
    bins = np.histogram_bin_edges(all_data, bins=120)

    plt.hist(y, color=color, bins=bins, density=True, alpha=0.75, label='Simulated')
    if both:
        plt.hist(z, color=color_bis, bins=bins, density=True, alpha=0.5, label='Real')

    plt.title("Histogram")
    plt.tight_layout()
    plt.legend()
    plt.show()

def tsdisplay_trie(y,z,both, figsize = (16,9), title = "", color = "",color_bis=""):
    y_bis=y.copy()
    z_bis=z.copy()
    y_bis.sort()
    z_bis.sort()
    tmp_data = pd.Series(y_bis)
    other_data=pd.Series(z_bis)
    plt.figure()
    tmp_data.plot(title = "$Log\ returns\ time\ series:\ " + title + "$", legend = False, color=color, linewidth=0.5, alpha=0.75)
    if (both):
        other_data.plot(title = "$Log\ returns\ time\ series:\ " + title + "$", legend = False, color=color_bis, linewidth=0.5, alpha=0.75)
    plt.show()

"""
s = np.array(dat_real).astype(float)
s = s[s != 0].copy()
t=s.copy()

tsdisplay(s, t,False,title = "BTC\ daily", color='green')


s = np.array(dat1).astype(float)
s = s[s != 0].copy()

tsdisplay(s,t,False,title = "MSM simulated daily returns", color='red')
tsdisplay(s,t,True,title = "MSM simulated daily returns", color='red',color_bis='green')

s = np.array(dat_real).astype(float)
s = s[s != 0].copy()
t=s.copy()

tsdisplay_simple(s, t,False,title = "BTC\ daily", color='green')


s = np.array(dat1).astype(float)
s = s[s != 0].copy()

tsdisplay_simple(s,t,False,title = "MSM simulated daily returns", color='red')
tsdisplay_simple(s,t,True,title = "MSM simulated daily returns", color='red',color_bis='green')
tsdisplay_trie(s,t,True,title="Returns triés",color='red',color_bis='green')

"""

print("Le modèle piecewise donne, pour le découpage sélectionné, une LL moyenne qui vaut : ", np.mean(LL_moy)," pour le BTC-USD")
#print(dates_rand_vol)
#plt.plot(taille_plage,LL_points)
#plt.show(block=True)


#Etude des données ETH
dat_real = pd.read_csv("ETH-USD.csv")
dat_real.rename(columns={'Date': 'DATE', 'Price': 'BTC_'}, inplace=True)                          
dat_real = dat_real.loc[dat_real.BTC_ != ","].BTC_.astype(float)
dat_real = np.array(dat_real)
dat2_rtn = dat_real[0:-1]
dat_real = np.log(dat_real[1:])-np.log(dat_real[0:-1])
dat_real = dat_real[dat_real != 0]
dat_real = dat_real[:,np.newaxis]

T=len(dat_real)
LL_moy=[]

for i in range (10):
    parameters, LL, niter, output = glo_min(kbar, dat_real, niter, temperature, stepsize)
    LL_moy.append(LL)
# name parameters for later use:
b = parameters[0]
m0 = parameters[1]
gamma_kbar = parameters[2]
sigma = parameters[3]

mean = np.mean(LL_moy)
sem = stats.sem(LL_moy)  # standard error of the mean
margin_of_error = stats.t.ppf(0.975, len(LL_moy)-1) * sem  # 0.975 pour 95%
print(f"Intervalle de confiance à 95% pour la LL du MSM sans piecewise sur des données ETH: {mean:.4f} ± {margin_of_error:.4f}")
dates_norm_hausse=[
'2017-11-09','2019-05-12'
'2019-05-13',	'2019-11-10'
'2019-11-11',	'2020-04-23'
'2020-04-24',	'2022-02-09'
'2022-02-10',	'2023-03-10'
'2023-03-11',	'2024-08-03'
'2024-08-04',	'2024-11-08'
'2024-11-09',	'2025-03-08']
dates_hausse=[0,549,731,896,1553,1582,1947,2459,2556,2676]
dates_norm_vol=['2018-12-08'	'2019-02-18'
'2019-02-19	''2019-06-03'
'2019-06-04'	'2019-11-07'
'2019-11-08'	'2020-02-23'
'2020-02-24'	'2020-05-05'
'2020-05-06'	'2020-12-27'
'2020-12-28'	'2021-03-03'
'2021-03-04'	'2021-05-08'
'2021-05-09'	'2021-10-30'
'2021-10-31'	'2022-05-30'
'2022-05-31'	'2022-12-23'
'2022-12-24'	'2024-03-20'
'2024-03-21'	'2024-11-29']
dates_vol=[0,394,466,571,836,908,1144,1210,1276,1451,1663,1870,2323,2676]
dates_rand_hausse=rd.sample(range(1, 2675), 8)
dates_rand_hausse =[0]+dates_rand_hausse+[2676]
dates_rand_hausse.sort()
dates_rand_vol=rd.sample(range(1,2675),12)
dates_rand_vol =[0]+dates_rand_vol+[2676]
dates_rand_vol.sort()
LL_moy=[]


for i in range (N_simul):
    dat1,vect_LL,ls_param=piecewise(dates_rand_vol,dat_real)
    LL_moy.append(vect_LL[-1])

print("Le modèle piecewise donne, pour le découpage sélectionné, une LL moyenne qui vaut : ",np.mean(LL_moy), " pour le ETH-USD")