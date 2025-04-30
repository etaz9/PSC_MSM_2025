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
import matplotlib.pyplot as plt
import random as rd
import copy
import openpyxl
from openpyxl.styles import Alignment
import csv
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
    w_t = stats.t.pdf(w_t/s,param_student)/s
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
    w_t = stats.t.pdf(w_t/s,param_student)/s
    
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
    dat = dat.reshape(-1,1)
    return (dat,g_s)
    

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
    ax.hist(tmp_data, color=color, bins = 120)
    if (both):
            ax.hist(pd.Series(z),color=color_bis,bins=120,alpha=0.5)
    plt.title("Histogram")
    # Fix layout of the plots:
    plt.tight_layout()

def ecart_hist(s,t):
    hist1,_=np.histogram(s,bins=120,density=True)
    hist2,_=np.histogram(t,bins=120,density=True)
    hist1+=1e-10
    hist2+=1e-10
    kl_divergence = stats.entropy(hist1, hist2)
    print("La divergence de Kullback Leibler vaut: ",kl_divergence)

"--------------------------OUVERTURE DES FICHIERS----------------------"

T=4086
debut=3654
end=4018
dat_real = pd.read_csv("BTC-USD.csv",header=None, names=['DATE', 'BTC_'])                           
dat_real = dat_real.loc[dat_real.BTC_ != ","].BTC_.astype(float)
dat_real = np.array(dat_real)
dat2_rtn = dat_real[0:-1]
dat_real = np.log(dat_real[1:])-np.log(dat_real[0:-1])
dat_real = dat_real[dat_real != 0]
dat_real = dat_real[:,np.newaxis]
dat_real=dat_real[debut:end]
mu=np.mean(dat_real)
dat_real=[elem -mu for elem in dat_real]
T=len(dat_real)


niter = 1
temperature = 1.0
stepsize = 1.0
vect_LL=[]
best_LL=0
best_kbar=0
best_df=0
# Initialisation de best_LL, best_kbar, best_df
best_LL = float('inf')
best_kbar = None
best_df = None

# Création d'un nouveau classeur Excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Résultats log-likelihood"

# Création des en-têtes
ws.cell(row=1, column=1).value = "kbar"
for df in range(1, 16):
    ws.cell(row=1, column=df+1).value = f"df={df}"

# Boucle pour remplir les données
for kbar in range(1, 8):  # kbar marque les lignes
    # Définir l'index de la ligne dans le fichier Excel
    excel_row = kbar + 1
    
    # Ajouter le numéro kbar dans la première colonne
    ws.cell(row=excel_row, column=1).value = f"k={kbar}"
    
    for df in range(1, 16):  # df marque les colonnes
        # Définir l'index de la colonne dans le fichier Excel
        excel_col = df + 1
        
        print(f"Traitement de kbar={kbar}, df={df}")
        param_student = df
        parameters, LL, niter, output = glo_min(kbar, dat_real, niter, temperature, stepsize)
        
        # Stockage des valeurs
        if LL < best_LL:
            best_LL = LL
            best_kbar = kbar
            best_df = df
        
        # Formatage de la valeur pour chaque case avec des sauts de ligne
        cell_text = f"LL= {LL}\nb= {parameters[0]}\nm0= {parameters[1]}\ngamma_kbar= {parameters[2]}\nsigma= {parameters[3]}"
        
        # Placer le texte dans la cellule
        cell = ws.cell(row=excel_row, column=excel_col)
        cell.value = cell_text
        
        # Configurer la cellule pour afficher correctement les sauts de ligne
        cell.alignment = Alignment(wrap_text=True, vertical='top')
    
# Ajuster automatiquement la largeur des colonnes
for col in ws.columns:
    max_length = 0
    column = col[0].column_letter
    for cell in col:
        if cell.value:
            max_length = max(max_length, len(str(cell.value).split('\n')[0]))
    adjusted_width = max_length + 2
    ws.column_dimensions[column].width = adjusted_width

# Ajuster la hauteur des lignes pour qu'elles affichent bien les sauts de ligne
for row in range(2, 9):  # Lignes 2 à 8 (pour kbar 1 à 7)
    ws.row_dimensions[row].height = 120  # Hauteur en points

# Enregistrer le fichier Excel
excel_filename = "Valeurs_log-likelihood_kbar_df_2024.xlsx"
try:
    wb.save(excel_filename)
    print(f"Fichier Excel créé avec succès: {excel_filename}")
    print(f"Meilleures valeurs trouvées: kbar={best_kbar}, df={best_df}, LL={best_LL}")
except PermissionError:
    # Si le fichier est ouvert ailleurs ou protégé
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"Valeurs_log-likelihood_kbar_df_2024_{timestamp}.xlsx"
    wb.save(excel_filename)
    print(f"Fichier Excel créé avec succès: {excel_filename}")

print(f"Meilleures valeurs trouvées: kbar={best_kbar}, df={best_df}, LL={best_LL}")

print(f"La meilleure log-likelihood est {best_LL},atteinte pour kbar={best_kbar} et df={best_df}")



