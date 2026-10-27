# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:07:32 2024

@author: cleme
"""
import importlib
import numpy as np
import scipy as sc
from scipy.special import jv
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpmath
from mpmath import *
import time
import pandas as pd

#%%
def J1(z):
    return jv(1, z)

#%%
def I(r, I0, d0):
    """Intensity profile with an airy distribution
    Args:
        r (nD array): position
        I0 (float): max intensity
        d0 (float): FWHM
    """
    arg = 1.62*2*r/d0
    B = 2*J1(arg)
    return(I0 *B*B/(arg*arg))

#%%
def Igaussian(r, I0, d0):
    """Intensity profile with an airy distribution
    Args:
        r (nD array): position
        I0 (float): max intensity
        d0 (float): FWHM
    """
    arg = -4*np.log(2)*r*r/(d0*d0)
    return(I0 *np.exp(arg))
#%%
def integ_max(t, d0,f,delta_y, n=2):
    """quantity to be integrated : DTT = integrande * sigma_ext
    Args:
        t (1D array): time
        d0 (float): FWHM
        f (float): modulation frequency
        delta_y (float): position modulation amplitude
    """
    if n%2==0:
        B = np.cos(2*n*np.pi*f*t)
    else :
        B = np.sin(2*n*np.pi*f*t)

    intensity = I(delta_y*np.sin(2*np.pi*f*t),1,d0)
    prefactor = 0.8316/(d0*d0)*f*np.pi
    return(prefactor * intensity* B)

#%%
def integ_maxgaussian(t, d0,f,delta_y, n=2):
    """quantity to be integrated : DTT = integrande * sigma_ext
    Args:
        t (1D array): time
        d0 (float): FWHM
        f (float): modulation frequency
        delta_y (float): position modulation amplitude
    """
    if n%2==0:
        B = np.cos(2*n*np.pi*f*t)
    else :
        B = np.sin(2*n*np.pi*f*t)

    intensity = Igaussian(delta_y*np.sin(2*np.pi*f*t),1,d0)
    #prefactor = 0.8316/(d0*d0)*f*2
    prefactor = 4*np.log(2)*2/(d0*d0)*f*2/np.pi
    return(prefactor * intensity* B)
#%%
def R2F_int(d0, delta_y):
    """R2F response in not-simplyfied form.
    We must use sigma_ext = DTT / R2F_int
    Args:
        d_y (float or 1D array): modulation amplitude
        d0 (float or 1D array): fwhm

    Returns:
        res (nD array): The max signal. 1D if d0 or delta_y is an array, 2D if both are, float otherwise
    """

    f = 1000 #modulation frequency: it doesn't change the result.
    n=2 #we work with second harmonic.
    T = np.linspace(-1/(2*f),1/(2*f), 200)

    # print(type(d0))
    # print(type(delta_y))
    # print('-----------------')

    if np.size(d0)==1 and np.size(delta_y)==1:
        integrande = integ_max(T,d0, f, delta_y, n=n)
        DTT = np.trapz(integrande,dx =T[1]-T[0])

    elif np.size(d0)==1:
        TT, DDelta_y = np.meshgrid(T,delta_y)
        integrande = integ_max(TT,d0,f,DDelta_y, n=n)
        DTT = np.trapz(integrande,dx =T[1]-T[0],axis=1)

    elif np.size(delta_y)==1:        
        TT, DD0 = np.meshgrid(T,d0)
        integrande = integ_max(TT,DD0,f,delta_y, n=n)
        DTT = np.trapz(integrande,dx =T[1]-T[0],axis=1)

    else :
        shape = (np.size(T),np.size(d0),np.size(delta_y))
        DDD0 = np.zeros(shape)
        DDDelta_y = np.zeros(shape)
        TTT = np.zeros(shape)

        for i in range(np.size(T)):
            TTT[i, :, :] = T[i]
        for j in range(np.size(d0)):
            DDDelta_y[:, :, j] = delta_y[j]
        for k in range(np.size(delta_y)):
            DDD0[:, k, :] =d0[k]

        integrande = integ_max(TTT,DDD0,f,DDDelta_y, n=n)
        
        DTT = np.trapz(integrande,dx =T[1]-T[0],axis=0)

    return(DTT)

    # print('FWHMs :')
    # for i in range(np.size(d0)):
    #     print(d0[i])
    # print('-------------')
    # print('R2f :')
    # for k in range(np.size(d0)):
    #     print(1/DTT[k])
#%%
def R2F_intgaussian(d0, delta_y):
    """R2F response in not-simplyfied form.
    We must use sigma_ext = DTT / R2F_int
    Args:
        d_y (float or 1D array): modulation amplitude
        d0 (float or 1D array): fwhm

    Returns:
        res (nD array): The max signal. 1D if d0 or delta_y is an array, 2D if both are, float otherwise
    """

    f = 1000 #modulation frequency: it doesn't change the result.
    n=2 #we work with second harmonic.
    T = np.linspace(-1/(2*f),1/(2*f), 200)

    # print(type(d0))
    # print(type(delta_y))
    # print('-----------------')

    if np.size(d0)==1 and np.size(delta_y)==1:
        integrande = integ_maxgaussian(T,d0, f, delta_y, n=n)
        DTT = np.trapz(integrande,dx =T[1]-T[0])

    elif np.size(d0)==1:
        TT, DDelta_y = np.meshgrid(T,delta_y)
        integrande = integ_maxgaussian(TT,d0,f,DDelta_y, n=n)
        DTT = np.trapz(integrande,dx =T[1]-T[0],axis=1)

    elif np.size(delta_y)==1:        
        TT, DD0 = np.meshgrid(T,d0)
        integrande = integ_maxgaussian(TT,DD0,f,delta_y, n=n)
        DTT = np.trapz(integrande,dx =T[1]-T[0],axis=1)

    else :
        shape = (np.size(T),np.size(d0),np.size(delta_y))
        DDD0 = np.zeros(shape)
        DDDelta_y = np.zeros(shape)
        TTT = np.zeros(shape)

        for i in range(np.size(T)):
            TTT[i, :, :] = T[i]
        for j in range(np.size(d0)):
            DDDelta_y[:, :, j] = delta_y[j]
        for k in range(np.size(delta_y)):
            DDD0[:, k, :] =d0[k]

        integrande = integ_maxgaussian(TTT,DDD0,f,DDDelta_y, n=n)
        
        DTT = np.trapz(integrande,dx =T[1]-T[0],axis=0)

    return(DTT)

    # print('FWHMs :')
    # for i in range(np.size(d0)):
    #     print(d0[i])
    # print('-------------')
    # print('R2f :')
    # for k in range(np.size(d0)):
    #     print(1/DTT[k])
#%%
"""
Example of code to generate the R2F calibration function for converting spectras

iterable = (0.62*x*1e-9 for x in [100,200,300])
d0=np.fromiter(iterable, float)
R2F_int(d0,300e-9)
"""
#%% Airy
# T = np.linspace(-1/(2*f),1/(2*f), 200)
# iterable = (0.62*x*1e-9 for x in [100,200,300])
iterable_deltay = np.linspace(0.0001,700,200)
d0=0.65*500
R2F_variable_deltay = R2F_int(d0,iterable_deltay)*1e6

deltay=300
iterable_lambda = np.linspace(375,1040,200)
iterable_d0 = 0.65*iterable_lambda
R2F_variable_lambda = R2F_int(iterable_d0,deltay)*1e6
#%% test avec profil Gaussien

iterable_deltay = np.linspace(0.0001,700,200)
d0=0.65*500
R2F_variable_deltaygaussian = R2F_intgaussian(d0,iterable_deltay)*1e6

deltay=300
iterable_lambda = np.linspace(375,1040,200)
iterable_d0 = 0.65*iterable_lambda
R2F_variable_lambdagaussian = R2F_intgaussian(iterable_d0,deltay)*1e6
#%%
iterable_r = np.linspace(-1000,1000,200)
intensity_airy=I(iterable_r,1,d0)
intensity_gaussian=Igaussian(iterable_r,1,d0)

plt.close('all')
plt.plot(iterable_r,intensity_airy,color='tab:red',linewidth=2.5)
plt.plot(iterable_r,intensity_gaussian,color='tab:blue',linewidth=2.5)
#-> ok les profils montent à 1 et se superposent bien
#%%
plt.close('all')
# plt.figure()
# plt.plot(iterable_deltay,test,color='tab:red',linewidth=2.5)

fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(13,4.5))

# fig,ax=plt.subplots(figsize=(6.7,5))
ax1.plot(iterable_deltay,R2F_variable_deltay,color='tab:red',linewidth=2.5)
ax1.set_xlim(0,700)
# ax.yaxis.set_ticks(np.arange(0,1.1,0.2))
# ax.set_ylim(0,1.05)
ax1.set_xlabel(r'$\delta_y$ (nm)',labelpad=10,fontsize=27)
ax1.set_ylabel(r'$R_{exp}$ ($10^{-6}$ $\mathrm{nm}^{-2}$)',fontsize=27)
ax1.tick_params(axis='x', labelsize=27)
ax1.tick_params(axis='y', labelsize=27)
ax1.text(0.05,0.9, "a)", transform=ax1.transAxes, 
            size=27)

# fig,ax=plt.subplots(figsize=(6.7,5))
ax2.plot(iterable_lambda,R2F_variable_lambda,color='tab:red',linewidth=2.5)
ax2.set_xlim(375,1040)
# ax.yaxis.set_ticks(np.arange(0,1.1,0.2))
# ax.set_ylim(0,1.05)
ax2.set_xlabel(r'$\lambda$ (nm)',labelpad=10,fontsize=27)
ax2.set_ylabel(r'$R_{exp}$ ($10^{-6}$ $\mathrm{nm}^{-2}$)',fontsize=27)
ax2.tick_params(axis='x', labelsize=27)
ax2.tick_params(axis='y', labelsize=27)
ax2.text(0.05,0.9, "b)", transform=ax2.transAxes, 
            size=27)

fig.tight_layout()

#%%
plt.close('all')
# plt.figure()
# plt.plot(iterable_deltay,test,color='tab:red',linewidth=2.5)

fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(13,4.5))

# fig,ax=plt.subplots(figsize=(6.7,5))
ax1.plot(iterable_deltay,R2F_variable_deltay,color='tab:red',linewidth=2.5)
ax1.plot(iterable_deltay,R2F_variable_deltaygaussian,color='tab:blue',linewidth=2.5)
ax1.set_xlim(0,700)
# ax.yaxis.set_ticks(np.arange(0,1.1,0.2))
# ax.set_ylim(0,1.05)
ax1.set_xlabel(r'$\delta_y$ (nm)',labelpad=10,fontsize=27)
ax1.set_ylabel(r'$R_{exp}$ ($10^{-6}$ $\mathrm{nm}^{-2}$)',fontsize=27)
ax1.tick_params(axis='x', labelsize=27)
ax1.tick_params(axis='y', labelsize=27)
ax1.text(0.05,0.9, "a)", transform=ax1.transAxes, 
            size=27)

# fig,ax=plt.subplots(figsize=(6.7,5))
ax2.plot(iterable_lambda,R2F_variable_lambda,color='tab:red',linewidth=2.5)
ax2.plot(iterable_lambda,R2F_variable_lambdagaussian,color='tab:blue',linewidth=2.5)
ax2.set_xlim(375,1040)
# ax.yaxis.set_ticks(np.arange(0,1.1,0.2))
# ax.set_ylim(0,1.05)
ax2.set_xlabel(r'$\lambda$ (nm)',labelpad=10,fontsize=27)
ax2.set_ylabel(r'$R_{exp}$ ($10^{-6}$ $\mathrm{nm}^{-2}$)',fontsize=27)
ax2.tick_params(axis='x', labelsize=27)
ax2.tick_params(axis='y', labelsize=27)
ax2.text(0.05,0.9, "b)", transform=ax2.transAxes, 
            size=27)

fig.tight_layout()

plt.show()




