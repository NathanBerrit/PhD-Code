import importlib
import numpy as np
import scipy as sc
from scipy.special import jv
from scipy.special import i1
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpmath
from mpmath import *
import time
import pandas as pd
import tempfile
import io


## Simulation treatment

def Lorentz_no_offset(x, xc, w, Area) :
    """Lorentzian resonance without offset. This function is defined just as he one in Origin.

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        Area (_type_): Total area of the curve
    """
    res = 2 * Area / np.pi * w * 1/(4*(x-xc)**2 + w**2)
    
    return res



def plot_fits(spectra, ax) :
    print(spectra['h'], spectra['D'])


    print('plotting')
    

    E_lisse = np.linspace(spectra['E'].min(), spectra['E'].max(), 1000)
    

    ax.plot(E_lisse, Lorentz_no_offset(E_lisse, spectra['res_fit']['E_c'], spectra['res_fit']['w'], spectra['res_fit']['Area']), color='red', label='fit')
    ax.scatter(spectra['E'], spectra['Ext'], color='black', label='data')
    ax.set_title('{:.2f}, h = {}, D = {}'.format(spectra['AR'], spectra['h'], spectra['D']))


    ax.axvline(spectra['E'][spectra['res_fit']['idx']], color='black', alpha=0.2, linestyle='--')


    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")



## Ar visualisation
