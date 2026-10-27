import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib
import io
import pandas as pd
import pickle








def plot_Lda_SPR_D_fixed(D0, Spectra_data, Lda_RBF, ax, offset = 0) :


    h_lisse = np.linspace(5, 60, 100)


    ax.plot(h_lisse, offset + Lda_RBF([(k, D0) for k in h_lisse]), label = 'D = {} nm'.format(D0))

    for key in Spectra_data :
        spectra = Spectra_data[key]
        
        if spectra['D'] == D0 :
            ax.scatter(spectra['h'], offset + spectra['res_fit']['Lda_c'], color='black')

    

def plot_Lda_SPR_h_fixed(h0, Spectra_data, Lda_RBF, ax, offset=0 ) :

    D_lisse = np.linspace(10, 150, 100)
    
    ax.plot(D_lisse, offset + Lda_RBF([(h0, k) for k in D_lisse]), label = 'h = {} nm'.format(h0))
    for key in Spectra_data :
        spectra = Spectra_data[key]
        
        if spectra['h'] == h0 :
            ax.scatter(spectra['D'], offset + spectra['res_fit']['Lda_c'], color='black')
