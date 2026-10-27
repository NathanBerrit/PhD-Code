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
import matplotlib


import matplotlib.colors
import plotly.offline
import plotly.graph_objs as go
import plotly


def plot_all_raw_spectra_interactive(Spectra_dic) :
    """_summary_

    Args:
        spectra (_type_): _description_
        ax (_type_): _description_
    """
    
    n_spectra = np.size([key for key in Spectra_dic])
    
    cm = plt.get_cmap('viridis')
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n_spectra)) for i in range(n_spectra)]

    compt = 0

    
    Spectra_plot = []
    for key in Spectra_dic :
        #fig, ax = plt.subplots(figsize = (9,4))
        


        spectra = Spectra_dic[key]
        
    
        Spectra_plot.append(go.Scatter(x =spectra['freq'], y=spectra['Q'], name=key, line=dict(color=Colors[compt])))

        compt +=1

    plotly.offline.plot(Spectra_plot)


def plot_all_raw_spectra(Spectra_dic, ax, plotted_key='Q') :
    """_summary_

    Args:
        spectra (_type_): _description_
        ax (_type_): _description_
    """
    
    n_spectra = np.size([key for key in Spectra_dic])
    
    cm = plt.get_cmap('viridis')
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n_spectra)) for i in range(n_spectra)]

    compt = 0

    

    for key in Spectra_dic :
        #fig, ax = plt.subplots(figsize = (9,4))
        


        spectra = Spectra_dic[key]
        
        if compt == 0 or compt == n_spectra-1 :
            ax.scatter(spectra['freq'], spectra[plotted_key], color = Colors[compt], label=key)
        else :
            pass
            ax.scatter(spectra['freq'], spectra[plotted_key], color = Colors[compt], label=key)
        compt +=1





def plot_all_acoustic_modes(Spectra_dic, ax, Q_max = -1, n_modes=3) :
    """_summary_

    Args:
        spectra (_type_): _description_
        ax (_type_): _description_
    """

    if Q_max == -1 :

        Q_max = np.max([np.max(Spectra_dic[key]['modes']['Q']) for key in Spectra_dic])
        print(Q_max)



    cm = plt.get_cmap('gist_rainbow')

    n = 10000
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n)) for i in range(n)]

    norm = matplotlib.colors.Normalize(vmin=0, vmax=Q_max)


        
    # Ajout de la barre de couleur sur l'axe ax
    cbar = ax.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax)
    # Étiquettes de la colorbar
    cbar.set_ticks([0, (0+Q_max)/2, Q_max])  # bornes + milieu
    cbar.set_ticklabels([f"{0}", "{:.1f}".format(Q_max/2), ">{:.1f}".format(Q_max)])

    compt = 0
    for key in Spectra_dic :
        #fig, ax = plt.subplots(figsize = (9,4))
        spectra = Spectra_dic[key]
        
        color_idx = [np.searchsorted(np.linspace(0, Q_max, n), val)-1 for val in spectra['modes']['Q']]

        color_list = [Colors[idx] for idx in color_idx]
        
        
        ax.scatter(spectra['AR']*np.ones(np.size(spectra['modes']['f'][:n_modes])), spectra['modes']['f_red'][:n_modes], color = color_list[:n_modes])


      
        compt +=1


