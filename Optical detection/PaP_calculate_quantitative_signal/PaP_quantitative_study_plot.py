import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib
from colour import Color

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\Optical detection\PaP_calculate_quantitative_signal')
import PaP_quantitative_thermal_study_func
importlib.reload(PaP_quantitative_thermal_study_func)
from PaP_quantitative_thermal_study_func import *



def print_acoustic_variation_fullnorm(ax, Data, key) :



    ax.scatter(Data['acoustic']['lambda_probe'],Data[key]['acoustic']['Normalized_total']*1e5, color = Data['color'], s=20, label = key )
    ax.errorbar(Data['acoustic']['lambda_probe'],Data[key]['acoustic']['Normalized_total']*1e5, yerr = np.abs(Data['acoustic']['Normalized_total_err']*1e5), color = Data['color'], linestyle='' )
    
    ax.axhline(0, color='black')
    ax.axvline(Data['lda_SPR']*1e9, color = Data['color'], linestyle='--')

    ax.set_xlabel('Lambda (nm)')
    ax.set_ylabel(r'$\frac{\Delta \sigma_{ext}}{\Delta T \cdot \sigma_{ext}}$ (10$^{-5}$ K$^{-1}$)')
    plt.legend(fontsize=10)


def print_acoustic_variation_T_norm(ax, Data, key) :

    ax.scatter(Data['acoustic']['lambda_probe'],Data['acoustic']['Delta_sigma_ac_normalized'],  color = Data['color'], s=15,  label = key )
    ax.errorbar(Data['acoustic']['lambda_probe'],Data['acoustic']['Delta_sigma_ac_normalized'], yerr = np.abs(Data['acoustic']['Delta_sigma_ac_normalized_err']),linestyle='', color = Data['color'] )
    
    ax.axhline(0, color='black')
    ax.axvline(Data['lda_SPR']*1e9, color = Data['color'], linestyle='--')

    ax.set_xlabel('Lambda (nm)')
    ax.set_ylabel(r'$\frac{\Delta \sigma_{ext}}{\Delta T}$ (nm$^2$ K$^{-1}$)')
    plt.legend(fontsize=10)





def print_acoustic_variation(ax, Data, key) :

    ax.scatter(Data['acoustic']['lambda_probe'],Data['acoustic']['Delta_sigma_ac'], color = Data['color'], s=15,  label = key )
    ax.errorbar(Data['acoustic']['lambda_probe'],Data['acoustic']['Delta_sigma_ac'], yerr = np.abs(Data['acoustic']['Delta_sigma_ac_err']),linestyle='', color = Data['color'] )
    
    ax.axhline(0, color='black')
    ax.axvline(Data['lda_SPR']*1e9, color = Data['color'], linestyle='--')

    ax.set_xlabel('Lambda (nm)')
    ax.set_ylabel(r'$\Delta \sigma_{ext}$ (nm$^2$s)')
    plt.legend(fontsize=10)











## Thermic part 

def print_thermic_variation_fullnorm(ax, Data, key) :

    ax.scatter(Data['thermic']['lambda_probe'],Data['thermic']['Normalized_total']*1e5, s=15, color = Data['color'], label = key )
    ax.errorbar(Data['thermic']['lambda_probe'],Data['thermic']['Normalized_total']*1e5, yerr = np.abs(Data['thermic']['Normalized_total_err']*1e5), color = Data['color'], linestyle='' )
    
    
    ax.axhline(0, color='black')
    ax.axvline(Data['lda_SPR']*1e9, color = Data['color'], linestyle='--')


    plt.legend(fontsize=10)


def print_thermic_variation_T_norm(ax, Data, key) :



    ax.scatter(Data['thermic']['lambda_probe'],Data['thermic']['Delta_sigma_th_normalized'], s=15, color = Data['color'], label = key )
    ax.errorbar(Data['thermic']['lambda_probe'],Data['thermic']['Delta_sigma_th_normalized'], yerr = np.abs(Data['thermic']['Delta_sigma_th_normalized_err']), color = Data['color'], linestyle='' )
    
    ax.axhline(0, color='black')
    ax.axvline(Data['lda_SPR']*1e9, color = Data['color'], linestyle='--')

    plt.legend(fontsize=10)





def print_thermic_variation(ax, Data, key) :



    ax.scatter(Data['thermic']['lambda_probe'],Data['thermic']['Delta_sigma_th']*1e5, s=15, color = Data['color'], label = key )
    ax.errorbar(Data['thermic']['lambda_probe'],Data['thermic']['Delta_sigma_th']*1e5, yerr = np.abs(Data['thermic']['Delta_sigma_th_err']*1e5), color = Data['color'], linestyle='' )
    
    ax.axhline(0, color='black')
    ax.axvline(Data['lda_SPR']*1e9, color = Data['color'], linestyle='--')


    plt.legend(fontsize=10)




### Plot The Traces

def plot_raw_traces(Data_dic, ax) :
    n_traces = len(Data_dic)
    # Colors1 = list(Color("red").range_to(Color("black"),n_traces//2))
    # Colors2 = list(Color("black").range_to(Color("blue"),n_traces - n_traces//2))
    # Colors = Colors1+Colors2
    Colors= list(Color("red").range_to(Color("blue"),n_traces))
    compt=0

    for key in Data_dic :
        Trace = Data_dic[key]

        ax.plot(Trace.delay, Trace.DTT*1e5, color=Colors[compt].rgb, label = str(Trace.lda) + ' nm')

        compt+=1
