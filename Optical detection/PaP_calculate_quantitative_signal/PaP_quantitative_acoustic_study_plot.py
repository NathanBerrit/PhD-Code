import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\Optical detection\PaP_calculate_quantitative_signal')
import PaP_quantitative_thermal_study_func 
importlib.reload(PaP_quantitative_thermal_study_func)
from PaP_quantitative_thermal_study_func import *





def plot_A_raw_coeff(Data_PaP, ax, alpha_PaP=1) :

    

    ax.errorbar(Data_PaP['detuning_normalised'], (Data_PaP['A_ac'])*alpha_PaP, (Data_PaP['A_ac_err'])*alpha_PaP, color='black', label='raw')

    ax.set_xlabel('Lambda (nm)')
    ax.set_ylabel('Acoustic amplitude (1e-5)')

    ax.axhline(0, color='black', linestyle='--',alpha=0.7)
    ax.axvline(0, color='black', linestyle='--',alpha=0.7)
    pass



def plot_A_corrected_coeff(Data_PaP, ax) :






    ax.errorbar(Data_PaP['detuning_normalised'], Data_PaP['A_ac_corr'], Data_PaP['A_ac_err_corr'], color='red', label='corrected')


    ax.set_xlabel('Lambda (nm)')
    ax.set_ylabel('Acoustic amplitude (1e-5)')

    ax.axhline(0, color='black', linestyle='--',alpha=0.7)
    ax.axvline(0, color='black', linestyle='--',alpha=0.7)

    pass



def plot_Delta_sigma_acoustic(Data_PaP, ax) :
        
    #ax.plot(Data_thermic['detuning_normalised'], Data_thermic['Delta_sigma_th'], color='red', label='Delta_sigma')
    ax.errorbar(Data_PaP['detuning_normalised'], Data_PaP['Delta_sigma_ac'], Data_PaP['Delta_sigma_ac_err'], color='red')
   

    ax.set_xlabel('Normalized detuning')
    ax.set_ylabel('$\Delta \sigma_{ext}$ (nm$^2$)')

    ax.axhline(0, color='black', linestyle='--',alpha=0.7)
    ax.axvline(0, color='black', linestyle='--',alpha=0.7)

    pass



def plot_E_pulse(Data_PaP, ax) :


    ax.scatter(Data_PaP['detuning_normalised'], Data_PaP['E_pulse']*1e12, color='red')


    ax.set_xlabel('Normalized detuning')
    ax.set_ylabel(r'$E_{pulse}$ (pJ)')





def plot_Delta_T(Data_PaP, ax) :


    ax.scatter(Data_PaP['detuning_normalised'], Data_PaP['Delta_T'], color='red')


    ax.set_xlabel('Normalized detuning')
    ax.set_ylabel(r'$\Delta T$ (K)')

def plot_Delta_sigma_acoustic_normalized(Data_PaP, ax) :


    ax.errorbar(Data_PaP['detuning_normalised'], Data_PaP['Delta_sigma_ac_normalized'], Data_PaP['Delta_sigma_ac_normalized_err'], color='red')


    ax.set_xlabel('Normalized detuning')
    ax.set_ylabel(r'$\frac{\Delta \sigma_{ext}}{\Delta T_0}$ (nm$^2$ K$^{-1}$)')

    ax.axhline(0, color='black', linestyle='--',alpha=0.7)
    ax.axvline(0, color='black', linestyle='--',alpha=0.7)


