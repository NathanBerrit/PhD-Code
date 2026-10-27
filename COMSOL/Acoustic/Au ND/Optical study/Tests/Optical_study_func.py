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
from scipy.signal import find_peaks, peak_widths








def parse_number(x):
    "Allow to read complex number when opening a csv with pandas."
    x = x.strip()

    try:
        # Essaye d'abord en float
        return complex(x.replace(" ", "").replace("i", "j"))
        
    except ValueError:
        try:
            # Normalise la notation : remplace 'i' par 'j' (et enlève espaces)
            return float(x)
        except ValueError:
            return np.nan  # ou gestion d'erreur différente


def get_COMSOL_time_study(dir_root, fname, verbose=False) :
    """The document is an excel table with COMSOL simulation iterationg on both D, e and lambda. All spectra must be completely calculated.
    
    Returns :
    res (dic) : a dictionnary, each key correspond to the simulated spectra of 1 nanodisk (in forme of a dic). 
    the key for each spectra dic are D, h, AR, Volume, Lambda, freq,E, Sigma_ext, Sigma_abs, Sigma_scat"""

    os.chdir(dir_root)

    

    print(fname)
        
    #df = pd.read_csv(fname + '.csv', dtype='str', skiprows=4, converters={0: parse_number})
    df = pd.read_csv(fname + '.csv', skiprows=4, converters={0: parse_number})
    # df = df.map(parse_number)
    data = df.to_numpy()
    #data = df.__array__()

    # Get the whole data
    res_raw = {}


    
    res_raw['time'] = np.real(data[:,0]) # in ps
    res_raw['u_r'] = np.real(data[:,1:])*1e9 # in nm

    return(res_raw)



def study_tf(signal, time, dt, height=0.3e-3, threshold=0.1e-3) :
        """make a dictionnary containing TF information.

        Args:
            signal (_type_): _description_
            time (_type_): _description_
            dt (_type_): _description_
            height (_type_): _description_
            threshold (_type_): _description_
        """

        dic = {}

        dic['freq'], dic['df'], dic['amp'], dic['phase'] =  make_tf(signal, time, dt)
        dic['peaks'], dic['width'], dic['left'], dic['right'] = compute_peaks(dic['amp']**2, dic['df'], height=height, threshold=threshold)
        return(dic)



def make_tf(signal, time, dt) :
    
    sp = np.fft.rfft(signal)
    freq = 1000*np.fft.rfftfreq(len(time),dt)  #frequency in GHz
    df = freq[1]-freq[0]


    sp_amp = np.absolute(sp)
    sp_phase = np.angle(sp)

    return(freq, df, sp_amp, sp_phase)



def compute_peaks(signal,df, height=0.3e-3, threshold=0.1e-3) :

    peaks, peak_dic = find_peaks(signal, height=height, threshold=threshold)

    width, width_height, left, right = peak_widths(signal, peaks, rel_height=0.5 )


    try :
        width = np.array(width)*df
        i_peak = peaks[0]
        

    except :
        pass

    return(peaks, width, left, right)

