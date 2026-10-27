import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib

from sklearn.metrics import root_mean_squared_error

from scipy.signal import find_peaks, peak_widths



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



def study_tf(signal, time, dt, height, threshold) :
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




def Osc4(t, A1,tau1, T1, phi1,A2,tau2, T2, phi2,A3,tau3, T3, phi3,A4,tau4, T4, phi4):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1) + \
            A2*np.exp(-t/tau2)*np.sin(2*np.pi/T2*t+phi2) + \
            A3*np.exp(-t/tau3)*np.sin(2*np.pi/T3*t+phi3) + \
            A4*np.exp(-t/tau4)*np.sin(2*np.pi/T4*t+phi4) 
def Osc3(t, A1,tau1, T1, phi1,A2,tau2, T2, phi2,A3,tau3, T3, phi3):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1) + \
            A2*np.exp(-t/tau2)*np.sin(2*np.pi/T2*t+phi2) + \
            A3*np.exp(-t/tau3)*np.sin(2*np.pi/T3*t+phi3)
def Osc2(t, A1,tau1, T1, phi1,A2,tau2, T2, phi2):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1) + \
            A2*np.exp(-t/tau2)*np.sin(2*np.pi/T2*t+phi2)
def Osc1(t, A1,tau1, T1, phi1):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1)



def remove_acoustic_modes(Trace, n_modes) :
    """Remove the thermal fit to the DTT data.

    Args:
        Trace_thermic (PaP_dataset): PaP_dataset used to do the thermal fit.
        n_exp (int): number of exponential used for the fit.
    """

    func = globals()['Osc' + str(n_modes)]
    
    exp_key = str(n_modes) + ' modes'

    DTT_mode_removed = Trace.DTT_acoustic - func(Trace.delay,*Trace.acoustic_fit[exp_key]['coeffs'])
    return(DTT_mode_removed)




def fit_signal_osc(signal_to_fit, delay, p0, Bounds_low, Bounds_high) :
    """make the damped oscillation fit for the function

    Args:
        signal_to_fit (1D array): signal
        delay (1D array): time
        p0 (tuple): initial guess for every mode.
        
        Bounds_low (list): list of the 6 bounds
        Bounds_high (list): list of the 6 bounds
        fit_funcs (list) : list of the 4 fitting functions.

    Returns:
        res (tuple): each res is a tuple (popt, pcov, err)
    """

    # Curve fitting
    popt1, pcov1 = curve_fit(Osc1, delay, signal_to_fit, p0=p0[0:4])
    popt2, pcov2 = curve_fit(Osc2, delay, signal_to_fit, p0=p0[0:8], bounds=(Bounds_low[0:8], Bounds_high[0:8]))                         
    popt3, pcov3 = curve_fit(Osc3, delay, signal_to_fit, p0=p0[0:12], bounds=(Bounds_low[0:12], Bounds_high[0:12]))
    popt4, pcov4 = curve_fit(Osc4, delay, signal_to_fit, p0=p0, bounds=(Bounds_low, Bounds_high))

    incert_1 = np.sqrt(np.diag(pcov1))
    incert_2 = np.sqrt(np.diag(pcov2))
    incert_3 = np.sqrt(np.diag(pcov3))
    incert_4 = np.sqrt(np.diag(pcov4))

    # Residual errors
    err_1 = (root_mean_squared_error(signal_to_fit,Osc1(delay,*popt1))/np.abs(np.max(signal_to_fit)))**2
    err_2 = (root_mean_squared_error(signal_to_fit,Osc2(delay,*popt2))/np.abs(np.max(signal_to_fit)))**2
    err_3 = (root_mean_squared_error(signal_to_fit,Osc3(delay,*popt3))/np.abs(np.max(signal_to_fit)))**2
    err_4 = (root_mean_squared_error(signal_to_fit,Osc4(delay,*popt4))/np.abs(np.max(signal_to_fit)))**2

    res1 = (popt1, incert_1, err_1)
    res2 = (popt2, incert_2, err_2)
    res3 = (popt3, incert_3, err_3)
    res4 = (popt4, incert_4, err_4)


    return (res1, res2, res3, res4)