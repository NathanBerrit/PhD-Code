import importlib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import io
import matplotlib
from scipy.optimize import fsolve
import scipy as sc
from mpmath import findroot
import mpmath
import matplotlib as mpl
from sklearn.metrics import root_mean_squared_error

from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit



def parse_number(s):
    if pd.isna(s):  # gestion des NaN éventuels
        return np.nan
    s = str(s).strip()
    s = s.replace("i", "j")  # Python utilise j au lieu de i
    try:
     
        return float(s)   # si c’est un réel
    except ValueError:
       
        return complex(s) # sinon, c’est un complexe
    



def open_COMSOL_results_frequency_study(dir_root, Fnames, param_list) :

    os.chdir(dir_root)

    Res_dic = {} 


    for fname in Fnames :


        
        df = pd.read_csv(fname + '.csv', skiprows=4, converters={0: parse_number})

        data = df.to_numpy()
        df = df.map(parse_number)
        data = df.to_numpy()


        
        # Get the whole data
        res_raw = {}


        compt=0
        for param in param_list :
            res_raw[param] = np.real(data[:,compt])
            compt+=1

        res_raw['freq'] = np.real(data[:,-2])
        res_raw['displacement'] = np.real(data[:,-1])



        if len(param_list) == 0 :
            Res_dic[fname] = res_raw


        else :
                
            compt = 0
 

            
            while compt <  np.size(res_raw['freq']) :
                
                idx_start = compt

                idx_end = scan_single_spectra(param_list, res_raw, idx_start)


                

                key = ' - '.join(["{}_{}".format(param, np.real(res_raw[param][idx_start])) for param in param_list])
                #print(key)
                Res_dic[key] = {}
            

                id=0
                for param in param_list :

                    Res_dic[key][param] = np.real(res_raw[param][idx_start])
                    id+=1

                Res_dic[key]['freq'] = res_raw['freq'][idx_start:idx_end].real
                Res_dic[key]['displacement'] = np.real(res_raw['displacement'][idx_start:idx_end])

                compt = idx_end +1

    return(Res_dic)



def scan_single_spectra(param_list, res_raw, idx_start) :
    init_value = [res_raw[key][idx_start] for key in param_list]

    compt = idx_start
    
    while [res_raw[key][compt] for key in param_list] == init_value and compt < np.size(res_raw[param_list[0]])-2 :
        compt+=1

    if compt == np.size(res_raw[param_list[0]])-2 :
        idx_end = compt+1

    else :
        idx_end=compt

    return(idx_end)






def Lorentz(x, xc, w, A, y0) :
    """Lorentzian resonance

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        A (_type_): height and intensity
        y0 (_type_): offset
    """
    res = y0 + A*w*w/(4*(x-xc)**2 + w**2)

    return res

def Lorentz_no_offset(x, xc, w, A) :
    """Lorentzian resonance without offset. A is the maximum of the function.

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        A (_type_): height and intensity
    """
    res = A*w*w*(1/(4*(x-xc)**2 + w**2))

    
    return res



def Lorentz4(x,offset, xc1, w1, A1, xc2, w2, A2, xc3, w3, A3, xc4, w4, A4):
    return  Lorentz(x, xc1, w1, A1, offset) + Lorentz_no_offset(x, xc2,w2,A2) + Lorentz_no_offset(x, xc3,w3,A3) + Lorentz_no_offset(x, xc4,w4,A4)

def Lorentz3(x,offset, xc1, w1, A1, xc2, w2, A2, xc3, w3, A3):
    return  Lorentz(x, xc1, w1, A1, offset) + Lorentz_no_offset(x, xc2,w2,A2) + Lorentz_no_offset(x, xc3,w3,A3)

def Lorentz2(x,offset, xc1, w1, A1, xc2, w2, A2):
    return  Lorentz(x, xc1, w1, A1, offset) + Lorentz_no_offset(x, xc2,w2,A2)

def Lorentz1(x,offset, xc1, w1, A1):
    return  Lorentz(x, xc1, w1, A1, offset)



def multi_lorentzian(x, *params):
    """Somme de N lorentziennes + offset.
    params = [offset, A1, x01, g1, A2, x02, g2, ...]"""
    y = np.full_like(x, params[0], dtype=float)
    for i in range(1, len(params), 3):
        y += Lorentz_no_offset(x, *params[i:i+3])
    return y



def fit_peaks(spectra, peaks, widths=None, key = None):
    """Fit de N lorentziennes, N = len(peaks)."""

    x, y = spectra['freq'], spectra['displacement']

    # Estimation des largeurs (à mi-hauteur -> demi-largeur)
    if widths is None:
        w = peak_widths(y, peaks, rel_height=0.5)[0]   # en indices
        dx = np.mean(np.diff(x))
        widths = w * dx 
    
    p0 = [np.min(y)]                       # offset
    lower = [-np.inf]
    upper = [np.inf]
    for pk, g in zip(peaks, widths):
        p0 += [x[pk], g, y[pk]-y.min()]
        lower += [ x[pk]-g, -np.inf,0]           # A>0, x0 dans la fenêtre, gamma>0
        upper += [ x[pk]+g, np.inf, np.inf]
    #print(p0)



    try:
        #popt, pcov = curve_fit(multi_lorentzian, x, y, p0=p0, maxfev=20000, bounds=(lower, upper))
        popt, pcov = curve_fit(multi_lorentzian, x, y, p0=p0, maxfev=20000)
    except RuntimeError as e:
        print(f"{key} : fit non convergé ({e})")
        popt, pcov = p0, np.zeros(np.shape(p0))
    

    

    
    return popt, pcov

def Lorentzian_fitting(data, p0, n_modes = 1 ) :



    X, Y = data['freq'], data['displacement']
    

    p0 = p0[0:1+3*n_modes]

    fit_func = globals()['Lorentz' + str(n_modes)]
    #fit_func = locals()['Lorentz' + str(n_modes)]()
    
    popt, pcov = curve_fit(fit_func, X, Y, p0=p0)


    incert = np.sqrt(np.diag(pcov))

    err= (root_mean_squared_error(Y,fit_func(X,*popt))/np.abs(np.max(Y)))**2

    res= (popt, incert, err)

    return(res)










def fit_acoustic(self, p0, Bounds_high, Bounds_low) :

    return None
    fit_func = [Osc1, Osc2, Osc3, Osc4]
    res1, res2, res3, res4 = fit_signal_osc(self.DTT_acoustic, self.delay, p0, Bounds_low, Bounds_high, fit_func)
    
    popt1, incert_1, err_1 = res1
    popt2, incert_2, err_2 = res2
    popt3, incert_3, err_3 = res3
    popt4, incert_4, err_4 = res4


    self.acoustic_fit = {
        '1 modes' : {},
        '2 modes' : {},
        '3 modes' : {},
        '4 modes' : {}
    }

    Numbers = ['1', '2', '3', '4']

    

    for n in Numbers :

        #print(locals()['popt' + n])
        self.acoustic_fit[n + ' modes']['coeffs'] = locals()['popt' + n]
        self.acoustic_fit[n + ' modes']['coeffs_err'] = locals()['incert_' + n]
        self.acoustic_fit[n + ' modes']['residual'] = locals()['err_' + n]


        for j in range(int(n)) :

            

            # Takes into account the offset from the electron peak to compute the phase phi.
            index_coeff = str(j+1)


            self.acoustic_fit[n + ' modes']['A' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 0]
            self.acoustic_fit[n + ' modes']['tau' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 1]
            self.acoustic_fit[n + ' modes']['T' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 2]
            self.acoustic_fit[n + ' modes']['phi' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 3]
            self.acoustic_fit[n + ' modes']['Q' + index_coeff] = np.pi*self.acoustic_fit[n + ' modes']['tau' + index_coeff]/self.acoustic_fit[n + ' modes']['T' + index_coeff]
            
            self.acoustic_fit[n + ' modes']['A' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 0]
            self.acoustic_fit[n + ' modes']['tau' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 1]
            self.acoustic_fit[n + ' modes']['T' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 2]
            self.acoustic_fit[n + ' modes']['phi' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 3]
            self.acoustic_fit[n + ' modes']['Q' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['Q' + index_coeff]*np.sqrt((self.acoustic_fit[n + ' modes']['tau' + index_coeff+'_err']/self.acoustic_fit[n + ' modes']['tau' + index_coeff])**2 + (self.acoustic_fit[n + ' modes']['T' + index_coeff+'_err']/self.acoustic_fit[n + ' modes']['T' + index_coeff])**2)







def fit_signal_osc(signal_to_fit, delay, p0, Bounds_low, Bounds_high, fit_funcs) :
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
    return None
    Osc1, Osc2, Osc3, Osc4 = fit_funcs[0], fit_funcs[1], fit_funcs[2], fit_funcs[3]

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



