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



def change_string_to_float(value, fill_value = 0) :

    if value =='--' :
            
            return fill_value
    else :
        try :
            return float(value)
        except :
            return value

def get_raw_data(dir_root, fname, fill_value=0) :
    """Program done open a .dat file extracted from origin with tab separators. The program takes into account that the decimal separator is ',', 
    and that there can be somme missing values that are decribed with '--' in the origin export. The missing values are replaced with fill_value.

    Args:
        dir_root (_type_): _description_
        fname (_type_): _description_
    """
    Data_dic = {}
    os.chdir(dir_root)

    with open(fname) as file:
        # Lire toutes les lignes du fichier
        lines = file.readlines()
        
        # Remplacer les virgules par des points et préparer les données
        cleaned_lines = [line.replace(',', '.').strip() for line in lines]

        data_stream = io.StringIO('\n'.join(cleaned_lines))
        # Charger les données en tant que tableau numpy


        data_col_name_values = np.genfromtxt(data_stream, delimiter='\t',dtype=str, filling_values = 0, max_rows=1)
        data_stream.seek(0) 
        data_values = np.genfromtxt(data_stream, delimiter='\t', skip_header=3,dtype=str)

        # print(data_values[0,:])

        # print(data_col_name_values)

  
    
    for i in range(np.size(data_col_name_values)) :
        key = data_col_name_values[i]
       
        n_values = np.size(data_values[:,i])
        Data_dic[key] = []
        
        for j in range(n_values) :
          
            Data_dic[key].append(change_string_to_float(data_values[j, i], fill_value = fill_value))
        
        try :
            Data_dic[key] = np.array(Data_dic[key])
        except :
            print(key)
            pass

    return(Data_dic)



def calculate_corrected_DTT(DTT, sigma, fwhm, alpha=2.55, verbose = False) :
    """Implements a correction of the raw pump_probe_signal to take into account that the transmitted probe beam is already 
    attenuated by the nano-object.

    Args:
        DTT (float or array): Signal to be corrected
        sigma (float): linear absorption of the nano-object at that scale in nm2
        fwhm (float): beam FWHM in nm
    """
    #print(fwhm, sigma)
    S = np.pi*fwhm**2/(4*np.log(2))
    S = np.pi*fwhm**2/(4*ln(2))
    

    alpha = alpha
    return(DTT*(1-(sigma/S))*alpha)




def compute_corrected_DTT_signal(Data_PaP, Spectra_dic, FWHM_calib_probe, WL_calib_probe, alpha=2.55, verbose=False) :
    """Implements a correction of the raw pump_probe_signal to take into account that the transmitted probe beam is already 
    attenuated by the nano-object. Here we sum the contribution of all thermic exponantial, to get a global coeffivient Ap"""

    n_values = np.size(Data_PaP['lambda_probe'])
    Data_PaP['Ap_corr'] = np.zeros(n_values)
    Data_PaP['Ap_err_corr'] = np.zeros(n_values)
    
    

    for i in range(n_values) :
        
        lda_probe = Data_PaP['lambda_probe'][i]

        idx_spectra = np.searchsorted(Spectra_dic['Lambda'],lda_probe)

        sigma = Spectra_dic['Sigma (nm2)'][idx_spectra]
        

        idx_fwhm_probe =  np.searchsorted(WL_calib_probe,lda_probe)
        fwhm_probe = FWHM_calib_probe[idx_fwhm_probe]


        Data_PaP['Ap_corr'][i] = calculate_corrected_DTT(Data_PaP['A_th'][i]+ Data_PaP['A2_th'][i], sigma, fwhm_probe, alpha=alpha, verbose = verbose)
        Data_PaP['Ap_err_corr'][i] = calculate_corrected_DTT(np.sqrt(Data_PaP['A_th_err'][i]**2 + Data_PaP['A2_th_err'][i]**2), sigma, fwhm_probe, alpha=alpha)
        if verbose :
            
            print('lda probe = {:.1f} nm'.format(lda_probe))
            print('Suface S probe = {:.1f} nm2'.format(np.pi*float(fwhm_probe)**2/(4*np.log(2))))
            print('Ap = {:.1f} e-5'.format(Data_PaP['A_th'][i]+ Data_PaP['A2_th'][i]))
            print('Ap error = {} e-5'.format(Data_PaP['A_th_err'][i] + Data_PaP['A2_th_err'][i]))
            print('Ap_corr = {:.1f} e-5'.format(Data_PaP['Ap_corr'][i]))
            print('Ap_corr error = {:.1f} e-5'.format(Data_PaP['Ap_err_corr'][i]))
            print('-----------------------------------------------------')


def Surface_value(fwhm) :

    return(np.pi*fwhm**2/(4*np.log(2)))

def DTT_to_delta_sigma(dtt, fwhm) :
    
    S = Surface_value(fwhm)

    return -S*dtt


def Compute_delta_sigma_thermic(Data_PaP, FWHM_calib_probe, WL_calib_probe, verbose=False) :
    """Calculated the maximum Delta sigma_ext value from the pump-probe signal.

    Args:
        Data_PaP (dic): raw pump-probe data
        FWHM_calib_probe (dic): _description_
    """
    
    n_values = np.size(Data_PaP['lambda_probe'])

    Data_PaP['Delta_sigma_th'] = np.zeros(n_values)
    Data_PaP['Delta_sigma_th_err'] = np.zeros(n_values)
    Data_PaP['DTT_to_Dsigma_coeff'] = np.zeros(n_values)

    for i in range(n_values) :

        lda_probe = Data_PaP['lambda_probe'][i]
        idx_fwhm_probe =  np.searchsorted(WL_calib_probe,lda_probe)
        fwhm_probe = FWHM_calib_probe[idx_fwhm_probe]
        

        Data_PaP['Delta_sigma_th'][i] = DTT_to_delta_sigma(Data_PaP['Ap_corr'][i]  *1e-5, fwhm_probe)
        Data_PaP['Delta_sigma_th_err'][i] = np.abs(DTT_to_delta_sigma(Data_PaP['Ap_err_corr'][i]  *1e-5, fwhm_probe))
        # Coeff from dtt signal to delta_sigma signal in nm2
        Data_PaP['DTT_to_Dsigma_coeff'][i]  = DTT_to_delta_sigma(1, fwhm_probe) 
    
        if verbose :
            print('lda probe = {:.1f} nm'.format(lda_probe))
            print('fwhm probe = {:.1f} nm'.format(fwhm_probe))
            print('Ap_corr = {:.1f} e-5'.format(Data_PaP['Ap_corr'][i]))
            print('Delta_sigma = {:.1f} nm2'.format(Data_PaP['Delta_sigma_th'][i]))
            print('Delta_sigma_err = {:.1f} nm2'.format(Data_PaP['Delta_sigma_th_err'][i]))
            print('-----------------------------------------------------')
     
def compute_E_pulse(Data_PaP, Pump_calib_data, t_chopper=2.55, f_rep=80e6, verbose=False) :


    n_values = np.size(Data_PaP['lambda_probe'])

    Data_PaP['E_pulse'] = np.zeros(n_values)

    if verbose :
        print('t chopper : {}'.format(t_chopper))
        print('frep = {:.1f} MHz'.format(f_rep*1e-6))


    for i in range(n_values) :
        

        x_lens = Data_PaP['x_LP2'][i]

        idx_lens_pos = np.searchsorted(Pump_calib_data['x_LP2'], x_lens)
        ## Get fwhm of pump
        x_lens_p = Pump_calib_data['x_LP2'][idx_lens_pos]
        x_lens_m = Pump_calib_data['x_LP2'][idx_lens_pos-1]
        
        t_obj_p = Pump_calib_data['Objective transmission'][idx_lens_pos]
        t_obj_m = Pump_calib_data['Objective transmission'][idx_lens_pos-1]

        t = (x_lens-x_lens_m)/(x_lens_p-x_lens_m)
        t_objective = ((1-t)*t_obj_m + t*t_obj_p)



        P_i = Data_PaP['P_ini'][i]*1e-6
        
        Data_PaP['E_pulse'][i] = P_i*t_chopper*t_objective/f_rep

        if verbose :
            lda_probe = Data_PaP['lambda_probe'][i]
            print('lda probe = {:.1f} nm'.format(lda_probe))
            print('x_lens = {} mm'.format(x_lens))
            print('t_objective = {}'.format(t_objective))
            print('E_pulse = {:.1e} J'.format(Data_PaP['E_pulse'][i]))
            print('-----------------------------------------------------')

def compute_Delta_T(Data_PaP, Pump_calib_data, sigma_abs_pump, C, verbose=True) :
    """_summary_

    Args:
        Data_PaP (dic): _description_
        Pump_calib_data (dic): _description_
        sigma_abs_pump (float): pump absorption effective surface in nm2.
    """

    n_values = np.size(Data_PaP['lambda_probe'])

    Data_PaP['Delta_T'] = np.zeros(n_values)

    for i in range(n_values) :



        x_lens = Data_PaP['x_LP2'][i]

        idx_lens_pos = np.searchsorted(Pump_calib_data['x_LP2'], x_lens)
        ## Get fwhm of pump
        x_lens_p = Pump_calib_data['x_LP2'][idx_lens_pos]
        x_lens_m = Pump_calib_data['x_LP2'][idx_lens_pos-1]
        
        fwhm_p = Pump_calib_data['FWHM'][idx_lens_pos]
        fwhm_m = Pump_calib_data['FWHM'][idx_lens_pos-1]

        t = (x_lens-x_lens_m)/(x_lens_p-x_lens_m)
        fwhm_pump = ((1-t)*fwhm_m + t*fwhm_p)


        # print(x_lens, x_lens_p, x_lens_m)
        # print(fwhm_pump, fwhm_p, fwhm_m)
        # print('-----------------')


        S_pump = np.pi*(fwhm_pump)**2/(4*np.log(2)) #In nm2. The dimension is compensated by sigma_pump


        E_abs = sigma_abs_pump*Data_PaP['E_pulse'][i]/S_pump
        Data_PaP['Delta_T'][i] = E_abs/C

        if verbose :
            lda_probe = Data_PaP['lambda_probe'][i]
            print('lda probe = {:.1f} nm'.format(lda_probe))
            print('x_lens = {} mm'.format(x_lens))
            print('fwhm pump : {:.1f} nm'.format(fwhm_pump))
            print('S_pump : {:.1f} nm2'.format(S_pump))
            print('Pump fluence : {:.2f} J/m2'.format(Data_PaP['E_pulse'][i]/S_pump*1e18))
            print('Delta T = {:.1f} K'.format(Data_PaP['Delta_T'][i]))
            print('-----------------------------------------------------')

def full_normalize_sigma(Data_PaP, Spectra_dic) :
    
    
    Data_PaP['Normalized_total'] = np.zeros(len(Data_PaP['lambda_probe']))
    Data_PaP['Normalized_total_err'] = np.zeros(len(Data_PaP['lambda_probe']))



    for k in range(len(Data_PaP['lambda_probe'])) :
        lda = Data_PaP['lambda_probe'][k]

        idx_sigma = np.searchsorted(Spectra_dic['Lambda'], lda)
        

        sigma = Spectra_dic['Sigma (nm2)'][idx_sigma]
        sigma_err = Spectra_dic['Sigma_err (nm2)'][idx_sigma]

        sigma_full_norm = Data_PaP['Delta_sigma_th_normalized'][k]/sigma
        sigma_full_norm_err = sigma_full_norm * (sigma_err/sigma + Data_PaP['Delta_sigma_th_normalized_err'][k]/Data_PaP['Delta_sigma_th_normalized'][k])

        Data_PaP['Normalized_total'][k] = sigma_full_norm
        Data_PaP['Normalized_total_err'][k] = sigma_full_norm_err




def save_treated_PaP_signal_thermic(Data_PaP, dir_to_save, fname) :
    """Save the treated quantitative Data. The conversion is calculated for the coefficients A1 and A2 of 2 decaying exponential, 
    but we want the quantitative initial extinction variation. This quantity is the sum of the A1 and A2 contribution.

    Args:
        Data_PaP (dic): dictionnary of PaP data
        dir_to_save (str): path to tave the file
        fname (str): name of the file
    """

    os.chdir(dir_to_save)
    df = pd.DataFrame({'lambda_probe (nm)' : Data_PaP['lambda_probe'],\
                       'detuning_normalized' : Data_PaP['detuning_normalised'],\
                       'Delta_T (K)' : Data_PaP['Delta_T'],\
                       'Delta_sigma_th (nm2)' : Data_PaP['Delta_sigma_th'],\
                       'Delta_sigma_th_err (nm2)' : Data_PaP['Delta_sigma_th_err'],\
                       'Delta_sigma_th_normalized (nm2 K-1)' : Data_PaP['Delta_sigma_th_normalized'] ,\
                       'Delta_sigma_th_normalized_err (nm2 K-1)' : Data_PaP['Delta_sigma_th_normalized_err'],\
                       'D_sigma / (DT * sigma)' : Data_PaP['Normalized_total'] , \
                       'D_sigma / (DT * sigma) error' : np.abs(Data_PaP['Normalized_total_err']), \
                       'From DTT to Delta_sigma coefficient (result in nm2)' : Data_PaP['DTT_to_Dsigma_coeff']

                        })
    df.to_csv(fname, index=False)

    with open('Treated_thermic_data'+ '.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(Data_PaP, outp, pickle.HIGHEST_PROTOCOL)
    pass



