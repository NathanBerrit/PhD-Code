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


########## Getting data ####################

def get_single_spectra_data(dir_root, fname) :
    os.chdir(dir_root)


    df = pd.read_csv(fname + '.csv', dtype='float64', skiprows=4)
    data = df.__array__()

    res = {}
    res['Lambda'] = data[:,0]*1e9
    res['freq'] = data[:,1]
    res['Sigma_scat'] = data[:,2]*1e15
    res['Sigma_abs'] = data[:,3]*1e15
    res['Sigma_ext'] = res['Sigma_abs'] + res['Sigma_scat']
    return res


def get_spectra_size(raw_res) :
    """Finds the size"""

    compt = 1

    val_ini = raw_res['Lambda'][0]

    val_final = raw_res['Lambda'][compt]

    while val_final > val_ini :
        compt +=1
        val_final = raw_res['Lambda'][compt]
        val_ini = raw_res['Lambda'][compt-1]

    return compt





def get_raw_size_sweep_data_Nathan(dir_root, fname) :
    """The document is an excel table with COMSOL simulation iterationg on both D, e and lambda. All spectra must be completely calculated.
    
    Returns :
    res (dic) : a dictionnary, each key correspond to the simulated spectra of 1 nanodisk (in forme of a dic). 
    the key for each spectra dic are D, h, AR, Volume, Lambda, freq,E, Sigma_ext, Sigma_abs, Sigma_scat"""

    os.chdir(dir_root)

    df = pd.read_csv(fname + '.csv', dtype='float64', skiprows=4)
    data = df.__array__()

    # Get the whole data
    res_raw = {}
    res = {}

    
    res_raw = {}
    res_raw['D'] = data[:,0]*1e9
    res_raw['h'] = data[:,1]*1e9

    res_raw['Lambda'] = data[:,2]*1e9
    res_raw['freq'] = data[:,3]*1e9
    res_raw['Sigma_scat'] = data[:,4]*1e18
    res_raw['Sigma_abs'] = data[:,5]*1e18
    res_raw['Sigma_ext'] = res_raw['Sigma_abs'] + res_raw['Sigma_scat']
    

    spectra_size = get_spectra_size(res_raw)

    for i in range(np.size(res_raw['D'])//spectra_size) :
        start_index = i*spectra_size
        key = 'ND_D_' + str(np.round(res_raw['D'][start_index], 1)) + '_h_' + str(np.round(res_raw['h'][start_index], 1))

        res[key] = {}
        res[key]['D'] = np.round(res_raw['D'][start_index], 1)
        res[key]['h'] = np.round(res_raw['h'][start_index], 1)
        res[key]['AR'] = np.round(res[key]['D'] / res[key]['h'], 1)
        res[key]['Lambda'] = res_raw['Lambda'][start_index:start_index+spectra_size]
        res[key]['freq'] = res_raw['freq'][start_index:start_index+spectra_size]
        res[key]['Sigma_scat'] = res_raw['Sigma_scat'][start_index:start_index+spectra_size]
        res[key]['Sigma_abs'] = res_raw['Sigma_abs'][start_index:start_index+spectra_size]
        res[key]['Sigma_ext'] = res_raw['Sigma_ext'][start_index:start_index+spectra_size]
        res[key]['Volume'] = np.pi*(res[key]['D']/2)**2 *res[key]['h']
        res[key]['E'] = 1240/res[key]['Lambda']

    return res






def get_raw_size_sweep_data_Aurelien(dir_root, fname) :

    """Récupère les data dans tout les fichiers listés dans h_list.

    Args:
        dir_root (str): chemin d'accès
        h_list (list): liste des h qu'on va exploiter
    Returns:
        res (1D array): toutes les grandeurs sont telles que lda_SPR[i] est évaluée en points[i,:]
    """

    os.chdir(dir_root)
    

    Spectra_data = {}


    with open(dir_root + '\\'+ fname + '.dat', 'r') as file:
        # Lire toutes les lignes du fichier
        lines = file.readlines()
        
        
        # Remplacer les virgules par des points et préparer les données
        cleaned_lines = [line.replace(',', '.').strip() for line in lines]

        data_stream = io.StringIO('\n'.join(cleaned_lines))
        # Charger les données en tant que tableau numpy



        
        data_col_name_values = np.genfromtxt(data_stream, delimiter='\t',dtype=str, filling_values = 0, max_rows=1)
        data_stream.seek(0)
        data_values = np.genfromtxt(data_stream, delimiter='\t', skip_header=1,dtype=float)
        #print(np.shape(data_col_name_values), np.shape(data_values))
    
    

    for i in range(np.size(data_col_name_values)) :


        if data_col_name_values[i].startswith('Ext') : ##It is an extinction spectra
            
            h_str =  data_col_name_values[i].split('_')[-3]
            D_str = data_col_name_values[i].split('_')[-1]


            key = 'ND_D_' + D_str + '_h_' + h_str
            
            
            # print(data_col_name_values[i])    

            # print(data_values[:,i])   
            
            
            #print('valuer ajoutée : h = {} nm et D = {} nm'.format(h_str, D_str))
            Spectra_data[key] = {}
            Spectra_data[key]['freq'] = data_values[:,0]
            Spectra_data[key]['lambda'] = data_values[:,1]
            Spectra_data[key]['E'] = data_values[:,2]


            Spectra_data[key]['Sigma_ext'] = data_values[:,i]
            Spectra_data[key]['Sigma_scat'] = data_values[:,(i-1)]
            Spectra_data[key]['Sigma_abs'] = data_values[:,(i-2)]

            Spectra_data[key]['h'] = float(h_str)
            Spectra_data[key]['D'] = float(D_str)
            Spectra_data[key]['AR'] = Spectra_data[key]['D']/Spectra_data[key]['h']
            Spectra_data[key]['Volume'] = np.pi*(Spectra_data[key]['D']/2)**2 *Spectra_data[key]['h'] 

    return Spectra_data

def get_treated_size_sweep_data(dir_path, filename) :


    try :
            os.chdir(dir_path)
            with open(filename + '.pkl', 'rb') as file:
                Trace = pickle.load(file)
        
                return Trace
        
    except :
        print("Error : there is no treated data")
        pass





############### Treating data ######################



def Lorentz_no_offset(x, xc, w, Area) :
    """Lorentzian resonance without offset. This function is defined just as he one in Origin.
    Changing w doesn't change the Area (no need to make the calculation for the tenth time)

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        Area (_type_): Total area of the curve
    """
    res = 2 * Area / np.pi * w * 1/(4*(x-xc)**2 + w**2)
    
    return res



def do_the_lorentz_fit(E, Ext, p0) :
    """Do a Lorentzian fit.

    Args:
        E (1D array): Energy
        Ext (1D array): Extinction spectra
        p0 (_type_): initial guess for (E_c, w, Area)

    Returns:
        res_fit (dic): dictionnary of the results
    """
    res_fit = {}
    
    try :

        res, pcov = curve_fit(Lorentz_no_offset, E, Ext, p0 = p0)
        
        

        #res, pcov = curve_fit(Lorentz_no_offset, Energy, Ext, p0 = p0)
        p_sigma = np.sqrt(np.diag(pcov))


        res_fit['E_c'] = res[0]
        res_fit['w'] = res[1]
        res_fit['Area'] = res[2]

        res_fit['E_c_err'] = p_sigma[0]
        res_fit['w_err'] = p_sigma[1]
        res_fit['Area_err'] = p_sigma[2]
        res_fit['converged'] = True
        

    except :
        res_fit['E_c'] = p0[0]
        res_fit['w'] = p0[1]
        res_fit['Area'] = p0[2]

        res_fit['E_c_err'] = 0
        res_fit['w_err'] = 0
        res_fit['Area_err'] = 0
        res_fit['converged'] = False
        print('failed !')

    E_left = res_fit['E_c'] - res_fit['w']/2
    E_right = res_fit['E_c'] + res_fit['w']/2

    Lda_left = 1240/E_left
    Lda_right = 1240/E_right

    res_fit["E_left"] = E_left
    res_fit["E_right"] = E_right

    res_fit["Lda_left"] = Lda_left
    res_fit["Lda_right"] =  Lda_right

    res_fit['Delta_lda'] = Lda_left-Lda_right
    res_fit['Lda_c'] = 1240/res_fit['E_c']

    # Maximum value of the function A
    res_fit['A'] = res_fit['Area']/res_fit['w'] / np.pi*2
    res_fit['A_err'] = res_fit['A'] * np.sqrt((res_fit['Area_err']/res_fit['Area'])**2 + (res_fit['w_err']/res_fit['w'])**2 )


    
    res_fit['Delta_lda_err'] = res_fit['Delta_lda']*res_fit['w_err']/res_fit['w']
    res_fit['Lda_c_err'] = res_fit['Lda_c']*res_fit['E_c_err']/res_fit['E_c']
    res_fit['coeffs'] = [res_fit['E_c'], res_fit['w'], res_fit['Area']]
    return res_fit




def fit_Lorentz_single_data(spectra) :
    """Spectra is a dictionnary defined in get_data_Simulation_raw. For big enough AR, we fit both side of resonance
    For small AR, there are a lot of interband response and the SPR is not Lorentzian anymore. Then I only fit the low energy side.

    Args:
        spectra (dic): single spectra raw data from COMSOL
    """

    Ext = spectra['Sigma_ext']
    Energy = spectra['E']
    idx_max = np.argmax(Ext)


    if spectra['AR'] <= 2 :
    
        
        indexes = np.where(Energy <= Energy[idx_max])
        
        Ext_cut_max = Ext[indexes]
        Energy_cut_max = Energy[indexes]

        #initial guess [E_c, w, Area]
        p0 = (Energy[idx_max], 0.2, Ext[idx_max]*np.pi*0.2/2)

        res_fit = do_the_lorentz_fit(Energy_cut_max, Ext_cut_max, p0)
        res_fit['idx'] = indexes


    else :

        #initial guess [E_c, w, Area]
        p0 = (Energy[idx_max], 0.2, Ext[idx_max]*np.pi*0.2/2)


        idx_cutoff = np.where(Ext > Ext.max()*0.7)

        indexes = np.where(Energy < np.max(Energy[idx_cutoff]))
        #print(idx_cutoff)
        Ext_to_fit = Ext[indexes]
        Energy_to_fit = Energy[indexes]

        res_fit = do_the_lorentz_fit(Energy_to_fit, Ext_to_fit, p0)
        res_fit['idx'] = indexes




    return(res_fit)


def fit_Lorentz_all_data(Spectra_data) :
    """Make a lorentzian fit on all spectra in Spectra_data

    Args:
        Spectra_data (_type_): _description_
    """

    for key in Spectra_data :
        
        spectra = Spectra_data[key]

        spectra['res_fit'] = fit_Lorentz_single_data(spectra)

        if not spectra['res_fit']['converged'] :
            print(key)
            print('-------------------')



################ Saving data #####################

def save_treated_simulation_data(Spectra_dic, dir_to_save, fname) :

    try :
        os.chdir(dir_to_save)

    except :
        os.makedirs(dir_to_save)
        os.chdir(dir_to_save)



    D = []
    h = []
    AR = []
    Volume = []
    Position = []
    Position_lambda = []
    W = []
    Area = []


    for key in Spectra_dic :
    

        spectra = Spectra_dic[key]

        D.append(spectra['D'])
        h.append(spectra['h'])
        AR.append(spectra['AR'])
        Volume.append(spectra['Volume'])
        Position.append(spectra['res_fit']['E_c'])
        Position_lambda.append(spectra['res_fit']['Lda_c'])
        W.append(spectra['res_fit']['w'])
        Area.append(spectra['res_fit']['Area'])   

    Dic_dataframe = {
        'D (nm)' : D, \
        'h (nm)' : h, \
        'AR' : AR, \
        'Volume (nm3)' : Volume, \
        'E_SPR (eV)' : Position, \
        'Lambda_SPR (nm)' : Position_lambda, \
        'w (eV)' : W, \
        'Area (eV.nm2)' : Area, \
    }


    df = pd.DataFrame(Dic_dataframe)
    df.to_csv(fname + '.csv', index=False)

    
    with open(fname + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(Spectra_dic, outp, pickle.HIGHEST_PROTOCOL)


    pass