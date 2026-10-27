import importlib
import numpy as np
import scipy as sc

import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpmath
from mpmath import *
import time
import pandas as pd
import tempfile
import io






def get_data_Simulations_raw(dir_root) :
    """Récupère les data dans tout les fichiers listés dans h_list.

    Args:
        dir_root (str): chemin d'accès
        h_list (list): liste des h qu'on va exploiter
    Returns:
        res (1D array): toutes les grandeurs sont telles que lda_SPR[i] est évaluée en points[i,:]
    """

    os.chdir(dir_root)
    

    Spectra_data = {}



    for root, dirs, files in os.walk(".", topdown=False):
        for f in files :

            

            h_str = f.split('_')[-1][:-4]
            Spectra_data[h_str] = {}

            with open(dir_root + r'\\'+ f, 'r') as file:
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
                    
                    # print(data_col_name_values[i])    

                    # print(data_values[:,i])   
                   
                    D_str = data_col_name_values[i].split('_')[-1]

                    #print('valuer ajoutée : h = {} nm et D = {} nm'.format(h_str, D_str))
                    Spectra_data[h_str][D_str] = {}
                    Spectra_data[h_str][D_str]['freq'] = data_values[:,0]
                    Spectra_data[h_str][D_str]['lambda'] = data_values[:,1]
                    Spectra_data[h_str][D_str]['E'] = data_values[:,2]


                    Spectra_data[h_str][D_str]['Ext'] = data_values[:,i]
                    Spectra_data[h_str][D_str]['Scat'] = data_values[:,(i-1)]
                    Spectra_data[h_str][D_str]['Abs'] = data_values[:,(i-2)]

                    Spectra_data[h_str][D_str]['h'] = float(h_str)
                    Spectra_data[h_str][D_str]['D'] = float(D_str)
                    Spectra_data[h_str][D_str]['AR'] = Spectra_data[h_str][D_str]['D']/Spectra_data[h_str][D_str]['h']
                    Spectra_data[h_str][D_str]['Volume'] = np.pi*(Spectra_data[h_str][D_str]['D']/2)**2 *Spectra_data[h_str][D_str]['h']

                    # print(Spectra_data[h_str][D_str]['freq'])
                    # print(Spectra_data[h_str][D_str]['Ext'])
                    

    return Spectra_data





def get_data_Simulations_raw_v2(dir_root) :
    """Récupère les data dans tout les fichiers listés dans h_list.

    Args:
        dir_root (str): chemin d'accès
        h_list (list): liste des h qu'on va exploiter
    Returns:
        res (1D array): toutes les grandeurs sont telles que lda_SPR[i] est évaluée en points[i,:]
    """

    os.chdir(dir_root)
    

    Spectra_data = {}



    for root, dirs, files in os.walk(".", topdown=False):
        for f in files :

            
            with open(dir_root + r'\\'+ f, 'r') as file:
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

                    if h_str not in Spectra_data :
                        Spectra_data[h_str] = {}
                    
                    
                    # print(data_col_name_values[i])    

                    # print(data_values[:,i])   
                   
                    D_str = data_col_name_values[i].split('_')[-1]

                    #print('valuer ajoutée : h = {} nm et D = {} nm'.format(h_str, D_str))
                    Spectra_data[h_str][D_str] = {}
                    Spectra_data[h_str][D_str]['freq'] = data_values[:,0]
                    Spectra_data[h_str][D_str]['lambda'] = data_values[:,1]
                    Spectra_data[h_str][D_str]['E'] = data_values[:,2]


                    Spectra_data[h_str][D_str]['Ext'] = data_values[:,i]
                    Spectra_data[h_str][D_str]['Scat'] = data_values[:,(i-1)]
                    Spectra_data[h_str][D_str]['Abs'] = data_values[:,(i-2)]

                    Spectra_data[h_str][D_str]['h'] = float(h_str)
                    Spectra_data[h_str][D_str]['D'] = float(D_str)
                    Spectra_data[h_str][D_str]['AR'] = Spectra_data[h_str][D_str]['D']/Spectra_data[h_str][D_str]['h']
                    Spectra_data[h_str][D_str]['Volume'] = np.pi*(Spectra_data[h_str][D_str]['D']/2)**2 *Spectra_data[h_str][D_str]['h']

                    # print(Spectra_data[h_str][D_str]['freq'])
                    # print(Spectra_data[h_str][D_str]['Ext'])
                    

    return Spectra_data




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

    return res_fit


def fit_Lorentz_single_data(spectra) :
    """Spectra is a dictionnary defined in get_data_Simulation_raw

    Args:
        spectra (_type_): _description_
    """

    Ext = spectra['Ext']
    Energy = spectra['E']
    idx_max = np.argmax(Ext)



    if spectra['AR'] <= 2 :

        Ext_cut_max = Ext[:idx_max+1]
        Energy_cut_max = Energy[:idx_max+1]

        #initial guess [E_c, w, Area]
        p0 = (Energy[idx_max], 0.2, Ext[idx_max]*np.pi*0.2/2)

        res_fit = do_the_lorentz_fit(Energy_cut_max, Ext_cut_max, p0)
        res_fit['idx'] = idx_max


    else :

        #initial guess [E_c, w, Area]
        p0 = (Energy[idx_max], 0.2, Ext[idx_max]*np.pi*0.2/2)


        idx_cutoff = np.where(Ext > Ext.max()*0.7)[0][-1]

        #print(idx_cutoff)
        Ext_to_fit = Ext[:idx_cutoff+1]
        Energy_to_fit = Energy[:idx_cutoff+1]

        res_fit = do_the_lorentz_fit(Energy_to_fit, Ext_to_fit, p0)
        res_fit['idx'] = idx_cutoff




    return(res_fit)


def fit_Lorentz_all_data(Spectra_data) :
    """Make a lorentzian fit on all spectra in Spectra_data

    Args:
        Spectra_data (_type_): _description_
    """

    for key in Spectra_data :
        for kkey in Spectra_data[key] :
            spectra = Spectra_data[key][kkey]

            spectra['res_fit'] = fit_Lorentz_single_data(spectra)



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
        for kkey in Spectra_dic[key] :

            spectra = Spectra_dic[key][kkey]

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
    df.to_csv(fname, index=False)
    pass