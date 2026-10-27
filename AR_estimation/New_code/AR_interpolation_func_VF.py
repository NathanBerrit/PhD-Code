import importlib
import numpy as np
import scipy as sc
from scipy.special import jv
from scipy.special import i1
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mpmath
from mpmath import *
import time
import pandas as pd
import tempfile
import io
import pickle



def get_treated_SPR_simu(dir_path, filename) :
    """Get the ductionnary of Simulation results for different sizes of the nanodisk"""
    try :
            os.chdir(dir_path)
            with open(filename + '.pkl', 'rb') as file:
                Trace = pickle.load(file)
        
                return Trace
        
    except :
        print("Error : there is no treated data")
        pass



def get_SPR_simu_data_and_interpolate(dir_root, fname, kernel='linear') :
    """Récupère les données de simulation pour plusieurs couples de points (h, D). Ensuite, fait une interpolation 2D.

    Args:
        dir_root (_type_): _description_
        fname (_type_): _description_
        kernel (str, optional): _description_. Defaults to 'linear'.

    Returns:
        Simu data (dic): Dicitonnary with simulations data
    """

    
    List_of_parameters = {}
    Simu_SPR = get_treated_SPR_simu(dir_root, fname)

    List_of_parameters['points'] = np.array([(Simu_SPR[key]['h'], Simu_SPR[key]['D']) for key in Simu_SPR])
    List_of_parameters['H'] = np.array([Simu_SPR[key]['h'] for key in Simu_SPR])
    List_of_parameters['D'] = np.array([Simu_SPR[key]['D'] for key in Simu_SPR])
    List_of_parameters['E_SPR'] = np.array([Simu_SPR[key]['E_SPR'] for key in Simu_SPR])
    List_of_parameters['Lda_SPR'] = np.array([Simu_SPR[key]['Lambda_SPR'] for key in Simu_SPR])
    List_of_parameters['W'] = np.array([Simu_SPR[key]['w_SPR'] for key in Simu_SPR])
    List_of_parameters['Aire_SPR'] = np.array([Simu_SPR[key]['Area_SPR'] for key in Simu_SPR])

    Lda_RBF = sc.interpolate.RBFInterpolator(List_of_parameters['points'], List_of_parameters['Lda_SPR'], kernel=kernel)
    E_RBF = sc.interpolate.RBFInterpolator(List_of_parameters['points'], List_of_parameters['E_SPR'], kernel=kernel)
    W_RBF = sc.interpolate.RBFInterpolator(List_of_parameters['points'], np.array(List_of_parameters['W'])*1e3, kernel = kernel)
    Aire_RBF = sc.interpolate.RBFInterpolator(List_of_parameters['points'], List_of_parameters['Aire_SPR'], kernel = kernel)

    return Simu_SPR, Lda_RBF, E_RBF, W_RBF, Aire_RBF, List_of_parameters['points']




def change_string_to_float(value) :

    try :
        return float(value)
    except :
        return 0

        
def get_ND_measures(dir_root, fname) :
    """Get the experimental data measured on the ND. They are extracted from the Origin synthesis file. Each column title 
    will be an entry of the returned dictionnary.
    
    All available keys on extracted data :

    AR	
    zone
    name	
    date	
    lda_SPR	
    E_SPR	
    w_SPR	
    Aire_SPR	
    D_estimated	
    h_estimated	
    D_estimated_1D (h = 25 nm)	
    Gsurf_estimated	
    lambda_probe	
    polar	
    tau	
    tau_err	
    T	
    T_err	
    phi	
    phi_err	
    f	
    f_red	
    AR_estimated	
    Q	
    Q_err	
    tau2	
    tau2_err	
    T2	
    T2_err	
    phi2	
    phi2_err	
    Q2	
    Q2_err	
    Comment
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
        data_values = np.genfromtxt(data_stream, delimiter='\t', skip_header=3,dtype=str, usecols=range(np.size(data_col_name_values)-1))

        # print(data_values[0,:])

        # print(data_col_name_values)
    
    for i in range(np.size(data_values[:,0])) :
        key = ' '.join(data_values[i,0:3])
        
        Data_dic[key] = {}
        Data_dic[key]['ND_name'] = key
        for j in range(np.size(data_col_name_values)-1) :

            if j > 3 :
                Data_dic[key][data_col_name_values[j]] = change_string_to_float(data_values[i, j])

            else :
                Data_dic[key][data_col_name_values[j]] = data_values[i, j]
            
    return Data_dic





def get_handmade_interpolation(dir_root, fname) :
    """Get the experimental data and extract the interpolation of AR that was made by hand. 
    They are tyhe Ar interpolation made for the ND for which there is a 2D solution, but the code didn't converge for unknown reasons.

    
    All available keys on extracted data :

    AR	
    zone
    name	
    date	
    lda_SPR	
    E_SPR	
    w_SPR	
    Aire_SPR	
    D_estimated	2D
    h_estimated	2D
    D_estimated_1D
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
        data_values = np.genfromtxt(data_stream, delimiter='\t', skip_header=2,dtype=str, usecols=range(np.size(data_col_name_values)-1))

        # print(data_values[0,:])

        # print(data_col_name_values)

    for i in range(np.size(data_values[:,0])) :
        key = ' '.join(data_values[i,0:3])
        
        Data_dic[key] = {}
        Data_dic[key]['ND_name'] = key
    
        for j in range(np.size(data_col_name_values)-1) :

            if j > 3 :
                Data_dic[key][data_col_name_values[j]] = change_string_to_float(data_values[i, j])

            else :
                Data_dic[key][data_col_name_values[j]] = data_values[i, j]
            
    return Data_dic





def add_handmade_interpolation_to_treated_data(Treated_data_dic, handmade_data_dic) :


    for key in Treated_data_dic :
        try :
            ND_handmade = handmade_data_dic[key]
            ND_treated = Treated_data_dic[key]
            
        
            if ND_handmade['D_estimated 2D'] != 0 :
                

                ND_treated['2D']['D'] = ND_handmade['D_estimated 2D']
                ND_treated['2D']['h'] = ND_handmade['h_estimated 2D']
                ND_treated['2D']['AR'] = ND_treated['2D']['D']/ND_treated['2D']['h']
                print('value added for {}'.format(key))

        except :
            pass
        
def get_ND_measures_Fabio(dir_root, fname) :
    """Get the experimental data measured on the ND. They are extracted from the Origin synthesis file. Each column title 
    will be an entry of the returned dictionnary.

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
        data_values = np.genfromtxt(data_stream, delimiter='\t', skip_header=3,dtype=str, usecols=range(np.size(data_col_name_values)-1))

        # print(data_values[0,:])

        # print(data_col_name_values)
        print(np.size(data_values[:,0]))
    for i in range(np.size(data_values[:,0])) :
        
        key = str(i) + ' -- square ' + data_values[i,0] + ' - case ' + data_values[i,1]
        
        Data_dic[key] = {}
        Data_dic[key]['ND_name'] = key
        
        for j in range(np.size(data_col_name_values)-1) :
            Data_dic[key][data_col_name_values[j]] = change_string_to_float(data_values[i, j])
            
    return Data_dic



    
        
def single_AR_estimation_1D(f, x_target, h=25, d_min=5, d_max=500, d_guess=50, ND_name= ' ', est_parameter ='') :
    """Find the estimated diameter for the object with h fixed, using the interpolation function f and target value x_target

    Args:
        f (interpolated func): interpolated function
        x_target (float): target value to be matched.
        h (int): Assumed thickness of the ND. Defaults to 21.
        d_min(float) : minimum diameter to fit
        d_max(float) : maximum diameter to fit
        d_guess (float) : initial guess
        est_parameter (str) : the fitted parameter for the estimation (SPR Area, SPR position...)
    """

    def func_D_1D(D, h=h) :
        Z = np.column_stack((np.full_like(D, h, dtype=float),D))
        
        return(f(Z))
    
    #print(func_D_1D(d_min, h=h)-x_target, func_D_1D(d_max, h=h)-x_target)
    try :
        res = sc.optimize.root_scalar(lambda x : (func_D_1D(x, h=h)-x_target), x0=[d_guess], bracket=[d_min, d_max])  
        if not res.converged :
            print('Erreur de convergence 1D sur {} avec le paramètre de fit : {}'.format(ND_name, est_parameter))
        
        else :
            return(res.root)
        
    except :
        print('Erreur de convergence 1D sur {} avec le paramètre de fit : {}'.format(ND_name, est_parameter))
        return 0
    

def AR_estimation_1D_Aire_SPR(Data_dic, Aire_RBF, h=21) :
    for key in Data_dic :
        ND_data = Data_dic[key]
        ND_data['1D'] = {}

        #print(ND_data['lda_SPR'])
        ND_data['1D']['h'] = h
        ND_data['1D']['D'] = single_AR_estimation_1D(Aire_RBF, ND_data['Aire_SPR'], h=h, ND_name=key, est_parameter='SPR Area')
        ND_data['1D']['AR'] = ND_data['1D']['D']  / h 


def AR_estimation_1D_Lda_SPR(Data_dic, Lda_RBF, h=21) :

    
    for key in Data_dic :
        ND_data = Data_dic[key]
        ND_data['1D_Lda'] = {}

        #print(ND_data['lda_SPR'])
        ND_data['1D_Lda']['h'] = h
        ND_data['1D_Lda']['D'] = single_AR_estimation_1D(Lda_RBF, ND_data['lda_SPR'], h=h, ND_name=key, est_parameter='SPR Lda')
        ND_data['1D_Lda']['AR'] = ND_data['1D_Lda']['D']  / h 

def single_AR_estimation_2D(f_lda, f_Aire_SPR,  lda_target, Aire_SPR_target, ND_name, d_guess=100, h_guess=25) :
    """Find the estimated diameter for the object with h fixed, using the interpolation function f and target value x_target

    Args:
        f (interpolated func): interpolated function
        x_target (float): target value to be matched.
        h (int): Assumed thickness of the ND. Defaults to 21.
        d_min(float) : minimum diameter to fit
        d_max(float) : maximum diameter to fit
        d_guess (float) : initial guess
    """

    def func_2D(vars ) :
        x, y = vars

        Z = np.column_stack((x, y))
        
        #print(x, y)
        return(np.array([f_lda(Z)[0]-lda_target, f_Aire_SPR(Z)[0]-Aire_SPR_target]))
    
    res = sc.optimize.root(lambda vars : func_2D(vars), x0=[d_guess, h_guess])  
    return(res)
    

def AR_estimation_2D(Data_dic, Lda_RBF, Aire_SPR_RBF, d_guess=100, h_guess=23) :
    """For every experimental data present in Data_dic, do a 2D interpolation to estimate the diameter and the
    thickness of the ND from interpolated data (Lda_RBF, Area_RBF) obtained through COMSOL numerical simulations

    Args:
        Data_dic (_type_): Experimental data
        Lda_RBF (_type_): Radial interpolation of SPR lda position thanks to (D,e)
        Aire_SPR_RBF (_type_): Radial interpolation of SPR area thanks to (D,e)
        d_guess (int, optional): guess diameter Defaults to 100.
        h_guess (int, optional): guessed thickness Defaults to 23.
    """
    for key in Data_dic :
        ND_data = Data_dic[key]
        ND_data['2D'] = {}

        res = single_AR_estimation_2D(Lda_RBF, Aire_SPR_RBF, ND_data['lda_SPR'], ND_data['Aire_SPR'], key, d_guess=d_guess, h_guess=h_guess)
        
        if not res.success or res.x[0]<=0 or res.x[1] <= 0 :
            print('Erreur de convergence 2D sur : {} - (Lda_SPR, Aire_SPR) = {}'.format(key, (ND_data['lda_SPR'], ND_data['Aire_SPR'])))
            ND_data['2D']['h'] = 1
            ND_data['2D']['D'] = 0

        else :
            #print('Convergence 2D : Disque {} (Lda_SPR = {:.1f} nm, Aire_SPR = {:.1f} nm2)'.format(key, ND_data['lda_SPR'], ND_data['Aire_SPR']))
            ND_data['2D']['h'] = res.x[0]
            ND_data['2D']['D'] = res.x[1]
        ND_data['2D']['AR'] = ND_data['2D']['D'] / ND_data['2D']['h']


def Gamma_surf_estimation(Data_dic, f_W) :

    for key in Data_dic :
        ND_data = Data_dic[key]

        ND_data['2D']['Gamma_surf'] = ND_data['w_SPR'] - f_W([[ND_data['2D']['h'], ND_data['2D']['D']]])[0]
        ND_data['1D']['Gamma_surf'] = ND_data['w_SPR'] - f_W([[ND_data['1D']['h'], ND_data['1D']['D']]])[0]
        ND_data['1D_Lda']['Gamma_surf'] = ND_data['w_SPR'] - f_W([[ND_data['1D_Lda']['h'], ND_data['1D_Lda']['D']]])[0]

        if ND_data['2D']['h'] == 0 or ND_data['2D']['D'] == 0 :
            ND_data['2D']['Gamma_surf'] = 0 

        if ND_data['1D']['h'] == 0 or ND_data['1D']['D'] == 0 :
            ND_data['1D']['Gamma_surf'] = 0 

        if ND_data['1D_Lda']['h'] == 0 or ND_data['1D_Lda']['D'] == 0 :
            ND_data['1D_Lda']['Gamma_surf'] = 0 

def save_interpolation_to_csv(Data_dic, dir_to_save, fname) :

    ND_name = np.array([Data_dic[key]['ND_name'] for key in Data_dic])

    D_estimation_1D_Area = np.array([Data_dic[key]['1D']['D'] for key in Data_dic])
    Gamma_surf_estimation_1D_Area = np.array([Data_dic[key]['1D']['Gamma_surf'] for key in Data_dic])
    D_estimation_1D_Lda = np.array([Data_dic[key]['1D_Lda']['D'] for key in Data_dic])
    Gamma_surf_estimation_1D_Lda = np.array([Data_dic[key]['1D_Lda']['Gamma_surf'] for key in Data_dic])
    D_estimation_2D = np.array([Data_dic[key]['2D']['D'] for key in Data_dic])
    h_estimation_2D = np.array([Data_dic[key]['2D']['h'] for key in Data_dic])
    Gamma_surf_estimation_2D = np.array([Data_dic[key]['2D']['Gamma_surf'] for key in Data_dic])
    
    AR_1D_Area = np.array([Data_dic[key]['1D']['AR'] for key in Data_dic])
    AR_1D_Lda = np.array([Data_dic[key]['1D_Lda']['AR'] for key in Data_dic])
    AR_2D = np.array([Data_dic[key]['2D']['AR'] for key in Data_dic])

    if not(os.path.exists(dir_to_save)) :
        os.mkdir(dir_to_save)
    os.chdir(dir_to_save)
    df = pd.DataFrame({'name' : ND_name, \
                        'D estimation 2D' : D_estimation_2D,\
                        'h estimation 2D' : h_estimation_2D, \
                        'D estimation 1D Area (h= 21 nm)' : D_estimation_1D_Area, \
                        'D estimation 1D Lda (h= 21 nm)' : D_estimation_1D_Lda, \
                        'Gamma_surf estimation 2D' : Gamma_surf_estimation_2D, \
                        'Gamma_surf estimation 1D Area' : Gamma_surf_estimation_1D_Area, \
                        'Gamma_surf estimation 1D Lda' : Gamma_surf_estimation_1D_Lda, \
                        'AR 2D ' : AR_2D ,\
                        'AR 1D Area' : AR_1D_Area ,\
                        ' AR 1D Lda' : AR_1D_Lda})
    df.to_csv(fname, index=False)
    pass




def save_interpolation_to_data_dic(Data_dic, dir_to_save, fname) :


    if not(os.path.exists(dir_to_save)) :
        os.mkdir(dir_to_save)
    os.chdir(dir_to_save)
    
    key_to_del = []

    for key in Data_dic :
        if Data_dic[key]['Q'] == 0 :
            key_to_del.append(key)

    for key in key_to_del :
        del Data_dic[key]
    with open(fname + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(Data_dic, outp, pickle.HIGHEST_PROTOCOL)

      
def print_AR_estimation_results(Data_dic) :

    for key in Data_dic :
        print('-------------------------------------------')
        print(Data_dic[key]['lda_SPR'], Data_dic[key]['Aire_SPR'])
        print((Data_dic[key]['2D']['h'], Data_dic[key]['2D']['D']), Data_dic[key]['2D']['AR'])



def get_treated_AR_data(dir_path, filename) :
    """Get a PaP_dataset object from the path.

    Args:
        dir_path (str): path
        filename (str): filename

    Returns:
        Trace: The PaP_dataset_object.
    """

    try :
        os.chdir(dir_path)
        with open(filename + '.pkl', 'rb') as file:
            AR_data_dic = pickle.load(file)

            return AR_data_dic
       
    except :
        print("Error : there is no treated data")
        pass
    