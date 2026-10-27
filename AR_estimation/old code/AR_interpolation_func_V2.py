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




      
def get_data_AR(dir_root, fname) :
    """Récupère les données des spectre simulées. Pour chaque couple (h,D) testé en simulation, on a extrait la position de la résonance, sa largeur et son aire.


    Args:
        dir_root (str): chemin d'accès
        fname (list): nom du fichier
    Returns:
        res (1D array): toutes les grandeurs sont telles que lda_SPR[i] est évaluée en points[i,:]
    """

    os.chdir(dir_root)
    

    H = set()
    D = set()
    points = []
    Lda_SPR = []
    E_SPR = []
    Aire_SPR = []
    W = []

    data = np.loadtxt(fname, skiprows=1, delimiter=',')

    for i in range(np.size(data[:,0])) :
        
        D.add(data[i,0])
        H.add(data[i,1])
        points.append([data[i,1], data[i,0]])
        E_SPR.append(data[i,4])
        Lda_SPR.append(data[i,5])
        W.append(data[i,6])
        Aire_SPR.append(data[i,7])



    #print(H)
    return np.array(points), np.array(H), np.array(D), np.array(E_SPR), np.array(Lda_SPR), np.array(W), np.array(Aire_SPR)




def get_simu_data_and_interpolate(dir_root, fname, kernel='linear') :
    """Récupère les données de simulation pour plusieurs couples de points (h, D). Ensuite, fait une interpolation 2D.

    Args:
        dir_root (_type_): _description_
        fname (_type_): _description_
        kernel (str, optional): _description_. Defaults to 'linear'.

    Returns:
        Simu data (dic): Dicitonnary with simulations data
    """

    
    Simu_data = {}
    points, H, D, E_SPR, Lda_SPR, W, Aire_SPR = get_data_AR(dir_root, fname)

    Simu_data['points'] = points
    Simu_data['H'] = H
    Simu_data['D'] = D
    Simu_data['E_SPR'] = E_SPR
    Simu_data['Lda_SPR'] = Lda_SPR
    Simu_data['W'] = W
    Simu_data['Aire_SPR'] = Aire_SPR


    Lda_RBF = sc.interpolate.RBFInterpolator(Simu_data['points'], Simu_data['Lda_SPR'], kernel=kernel)
    E_RBF = sc.interpolate.RBFInterpolator(Simu_data['points'], Simu_data['E_SPR'], kernel=kernel)
    W_RBF = sc.interpolate.RBFInterpolator(Simu_data['points'], np.array(Simu_data['W'])*1e3, kernel = kernel)
    Aire_RBF = sc.interpolate.RBFInterpolator(Simu_data['points'], Simu_data['Aire_SPR'], kernel = kernel)

    return Simu_data, Lda_RBF, E_RBF, W_RBF, Aire_RBF




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
    

def AR_estimation_1D_Aire_SPR(Data_dic, Aire_RBF, h=25) :


    
    for key in Data_dic :
        ND_data = Data_dic[key]
        ND_data['1D'] = {}

        #print(ND_data['lda_SPR'])
        ND_data['1D']['h'] = h
        ND_data['1D']['D'] = single_AR_estimation_1D(Aire_RBF, ND_data['Aire_SPR'], h=h, ND_name=key, est_parameter=' SPR Area')
        ND_data['1D']['AR'] = ND_data['1D']['D']  / h 



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

    for key in Data_dic :
        ND_data = Data_dic[key]
        ND_data['2D'] = {}

        res = single_AR_estimation_2D(Lda_RBF, Aire_SPR_RBF, ND_data['lda_SPR'], ND_data['Aire_SPR'], key, d_guess=d_guess, h_guess=h_guess)
        
        if not res.success or res.x[0]<=0 or res.x[1] <= 0 :
            print('Erreur de convergence 2D sur : {}'.format(key))
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

        if ND_data['2D']['h'] == 0 or ND_data['2D']['D'] == 0 :
            ND_data['2D']['Gamma_surf'] = 0 

        if ND_data['1D']['h'] == 0 or ND_data['1D']['D'] == 0 :
            ND_data['1D']['Gamma_surf'] = 0 


def save_interpolation_to_csv(Data_dic, dir_to_save, fname) :

    ND_name = np.array([Data_dic[key]['ND_name'] for key in Data_dic])

    D_estimation_1D = np.array([Data_dic[key]['1D']['D'] for key in Data_dic])
    Gamma_surf_estimation_1D = np.array([Data_dic[key]['1D']['Gamma_surf'] for key in Data_dic])
    D_estimation_2D = np.array([Data_dic[key]['2D']['D'] for key in Data_dic])
    h_estimation_2D = np.array([Data_dic[key]['2D']['h'] for key in Data_dic])
    Gamma_surf_estimation_2D = np.array([Data_dic[key]['2D']['Gamma_surf'] for key in Data_dic])
    
    
    os.chdir(dir_to_save)
    df = pd.DataFrame({'name' : ND_name, \
                        'D estimation 2D' : D_estimation_2D,\
                        'h estimation 2D' : h_estimation_2D, \
                       'Gamma_surf estimation 2D' : Gamma_surf_estimation_2D, \
                        'D estimation 1D (h= {} nm)' : D_estimation_1D, \
                        'Gamma_surf estimation 1D' : Gamma_surf_estimation_1D})
    df.to_csv(fname, index=False)
    pass




def save_interpolation_to_data_dic(Data_dic, dir_to_save, fname) :

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
    