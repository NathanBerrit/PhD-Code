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


def get_treated_data_dictionnary_dataset(dir_path, filename) :
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
            Trace = pickle.load(file)
    
            return Trace
    except :
        print("Error : there is no treated data")
        pass


def reorder_all_traces_dic(Data_dic) :

    list = []
    for key in Data_dic :
        list.append(float(key))

    array = np.array(list)
    array.argsort()
    #print(array[array.argsort()])

    reordered_dic = {str(int(k)) : Data_dic[str(int(k))] for k in array[array.argsort()]}
    return(reordered_dic)
    


def get_all_raw_traces(dir_root) :


    Traces_dic = {}
    os.chdir(dir_root)
    
    dir_list = os.listdir()

    for key in dir_list:
        if key.endswith('treated') :
            lda = key.split('_')[-2]
            os.chdir(key)
            
            with open('Trace_acoustic_fitted' + '.pkl', 'rb') as file :
                
                Trace = pickle.load(file)
                Trace.lda = float(lda)
                Traces_dic[lda] = Trace

            os.chdir(dir_root)
    return(Traces_dic)

    
def get_treated_data_acoustic(dir_root, fname) :
    """
    Args:
        dir_root (_type_): _description_
        fname (_type_): _description_
    """
    os.chdir(dir_root)
    raw_data_acoustic = np.array(pd.read_csv(fname, skiprows=0))
    Data_acoustic = {}

    Data_acoustic['lambda'] = raw_data_acoustic[:,0]
    Data_acoustic['detuning_normalized'] = raw_data_acoustic[:,1]
    Data_acoustic['Delta T'] = raw_data_acoustic[:,2]
    Data_acoustic['Delta_sigma_ac'] = raw_data_acoustic[:,3]
    Data_acoustic['Delta_sigma_ac_err'] = raw_data_acoustic[:,4]
    Data_acoustic['Delta_sigma_ac_norm'] = raw_data_acoustic[:,5]
    Data_acoustic['Delta_sigma_ac_norm_err'] = raw_data_acoustic[:,6]
    Data_acoustic['Delta_sigma_ac_fullnorm'] = raw_data_acoustic[:,7]
    Data_acoustic['Delta_sigma_ac_fullnorm_err'] = raw_data_acoustic[:,8]

    return Data_acoustic



def get_treated_data_thermic(dir_root, fname) :
    """
    Args:
        dir_root (_type_): _description_
        fname (_type_): _description_
    """
    os.chdir(dir_root)
    raw_data_acoustic = np.array(pd.read_csv(fname, skiprows=0))
    Data_acoustic = {}

    Data_acoustic['lambda'] = raw_data_acoustic[:,0]
    Data_acoustic['detuning_normalized'] = raw_data_acoustic[:,1]
    Data_acoustic['Delta T'] = raw_data_acoustic[:,2]
    Data_acoustic['Delta_sigma_th'] = raw_data_acoustic[:,3]
    Data_acoustic['Delta_sigma_th_err'] = raw_data_acoustic[:,4]
    Data_acoustic['Delta_sigma_th_norm'] = raw_data_acoustic[:,5]
    Data_acoustic['Delta_sigma_th_norm_err'] = raw_data_acoustic[:,6]
    Data_acoustic['Delta_sigma_th_fullnorm'] = raw_data_acoustic[:,7]
    Data_acoustic['Delta_sigma_th_fullnorm_err'] = raw_data_acoustic[:,8]
    return Data_acoustic






def get_simulation_dsigma_thermic(dir_root, fname) :

    os.chdir(dir_root)

    
    with open(fname + '.dat', 'r') as file:
        # Lire toutes les lignes du fichier
        lines = file.readlines()

        # Remplacer les virgules par des points et préparer les données
        cleaned_lines = [line.replace(',', '.').strip() for line in lines]

        data_stream = io.StringIO('\n'.join(cleaned_lines))
        # Charger les données en tant que tableau numpy
        data = np.genfromtxt(data_stream, delimiter='\t', skip_header=0,dtype=float)
    #print(h)
    #print(data)

    Data_simul = {}

    Data_simul['Lambda'] = data[:,0]
    Data_simul['Stoll'] = data[:,1]
    Data_simul['Cahill'] = data[:,2]

    return Data_simul



def get_treated_experimental_data(dir_root, lda_SPR, color = "#ffa826") :
    
    
    dir_root_treated_data = dir_root + r'\Extinction variation calculation'
    dir_root_exp_traces = dir_root + r'\PaP_study'
    dir_root_simul_data = dir_root + r'\Simulation results'
    fname_acoustic = 'Treated_acoustic_data'
    fname_thermic = 'Treated_thermic_data'

    Data = {}
    Data['Simul'] = {}
    Data['lda_SPR'] = lda_SPR
    Data['color'] = color
    Data['acoustic']  = get_treated_data_dictionnary_dataset(dir_root_treated_data, fname_acoustic)
    Data['thermic']  = get_treated_data_dictionnary_dataset(dir_root_treated_data, fname_thermic)
    Data['All traces'] = get_all_raw_traces(dir_root_exp_traces)
    Data['All traces'] = reorder_all_traces_dic(Data['All traces'])
    #Data['AR3.5-5']['Simul']['thermic'] = get_simulation_dsigma_thermic(dir_root_simul_data_1, 'd_sigma_simulation_thermic')

    return Data