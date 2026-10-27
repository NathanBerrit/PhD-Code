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
fontsize=30
import matplotlib as mpl



def parse_number(s):
    if pd.isna(s):  # gestion des NaN éventuels
        return np.nan
    s = str(s).strip()
    s = s.replace("i", "j")  # Python utilise j au lieu de i
    try:
     
        return float(s)   # si c’est un réel
    except ValueError:
       
        return complex(s) # sinon, c’est un complexe
    



def open_COMSOL_results(dir_root, Fnames, param_list) :
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
        res_raw['Q'] = np.real(data[:,-1])



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
            Res_dic[key]['freq_red'] = res_raw['freq'][idx_start:idx_end].real*Res_dic[key]['e']
            Res_dic[key]['Q'] = np.real(res_raw['Q'][idx_start:idx_end])

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




def get_modes_with_Q_selection(Spectra_dic, Q_th=20) :
    """Select the modes with a Q factor high enough

    Args:
        Spectra_dic (_type_): _description_
        Q_th : threshold Quality factor to consider that it's a mode
    """

    for key in Spectra_dic :
        spectra = Spectra_dic[key]


        spectra['modes'] = {}
        indexes = np.where(((spectra['Q']>Q_th) & (spectra['freq']>1)))
        
        spectra['modes']['f']  = spectra['freq'][indexes]
        spectra['modes']['f_red']  = spectra['freq_red'][indexes]
        spectra['modes']['Q'] = spectra['Q'][indexes]

    pass



def get_max_Q(Spectra_dic, freq_low_func, freq_high_func, AR_min=2, AR_max=5, n_modes=2 ) :

    AR = []
    Q_list = []
    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        ar = spectra['D']/spectra['e']

        if ar>= AR_min and ar<=AR_max :


            
            idx_freq_lim = np.where(((spectra['modes']['f_red']<=freq_high_func(ar)) & (spectra['modes']['f_red']>=freq_low_func(ar))))
            
            if np.size(idx_freq_lim)>0 :
                AR.append(ar)
                Q_list.append(np.max(spectra['modes']['Q'][idx_freq_lim][:n_modes]))

    AR = np.array(AR)
    Q_array = np.array(Q_list)

    return(AR, Q_array)



def get_Q_specific_bands(Spectra_dic, freq_low_func, freq_high_func, AR_min=2, AR_max=5 ) :

    AR = []
    Q_list = []
    freq_list = []

    dV_list = []
    V_list = []

    for key in Spectra_dic :
        spectra = Spectra_dic[key]
        ar = spectra['D']/spectra['e']

        if ar>= AR_min and ar<=AR_max :

            idx_freq_lim = np.where(((spectra['modes']['f_red']<=freq_high_func(ar)) & (spectra['modes']['f_red']>=freq_low_func(ar))))
            

            for idx in idx_freq_lim[0] :
                
                AR.append(ar)

                freq_list.append(spectra['modes']['f'][idx])
                Q_list.append(spectra['modes']['Q'][idx])


    
    AR = np.array(AR)
    Q_array = np.array(Q_list)
    freq_array = np.array(freq_list)

    return(AR, freq_array, Q_array)
