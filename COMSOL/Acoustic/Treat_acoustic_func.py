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

def scan_single_spectra(idx_start, raw_res, key_to_scan, verbose=False) :
    """Finds the size of a single spectra stored in raw_res.

    Args:
        idx_start (int): start index for this spectra in raaw_res
        raw_res (dic): the raw simulation data containing different spectra of different size
        key_to_scan : the key that will be scanned in order to detect the end of the spectra

    Returns:
        end_index : end index for the spectra.
    """
    compt = idx_start

    val_ini = (raw_res['D'][compt], raw_res['e'][compt])

    val_final = (raw_res['D'][compt+1], raw_res['e'][compt+1])

    if verbose :
        print('idx start : {}'.format(idx_start))
        print('value start : {}'.format(raw_res[key_to_scan][idx_start]))


    while val_ini == val_final and compt < np.size(raw_res[key_to_scan])-2 :
        compt +=1
        val_final = (raw_res['D'][compt+1], raw_res['e'][compt+1])
        val_ini = (raw_res['D'][compt], raw_res['e'][compt])


    if compt == np.size(raw_res[key_to_scan])-2 :
        end_index = compt+1

    else :
        end_index=compt
    if verbose :
        print('end index : {}'.format(end_index))
        print('value end : {}'.format(raw_res[key_to_scan][end_index]))
    

    return end_index

def parse_number(s):
    if pd.isna(s):  # gestion des NaN éventuels
        return np.nan
    s = str(s).strip()
    s = s.replace("i", "j")  # Python utilise j au lieu de i
    try:
        return float(s)   # si c’est un réel
    except ValueError:
        return complex(s) # sinon, c’est un complexe


def get_raw_size_sweep_data_Nathan(dir_root, Fname_list, verbose=False) :
    """The document is an excel table with COMSOL simulation iterationg on both D, e and lambda. All spectra must be completely calculated.
    
    Returns :
    res (dic) : a dictionnary, each key correspond to the simulated spectra of 1 nanodisk (in forme of a dic). 
    the key for each spectra dic are D, h, AR, Volume, Lambda, freq,E, Sigma_ext, Sigma_abs, Sigma_scat"""

    os.chdir(dir_root)

    res = {}

    for fname in Fname_list :
            
        #df = pd.read_csv(fname + '.csv', dtype='str', skiprows=4, converters={0: parse_number})
        df = pd.read_csv(fname + '.csv', skiprows=4, converters={0: parse_number})
        df = df.map(parse_number)
        data = df.to_numpy()
        #data = df.__array__()

        # Get the whole data
        res_raw = {}
        
        res_raw = {}


        
        res_raw['D'] = np.real(data[:,0])*1e9 # in nm
        res_raw['e'] = np.real(data[:,1])*1e9 # in nm

        res_raw['eigen_freq'] = data[:,2]*1e-9 # in GHz
        res_raw['Q'] = np.real(data[:,3]) # projection on dilatation mode (in percentage)


        compt = 0


        key_to_scan = 'eigen_freq'


        # print(res_raw['D'])
        # print(res_raw['h'])
        # print(res_raw['eigen_freq'])
        # print(res_raw['projection'])

        print('Total size of raw data : {}'.format(np.size(res_raw[key_to_scan])))
        while compt <  np.size(res_raw['D']) :
            
            
            if verbose :
                print('----------- Find spectra size -----------')
            start_index = compt

            end_index = scan_single_spectra(compt, res_raw, key_to_scan, verbose = verbose)

            compt= end_index + 1
            key = 'ND_D_' + str(np.round(res_raw['D'][start_index], 1)) + '_e_' + str(np.round(res_raw['e'][start_index], 1))


            res[key] = {}
            res[key]['D'] = np.round(res_raw['D'][start_index], 3)
            res[key]['e'] = np.round(res_raw['e'][start_index], 3)
            res[key]['AR'] = np.round(res[key]['D'] / res[key]['e'], 3)
            res[key]['eigen_freq'] = res_raw['eigen_freq'][start_index:end_index+1]
            res[key]['freq'] = np.real(res_raw['eigen_freq'][start_index:end_index+1])
            res[key]['tau'] = 1/(np.imag(res_raw['eigen_freq'][start_index:end_index+1])*2*np.pi)
            res[key]['Q'] = res[key]['tau'] * res[key]['freq']*np.pi

    return res



def get_raw_size_sweep_data_Nathan_with_dV(dir_root, Fname_list, verbose=False) :
    """The document is an excel table with COMSOL simulation iterationg on both D, e and lambda. All spectra must be completely calculated.
    
    Returns :
    res (dic) : a dictionnary, each key correspond to the simulated spectra of 1 nanodisk (in forme of a dic). 
    the key for each spectra dic are D, h, AR, Volume, Lambda, freq,E, Sigma_ext, Sigma_abs, Sigma_scat"""

    os.chdir(dir_root)

    res = {}

    for fname in Fname_list :
            
        #df = pd.read_csv(fname + '.csv', dtype='str', skiprows=4, converters={0: parse_number})
        df = pd.read_csv(fname + '.csv', dtype='str', skiprows=4, converters={0: parse_number})
        df = df.applymap(parse_number)
        data = df.to_numpy()
        #data = df.__array__()

        # Get the whole data
        res_raw = {}
        
        res_raw = {}


        
        res_raw['D'] = np.real(data[:,0])*1e9 # in nm
        res_raw['e'] = np.real(data[:,1])*1e9 # in nm

        res_raw['eigen_freq'] = data[:,2]*1e-9 # in GHz
        res_raw['Q'] = np.real(data[:,3]) # projection on dilatation mode (in percentage)

        res_raw['Delta_V'] = np.abs(data[:,4]) #Volume change
        res_raw['V'] = np.real(data[:,5]) # Volume
        compt = 0


        key_to_scan = 'eigen_freq'


        # print(res_raw['D'])
        # print(res_raw['h'])
        # print(res_raw['eigen_freq'])
        # print(res_raw['projection'])

        print('Total size of raw data : {}'.format(np.size(res_raw[key_to_scan])))
        while compt <  np.size(res_raw['D']) :
            
            
            if verbose :
                print('----------- Find spectra size -----------')
            start_index = compt

            end_index = scan_single_spectra(compt, res_raw, key_to_scan, verbose = verbose)

            compt= end_index + 1
            key = 'ND_D_' + str(np.round(res_raw['D'][start_index], 1)) + '_e_' + str(np.round(res_raw['e'][start_index], 1))


            res[key] = {}
            res[key]['D'] = np.round(res_raw['D'][start_index], 3)
            res[key]['e'] = np.round(res_raw['e'][start_index], 3)
            res[key]['AR'] = np.round(res[key]['D'] / res[key]['e'], 3)
            res[key]['eigen_freq'] = res_raw['eigen_freq'][start_index:end_index+1]
            res[key]['freq'] = np.real(res_raw['eigen_freq'][start_index:end_index+1])
            res[key]['tau'] = 1/(np.imag(res_raw['eigen_freq'][start_index:end_index+1])*2*np.pi)
            res[key]['Q'] = res[key]['tau'] * res[key]['freq']*np.pi
            res[key]['dV'] = res_raw['Delta_V'][start_index:end_index+1]
            res[key]['V'] = res_raw['V'][start_index:end_index+1]
    return res





def get_modes_with_expansion_projection(Spectra_dic, threshold = 0.05, height=0.1) :
    """find the mode frequency and position for each spectra

    Args:
        Spectra_dic (_type_): _description_
    """

    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        peaks, peaks_dic = scipy.signal.find_peaks(spectra['projection'], height=height, threshold=threshold)

        spectra['modes'] = {}
        spectra['modes']['f'] = np.real(spectra['eigen_freq'][peaks])
        spectra['modes']['tau'] = np.imag(spectra['eigen_freq'][peaks])
        spectra['modes']['projection'] = spectra['projection'][peaks]
    pass


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
        spectra['modes']['tau']  = spectra['tau'][indexes]
        spectra['modes']['Q'] = spectra['Q'][indexes]

    pass


def get_modes_with_Q_selection_with_dV(Spectra_dic, Q_th=20) :
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
        spectra['modes']['tau']  = spectra['tau'][indexes]
        spectra['modes']['Q'] = spectra['Q'][indexes]
        spectra['modes']['dV'] = spectra['dV'][indexes]
        spectra['modes']['V'] = spectra['V'][indexes]
    pass




def get_max_Q(Spectra_dic, freq_low_func, freq_high_func, AR_min=2, AR_max=5, n_modes=2 ) :

    AR = []
    Q_list = []
    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        if spectra['AR']>= AR_min and spectra['AR']<=AR_max :


            
            idx_freq_lim = np.where(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            
            if np.size(idx_freq_lim)>0 :
                AR.append(spectra['AR'])
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

        if spectra['AR']>= AR_min and spectra['AR']<=AR_max :

            idx_freq_lim = np.where(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            

            for idx in idx_freq_lim[0] :
                
                AR.append(spectra['AR'])

                freq_list.append(spectra['modes']['f'][idx])
                Q_list.append(spectra['modes']['Q'][idx])


    
    AR = np.array(AR)
    Q_array = np.array(Q_list)
    freq_array = np.array(freq_list)

    return(AR, freq_array, Q_array)




def get_Q_specific_bands_v2(Spectra_dic, freq_low_func, freq_high_func, Mode_dic, AR_min=2, AR_max=5 ) :

    AR_crossing = Mode_dic['AR_crossing']
    Crossing = Mode_dic['Crossing']

    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        spectra['specific_modes'] = {}
        if spectra['AR']>= AR_min and spectra['AR']<=AR_max :

            good_freq = np.where(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            idx_freq = np.argwhere(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            

            idx_ord_freq = np.argsort(good_freq)[0]
           
            if Crossing and (spectra['AR'] < AR_crossing) : #Order depend of which side of the crossing we are
                idx_ord_freq = idx_ord_freq[::-1] #then we fli the order of the indexes

            compt=0
            for index in idx_ord_freq :

                compt +=1
                spectra['specific_modes'][str(compt)] = {}
                spectra['specific_modes'][str(compt)]['f'] = spectra['modes']['f'][idx_freq][index][0]
                spectra['specific_modes'][str(compt)]['Q'] = spectra['modes']['Q'][idx_freq][index][0]
    print(spectra['specific modes'])

    pass



def get_Q_specific_bands_v2_with_dV(Spectra_dic, freq_low_func, freq_high_func, Mode_dic, AR_min=2, AR_max=5 ) :

    AR_crossing = Mode_dic['AR_crossing']
    Crossing = Mode_dic['Crossing']

    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        spectra['specific_modes'] = {}
        if spectra['AR']>= AR_min and spectra['AR']<=AR_max :

            good_freq = np.where(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            idx_freq = np.argwhere(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            

            idx_ord_freq = np.argsort(good_freq)[0]
           
            if Crossing and (spectra['AR'] < AR_crossing) : #Order depend of which side of the crossing we are
                idx_ord_freq = idx_ord_freq[::-1] #then we fli the order of the indexes

            compt=0
            for index in idx_ord_freq :

                compt +=1
                spectra['specific_modes'][str(compt)] = {}
                spectra['specific_modes'][str(compt)]['f'] = spectra['modes']['f'][idx_freq][index][0]
                spectra['specific_modes'][str(compt)]['Q'] = spectra['modes']['Q'][idx_freq][index][0]
                spectra['specific_modes'][str(compt)]['dV'] = spectra['modes']['dV'][idx_freq][index][0]
                spectra['specific_modes'][str(compt)]['V'] = spectra['modes']['V'][idx_freq][index][0]

    pass

def get_Q_specific_bands_with_dV(Spectra_dic, freq_low_func, freq_high_func, AR_min=2, AR_max=5 ) :

    AR = []
    Q_list = []
    freq_list = []

    dV_list = []
    V_list = []

    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        if spectra['AR']>= AR_min and spectra['AR']<=AR_max :

            idx_freq_lim = np.where(((spectra['modes']['f']<=freq_high_func(spectra['AR'])) & (spectra['modes']['f']>=freq_low_func(spectra['AR']))))
            

            for idx in idx_freq_lim[0] :
                
                AR.append(spectra['AR'])

                freq_list.append(spectra['modes']['f'][idx])
                Q_list.append(spectra['modes']['Q'][idx])
                dV_list.append(spectra['modes']['dV'][idx])
                V_list.append(spectra['modes']['V'][idx])

    
    AR = np.array(AR)
    Q_array = np.array(Q_list)
    freq_array = np.array(freq_list)
    dV = np.array(dV_list)
    V = np.array(V_list)
    return(AR, freq_array, Q_array, dV, V)




def save_Q_with_AR_to_csv(dir_to_save, fname, Q, AR, freq) :

    if not(os.path.exists(dir_to_save)) :
        print(1)
        os.mkdir(dir_to_save)
    os.chdir(dir_to_save)
    df = pd.DataFrame({'AR' : AR, \
                       'freq' : freq,\
                        'Q' : Q,\
                        })
    df.to_csv(fname + '.csv', index=False)


    pass



def save_Q_with_AR_to_csv_v2(dir_to_save, fname, spectra) :

    AR_1 = []
    freq_1 = []
    Q_1 = []
    dV_1 = []
    V_1 = []


    AR_2 = []
    freq_2 = []
    Q_2 = []
    dV_2 = []
    V_2 = []

    for key in spectra :

        try :
            mode1 = spectra[key]['specific_modes']['1']
            AR_1.append(spectra[key]['AR'])
            freq_1.append(mode1['f'])
            Q_1.append(mode1['Q'])
            dV_1.append(mode1['dV'])
            V_1.append(mode1['V'])

        except :
            pass

        try :
            mode2 = spectra[key]['specific_modes']['2']
            AR_2.append(spectra[key]['AR'])
            freq_2.append(mode2['f'])
            Q_2.append(mode2['Q'])
            dV_2.append(mode2['dV'])
            V_2.append(mode2['V'])

        except :
            pass

    if not(os.path.exists(dir_to_save)) :
        print(1)
        os.mkdir(dir_to_save)
    os.chdir(dir_to_save)

    df = pd.DataFrame({'AR_1' : pd.Series(np.array(AR_1)), \
                       'freq_1' : pd.Series(np.array(freq_1)),\
                        'Q_1' : pd.Series(np.array(Q_1)),\
                        'dV_1' : pd.Series(np.array(dV_1)), \
                        'V_1' : pd.Series(np.array(V_1)), \
                        
                        'AR_2' : pd.Series(np.array(AR_2)), \
                       'freq_2' : pd.Series(np.array(freq_2)),\
                        'Q_2' : pd.Series(np.array(Q_2)),\
                        'dV_2' : pd.Series(np.array(dV_2)), \
                        'V_2' :pd.Series(np.array(V_2))
                        })
    df.to_csv(fname + '.csv', index=False)


    pass





def save_freq_with_AR_to_csv(dir_to_save, fname, Spectra_dic) :

    if not(os.path.exists(dir_to_save)) :
        os.mkdir(dir_to_save)
    os.chdir(dir_to_save)

    AR = []
    freq = []
    Q = []


    for key in Spectra_dic :
        spectra = Spectra_dic[key]

        for i in range(np.size(spectra['modes']['f'])) :
            AR.append(spectra['AR'])
            freq.append(spectra['modes']['f'][i])
            Q.append(spectra['modes']['Q'][i])
    
    AR = np.array(AR)
    freq = np.array(freq)
    Q = np.array(Q)

    df = pd.DataFrame({'AR' : AR, \
                       'freq' : freq, \
                        'Q' : Q,\
                        })
    df.to_csv(fname + '.csv', index=False)

    
    pass




def get_acoustic_simu_data_and_interpolate(root_simu_acoustic, fname) :
    """open the csv with the data

    Args:
        root_simu_acoustic (_type_): _description_
        fname (_type_): _description_
    """

    os.chdir(root_simu_acoustic)

    data = np.genfromtxt(fname + '.csv', delimiter=',', skip_header=1)

    AR, freq, Q = data[:,0], data[:,1], data[:,2]
    return(AR, freq, Q)