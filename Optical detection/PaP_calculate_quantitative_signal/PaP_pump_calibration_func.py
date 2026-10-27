import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)



def get_treated_SMS_maps(dir_root) :
    
    
    Data = {}
    dir_treated_data = dir_root + r'\Treated_SMS'


    try :
        os.chdir(dir_treated_data)
    except :
        return("Error : there is no treated data")


    root, dirs, files = list(os.walk(dir_treated_data))[0]

    for name in files:
        if name.endswith('.pkl'):
            with open(name, 'rb') as file:
                sms = pickle.load(file)
                Data[sms.fname] = sms

    return Data
        





def create_fwhm_dataset(source, Data) :
    """Create directories of FWHM and amplitude estimation on all data for measures done with e specific source (OPO, SHG, Chameleon)

    Args:
        source (str): identify the source. Will be compared to sms.source.
        Data (dic): directory of all sms maps.

    Returns:
        FWHM_1D (dic) : dic of 1D fit. Keys are wl.  Each entry is a list of measured FWHM for that wl
        FWHM_2D (dic) : dic of 2D fit. Keys are wl. Each entry is a list of measured FWHM for that wl
        FWHM_peak (dic) : dic of peak fit. Keys are wl. Each entry is a list of measured FWHM for that wl
        amp_1D (dic) : dic of 1D fit. Keys are wl. Each entry is a list of measured amp for that wl
        amp_2D (dic) : dic of 2D fit. Keys are wl.  Each entry is a list of measured amp for that wl
    """

    FWHM_2D = {}
    FWHM_1D = {}
    FWHM_err_1D = {}
    FWHM_err_2D = {}
    FWHM_peak = {}
    amp_1D = {}
    amp_2D = {}


    for key in Data :
        sms = Data[key]

        if sms.source == source :
            if str(sms.wl) in FWHM_2D :
                FWHM_2D[str(sms.wl)].append(sms.FWHM_2D)
                FWHM_err_2D[str(sms.wl)].append(sms.FWHM_err_2D)
                amp_2D[str(sms.wl)].append(sms.amp_2D)
                
                FWHM_1D[str(sms.wl)].append(sms.FWHM_1D)
                FWHM_err_1D[str(sms.wl)].append(sms.std_wid)
                amp_1D[str(sms.wl)].append(sms.amp_1D)
                FWHM_peak[str(sms.wl)].append(sms.FWHM_peak)


            else :
                FWHM_2D[str(sms.wl)] = [sms.FWHM_2D]
                FWHM_err_2D[str(sms.wl)] = [sms.FWHM_err_2D]
                amp_2D[str(sms.wl)] = [sms.amp_2D]
                FWHM_1D[str(sms.wl)] = [sms.FWHM_1D]
                FWHM_err_1D[str(sms.wl)] = [sms.std_wid]
                amp_1D[str(sms.wl)] = [sms.amp_1D]
                FWHM_peak[str(sms.wl)] = [sms.FWHM_peak]

    return FWHM_2D, FWHM_err_2D, FWHM_1D, FWHM_err_1D, FWHM_peak, amp_1D, amp_2D







def mean_on_multiple_data(Data_dic) :
    """Make a mean on a Data dic (fwhm, amplitude...) for each wavelength involved, when there are multiple measurements.

    Args:
        Data_dic (dic): dic of FWHM. keys are wl. Each entry is a list of data for that wl.

    Return:
        Wl : list of wl in dic
        Data_mean (list): list of mean, ordered
    """
    WL = []
    Data_mean = []

    for key in Data_dic :
        WL.append(float(key))
        Data_mean.append(np.mean(Data_dic[key]))

    try:
        WL_sorted, Data_mean_sorted = (list(t) for t in zip(*sorted(zip(WL, Data_mean))))
        
        return(WL_sorted, Data_mean_sorted)
    except :
        return([], [])



def save_treated_FWHM(dir_root, WL, FWHM_1D_mean, FWHM_2D_mean, FWHM_1D, FWHM_2D, FWHM_err_1D, FWHM_err_2D) :
    """Saves the measured FWHM on a sms dataset. A mean has been done when there are several maps in the same wavelength. Each mean is separated by source.

    Args:
        dir_root (str): root folder of the sms dataset
        WL (dictionnary): dictionnary of wavelength
        FWHM_1D_mean (dictionnary): dictionnary of 1D FWHM fit, averaged on same WL
        FWHM_2D_mean (dictionnary): dictionnary of 2D FWHM fit, averaged on same WL
    """
    dir_save_FWHM = dir_root + r'/Treated_FWHM'
    if not(os.path.exists(dir_save_FWHM)) :
        os.mkdir(dir_save_FWHM)
 
    os.chdir(dir_save_FWHM)
    
    header = 'Wavelength (nm), FWHM fit 1D (nm), FWHM fit 2D (nm) \n \
        ------------------------------ \n \
        -------------------------'
    

    for source_key in WL :
            #Save 1 txt for each source.

            wl = WL[source_key]
            n_wl = np.size(WL[source_key])

            res_1D = np.array(FWHM_1D_mean[source_key])
            res_2D = np.array(FWHM_2D_mean[source_key])

            
            Results = np.zeros(shape=(n_wl, 3))
            Results[:,0] = wl
            Results[:,1] = res_1D
            Results[:,2] = res_2D


            fname= source_key + '.dat'


            np.savetxt(fname, Results, header=header, delimiter=',', fmt='%d')
    
    save_object(WL, 'WL.pkl')
    save_object(FWHM_1D_mean, 'FWHM_1D_mean.pkl')
    save_object(FWHM_2D_mean, 'FWHM_2D_mean.pkl')
    save_object(FWHM_1D, 'FWHM_1D.pkl')
    save_object(FWHM_2D, 'FWHM_2D.pkl')
    save_object(FWHM_err_1D, 'FWHM_err_1D.pkl')
    save_object(FWHM_err_2D, 'FWHM_err_2D.pkl')



def incert_type_A(res) :
    """Compute the 95% interval of trust for a set of measurement res.

    Args:
        res (1d array): measured values
    
    Return:
        delta = the 95% trust interval. result = mean(res) +- delta
    """

    k_list = [15, 12.7, 4.3, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26] #prise en compte du nb de mesures
    n_mes = np.size(res)
    res_mean = np.mean(res)

    if n_mes > 1 :
        sigma = np.sqrt(np.sum((res-res_mean)**2)/(n_mes-1)) #écart type

        if n_mes > 8 :
            k = 2
        else :
            k = k_list[n_mes+1]

    delta = sigma/np.sqrt(n_mes)*k #incertitudes type

    return(delta)





def print_results(WL, FWHM, amp) :
    for key in WL :
        print('------------------ {} -----------------'.format(key))
        for i in range(len(WL[key])) :

            print('Lambda : {:.1f} nm, FWHM : {:.1f} nm, amp = {:.1f}e-3'.format(WL[key][i], FWHM[key][i]*1e3, amp[key][i]*1000))
        


def get_FWHM_data(dir_root, Source_list = ['OPO', 'SHG', 'CHA']) :
    """Extract all FWHM data from treated sms maps.

    Args:
        dir_root (str): root directory containing treated sms maps.

    Returns :
        FWHM_data (dir) : directory containint all relevant FWHM informations. Each entry is a directory divided by source, containing lists of corresponding informations.
    """
    Data = get_treated_SMS_maps(dir_root)


    FWHM_2D = {}
    FWHM_1D = {}
    FWHM_err_1D = {}
    FWHM_err_2D = {}
    FWHM_peak = {}
    amp_1D = {}
    amp_2D = {}

    for source_key in Source_list :

        FWHM_2D[source_key],FWHM_err_2D[source_key], FWHM_1D[source_key], FWHM_err_1D[source_key], FWHM_peak[source_key], amp_1D[source_key], amp_2D[source_key] = create_fwhm_dataset(source_key, Data)



    WL = {}
    FWHM_2D_mean = {}
    FWHM_1D_mean = {}
    FWHM_peak_mean = {}
    amp_1D_mean = {}
    amp_2D_mean = {}

    for source_key in Source_list :
        
        WL[source_key], FWHM_2D_mean[source_key] = mean_on_multiple_data(FWHM_2D[source_key])
        WL[source_key], FWHM_1D_mean[source_key] = mean_on_multiple_data(FWHM_1D[source_key])
        WL[source_key], FWHM_peak_mean[source_key] = mean_on_multiple_data(FWHM_peak[source_key])
        WL[source_key], amp_1D_mean[source_key] = mean_on_multiple_data(amp_1D[source_key])
        WL[source_key], amp_2D_mean[source_key] = mean_on_multiple_data(amp_2D[source_key])


    FWHM_data = {}

    FWHM_data['WL'] = WL
    FWHM_data['FWHM_1D'] = FWHM_1D
    FWHM_data['FWHM_err_1D'] = FWHM_err_1D
    FWHM_data['FWHM_1D_mean'] = FWHM_1D_mean
    FWHM_data['FWHM_2D'] = FWHM_2D
    FWHM_data['FWHM_err_2D'] = FWHM_err_2D
    FWHM_data['FWHM_2D_mean'] = FWHM_2D_mean
    FWHM_data['FWHM_peak'] = FWHM_peak
    FWHM_data['FWHM_peak_mean'] = FWHM_peak_mean
    FWHM_data['amp_2D_mean'] = amp_2D_mean
    FWHM_data['amp_2D'] = amp_2D
    FWHM_data['amp_1D_mean'] = amp_1D_mean
    FWHM_data['amp_1D'] = amp_1D



    return(FWHM_data)