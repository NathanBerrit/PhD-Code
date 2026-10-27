import os
import numpy as np
#import pickle
import scipy as sc
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib
from matplotlib.transforms import Bbox

import matplotlib.colors
import plotly.offline
import plotly.graph_objs as go
import pandas as pd



#Defined in eV
def eps_1_drude(E, gamma, w_p) :
    return(1 - (w_p**2)/(E**2+gamma**2))
def eps_2_drude(E, gamma, w_p) :
    return((gamma*w_p**2)/(E*(E**2 + gamma**2)))


def fermi_dist(E, E_f, T_e) :
    """Fermi occupation

    Args:
        E (_type_): in eV
        E_f (_type_): in eV
        T_e (_type_): in K

    Returns:
        _type_: _description_
    """
    return 1/(np.exp((E-E_f)*sc.constants.e/(sc.constants.k*T_e))+1)





def Jdp_Ag(w, T=300, n_energy=1000, **kwargs) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """

    params = {**kwargs}


    Ef = params['Ef'] #in eV

    m_p_perp = params['m_p_perp']
    m_p_par = params['m_p_par']
    m_d_perp = params['m_d_perp']
    m_d_par = params['m_d_par']

    #gaps in eV
    wf = params['wf'] 
    wo = params['wo']

    if np.size(w) ==1 :
        w = np.array([w])

    E_min = Ef - 5 #superior integration limit no need to go further than that thanks to the fermi-dirac term
    Emax_inf = (Ef - wf + (m_d_par/(m_d_par - m_p_par))*(w-wo)) #for hw<hwo
    Emax_sup = (Ef - wf + (m_d_perp/(m_d_perp + m_p_perp))*(w-wo)) # for hw>hwo

    Emax = (w < wo) * Emax_inf + (w >= wo) * Emax_sup
    

    inf_born = np.sqrt((Emax_sup - Emax)) #Inferior integration limit
    sup_born = np.sqrt((Emax_sup - E_min))


    Integral_array = inf_born[:, None] + (sup_born[:, None] - inf_born[:, None]) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    #print('---------------')
    #print(np.shape(Integral_array))
    #print(Integral_array)


    
    #Form factor
    Fdp = np.sqrt((m_p_perp*m_p_par*m_d_perp*m_d_par) / (m_p_par*m_d_perp + m_p_perp*m_d_par)) * sc.constants.m_e

    B = np.sqrt(m_p_perp*m_d_perp/(m_p_perp + m_d_perp) * sc.constants.m_e)
    C = 16*np.pi**2 * sc.constants.hbar**(5/2)

    chi = Fdp*B/C
  
    integrande = 2*chi*(1-fermi_dist(Emax_sup[:, None]-Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)



def Jps_Ag(w, T=300, n_energy=1000, **kwargs) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """
    params = {**kwargs}
    
    
    Ef = params['Ef'] #in eV

    m_p_perp = params['m_p_perp']
    m_p_par = params['m_p_par']
    m_s_perp = params['m_s_perp']
    m_s_par = params['m_s_par']

    #gaps in eV
    wf = params['wf'] 
    wg = params['wg']


    
    if np.size(w) ==1 :
        w = np.array([w])

    E_max = Ef + 5 #superior integration limit no need to go further than that thanks to the fermi-dirac term
    Emin_inf = (Ef - wf + (m_s_perp/(m_s_perp - m_p_perp))*(wg+wf-w)) #for hw<hwo
    Emin_sup = (Ef - wf + (m_s_par/(m_s_par + m_p_par))*(wg+wf-w)) # for hw>hwo

    Emin = (w < (wf+wg)) * Emin_inf + (w >= wf+wg) * Emin_sup
    #Emin =Emin_inf 
    

    inf_born = np.sqrt((Emin - Emin_inf)) #Inferior integration limit
    sup_born = np.sqrt((E_max - Emin_inf))


    Integral_array = inf_born[:, None] + (sup_born[:, None] - inf_born[:, None]) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    #print('---------------')
    #print(np.shape(Integral_array))
    #print(Integral_array)


    
    #Form factor
    Fps = np.sqrt((m_p_perp*m_p_par*m_s_perp*m_s_par) / (m_p_par*m_s_perp + m_p_perp*m_s_par)) * sc.constants.m_e

    B = np.sqrt(m_s_perp*m_p_perp/(m_s_perp - m_p_perp) * sc.constants.m_e)
    C = 16*np.pi**2 * sc.constants.hbar**(5/2)

  
    chi = Fps*B/C

    integrande = 2*chi*(fermi_dist(Emin_inf[:, None]+Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)





def create_fit_func_Ag(Ag_band_info, free_param_list) :

    
    def fit_func_total(E, **kwargs) :

        params = {**Ag_band_info, **kwargs}

        return(eps2_from_interband_Ag(E, **params))


    def func(E,*par):
        par_dict = {}
        for i in range(len(free_param_list)):
            par_dict[free_param_list[i]]=par[i]
        return fit_func_total(E,**par_dict) #only values of the free parameters are updated
    
    return func



def make_Ag_interband_fit(Ag_band_info, free_param_list, Xdata, Ydata) :
    """Make the fit of interband for Ag, leaving the parameters listed in free_param_list free

    Args:
        Ag_band_info (dic): Dictionnary of initial values for each interband parameters
        free_param_list (list of str): list of parameters to let vary to do the fit
        Pdp_ini (float): initial guess for Pdp
        Pps_ini (float): initial guess for Pps
        Xdata (1D array): reference data to fit
        Ydata (1D array): reference data to fit
    """
    
        
    func = create_fit_func_Ag(Ag_band_info, free_param_list)


    #create a dictionnary of initial value
    init_vals_dict ={**Ag_band_info}

    #Among this list, we select the free parameters to be fitted
    init_vals = [init_vals_dict[param] for param in free_param_list]


    print('------------Initial parameters ------------------')
    k=0
    for param in free_param_list :
        print('key : {} / value : {}'.format(param, init_vals[k]))
        k+=1

    print('------------------------------------------------')
    #We make the fit and save the errors
    best_vals, pcov = curve_fit(func, Xdata, Ydata, p0=init_vals)  

    #Now we save the results of the fit in the sms object
    sigma_err = np.sqrt(np.diag(pcov))
    chi_sq=((Ydata-func(Xdata,*best_vals))**2).sum()

    SStot=((Ydata-np.average(Ydata))**2).sum()
    R2 = 1-chi_sq/SStot


    Ag_info_fitted = {}

    for key in Ag_band_info :
        Ag_info_fitted[key] = Ag_band_info[key]

    fitted_vals_err = {}
    
    print('------------Fitted parameters ------------------')
    c=0
    for param in free_param_list :
        print('key : {} / value : {} / error : {}'.format(param, best_vals[c], sigma_err[c]))
        Ag_info_fitted[param] = best_vals[c]
        fitted_vals_err[param] = sigma_err[c]
        c+=1

    print('----------------------------------------------')

    return Ag_info_fitted, fitted_vals_err



def eps2_from_interband_Ag(E, **kwargs) :

    params = {**kwargs}


    Pdp = params['Pdp']
    Pps = params['Pps']

    prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

    Jps = Jps_Ag(E, **params)
    Jdp = Jdp_Ag(E, **params)

    return(prefactor*(Pdp**2 * Jdp + Pps**2 * Jps*2)/E**2)





############################# Au fitting ###################



def Jdp_L_Au(w, T=300, n_energy=1000, **kwargs) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """

    params = {**kwargs}


    Ef = params['Ef'] #in eV

    m_p_perp_L = params['m_p_perp_L']
    m_p_par_L = params['m_p_par_L']
    m_d_perp_L = params['m_d_perp_L']
    m_d_par_L = params['m_d_par_L']

    #gaps in eV
    wf_L = params['wf_L'] 
    wo_L = params['wo_L']


    if np.size(w) ==1 :
        w = np.array([w])

    E_min = Ef - 5 #superior integration limit no need to go further than that thanks to the fermi-dirac term
    Emax_inf = (Ef - wf_L + (m_d_par_L/(m_d_par_L - m_p_par_L))*(w-wo_L)) #for hw<hwo
    Emax_sup = (Ef - wf_L + (m_d_perp_L/(m_d_perp_L + m_p_perp_L))*(w-wo_L)) # for hw>hwo

    Emax = (w < wo_L) * Emax_inf + (w >= wo_L) * Emax_sup
    

    inf_born = np.sqrt((Emax_sup - Emax)) #Inferior integration limit
    sup_born = np.sqrt((Emax_sup - E_min))


    Integral_array = inf_born[:, None] + (sup_born[:, None] - inf_born[:, None]) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    #print('---------------')
    #print(np.shape(Integral_array))
    #print(Integral_array)


    
    #Form factor
    Fdp = np.sqrt((m_p_perp_L*m_p_par_L*m_d_perp_L*m_d_par_L) / (m_p_par_L*m_d_perp_L + m_p_perp_L*m_d_par_L)) * sc.constants.m_e

    B = np.sqrt(m_p_perp_L*m_d_perp_L/(m_p_perp_L + m_d_perp_L) * sc.constants.m_e)
    C = 16*np.pi**2 * sc.constants.hbar**(5/2)

    chi = Fdp*B/C
  
    integrande = 2*chi*(1-fermi_dist(Emax_sup[:, None]-Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)







def Jps_L_Au(w, T=300, n_energy=1000, **kwargs) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """
    params = {**kwargs}
    
    
    Ef = params['Ef'] #in eV

    m_p_perp_L = params['m_p_perp_L']
    m_p_par_L = params['m_p_par_L']
    m_s_perp_L = params['m_s_perp_L']
    m_s_par_L = params['m_s_par_L']

    #gaps in eV
    wf_L = params['wf_L'] 
    wg_L = params['wg_L']


    
    if np.size(w) ==1 :
        w = np.array([w])

    E_max = Ef + 5 #superior integration limit no need to go further than that thanks to the fermi-dirac term
    Emin_inf = (Ef - wf_L + (m_s_perp_L/(m_s_perp_L - m_p_perp_L))*(wg_L+wf_L-w)) #for hw<hwo
    Emin_sup = (Ef - wf_L + (m_s_par_L/(m_s_par_L + m_p_par_L))*(wg_L+wf_L-w)) # for hw>hwo

    Emin = (w < (wf_L+wg_L)) * Emin_inf + (w >= wf_L+wg_L) * Emin_sup
    #Emin =Emin_inf 
    

    inf_born = np.sqrt((Emin - Emin_inf)) #Inferior integration limit
    sup_born = np.sqrt((E_max - Emin_inf))


    Integral_array = inf_born[:, None] + (sup_born[:, None] - inf_born[:, None]) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    #print('---------------')
    #print(np.shape(Integral_array))
    #print(Integral_array)


    
    #Form factor
    Fps = np.sqrt((m_p_perp_L*m_p_par_L*m_s_perp_L*m_s_par_L) / (m_p_par_L*m_s_perp_L + m_p_perp_L*m_s_par_L)) * sc.constants.m_e

    B = np.sqrt(m_s_perp_L*m_p_perp_L/(m_s_perp_L - m_p_perp_L) * sc.constants.m_e)
    C = 16*np.pi**2 * sc.constants.hbar**(5/2)

  
    chi = Fps*B/C

    integrande = 2*chi*(fermi_dist(Emin_inf[:, None]+Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)





def Jdp_X_Au(w, T=300, n_energy=1000, **kwargs) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """

    params = {**kwargs}


    Ef = params['Ef'] #in eV

    m_p_perp_X = params['m_p_perp_X']
    m_p_par_X = params['m_p_par_X']
    m_d_perp_X = params['m_d_perp_X']
    m_d_par_X = params['m_d_par_X']

    #gaps in eV
    wf_X = params['wf_X'] 
    wg_X = params['wg_X']


    if np.size(w) ==1 :
        w = np.array([w])

    E_min = Ef - 5 #superior integration limit no need to go further than that thanks to the fermi-dirac term
    Emax_inf = (Ef + wg_X + (m_d_par_X/(m_d_par_X - m_p_par_X))*(w - wg_X - wf_X)) #for hw<hwo
    Emax_sup = (Ef + wg_X + (m_d_perp_X/(m_d_perp_X + m_p_perp_X))*(w - wg_X - wf_X)) # for hw>hwo

    Emax = (w < (wf_X+wg_X)) * Emax_inf + (w >= (wf_X+wg_X)) * Emax_sup
    


    

    inf_born = np.sqrt((Emax_sup - Emax)) #Inferior integration limit
    sup_born = np.sqrt((Emax_sup - E_min))





    Integral_array = inf_born[:, None] + (sup_born[:, None] - inf_born[:, None]) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    #print('---------------')
    #print(np.shape(Integral_array))
    #print(Integral_array)


    
    #Form factor
    Fdp = np.sqrt((m_p_perp_X*m_p_par_X*m_d_perp_X*m_d_par_X) / (m_p_par_X*m_d_perp_X + m_p_perp_X*m_d_par_X)) * sc.constants.m_e

    B = np.sqrt(m_p_perp_X*m_d_perp_X/(m_p_perp_X + m_d_perp_X) * sc.constants.m_e)
    C = 16*np.pi**2 * sc.constants.hbar**(5/2)




    chi = Fdp*B/C
  
    integrande = 2*chi*(1-fermi_dist(Emax_sup[:, None]-Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)





def create_fit_func_Au(Au_band_info, free_param_list) :

    
    def fit_func_total(E, **kwargs) :

        params = {**Au_band_info, **kwargs}

        return(eps2_from_interband_Au(E, **params))


    def func(E,*par):
        par_dict = {}
        for i in range(len(free_param_list)):
            par_dict[free_param_list[i]]=par[i]
        return fit_func_total(E,**par_dict) #only values of the free parameters are updated
    
    return func



def make_Au_interband_fit(Au_band_info, free_param_list, Xdata, Ydata) :
    """Make the fit of interband for Au, leaving the parameters listed in free_param_list free

    Args:
        Ag_band_info (dic): Dictionnary of initial values for each interband parameters
        free_param_list (list of str): list of parameters to let vary to do the fit
        Pdp_ini (float): initial guess for Pdp
        Pps_ini (float): initial guess for Pps
        Xdata (1D array): reference data to fit
        Ydata (1D array): reference data to fit
    """
    
        
    func = create_fit_func_Au(Au_band_info, free_param_list)


    #create a dictionnary of initial value
    init_vals_dict ={**Au_band_info}

    #Among this list, we select the free parameters to be fitted
    init_vals = [init_vals_dict[param] for param in free_param_list]


    print('------------Initial parameters ------------------')
    k=0
    for param in free_param_list :
        print('key : {} / value : {}'.format(param, init_vals[k]))
        k+=1

    print('------------------------------------------------')
    #We make the fit and save the errors
    best_vals, pcov = curve_fit(func, Xdata, Ydata, p0=init_vals)  

    #Now we save the results of the fit in the sms object
    sigma_err = np.sqrt(np.diag(pcov))
    chi_sq=((Ydata-func(Xdata,*best_vals))**2).sum()

    SStot=((Ydata-np.average(Ydata))**2).sum()
    R2 = 1-chi_sq/SStot


    Au_info_fitted = {}

    for key in Au_band_info :
        Au_info_fitted[key] = Au_band_info[key]

    fitted_vals_err = {}
    
    print('------------Fitted parameters ------------------')
    c=0
    for param in free_param_list :
        print('key : {} / value : {} / error : {}'.format(param, best_vals[c], sigma_err[c]))
        Au_info_fitted[param] = best_vals[c]
        fitted_vals_err[param] = sigma_err[c]
        c+=1

    print('----------------------------------------------')

    return Au_info_fitted, fitted_vals_err

def eps2_from_interband_Au(E, **kwargs) :

    params = {**kwargs}


    Pdp_L = params['Pdp_L']
    Pps_L = params['Pps_L']
    Pdp_X = params['Pdp_X']

    prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

    Jps_L = Jps_L_Au(E, **params)
    Jdp_L = Jdp_L_Au(E, **params)
    Jdp_X = Jdp_X_Au(E, **params)

    return(prefactor*(Pdp_L**2 * Jdp_L + Pps_L**2 * Jps_L*2 + Pdp_X**2 * Jdp_X)/E**2)

def eps2_Jdp_L_interband_Au(E, **kwargs) :

    params = {**kwargs}


    Pdp_L = params['Pdp_L']
    Pps_L = params['Pps_L']
    Pdp_X = params['Pdp_X']

    prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

    Jps_L = Jps_L_Au(E, **params)
    Jdp_L = Jdp_L_Au(E, **params)
    Jdp_X = Jdp_X_Au(E, **params)

    return(prefactor*(Pdp_L**2 * Jdp_L)/E**2)

def eps2_Jps_L_interband_Au(E, **kwargs) :

    params = {**kwargs}


    Pdp_L = params['Pdp_L']
    Pps_L = params['Pps_L']
    Pdp_X = params['Pdp_X']

    prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

    Jps_L = Jps_L_Au(E, **params)
    Jdp_L = Jdp_L_Au(E, **params)
    Jdp_X = Jdp_X_Au(E, **params)

    return(prefactor*(Pps_L**2 * Jps_L*2)/E**2)

def eps2_Jdp_X_interband_Au(E, **kwargs) :

    params = {**kwargs}


    Pdp_L = params['Pdp_L']
    Pps_L = params['Pps_L']
    Pdp_X = params['Pdp_X']

    prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

    Jps_L = Jps_L_Au(E, **params)
    Jdp_L = Jdp_L_Au(E, **params)
    Jdp_X = Jdp_X_Au(E, **params)

    return(prefactor*(Pdp_X**2 * Jdp_X)/E**2)



def save_eps2_fitted(dir_to_save, fname, Fit_dic_info) :

    try :
        os.chdir(dir_to_save)
    except :
        os.mkdir(dir_to_save)
        os.chdir(dir_to_save)


    E = np.linspace(1, 5, 2000)
    data = {}
    data['E'] = E
    data['eps2_Ldp'] = eps2_Jdp_L_interband_Au(E, **Fit_dic_info)
    data['eps2_Lps'] = eps2_Jps_L_interband_Au(E, **Fit_dic_info)
    data['eps2_Xdp'] = eps2_Jdp_X_interband_Au(E, **Fit_dic_info)
    data['eps2ib'] = eps2_from_interband_Au(E, **Fit_dic_info)
    data['eps2_drude'] = eps_2_drude(E, Fit_dic_info['gamma0'], Fit_dic_info['wp']) 
    data['eps_2_tot'] = data['eps2ib'] + data['eps2_drude']
    

    df = pd.DataFrame.from_dict(data)
    df.to_csv(fname, index=False)


def save_fit_parameters(dir_to_save, fname, Fit_dic_info) :

    try :
        os.chdir(dir_to_save)
    except :
        os.mkdir(dir_to_save)
        os.chdir(dir_to_save)


    data = {}
    for key in Fit_dic_info :
        print(key, Fit_dic_info[key])
        data[key] = [Fit_dic_info[key]] #Need to make it a list for Dataframe.from_dic to work
    df = pd.DataFrame.from_dict(data)
    df.to_csv(fname, index=False)