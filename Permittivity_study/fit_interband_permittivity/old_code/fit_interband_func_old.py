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


def Ddp_Ag(E, w, Ag_band_info) :
    """Compute the EDJOS of Ag from initial energy E to final E - hw for the d -> p transition

    Args:
        E (float array): initial Energy in eV
        hbar_w (float): energy gain in eV
        Ag_band_info (dic): dictionnary containing all band information for Silver
    """

    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_d_perp = Ag_band_info['m_d_perp']
    m_d_par = Ag_band_info['m_d_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wo = Ag_band_info['wo']
    
    Ef = Ag_band_info['Ef'] #in eV
    
    #Form factor
    Fdp = np.sqrt((m_p_perp*m_p_par*m_d_perp*m_d_par) / (m_p_par*m_d_perp + m_p_perp*m_d_par)) * sc.constants.m_e


    A =  ((w-wo-wf - (E-Ef))/m_p_perp) * sc.constants.e / sc.constants.m_e
    B = ((E-Ef) + wf)/m_d_perp * sc.constants.e / sc.constants.m_e
    C = 16*np.pi**2 * sc.constants.hbar**3

  
    res = Fdp/(C*np.sqrt(A-B))
    
  

    return res




def Jdp_Ag(w, Ag_band_info, T=300, n_energy=1000) :
    """Computes the JDOS of the d->p transition of Ag.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """


    Ef = Ag_band_info['Ef'] #in eV
    Emin = Ef - 5

    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_d_perp = Ag_band_info['m_d_perp']
    m_d_par = Ag_band_info['m_d_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wo = Ag_band_info['wo']

    
    if np.size(w) ==1 :
        w = np.array([w])


    Emax = (w < wo) * (Ef - wf + (m_d_par/(m_d_par - m_p_par))*(w-wo)) + (w >= wo) * (Ef - wf + (m_d_perp/(m_d_perp + m_p_perp))*(w-wo))
    
    Emax = Emax-0.01

    E_2D = Emin + (Emax[:, None] - Emin) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    
    w_2D = np.outer(w,np.ones(n_energy))

    
    Ddp = Ddp_Ag(E_2D, w_2D, Ag_band_info)

    #print(Ddp)
    #print('Ddp : {}'.format(Ddp))
    
    fermi_occup = (1- fermi_dist(E_2D, Ef, T))
    #print('fermi_occupation : {}'.format(fermi_occup))

    integrande = Ddp * fermi_occup
    #print('Integrande : {}'.format(integrande))
    

    res = sc.integrate.trapezoid(integrande, x=E_2D, axis=1)*sc.constants.e
    return res





def Jdp_Ag_v2(w, Ag_band_info, T=300, n_energy=1000) :
    """Computes the JDOS of the d->p transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """


    Ef = Ag_band_info['Ef'] #in eV
    

    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_d_perp = Ag_band_info['m_d_perp']
    m_d_par = Ag_band_info['m_d_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wo = Ag_band_info['wo']
    
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
    C = 16*np.pi**2 * sc.constants.hbar**3

  
    chi = Fdp*B/C
  

    
    integrande = 2*chi*(1-fermi_dist(Emax_sup[:, None]-Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)


def Dps_Ag(E, w, Ag_band_info) :
    """Compute the EDJOS of Ag from initial energy E to final E - hw for the d -> p transition

    Args:
        E (float array): initial Energy in eV
        hbar_w (float): energy gain in eV
        Ag_band_info (dic): dictionnary containing all band information for Silver
    """

    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_s_perp = Ag_band_info['m_s_perp']
    m_s_par = Ag_band_info['m_s_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wg = Ag_band_info['wg']
    
    Ef = Ag_band_info['Ef'] #in eV
    
    #Form factor
    Fps = np.sqrt((m_p_perp*m_p_par*m_s_perp*m_s_par) / (m_p_par*m_s_perp + m_p_perp*m_s_par)) * sc.constants.m_e
    A = ((E-Ef) + w - wg)/m_p_perp
    B = ((E-Ef) + wf)  / m_s_perp
    C = 16*np.pi**2 * sc.constants.hbar**3


  
    res = Fps/(C*np.sqrt(A-B))
    
  

    return res



def Jps_Ag(w, Ag_band_info, T=300, n_energy=1000) :
    """Computes the JDOS of the d->p transition of Ag.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """


    Ef = Ag_band_info['Ef'] #in eV
    Emin = Ef - 5

    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_s_perp = Ag_band_info['m_s_perp']
    m_s_par = Ag_band_info['m_s_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wg = Ag_band_info['wg']

    
    if np.size(w) ==1 :
        w = np.array([w])


    Emin_inf = (Ef - wf + (m_s_perp/(m_s_perp - m_p_perp))*(wg+wf-w)) #for hw<hwo
    Emin_sup = (Ef - wf + (m_s_par/(m_s_par + m_p_par))*(wg+wf-w)) # for hw>hwo

    Emin = (w < (wf+wg)) * Emin_inf + (w >= wf+wg) * Emin_sup
    
    Emin+=0.0001

    Emax = Ef+5

    E_2D = Emin[:, None] + (Emax - Emin[:, None]) * np.linspace(0, 1, n_energy) #Creates a 2D array, each line is a linspace between Emin and Emax

    
    w_2D = np.outer(w,np.ones(n_energy))

    
    Dps = Dps_Ag(E_2D, w_2D, Ag_band_info)

    #print(Ddp)
    #print('Ddp : {}'.format(Ddp))
    
    fermi_occup = fermi_dist(E_2D, Ef, T)
    #print('fermi_occupation : {}'.format(fermi_occup))

    integrande = Dps * fermi_occup
    #print('Integrande : {}'.format(integrande))
    

    res = sc.integrate.trapezoid(integrande, x=E_2D, axis=1)*sc.constants.e
    return res





def Jps_Ag_v2(w, Ag_band_info, T=300, n_energy=1000) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """


    Ef = Ag_band_info['Ef'] #in eV
    

    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_s_perp = Ag_band_info['m_s_perp']
    m_s_par = Ag_band_info['m_s_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wg = Ag_band_info['wg']

        
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
    C = 16*np.pi**2 * sc.constants.hbar**3

  
    chi = Fps*B/C

    integrande = 2*chi*(fermi_dist(Emin_inf[:, None]+Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)







def Jps_Ag_v3(w, wo, wg, wf, Ef,  m_p_perp, m_p_par, m_s_perp , m_s_par, m_d_perp , m_d_par, gamma0, wp, T=300, n_energy=1000) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """



    
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
    C = 16*np.pi**2 * sc.constants.hbar**3

  
    chi = Fps*B/C

    integrande = 2*chi*(fermi_dist(Emin_inf[:, None]+Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)





def Jdp_Ag_v3(w, wo, wg, wf, Ef,  m_p_perp, m_p_par, m_s_perp , m_s_par, m_d_perp , m_d_par, gamma0, wp, T=300, n_energy=1000) :
    """Computes the JDOS of the p->s transition of Ag, using a change of variable to avoid the pole in the integration.

    Args:
        w (float) : photon energy in eV
        Ag_band_info (_type_): _description_
        T (float):Temperature in Kelvin
    """


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
    C = 16*np.pi**2 * sc.constants.hbar**3

  
    chi = Fdp*B/C
  

    
    integrande = 2*chi*(1-fermi_dist(Emax_sup[:, None]-Integral_array**2, Ef, T_e=T))
    return sc.integrate.trapezoid(integrande, x=Integral_array, axis=1)*np.sqrt(sc.constants.e)





def create_fit_func_Ag(Ag_band_info, free_param_list, Pdp_ini=1, Pps_ini=1) :

    Ef = Ag_band_info['Ef'] #in eV
    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_s_perp = Ag_band_info['m_s_perp']
    m_s_par = Ag_band_info['m_s_par']
    m_d_perp = Ag_band_info['m_d_perp']
    m_d_par = Ag_band_info['m_d_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wg = Ag_band_info['wg']
    wo = Ag_band_info['wo']
    
    wp = Ag_band_info['wp']
    gamma0 = Ag_band_info['gamma0']
    

    def fit_func_total(E, Pdp = Pdp_ini, Pps=Pps_ini, wo=wo, wg=wg, wf=wf, Ef = Ef,  m_p_perp = m_p_perp, m_p_par = m_p_par, m_s_perp = m_s_perp, m_s_par = m_s_par, m_d_perp =  m_d_perp, m_d_par = m_d_par, wp=wp, gamma0=gamma0) :

        prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)
        
        Jps = Jps_Ag_v3(E, wo, wg, wf, Ef,  m_p_perp, m_p_par, m_s_perp , m_s_par, m_d_perp , m_d_par, gamma0, wp,)
        Jdp = Jdp_Ag_v3(E, wo, wg, wf, Ef,  m_p_perp, m_p_par, m_s_perp , m_s_par, m_d_perp , m_d_par, gamma0, wp,)

        return(prefactor*(Pdp**2 * Jdp + Pps**2 * Jps*2)/E**2)


    def func(E,*par):
        par_dict = {}
        for i in range(len(free_param_list)):
            par_dict[free_param_list[i]]=par[i]
        return fit_func_total(E,**par_dict) #only values of the free parameters are updated
    
    return func







def make_Ag_interband_fit(Ag_band_info, free_param_list, Pdp_ini, Pps_ini, Xdata, Ydata) :
    """Make the fit of interband for Ag, leaving the parameters listed in free_param_list free

    Args:
        Ag_band_info (dic): Dictionnary of initial values for each interband parameters
        free_param_list (list of str): list of parameters to let vary to do the fit
        Pdp_ini (float): initial guess for Pdp
        Pps_ini (float): initial guess for Pps
        Xdata (1D array): reference data to fit
        Ydata (1D array): reference data to fit
    """
    func = create_fit_func_Ag(Ag_band_info, free_param_list, Pdp_ini=Pdp_ini, Pps_ini=Pps_ini)


    #create a dictionnary of initial value
    init_vals_dict ={**Ag_band_info, 'Pdp' : Pdp_ini, 'Pps' : Pps_ini}

    #Among this list, we select the free parameters to be fitted
    init_vals = [init_vals_dict[param] for param in free_param_list]

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
    c=0
    for param in free_param_list :
        if param != 'Pps' and param != 'Pdp' :
            Ag_info_fitted[param] = best_vals[c]
            fitted_vals_err[param] = sigma_err[c]
        elif param == 'Pdp' :
            Pdp = best_vals[c]
        elif param == 'Pps' :
            Pps = best_vals[c]

        c+=1


    return Ag_info_fitted, fitted_vals_err, Pdp, Pps




# def eps2_from_interband(E, Pdp, Pps, **kwargs) :

#     params = {**kwargs}
#     prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

#     Jps = Jps_Ag_v3(E, **params)
#     Jdp = Jdp_Ag_v3(E, **params)

#     return(prefactor*(Pdp**2 * Jdp + Pps**2 * Jps*2)/E**2)




def eps2_from_interband(E, Pdp, Pps, Ag_band_info) :


    Ef = Ag_band_info['Ef'] #in eV
    m_p_perp = Ag_band_info['m_p_perp']
    m_p_par = Ag_band_info['m_p_par']
    m_s_perp = Ag_band_info['m_s_perp']
    m_s_par = Ag_band_info['m_s_par']
    m_d_perp = Ag_band_info['m_d_perp']
    m_d_par = Ag_band_info['m_d_par']

    #gaps in eV

    wf = Ag_band_info['wf'] 
    wg = Ag_band_info['wg']
    wo = Ag_band_info['wo']
    
    wp = Ag_band_info['wp']
    gamma0 = Ag_band_info['gamma0']
    
    prefactor = 8*np.pi**2 * sc.constants.hbar**2 / (sc.constants.epsilon_0*sc.constants.m_e**2)

    Jps = Jps_Ag_v3(E, wo, wg, wf, Ef,  m_p_perp, m_p_par, m_s_perp , m_s_par, m_d_perp , m_d_par, gamma0, wp,)
    Jdp = Jdp_Ag_v3(E, wo, wg, wf, Ef,  m_p_perp, m_p_par, m_s_perp , m_s_par, m_d_perp , m_d_par, gamma0, wp,)

    return(prefactor*(Pdp**2 * Jdp + Pps**2 * Jps*2)/E**2)
