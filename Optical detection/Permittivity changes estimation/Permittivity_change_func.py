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


def get_permittivity_tabulated_data(dir_root, fname) :
    """Get the permittivity from refractive index.info website

    Args:
        dir_root (str): root directory
        fname (str): filename

    """
    os.chdir(dir_root)
    f = pd.read_csv(fname)

    data = {}

    data['lambda'] = (f['wl'].__array__())*1e3
    data['eV'] = 1240/data['lambda']
    data['n'] = f['n'].__array__()
    data['k'] = f['k'].__array__()
    data['eps_1'] = data['n']**2 - data['k']**2
    data['eps_2'] = 2*data['k']*data['n']

    return data




def get_plasmonic_derivatives(dir_root, fname) :
    """Get the permittivity from refractive index.info website

    Args:
        dir_root (str): root directory
        fname (str): filename

    """
    os.chdir(dir_root)
    f = pd.read_csv(fname, header=0, skiprows=[1,2])

    data = {}

    data['lambda'] = (f['lda'].__array__())
    data['eV'] = 1240/data['lambda']
    data['dsigma/deps1'] = f['dsigma/deps1'].__array__()
    data['dsigma/deps2'] = f['dsigma/deps2'].__array__()
    
    return data

def save_treated_acoustic_sensitivity(Data, dir_to_save, fname) :
    """Here we write :
    \Delta \varepsilon_1(\lambda) = \left( a^D(\lambda) + a^{ib}(\lambda) \right) \left( \frac{dV}{V} \right)
    \Delta \varepsilon_2(\lambda) = \left( b^D(\lambda) + b^{ib}(\lambda) \right) \left( \frac{dV}{V} \right)

    Args:
        Data_PaP (_type_): _description_
        dir_to_save (_type_): _description_
        fname (_type_): _description_
    """
    
    os.chdir(dir_to_save)
    df = pd.DataFrame({'lambda (nm)' : Data['lambda'],\
                       'Energy (eV)' : Data['eV'],\
                       'a^D' : Data['a_D'],\
                       'b^D' : Data['b_D'],\
                       'a^ib' : Data['a_ib'],\
                       'b^ib' : Data['b_ib'],\

                        })
    df.to_csv(fname, index=False)
    pass




def get_eps2_ib(dir_root, fname) :
    """Get the permittivity from refractive index.info website

    Args:
        dir_root (str): root directory
        fname (str): filename

    """
    os.chdir(dir_root)
    f = pd.read_csv(fname, header=0, skiprows=[1,2])

    data = {}

    data['eV'] = (f['E'].__array__())
    data['Lambda'] = 1240/data['eV']
    data['eps2_Lps'] = f['eps2_Lps'].__array__()
    data['eps2_Ldp'] = f['eps2_Ldp'].__array__()
    data['eps2_Xdp'] = f['eps2_Xdp'].__array__()
    data['eps2_ib'] = data['eps2_Xdp'] + data['eps2_Ldp'] + data['eps2_Lps']

    data['eps2_Drude'] = f['eps2_drude'].__array__()
    
    return data





def save_treated_interband_contribution(Permittivity_variations, dir_to_save, fname) :
    """Here we write :
    \Delta \varepsilon_1(\lambda) = \left( a^D(\lambda) + a^{ib}(\lambda) \right) \left( \frac{dV}{V} \right)
    \Delta \varepsilon_2(\lambda) = \left( b^D(\lambda) + b^{ib}(\lambda) \right) \left( \frac{dV}{V} \right)

    Args:
        Data_PaP (_type_): _description_
        dir_to_save (_type_): _description_
        fname (_type_): _description_
    """
    
    Energy = Permittivity_variations['ib']['Energy']
    os.chdir(dir_to_save)
    df = pd.DataFrame({'lambda (nm)' : Permittivity_variations['ib']['lambda'],\
                       'Energy (eV)' : Permittivity_variations['ib']['Energy'],\
                       'deps2/dE - Ldp' : Permittivity_variations['ib']['deps2/deV']['Ldp'](Energy),\
                       'deps2/dE - Lps' : Permittivity_variations['ib']['deps2/deV']['Lps'](Energy),\
                       'deps2/dE - Xdp' : Permittivity_variations['ib']['deps2/deV']['Xdp'](Energy),\
                     
                       'deps1/dE - Ldp' : Permittivity_variations['ib']['deps1/deV']['Ldp'](Energy),\
                       'deps1/dE - Lps' : Permittivity_variations['ib']['deps1/deV']['Lps'](Energy),\
                       'deps1/dE - Xdp' : Permittivity_variations['ib']['deps1/deV']['Xdp'](Energy),\
  
                       'delta eps2 - Ldp' : Permittivity_variations['ib']['delta_eps2']['Ldp'](Energy),\
                       'delta eps2 - Lps' : Permittivity_variations['ib']['delta_eps2']['Lps'](Energy),\
                       'delta eps2 - Xdp' : Permittivity_variations['ib']['delta_eps2']['Xdp'](Energy),\
                       'delta eps2 - total' : Permittivity_variations['ib']['delta_eps2']['total'](Energy),\
                                              
                       'delta eps1 - Ldp' : Permittivity_variations['ib']['delta_eps1']['Ldp'](Energy),\
                       'delta eps1 - Lps' : Permittivity_variations['ib']['delta_eps1']['Lps'](Energy),\
                       'delta eps1 - Xdp' : Permittivity_variations['ib']['delta_eps1']['Xdp'](Energy),\
                       'delta eps1 - total' : Permittivity_variations['ib']['delta_eps1']['total'](Energy),\
                                              


                        })
    df.to_csv(fname, index=False)
    pass



def get_treated_interband_contribution(dir_root, fname) :
    
    os.chdir(dir_root)
    f = pd.read_csv(fname, header=0, skiprows=[1,2])

    Permittivity_variations = {}
    Permittivity_variations['deps2/deV'] = {}
    Permittivity_variations['deps1/deV'] = {}
    Permittivity_variations['delta_eps2'] = {}
    Permittivity_variations['delta_eps1'] = {}

    Permittivity_variations['lambda'] = f['Energy (eV)'].__array__()
    Permittivity_variations['Energy'] = f['Energy (eV)'].__array__()


    Permittivity_variations['deps2/deV']['Ldp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['deps2/dE - Ldp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['deps2/deV']['Lps'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['deps2/dE - Lps'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['deps2/deV']['Xdp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['deps2/dE - Xdp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['deps1/deV']['Ldp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['deps1/dE - Ldp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['deps1/deV']['Lps'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['deps1/dE - Lps'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['deps1/deV']['Xdp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['deps1/dE - Xdp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)

    Permittivity_variations['delta_eps2']['Ldp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps2 - Ldp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps2']['Lps'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps2 - Lps'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps2']['Xdp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps2 - Xdp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps1']['Ldp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps1 - Ldp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps1']['Lps'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps1 - Lps'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps1']['Xdp'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps1 - Xdp'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps1']['total'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps1 - total'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['delta_eps2']['total'] = sc.interpolate.interp1d( Permittivity_variations['Energy'], f['delta eps2 - total'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    return(Permittivity_variations)


def get_interband_contribution_Stoll_2014(dir_root) :
    
    os.chdir(dir_root)
    fname_eps1 = 'interband_permittivity_change_stoll_2014_eps1.csv'
    fname_eps2 = 'interband_permittivity_change_stoll_2014_eps2.csv'


    Data ={}
    Data['eps1'] ={}
    Data['eps2'] ={}

    os.chdir(dir_root)
    f = pd.read_csv(fname_eps1,sep=";", decimal=",")
    Data['eps1']['lambda'] = (f['lambda'].__array__().astype('float64'))*1e-9
    Data['eps1']['Delta_eps1'] = f['Delta eps1 ib (1K)'].__array__().astype('float64')


    os.chdir(dir_root)
    f2 = pd.read_csv(fname_eps2,sep=";", decimal=",")
    Data['eps2']['lambda'] = (f2['lambda'].__array__())*1e-9
    Data['eps2']['Delta_eps2'] = f2['Delta eps2 ib (1K)'].__array__()


    f_eps_1_ib = sc.interpolate.interp1d(1240/Data['eps1']['lambda']*1e-9,Data['eps1']['Delta_eps1'], fill_value=0, bounds_error=False)
    f_eps_2_ib = sc.interpolate.interp1d(1240/Data['eps2']['lambda']*1e-9,Data['eps2']['Delta_eps2'], fill_value=0, bounds_error=False)


    Permittivity_variations = {}
    Permittivity_variations['delta_eps2'] = {}
    Permittivity_variations['delta_eps1'] = {}
    Permittivity_variations['lambda'] = f2['lambda'].__array__()
    Permittivity_variations['Energy'] =  1240/(Permittivity_variations['lambda']*1e9)
    Permittivity_variations['delta_eps1']['total'] = f_eps_1_ib
    Permittivity_variations['delta_eps2']['total'] = f_eps_2_ib


    return(Permittivity_variations)



def get_deps_dT_Stoll_2014(dir_root) :
    
    os.chdir(dir_root)
    fname_eps1 = 'interband_permittivity_change_stoll_2014_eps1.csv'
    fname_eps2 = 'interband_permittivity_change_stoll_2014_eps2.csv'


    Data ={}
    Data['eps1'] ={}
    Data['eps2'] ={}

    os.chdir(dir_root)
    f = pd.read_csv(fname_eps1,sep=";", decimal=",")

    os.chdir(dir_root)
    f2 = pd.read_csv(fname_eps2,sep=";", decimal=",")

    f_eps_1_ib = sc.interpolate.interp1d(1240/(f['lambda'].__array__().astype('float64')),f['Delta eps1 tot (1K)'].__array__(), fill_value=0, bounds_error=False)
    f_eps_2_ib = sc.interpolate.interp1d(1240/(f2['lambda'].__array__().astype('float64')),f2['Delta eps2 tot (1K)'].__array__(), fill_value=0, bounds_error=False)

    Permittivity_variations = {}
    Permittivity_variations['lambda'] = f2['lambda'].__array__()
    Permittivity_variations['Energy'] =  1240/(Permittivity_variations['lambda']*1e9)
    Permittivity_variations['deps1/dT'] = f_eps_1_ib
    Permittivity_variations['deps2/dT'] = f_eps_2_ib


    return(Permittivity_variations)

def get_Cahill_data(dir_root, fname) :
    
    os.chdir(dir_root)
    f = pd.read_csv(fname, header=0, skiprows=[1,2])

    Permittivity_variations = {}
    Permittivity_variations['lambda'] = f['nm'].__array__()
    
    Permittivity_variations['deps1/dT'] = sc.interpolate.interp1d(1240/Permittivity_variations['lambda'], f['deps1/dT'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)
    Permittivity_variations['deps2/dT'] = sc.interpolate.interp1d(1240/Permittivity_variations['lambda'], f['deps2/dT'].__array__(), fill_value=0, kind='quadratic', bounds_error=False)

    return(Permittivity_variations)





def Delta_eps_e_density(w_p, gamma_0) :
    """Return the electron density contribution to the change of eps1,eps2 for a unitary change of volume dV/V.

    It returns a function that depends on energy in eV.
    """
    def eps_1_drude(w, gamma=gamma_0, w_p=w_p) :
        return(1 - (w_p**2)/(w**2+gamma**2))
    def eps_2_drude(w, gamma=gamma_0, w_p=w_p) :
        return((gamma*w_p**2)/(w*(w**2 + gamma**2)))

    # in rad/s
    def a_drude_rad(w) :
        return (1- eps_1_drude(w))
    def b_drude_rad(w) :
        return(-eps_2_drude(w))

    # in eV
    def a_drude_energy(energy) :
        return (1- eps_1_drude(energy* sc.constants.e / sc.constants.hbar  ))
    def b_drude_energy(energy) :
        return(-eps_2_drude(energy * sc.constants.e / sc.constants.hbar ))

    # in meters
    def a_drude_wavelength(lda) :
        return (1- eps_1_drude(2*np.pi*sc.constants.c /lda))
    def b_drude_wavelength(lda) :
        return(-eps_2_drude(2*np.pi*sc.constants.c /lda))

    return a_drude_energy, b_drude_energy


def Delta_eps_e_ph_scattering(w_p, gamma_0, dgamma_dT) :
    """Return the e-ph scattering rate change for a lattice temperature elevation of 1 K

    Args:
        w_p (_type_): _description_
        gamma_0 (_type_): _description_
        dgamma_dT (_type_): _description_
    """

    def eps_2_drude(w, gamma=gamma_0, w_p=w_p) :
        return((gamma*w_p**2)/(w*(w**2 + gamma**2)))

    def deps2_dT_scattering(energy) :
        return(eps_2_drude(energy * sc.constants.e / sc.constants.hbar, gamma=gamma_0 + dgamma_dT ) - eps_2_drude(energy * sc.constants.e / sc.constants.hbar, gamma=gamma_0 ))

    def deps1_dT_scattering(energy) :
        return(0*energy)

    return deps1_dT_scattering, deps2_dT_scattering
