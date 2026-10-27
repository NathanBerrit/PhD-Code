import importlib
import numpy as np
import scipy as sc
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
#Mes imports

#os.chdir(r'D:\Documents\Code\python\Modules')


#import Slider_plot
#importlib.reload(Slider_plot)
#from Slider_plot import *






def compute_heating_changes(Permittivity_change, Spectra_info, band_contribution = 'both', Ldp_coeff=-1.7e-4, Xdp_coeff = -3.2e-4, Lps_coeff=0, with_dilatation=False) :
    """Compute the change in extinction spectra for a 1K temperature increase"""

    alpha = 1.42e-5 #gold linear thermal expansion coeff
    Ds = {}
    Ds['Lambda'] = Spectra_info['lambda']

    deps1_e_density = Permittivity_change['Drude']['e density']['delta_eps1'](1240/Ds['Lambda']*1e-9)*3*alpha
    deps2_e_density = Permittivity_change['Drude']['e density']['delta_eps2'](1240/Ds['Lambda']*1e-9)*3*alpha

    deps_2_e_ph_scattering = Permittivity_change['Drude']['e-ph scattering']['delta_eps2'](1240/Ds['Lambda']*1e-9)

    deps1_interband_Ldp = -Permittivity_change['ib']['deps1/deV']['Ldp'](1240/Ds['Lambda']*1e-9) * Ldp_coeff
    deps1_interband_Lps = -Permittivity_change['ib']['deps1/deV']['Lps'](1240/Ds['Lambda']*1e-9) * Lps_coeff
    deps1_interband_Xdp = -Permittivity_change['ib']['deps1/deV']['Xdp'](1240/Ds['Lambda']*1e-9) * Xdp_coeff

    deps2_interband_Ldp = -Permittivity_change['ib']['deps2/deV']['Ldp'](1240/Ds['Lambda']*1e-9) * Ldp_coeff
    deps2_interband_Lps = -Permittivity_change['ib']['deps2/deV']['Lps'](1240/Ds['Lambda']*1e-9) * Lps_coeff
    deps2_interband_Xdp = -Permittivity_change['ib']['deps2/deV']['Xdp'](1240/Ds['Lambda']*1e-9) * Xdp_coeff

    if band_contribution == 'both' :
        deps1_interband = deps1_interband_Ldp + deps1_interband_Lps + deps1_interband_Xdp
        deps2_interband = deps2_interband_Ldp + deps2_interband_Lps + deps2_interband_Xdp

    if band_contribution == 'Ldp' :
        deps1_interband = deps1_interband_Ldp 
        deps2_interband = deps2_interband_Ldp 

    if band_contribution == 'Stoll' :
        deps1_interband = Permittivity_change['Stoll ib']['delta_eps1']['total'](1240/Ds['Lambda']*1e-9)
        deps2_interband = Permittivity_change['Stoll ib']['delta_eps2']['total'](1240/Ds['Lambda']*1e-9)
 
    if band_contribution == 'nothing' :
        deps1_interband = 0
        deps2_interband = 0
 
 


    deps1_tot = (deps1_e_density+ deps1_interband)
    deps2_tot = (deps2_e_density + deps2_interband + deps_2_e_ph_scattering)



    if with_dilatation :
        Ds['geometry'] = Spectra_info['dsigma_dilatation']
    else :
        Ds['geometry'] = np.zeros(np.size(Ds['Lambda']))
    Ds['eps1'] = Spectra_info['dsigma/deps1'] * deps1_tot
    Ds['eps2'] =  Spectra_info['dsigma/deps2'] * deps2_tot 
    Ds['tot'] = Ds['eps2'] + Ds['eps1'] + Ds['geometry']
    return Ds


def compute_heating_changes_Cahill(Permittivity_change, Spectra_info, with_dilatation=False) :
    """Compute the change in extinction spectra for a 1K temperature increase"""

    alpha = 1.42e-5 #gold linear thermal expansion coeff
    Ds = {}
    Ds['Lambda'] = Spectra_info['lambda']


    deps1 = Permittivity_change['Cahill']['deps1/dT'](1240/Ds['Lambda']*1e-9)
    deps2 = Permittivity_change['Cahill']['deps2/dT'](1240/Ds['Lambda']*1e-9)
 

    if with_dilatation :
        Ds['geometry'] = Spectra_info['dsigma_dilatation']
    else :
        Ds['geometry'] = np.zeros(np.size(Ds['Lambda']))


    Ds['eps1'] = Spectra_info['dsigma/deps1'] * (deps1)
    Ds['eps2'] =  Spectra_info['dsigma/deps2'] * (deps2)
    Ds['tot'] = Ds['eps2'] + Ds['eps1'] + Ds['geometry']
    return Ds




def compute_heating_changes_Stoll(Permittivity_change, Spectra_info, with_dilatation=False) :
    """Compute the change in extinction spectra for a 1K temperature increase"""

    alpha = 1.42e-5 #gold linear thermal expansion coeff
    Ds = {}
    Ds['Lambda'] = Spectra_info['lambda']


    deps1 = Permittivity_change['Stoll']['deps1/dT'](1240/Ds['Lambda']*1e-9)
    deps2 = Permittivity_change['Stoll']['deps2/dT'](1240/Ds['Lambda']*1e-9)
 

    if with_dilatation :
        Ds['geometry'] = Spectra_info['dsigma_dilatation']
    else :
        Ds['geometry'] = np.zeros(np.size(Ds['Lambda']))


    Ds['eps1'] = Spectra_info['dsigma/deps1'] * (deps1)
    Ds['eps2'] =  Spectra_info['dsigma/deps2'] * (deps2)
    Ds['tot'] = Ds['eps2'] + Ds['eps1'] + Ds['geometry']
    return Ds

def compute_dilatation_changes(Permittivity_change, Spectra_info, band_contribution = 'both', Ldp_coeff=-1.7e-4, Xdp_coeff = -3.2e-4, Lps_coeff=0) :
    """Compute the change in extinction spectra for a 1K temperature increase"""

    alpha = 1.42e-5 #gold linear thermal expansion coeff
    Ds = {}
    Ds['Lambda'] = Spectra_info['lambda']


    deps1_e_density = Permittivity_change['Drude']['e density']['delta_eps1'](1240/Ds['Lambda']*1e-9)*3*alpha
    deps2_e_density = Permittivity_change['Drude']['e density']['delta_eps2'](1240/Ds['Lambda']*1e-9)*3*alpha

    deps1_interband_Ldp = -Permittivity_change['ib']['deps1/deV']['Ldp'](1240/Ds['Lambda']*1e-9) * Ldp_coeff 
    deps1_interband_Lps = -Permittivity_change['ib']['deps1/deV']['Lps'](1240/Ds['Lambda']*1e-9) * Lps_coeff 
    deps1_interband_Xdp = -Permittivity_change['ib']['deps1/deV']['Xdp'](1240/Ds['Lambda']*1e-9) * Xdp_coeff 

    deps2_interband_Ldp = -Permittivity_change['ib']['deps2/deV']['Ldp'](1240/Ds['Lambda']*1e-9) * Ldp_coeff 
    deps2_interband_Lps = -Permittivity_change['ib']['deps2/deV']['Lps'](1240/Ds['Lambda']*1e-9) * Lps_coeff 
    deps2_interband_Xdp = -Permittivity_change['ib']['deps2/deV']['Xdp'](1240/Ds['Lambda']*1e-9) * Xdp_coeff 


    if band_contribution == 'both' :
        deps1_interband = (deps1_interband_Ldp + deps1_interband_Lps + deps1_interband_Xdp)
        deps2_interband = (deps2_interband_Ldp + deps2_interband_Lps + deps2_interband_Xdp)

    if band_contribution == 'Ldp' :
        deps1_interband = (deps1_interband_Ldp)
        deps2_interband = (deps2_interband_Ldp)



    if band_contribution == 'Stoll' :
        deps1_interband = Permittivity_change['Stoll ib']['delta_eps1']['total'](1240/Ds['Lambda']*1e-9)
        deps2_interband = Permittivity_change['Stoll ib']['delta_eps2']['total'](1240/Ds['Lambda']*1e-9)


    if band_contribution == 'nothing' :
        deps1_interband = 0
        deps2_interband = 0



    Ds['geometry'] = Spectra_info['dsigma_dilatation']
    Ds['eps1'] = Spectra_info['dsigma/deps1'] * (deps1_e_density+ deps1_interband)
    Ds['eps2'] =  Spectra_info['dsigma/deps2'] * (deps2_e_density + deps2_interband)
    Ds['tot'] = Ds['eps2'] + Ds['eps1'] + Ds['geometry']
    return Ds




def compute_contraction_changes(Permittivity_change, Spectra_info, band_contribution = 'both', Ldp_coeff=-1.7e-4, Xdp_coeff = -3.2e-4, Lps_coeff=0) :
    """Compute the change in extinction spectra for a 1K temperature increase"""

    alpha = 1.42e-5 #gold linear thermal expansion coeff
    Ds = {}
    Ds['Lambda'] = Spectra_info['lambda']
    Ds['geometry'] = Spectra_info['dsigma_contraction']
    Ds['eps1'] = Spectra_info['dsigma/deps1'] * 0
    Ds['eps2'] =  Spectra_info['dsigma/deps2'] * 0
    Ds['tot'] = Ds['eps2'] + Ds['eps1'] + Ds['geometry']
    return Ds




def plot_3_modes(ax, Permittivity_change, Spectra_info, band_contribution = 'both', Ldp_coeff=-1.7e-4, Xdp_coeff = -3.2e-4, Lps_coeff=0, lda_SPR=630) :
    
    Ds_heating = compute_heating_changes(Permittivity_change, Spectra_info, band_contribution=band_contribution, Ldp_coeff=Ldp_coeff, Xdp_coeff=Xdp_coeff, Lps_coeff=Lps_coeff)
    Ds_dilatation = compute_dilatation_changes(Permittivity_change, Spectra_info, band_contribution=band_contribution, Ldp_coeff=Ldp_coeff, Xdp_coeff=Xdp_coeff, Lps_coeff=Lps_coeff)
    Ds_contraction = compute_contraction_changes(Permittivity_change, Spectra_info, band_contribution=band_contribution, Ldp_coeff=Ldp_coeff, Xdp_coeff=Xdp_coeff, Lps_coeff=Lps_coeff)

    
    ax[0].plot(Ds_heating['Lambda']*1e9, Ds_heating['eps1']*1e18, color='#377Eb8', linestyle='dotted')
    ax[0].plot(Ds_heating['Lambda']*1e9, Ds_heating['eps2']*1e18, color='#E03E22', linestyle='dashed')
    ax[0].plot(Ds_heating['Lambda']*1e9, Ds_heating['geometry']*1e18, color='#ECA208', linestyle='dashdot')
    ax[0].plot(Ds_heating['Lambda']*1e9, Ds_heating['tot']*1e18, color="#000000",)
    ax[0].axvline(lda_SPR, linestyle='--', color='black')    

    ax[1].plot(Ds_dilatation['Lambda']*1e9, -Ds_dilatation['eps1']*1e18, color='#377Eb8', linestyle='dotted')
    ax[1].plot(Ds_dilatation['Lambda']*1e9, -Ds_dilatation['eps2']*1e18, color='#E03E22', linestyle='dashed')
    ax[1].plot(Ds_dilatation['Lambda']*1e9, -Ds_dilatation['geometry']*1e18, color='#ECA208', linestyle='dashdot')
    ax[1].plot(Ds_dilatation['Lambda']*1e9, -Ds_dilatation['tot']*1e18, color="#000000",)
    ax[1].axvline(lda_SPR, linestyle='--', color='black')

    ax[2].plot(Ds_contraction['Lambda']*1e9, Ds_contraction['eps1']*1e18, color='#377Eb8', linestyle='dotted')
    ax[2].plot(Ds_contraction['Lambda']*1e9, Ds_contraction['eps2']*1e18, color='#E03E22', linestyle='dashed')
    ax[2].plot(Ds_contraction['Lambda']*1e9, Ds_contraction['geometry']*1e18, color='#ECA208', linestyle='dashdot')
    ax[2].plot(Ds_contraction['Lambda']*1e9, Ds_contraction['tot']*1e18, color="#000000",)
    ax[2].axvline(lda_SPR, linestyle='--', color='black')
    pass




def get_COMSOL_spectra_with_deps1_deps2(dir_root, ND_name, spectra_name_root) :
    """Get COMSOL spectra results 

    Args:
        dir_root (_type_): _description_
        ND_name (_type_): _description_
        spectra_name_root (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    os.chdir('\\'.join([dir_root, ND_name, 'detailed spectra']))
    f = pd.read_csv(spectra_name_root + '.csv', skiprows = 4 )
    f_eps2 = pd.read_csv(spectra_name_root + ' - deps2 0,001eps2' + '.csv', skiprows = 4 )
    f_eps1 = pd.read_csv(spectra_name_root + ' - deps1 0,001eps1' + '.csv', skiprows = 4 )


    data = {}

    data['lambda'] = (f['lambda (m)'].__array__())
    data['energy'] = 1240/data['lambda']*1e-9
    data['sigma_scat'] = (f['intermediate surface scattering (s^6*A^2/kg^2)'].__array__())
    data['sigma_abs'] = (f['abs (s^6*A^2/kg^2)'].__array__())
    data['sigma_ext'] = data['sigma_abs']+data['sigma_scat']


    d_ext_with_deps1 = ((f_eps1['intermediate surface scattering (s^6*A^2/kg^2)'].__array__()) + (f_eps1['abs (s^6*A^2/kg^2)'].__array__()))
    deps1 = f_eps1['deps1'].__array__()
    data['dsigma/deps1'] = (d_ext_with_deps1 - data['sigma_ext'])/deps1

    d_ext_with_deps2 = ((f_eps2['intermediate surface scattering (s^6*A^2/kg^2)'].__array__()) + (f_eps2['abs (s^6*A^2/kg^2)'].__array__()))
    deps2 = f_eps2['deps2'].__array__()
    data['dsigma/deps2'] = (d_ext_with_deps2 - data['sigma_ext'])/deps2

    return data


