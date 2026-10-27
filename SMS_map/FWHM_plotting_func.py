import os
import numpy as np
#import pickle
import importlib
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *

import itertools

import matplotlib.image
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\SMS_map')
import FWHM_treatment_func
importlib.reload(FWHM_treatment_func)
from FWHM_treatment_func import *

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\SMS_map')
import SMS_map
importlib.reload(SMS_map)
from SMS_map import *

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\SMS_map')
import DTT_func
importlib.reload(DTT_func)
from DTT_func import *




def plot_single_calib(dir_root, Setup_info, Plot_info, Source_list) :
    


    
    fig, ax = plt.subplots(num=None, figsize=(13, 7), dpi=80, facecolor='w', edgecolor='k')

    fontsize = Plot_info['fontsize']
    Couleurs = Plot_info['Couleurs']
    Markers = Plot_info['Markers']

    FWHM_data = get_FWHM_data(dir_root, Source_list)
    SMS_data = get_treated_SMS_maps(dir_root)
    
    WL_min = np.min(np.array([np.min(np.array(FWHM_data['WL'][key])) for key in FWHM_data['WL'] if len(FWHM_data['WL'][key]) != 0]))
    WL_max = np.max(np.array([np.max(np.array(FWHM_data['WL'][key])) for key in FWHM_data['WL'] if len(FWHM_data['WL'][key]) != 0]))
    print(WL_max)
    xplot = np.array([WL_min, WL_max])


    plt.plot(xplot,0.515/Setup_info['NA']*xplot,color='black', linestyle='--',linewidth=2,label='Limite de diffraction') # diffraction limit


        
    for key in SMS_data :
        sms = SMS_data[key]
        plt.errorbar(sms.wl, sms.FWHM_2D, sms.FWHM_err_2D,capsize=2,  color=Couleurs[sms.source][1], zorder=1)
        plt.scatter(sms.wl, sms.FWHM_2D, marker = Markers[sms.source],s=50, color=Couleurs[sms.source][0], zorder=2)
    


    plt.grid('on')

    SHG_marker = mlines.Line2D([], [], color='blue', marker=Markers['SHG'], linestyle='None',
                            markersize=15, label='SHG')
    OPO_marker = mlines.Line2D([], [], color='green', marker=Markers['OPO'], linestyle='None',
                            markersize=20, label='OPO')
    CHA_marker = mlines.Line2D([], [], color='red', marker=Markers['CHA'], linestyle='None',
                            markersize=20, label='CHA')

    black_line = mlines.Line2D([], [], color='black',linewidth = 2, linestyle='--', label='Limite de diffraction')

    plt.legend(handles=[SHG_marker, OPO_marker, black_line])

    #plt.legend( loc='upper left')
    plt.xlim(WL_min-30,WL_max+30)
    plt.ylim(bottom =0, top= 1000)
    plt.xlabel('Wavelength (nm)', fontsize = fontsize)
    plt.ylabel('$d_0$ (nm)', fontsize = fontsize)
    plt.title('FWHM Calibration', fontsize=fontsize)
    plt.tick_params(axis = 'both', labelsize = fontsize)# Taille de la légende
   
    


    
    return fig, ax




def plot_compare_1D_2D_fit(dir_root, Setup_info, Plot_info, Source_list) :
    
    fig, ax = plt.subplots(num=None, figsize=(13, 7), dpi=80, facecolor='w', edgecolor='k')

    fontsize = Plot_info['fontsize']
    Couleurs = Plot_info['Couleurs']
    Markers = Plot_info['Markers']

    FWHM_data = get_FWHM_data(dir_root, Source_list)
    SMS_data = get_treated_SMS_maps(dir_root)
    
    WL_min = np.min(np.array([np.min(np.array(FWHM_data['WL'][key])) for key in FWHM_data['WL'] if len(FWHM_data['WL'][key]) != 0]))
    WL_max = np.max(np.array([np.max(np.array(FWHM_data['WL'][key])) for key in FWHM_data['WL'] if len(FWHM_data['WL'][key]) != 0]))

    xplot = np.array([WL_min, WL_max])


    plt.plot(xplot,0.515/Setup_info['NA']*xplot,color='black', linestyle='--',linewidth=2,label='Limite de diffraction') # diffraction limit


        
    for key in SMS_data :
        sms = SMS_data[key]
        #plt.errorbar(sms.wl, sms.FWHM_1D, sms.std_wid, capsize=3, color=Couleurs[sms.source][1])

        # plt.scatter(sms.wl, sms.FWHM_1D, marker = Markers[sms.source], color='black')
        # plt.scatter(sms.wl, sms.FWHM_2D, marker = Markers[sms.source], color='#de425b')
    
    for source_key in FWHM_data['WL'] :
        
        plt.plot(FWHM_data['WL'][source_key], FWHM_data['FWHM_1D_mean'][source_key], marker = Markers[source_key], color='black')
        plt.plot(FWHM_data['WL'][source_key], FWHM_data['FWHM_2D_mean'][source_key], marker = Markers[source_key], color='#de425b')



    plt.grid('on')
    
    marker_1D = mlines.Line2D([], [], color='black', marker=Markers['SHG'], linestyle='None',
                            markersize=15, label='1D')
    marker_2D = mlines.Line2D([], [], color='#de425b', marker=Markers['SHG'], linestyle='None',
                            markersize=15, label='2D')

    
    plt.legend(handles=[marker_1D, marker_2D])

    #plt.legend( loc='upper left')
    plt.xlim(WL_min-30,WL_max+30)
    plt.ylim(bottom =0, top= 1000)
    plt.xlabel('Wavelength (nm)', fontsize = fontsize)
    plt.ylabel('$d_0$ (nm)', fontsize = fontsize)
    plt.title('FWHM Calibration', fontsize=fontsize)
    plt.tick_params(axis = 'both', labelsize = fontsize)# Taille de la légende
   
    


    #plt.show()
    return fig, ax







def plot_compare_calib(Dir_root_list, Setup_info, Plot_info, SSource_list) :
    
    fig, ax = plt.subplots(num=None, figsize=(13, 7), dpi=80, facecolor='w', edgecolor='k')
    fontsize = Plot_info['fontsize']

    n_calib = len(Dir_root_list)
    cm = plt.get_cmap('Accent')
    
    colors = [cm(1.*i/n_calib) for i in range(n_calib)]
    markers = ['o','s','v', 'o', 's']

    Legend_items = []

    WL_min = []
    WL_max = []

    for n in range(len(Dir_root_list)) :
        dir_root = Dir_root_list[n]
        FWHM_data = get_FWHM_data(dir_root, Source_list=SSource_list[n])
        #print(FWHM_data['WL'])
        SMS_data = get_treated_SMS_maps(dir_root)

        WL_min = np.min(np.array([np.min(np.array(FWHM_data['WL'][key])) for key in FWHM_data['WL'] if len(FWHM_data['WL'][key]) != 0]))
        WL_max = np.max(np.array([np.max(np.array(FWHM_data['WL'][key])) for key in FWHM_data['WL'] if len(FWHM_data['WL'][key]) != 0]))

        m = markers[n%n_calib]
        c = colors[n]
        for key in SMS_data :
            sms = SMS_data[key]
            #plt.errorbar(sms.wl, sms.FWHM_1D, sms.std_wid, capsize=3, color=Couleurs[sms.source][1])
            
            plt.scatter(sms.wl, sms.FWHM_1D, marker = m, color=c)

        Legend_items.append(mlines.Line2D([], [], color=c, marker=m, linestyle='None',
                            markersize=15, label=os.path.basename(os.path.normpath(dir_root))))

        if n==0 :
            wl_min = np.min(np.array(WL_min))
            wl_max = np.max(np.array(WL_max))
        else :
            wl_min = np.min([wl_min, np.min(np.array(WL_min))])
            wl_max = np.max([wl_max, np.max(np.array(WL_max))])
        #print(wl_min, wl_max)

    xplot = np.array([wl_min, wl_max])


    plt.plot(xplot,0.515/Setup_info['NA']*xplot,color='black', linestyle='--',linewidth=2,label='Limite de diffraction') # diffraction limit


    Legend_items.append(mlines.Line2D([], [], color='black',linewidth = 2, linestyle='--', label='Limite de diffraction'))

    plt.legend(handles=Legend_items)

    #plt.legend( loc='upper left')
    plt.xlim(wl_min-30,wl_max+30)
    plt.grid('on')
    plt.ylim(bottom =0, top= 1000)
    plt.xlabel('Wavelength (nm)', fontsize = fontsize)
    plt.ylabel('$d_0$ (nm)', fontsize = fontsize)
    plt.title('FWHM Calibration', fontsize=fontsize)
    plt.tick_params(axis = 'both', labelsize = fontsize)# Taille de la légende
    
    pass