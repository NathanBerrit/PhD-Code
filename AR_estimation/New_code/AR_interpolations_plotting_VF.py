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
import matplotlib
import matplotlib.image
import matplotlib.lines as mlines
import matplotlib.patches as mpatches




def Lorentz_no_offset(x, xc, w, Area) :
    """Lorentzian resonance without offset. This function is defined just as he one in Origin.

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        Area (_type_): Total area of the curve
    """
    res = 2 * Area / np.pi * w * 1/(4*(x-xc)**2 + w**2)
    
    return res


def plot_func_interp(f, H, D, HH, DD, points, points_data, title, contour_specif=[]) :


    fig, ax = plt.subplots( figsize=(9,5))


    cm = ax.pcolormesh(HH, DD, f(points).reshape(np.shape(HH)))
    #cs = ax.contour(H, D, f(points).reshape(np.shape(HH)), levels=10, colors='red', alpha=0.6)
    #ax.clabel(cs, inline=1, fontsize=10)
    cs = ax.contour(H, D, DD/HH, levels=[ 2.3, 3, 4], colors='black', alpha=0.6)
    ax.clabel(cs, inline=1, fontsize=10)
    cs = ax.contour(H, D, f(points).reshape(np.shape(HH)), levels=contour_specif, colors='red', alpha=1)
    ax.clabel(cs, inline=1, fontsize=10)
    ax.scatter(*zip(*points_data), c='k', marker = '.')
    ax.set_title('Interpolation RBF {}'.format(title))
    fig.colorbar(cm)
    ax.set_xlabel('h (nm)')
    ax.set_ylabel('D (nm)')
    plt.tight_layout()






def plot_2_level_lines(Lda_RBF, Aire_RBF, H, D, HH, DD, points, points_data,  contour_specif_lda=[], contour_specif_Area = [],title = '') :


    fig, ax = plt.subplots( figsize=(9,5))



    cs = ax.contour(H, D, Lda_RBF(points).reshape(np.shape(HH)), levels=contour_specif_lda, colors='red', alpha=1)
    cs = ax.contour(H, D, Aire_RBF(points).reshape(np.shape(HH)), levels=contour_specif_Area, colors='black', alpha=1)

    ax.clabel(cs, inline=1, fontsize=10)
    ax.scatter(*zip(*points_data), c='k', marker = '.')
    ax.set_title('Interpolation RBF {}'.format(title))

    ax.set_xlabel('h (nm)')
    ax.set_ylabel('D (nm)')
    plt.tight_layout()

def plot_f_red_AR(Data_dic, ax, h_1D) :
    """_summary_

    Args:
        Data_dic (_type_): Experimental data with treated AR
        ax (_type_): _description_
        h_1D (_type_): _description_
    """
    
    n = len(Data_dic)
    cm = plt.get_cmap('gist_rainbow')
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n)) for i in range(n)]

    compt=0

    for key in Data_dic :


        AR_1D = Data_dic[key]['1D_estimation']['AR']
        h_1D = Data_dic[key]['1D_estimation']['h']
        AR_2D = Data_dic[key]['2D_estimation']['AR']
        h_2D = Data_dic[key]['2D_estimation']['h']
        Freq =  Data_dic[key]['f']

        
        if AR_1D*Freq >=1e-10 :
            ax.scatter(AR_1D, Freq*h_1D*1e-9, color=Colors[compt], marker ='.')
         

        if AR_2D*Freq >= 1e-10 :
            ax.scatter(AR_2D, Freq*h_2D*1e-9, color=Colors[compt], marker='v')
       
        compt += 1
    ax.set_xlabel('AR')
    ax.set_ylabel('f_red (Hz/m) ')

    marker_1D = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                            markersize=10, label='1D estimation : h = {}'.format(h_1D))
    marker_2D = mlines.Line2D([], [], color='black', marker='v', linestyle='None',
                            markersize=10, label='2D estimation')
    ax.legend(handles=[marker_1D, marker_2D])
    

def plot_E_SPR_with_AR_fit_2D_and_1D_lda_SPR(Data_dic, ax, h_1D, old_legend = '') :


    n = len(Data_dic)
    cm = plt.get_cmap('gist_rainbow')
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n)) for i in range(n)]

    compt=0
    
    for key in Data_dic :


        AR_1D = Data_dic[key]['1D_estimation']['AR']
        
        AR_2D = Data_dic[key]['2D_estimation']['AR']
        
        E_SPR =  Data_dic[key]['E_SPR']

        
        if AR_1D*E_SPR >=1e-10 :
            ax.scatter(AR_1D, E_SPR, color=Colors[compt], marker ='.', s=100)
         

        if AR_2D*E_SPR >= 1e-10 :
            ax.scatter(AR_2D, E_SPR, color=Colors[compt], marker='v', s=30)
       
        compt += 1
    ax.set_xlabel('AR')
    ax.set_ylabel('E_SPR (eV) ')

    marker_1D = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                            markersize=10, label='1D estimation (h = {} nm)'.format(h_1D))
    marker_2D = mlines.Line2D([], [], color='black', marker='v', linestyle='None',
                            markersize=10, label='2D estimation')
    legend = ax.legend(handles=[marker_1D, marker_2D], loc='lower left', fontsize=10)



        
    try :
        if old_legend != '' :
            ax.add_artist(old_legend)
    except :
        pass
    return legend



def plot_E_SPR_with_AR_fit_1D_Aire(Data_dic, ax, h_1D, old_legend = '') :


    n = len(Data_dic)
    cm = plt.get_cmap('gist_rainbow')
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n)) for i in range(n)]

    compt=0
    
    for key in Data_dic :


        AR_1D = Data_dic[key]['1D_estimation_Aire_SPR']['AR']
        

        E_SPR =  Data_dic[key]['E_SPR']

        
        if AR_1D*E_SPR >=1e-10 :
            ax.scatter(AR_1D, E_SPR, color=Colors[compt], marker ='.', s=100)
         
        compt += 1
    ax.set_xlabel('AR')
    ax.set_ylabel('E_SPR (eV) ')

    marker_1D = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                            markersize=10, label='1D estimation (h = {} nm)'.format(h_1D))

    legend = ax.legend(handles=[marker_1D], loc='lower left', fontsize=10)



        
    try :
        if old_legend != '' :
            ax.add_artist(old_legend)
    except :
        pass
    return legend
    

def plot_E_SPR_with_AR_simulation(Simu_data, ax, old_legend='', h_list = [5, 10, 15, 20, 25, 30]) :
    
    
    H_dic = {}
    for (i, (h, D)) in enumerate(Simu_data['points']) :
        
        if str(h) in H_dic :
            H_dic[str(h)]['D'].append(D)
            H_dic[str(h)]['E_SPR'].append(Simu_data['E_SPR'][i])

        else :
            H_dic[str(h)] = {}
            H_dic[str(h)]['D'] = [D]
            H_dic[str(h)]['E_SPR'] = [Simu_data['E_SPR'][i]]




    for h_key in H_dic :
        
        h = float(h_key)

        D = np.array(H_dic[h_key]['D'])
        E_SPR = np.array(H_dic[h_key]['E_SPR'])



        D_sorted, E_SPR_sorted = (list(t) for t in zip(*sorted(zip(D, E_SPR))))
        D_sorted = np.array(D_sorted)
        E_SPR_sorted = np.array(E_SPR_sorted)

        if h in h_list :
            ax.plot(D_sorted/h, E_SPR_sorted, label='h = {} nm'.format(h))



    
    legend = ax.legend(loc='upper right', fontsize=10)
        
    try :
        if old_legend != '' :
            ax.add_artist(old_legend)
    except :
        pass

    return legend

def plot_E_SPR_with_AR_simulation_prolongated(E_RBF, ax, old_legend='', AR_min=0.1, AR_max=6, h_list = [5, 10, 15, 20, 25, 30]) :

    for h in h_list :
        D_min = h*AR_min
        D_max = h*AR_max

        D = np.linspace(D_min, D_max, 500)


        def E_RBF_D_1D(D, h=h) :
            Z = np.column_stack((np.full_like(D, h, dtype=float),D))
        
            return(E_RBF(Z))
        
        ax.plot(D/h, E_RBF_D_1D(D), label='h = {} nm'.format(h), linestyle='--')

    legend = ax.legend(loc='upper right', fontsize=10)
    
    try :
        if old_legend != '' :
            ax.add_artist(old_legend)
    except :
        pass


        return legend



def plot_interpolation_result_2D(f_RBF, H, D, HH, DD, points, Exp_data, contour_specif=[570, 600, 620, 640], title='Lambda SPR') :

    fig, ax = plt.subplots( figsize=(9,4))



    
    

    cm = ax.pcolormesh(HH, DD, f_RBF(points).reshape(np.shape(HH)))
    cs = ax.contour(H, D, DD/HH, levels=[ 2.3], colors='black', alpha=0.6)
    ax.clabel(cs, inline=1, fontsize=10)
    cs = ax.contour(H, D, f_RBF(points).reshape(np.shape(HH)), levels=contour_specif, colors='red', alpha=1)
    ax.clabel(cs, inline=1, fontsize=10)

    for key in Exp_data :
        ax.scatter(Exp_data[key]['1D_estimation']['h'], Exp_data[key]['1D_estimation']['D'], color='white', marker='v')
        ax.scatter(Exp_data[key]['2D_estimation']['h'], Exp_data[key]['2D_estimation']['D'], color='red', marker='v')
    
    #ax.scatter(*zip(*points_data), c='k', marker = '.')
    ax.set_title('Interpolation RBF {}'.format(title))


    fig.colorbar(cm)
    ax.set_xlabel('h (nm)')
    ax.set_ylabel('D (nm)')



    marker_1D = mlines.Line2D([], [], color='white', marker='v', linestyle='None',
                            markersize=10, label='1D estimation (h = {} nm)'.format(Exp_data[key]['1D_estimation']['h']))
    marker_2D = mlines.Line2D([], [], color='red', marker='v', linestyle='None',
                            markersize=10, label='2D estimation')
    legend = ax.legend(handles=[marker_1D, marker_2D], loc='lower right', fontsize=10)

    plt.tight_layout()







def plot_Q_with_lda_spr(Data_dic,Color_dic, ax) :

    for key in Data_dic :
        data = Data_dic[key]




        compt=0
        for kkey in data :
            spectra = data[kkey]

            if spectra['Q'] != 0 :
                if compt==0 :
                    ax.scatter(spectra['lda_SPR'], spectra['Q'], c = Color_dic[key], label=key)
                else :
                    ax.scatter(spectra['lda_SPR'], spectra['Q'], c = Color_dic[key])

                ax.errorbar(spectra['lda_SPR'], spectra['Q'], yerr = spectra['Q_err'], color=Color_dic[key])
            compt+=1

        ax.set_xlabel('$\lambda_{SPR}$ (nm)')
        ax.set_ylabel('Q')
        ax.grid()


        # os.chdir(r'H:\commun\Experiments (from 2018)\Nanoscope 2\Projects\ULTRASINGLE\Measurements\AuND_specific_AR_Marguet\Quartz')

        # plt.savefig('test.png')
        # plt.savefig('test.pdf')
    pass



def plot_Q_with_AR_1D(Data_dic,Color_dic, ax) :

    for key in Data_dic :
        data = Data_dic[key]




        compt=0
        for kkey in data :
            spectra = data[kkey]

            if spectra['Q'] != 0 :
                if compt==0 :
                    ax.scatter(spectra['1D_estimation_Aire_SPR']['AR'], spectra['Q'], c = Color_dic[key], label=key)
                else :
                    ax.scatter(spectra['1D_estimation_Aire_SPR']['AR'], spectra['Q'], c = Color_dic[key])

                ax.errorbar(spectra['1D_estimation_Aire_SPR']['AR'], spectra['Q'], yerr = spectra['Q_err'], color=Color_dic[key])
            compt+=1

        ax.set_xlabel('A.R.')
        ax.set_ylabel('Q')
        ax.grid()


        # os.chdir(r'H:\commun\Experiments (from 2018)\Nanoscope 2\Projects\ULTRASINGLE\Measurements\AuND_specific_AR_Marguet\Quartz')

        # plt.savefig('test.png')
        # plt.savefig('test.pdf')
    pass



def plot_y_with_x_AR_for_all_sample(ax, Exp_Data, key_x, key_y, Key_sample_list, Plot_dic) :

    
    ax.set_xlabel('AR')
    ax.set_ylabel(key_y)

    for key in Exp_Data :

        exp_data = Exp_Data[key]


        idx = np.where(np.array([exp_data[kkey][key_y] for kkey in exp_data]) != 0)[0]

        X = np.array([exp_data[kkey]['1D']['AR'] for kkey in exp_data])[idx]
        Y = 1/np.array([(exp_data[kkey][key_y]) for kkey in exp_data])[idx]*21*1e3

        plt.scatter(X, Y, color=Plot_dic[key]['main'],marker = Plot_dic[key]['marker'], label = key)

    pass



def print_2D_AR_interpolations(exp_data, points, ax) :
    ax.set_xlabel('h (nm)')
    ax.set_ylabel('D (nm)')
    

    ax.scatter(points[:,0], points[:,1], color='blue', label = 'Simu')
    ax.scatter([exp_data[kkey]['2D']['h'] for kkey in exp_data], [exp_data[kkey]['2D']['D'] for kkey in exp_data], color='red', label = '2D interpolation')

    ax.set_xlim((10, 50))
    ax.set_ylim((10, 140))



def print_1D_Area_AR_interpolations(exp_data, points, ax) :
    ax.set_xlabel('h (nm)')
    ax.set_ylabel('D (nm)')
    

    ax.scatter(points[:,0], points[:,1], color='blue', label = 'Simu')
    ax.scatter([exp_data[kkey]['1D']['h'] for kkey in exp_data], [exp_data[kkey]['1D']['D'] for kkey in exp_data], color='red', label = '2D interpolation')

    ax.set_xlim((10, 50))
    ax.set_ylim((10, 140))


def print_1D_Lda_AR_interpolations(exp_data, points, ax) :
    ax.set_xlabel('h (nm)')
    ax.set_ylabel('D (nm)')
    

    ax.scatter(points[:,0], points[:,1], color='blue', label = 'Simu')
    ax.scatter([exp_data[kkey]['1D_Lda']['h'] for kkey in exp_data], [exp_data[kkey]['1D_Lda']['D'] for kkey in exp_data], color='red', label = '2D interpolation')

    ax.set_xlim((10, 50))
    ax.set_ylim((10, 140))
