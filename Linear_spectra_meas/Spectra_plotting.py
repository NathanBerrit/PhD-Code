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



os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\Linear_spectra_meas')
from Spectra_computation_func import *
import Spectra_computation_func
importlib.reload(Spectra_computation_func)
from Spectra_computation_func  import *





def spectra_simple_plot(data, Plot_info) :
    """Makes a simple plot of a spectra from its data.

    Args:
        data (dic): dictionnary with the spectra info.

    Returns:
        fig (figure) : matplotlib associated figure.
        ax (figure) : matplotlib associated axes object.
    """

   

    fig, ax = plt.subplots()    

    for source_key in data['Source_list'] :

        plt.scatter(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,s = 15,color=Plot_info['Couleurs'][source_key][0], label=source_key)
        plt.errorbar(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15, yerr = data['Sigma_err'][source_key]*1e15, linestyle='',alpha=0.3, color='black')
    

    plt.ylim(bottom = 0)

    L1 = [data['Lambda'][kkey] for kkey in data['Lambda']]
    All_lambda = np.concatenate(L1)
    E_min = 1240/((np.max(All_lambda)*1e9))-0.1
    E_max = 1240/(np.min(All_lambda)*1e9)+0.1

    plt.xlim((E_min, E_max))
    plt.ylabel("$\sigma_{ext}$ ($10^3$ nm2)")
    plt.xlabel('Energy (eV)')

    
    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")
    plt.grid('on')
    plt.subplots_adjust(top=0.7)
    plt.title(data['title'])
    plt.legend()

    return(fig, ax)




def all_spectra_plot(Data, Plot_info) :
    """Plot all the spectra of the dataset Data.

    Args:
        Data (dic): each key links to one spectra dic

    Returns:
        dif, ax: the figure and axes.
    """

    compt=0

    n_spectra = len(Data)
    cm = plt.get_cmap('Accent')
    Colors = [cm(1.*i/n_spectra) for i in range(n_spectra)]

    Lambda_min = []
    Lambda_max = []

    fig, ax = plt.subplots()
    
    #ax.set_prop_cycle(color=[cm(1.*i/n_spectra) for i in range(n_spectra)])
    for key in Data :
        data = Data[key]

        #Regroup all sources points in 1
        
    
        WL = np.concatenate([data['Lambda'][kkey] for kkey in data['Lambda']])
        Lambda_min.append(np.min(WL))
        Lambda_max.append(np.max(WL))
       
       
        #Make the plot
        label_compt=0
        for source_key in data['Lambda'] :
            # if label_compt==0 :
            #     plt.plot(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,color=Colors[compt], marker = Plot_info['Markers'][source_key], label = key)
            # else :
            #     plt.plot(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,color=Colors[compt], marker = Plot_info['Markers'][source_key])
            if label_compt==0 :
                plt.plot(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,color=Colors[compt], label = key)
            else :
                plt.plot(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,color=Colors[compt])

            #plt.errorbar(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15, yerr = data['Sigma_err'][source_key]*1e15, linestyle='',alpha=0.3, color='black')
            label_compt += 1

        compt +=1

    plt.ylim(bottom = 0)

    lambda_min = np.min(np.array(Lambda_min))
    lambda_max = np.min(np.array(Lambda_max))

    E_min = 1240/((lambda_max*1e9))-0.1
    E_max = 1240/(lambda_min*1e9)+0.1

    plt.xlim((E_min, E_max))
    plt.ylabel("$\sigma_{ext}$ ($10^3$ nm2)")
    plt.xlabel('Energy (eV)')

    
    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")
    plt.grid('on')
    plt.legend()
    
    return fig, ax



def plot_lorentz_single(data, res_fit, Plot_info, key) :

    

    fig, ax = plt.subplots()    

    for source_key in data['Source_list'] :

        plt.scatter(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,s = 15,color=Plot_info['Couleurs'][source_key][0], label=source_key)
       
        plt.errorbar(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15, yerr = data['Sigma_err'][source_key]*1e15, linestyle='',alpha=0.3, color='black')
   
    

    plt.ylim(bottom = 0)

    L1 = [data['Lambda'][kkey] for kkey in data['Lambda']]
    All_lambda = np.concatenate(L1)
    E_min = 1240/((np.max(All_lambda)*1e9))-0.1
    E_max = 1240/(np.min(All_lambda)*1e9)+0.1

    E = np.linspace(1240/(All_lambda.min()*1e9), 1240/(All_lambda.max()*1e9), 1000)
    plt.plot(E, Lorentz(E, res_fit['E_c'], res_fit['w'], res_fit['A'], 0 )*1e15, color='black')


    
    offset=0.08
    plt.text(res_fit['E_c'], 0.1 + 3*offset, '$\Delta$E = {:.2f} $\pm$ {:.2f} meV'.format(res_fit['w']*1e3, res_fit['w_err']*1e3),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)
    plt.text(res_fit['E_c'], 0.1 + 2*offset, '$\lambda_c$ = {:.2f} $\pm$ {:.2f} nm'.format(res_fit['Lda_c'], res_fit['Lda_c_err']),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)
    plt.text(res_fit['E_c'], 0.1 + 1*offset, '$\lambda_g$ = {:.2f} nm, $\lambda_d$ = {:.2f} nm'.format(res_fit['Lda_left'],res_fit['Lda_right']),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)
    plt.text(res_fit['E_c'], 0.1 + 0*offset, 'integral = {:.2f} $\pm$ {:.2f} eV*nm^2'.format(res_fit['integral']*1e18, res_fit['integral_err']*1e18),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)


    

    plt.xlim((E_min, E_max))
    plt.ylabel("$\sigma_{ext}$ ($10^3$ nm2)")
    plt.xlabel('Energy (eV)')
    plt.title(key)
    
    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")

    plt.grid('on')
    plt.subplots_adjust(top=0.7)
    #plt.title(key + ' - ' + data['title'])
    

    plt.legend(fontsize=10)
    plt.show()
    return(fig, ax)



def plot_lorentz_single_nofit(data, res_fit, Plot_info, key) :

    

    fig, ax = plt.subplots()    

    for source_key in data['Source_list'] :

        plt.scatter(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,s = 15,color=Plot_info['Couleurs'][source_key][0], label=source_key)
       
        plt.errorbar(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15, yerr = data['Sigma_err'][source_key]*1e15, linestyle='',alpha=0.3, color='black')
   
    

    plt.ylim(bottom = 0)

    L1 = [data['Lambda'][kkey] for kkey in data['Lambda']]
    All_lambda = np.concatenate(L1)
    E_min = 1240/((np.max(All_lambda)*1e9))-0.1
    E_max = 1240/(np.min(All_lambda)*1e9)+0.1


    
 
    

    plt.xlim((E_min, E_max))
    plt.ylabel("$\sigma_{ext}$ ($10^3$ nm2)")
    plt.xlabel('Energy (eV)')

    
    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")

    plt.grid('on')
    plt.subplots_adjust(top=0.7)
    #plt.title(key + ' - ' + data['title'], size=10)
    

    plt.legend(fontsize=10)
    plt.show()
    return(fig, ax)



def plot_spectra_together_with_fit(key_list, Data, Res_fit, Plot_info) :

    

    fig, ax = plt.subplots(figsize=(9,6))    

    compt = 0

    n_spectra = len(key_list)
    cm = plt.get_cmap('Accent')
    Colors = [cm(1.*i/n_spectra) for i in range(n_spectra)]


    E_min_tot = 100
    E_max_tot = 0

    for key in key_list :

        
        
        
        compt +=1
        data = Data[key]
        res_fit = Res_fit[key]
    
    
    
        for source_key in data['Source_list'] :

            plt.scatter(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,s = 15,color=Colors[compt-1])
        
            plt.errorbar(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15, yerr = data['Sigma_err'][source_key]*1e15, linestyle='',alpha=0.3)
    
        


        if compt == 1 :
                offset=0.08
                plt.text(res_fit['E_c'], 0.1 + 3*offset, '$\Delta$E = {:.2f} $\pm$ {:.2f} meV'.format(res_fit['w']*1e3, res_fit['w_err']*1e3),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)
                plt.text(res_fit['E_c'], 0.1 + 2*offset, '$\lambda_c$ = {:.2f} $\pm$ {:.2f} nm'.format(res_fit['Lda_c'], res_fit['Lda_c_err']),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)
                plt.text(res_fit['E_c'], 0.1 + 1*offset, '$\lambda_g$ = {:.2f} nm, $\lambda_d$ = {:.2f} nm'.format(res_fit['Lda_left'],res_fit['Lda_right']),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)
                plt.text(res_fit['E_c'], 0.1 + 0*offset, 'integral = {:.2f} $\pm$ {:.2f} eV*nm^2'.format(res_fit['integral']*1e18, res_fit['integral_err']*1e18),transform=ax.get_xaxis_transform(), horizontalalignment='center', fontsize=10)



        plt.ylim(bottom = 0)

        L1 = [data['Lambda'][kkey] for kkey in data['Lambda']]
        All_lambda = np.concatenate(L1)
        E_min = 1240/((np.max(All_lambda)*1e9))-0.1
        E_max = 1240/(np.min(All_lambda)*1e9)+0.1




        E = np.linspace(1240/(All_lambda.min()*1e9), 1240/(All_lambda.max()*1e9), 1000)
        plt.plot(E, Lorentz(E, res_fit['E_c'], res_fit['w'], res_fit['A'], 0 )*1e15, color=Colors[compt-1], label = key)


        

        if E_min_tot > min(E_min + 0.1, E_min_tot) :
            E_min_tot =  min(E_min + 0.1, E_min_tot)

        if E_max_tot <  max(E_max - 0.1, E_max_tot) :
            E_max_tot =  max(E_max - 0.1, E_max_tot)
        

    plt.xlim((E_min_tot - 0.1, E_max_tot + 0.1))
    plt.ylabel("$\sigma_{ext}$ ($10^3$ nm2)")
    plt.xlabel('Energy (eV)')

    
    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")

    plt.grid('on')
    plt.subplots_adjust(top=0.7)
    #plt.title(data['title'])
    

    plt.legend(loc='upper left', fontsize=10)
    
    plt.show()
    return(fig, ax)




def plot_spectra_together(key_list, Data, Res_fit, Plot_info) :

    

    fig, ax = plt.subplots(figsize=(9,6))    

    compt = 0

    n_spectra = len(key_list)
    cm = plt.get_cmap('Accent')
    Colors = [cm(1.*i/n_spectra) for i in range(n_spectra)]


    E_min_tot = 100
    E_max_tot = 0

    for key in key_list :

        
        compt +=1
        data = Data[key]
        res_fit = Res_fit[key]
    
    
        a=0
        for source_key in data['Source_list'] :
            if a==0 :
                #plt.scatter(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,s = 15,color=Colors[compt-1], label = key)
                plt.plot(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,color=Colors[compt-1], label = key)
            else :
                #plt.scatter(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,s = 15,color=Colors[compt-1])
                plt.plot(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15,color=Colors[compt-1])
            plt.errorbar(1240/(data['Lambda'][source_key]*1e9), data['Sigma'][source_key]*1e15, yerr = data['Sigma_err'][source_key]*1e15, linestyle='',alpha=0.3)
            a+=1

        plt.ylim(bottom = 0)

        L1 = [data['Lambda'][kkey] for kkey in data['Lambda']]
        All_lambda = np.concatenate(L1)
        E_min = 1240/((np.max(All_lambda)*1e9))-0.1
        E_max = 1240/(np.min(All_lambda)*1e9)+0.1

        E_min_tot =  min(E_min + 0.1, E_min_tot)
        E_max_tot =  max(E_max - 0.1, E_max_tot)
        

    plt.xlim((E_min_tot - 0.1, E_max_tot + 0.1))
    plt.ylabel("$\sigma_{ext}$ ($10^3$ nm2)")
    plt.xlabel('Energy (eV)')

    
    def convert(x) :
        return 1240/x
    
    ax2 = ax.secondary_xaxis("top", functions=(convert, convert))

    ax2.set_xlabel("Lambda (nm)")

    plt.grid('on')
    plt.subplots_adjust(top=0.7)
    #plt.title(data['title'])
    

    plt.legend(loc='upper left', fontsize=10)
    
    plt.show()
    return(fig, ax)
