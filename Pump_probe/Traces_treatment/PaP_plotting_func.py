import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib
from matplotlib.transforms import Bbox

import matplotlib.colors
import plotly.offline
import plotly.graph_objs as go

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\Pump_probe\Traces_treatment')
import P_and_P_func
importlib.reload(P_and_P_func)
from P_and_P_func import *



def save_in_dir(fig, dir, title) :

    plt.figure(fig.number) #set the wanted figure active
    os.chdir(dir)
    plt.savefig(title + '.png')
    plt.savefig(title + '.pdf')

    pass




def Plot_all_traces_2(Trace) :

    fig = plt.figure(figsize = (13,5))
    
    

    n_traces = len(Trace.All_traces_data)


    
    cm = plt.get_cmap('gist_rainbow')
    Colors = [matplotlib.colors.to_hex(cm(1.*i/n_traces)) for i in range(n_traces)]

    Trace_plot = []

    compt = 0
    for key in Trace.All_traces_data :

        color = Colors[compt]
        trace = Trace.All_traces_data[key]

        Trace_plot.append(go.Scatter(x =trace['t'], y=trace['DTT'], name=key[0:8] + '_' + key[-11:], line=dict(color=color)))
        compt+=1
    plt.xlabel('t (ps)')
    plt.ylabel('DTT (1e-5)')
    plotly.offline.plot(Trace_plot)


    
    pass




def Plot_all_traces(Trace) :

    fig = plt.figure(figsize = (13,5))
    
    

    n_traces = len(Trace.All_traces_data)



    cm = plt.get_cmap('gist_rainbow')
    Colors = [cm(1.*i/n_traces) for i in range(n_traces)]

    

    compt = 0
    for key in Trace.All_traces_data :

        color = Colors[compt]
        trace = Trace.All_traces_data[key]
        
        plt.plot(trace['t'], trace['DTT'], color=color, label = key)
        compt+=1

    legend = plt.legend(framealpha=0.5,bbox_to_anchor=(0.5, 0.5), loc='center left',ncol=1, fontsize=10)
    plt.xlabel('t (ps)')
    plt.ylabel('DTT (1e-5)')

    d = {"down" : 100, "up" : -100}
    def func(evt):
        if legend.contains(evt):
            bbox = legend.get_bbox_to_anchor()
            bbox = Bbox.from_bounds(bbox.x0, bbox.y0+d[evt.button], bbox.width, bbox.height)
            tr = legend.axes.transAxes.inverted()
            legend.set_bbox_to_anchor(bbox.transformed(tr))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", func)
    
    #plt.tight_layout()


    
    return fig


def Plot_mean_before_pulse(Trace, n_i, n_f, mean_start) :

    fig, ax = plt.subplots(1, 2, figsize = (11,4),width_ratios=[2,1])
    
    

    #ax[0].scatter(Trace.delay_tot, Trace.DTT_tot*1e5, s=5, color = 'black',label= 'Mean trace')
    ax[0].plot(Trace.delay_tot, Trace.DTT_tot*1e5,  color = 'black',label= 'Mean trace', marker='.')
    ax[0].plot(Trace.delay_tot[n_i:n_f], [mean_start*1e5]*(n_f-n_i), label='average before pump = {}E-5'.format(round(1e5*mean_start,2)))

    ax[0].scatter(Trace.delay_tot[Trace.t0_index], Trace.DTT_tot[Trace.t0_index]*1e5, color='red', s=15)

    ax[0].set_ylabel('DTT (1e-5)')
    ax[0].set_xlabel('t (ps)')
    ax[0].grid('on')
    ax[0].set_title('Mean on {} traces'.format(Trace.n_traces))
    ax[0].legend(fontsize=10)



   

    ax[1].scatter(Trace.delay_tot[(Trace.t0_index+1):Trace.t0_index+30], Trace.DTT_tot[(Trace.t0_index+1):Trace.t0_index+30]*1e5, color='black', s=5)
    ax[1].set_title("Zoom after excitation")
    ax[1].set_ylabel('DTT (1e-5)')
    ax[1].set_xlabel('t (ps)')
    ax[1].grid('on')

    plt.tight_layout()
    return fig




def Plot_delay(Trace, ax) :
    ax.plot(Trace.delay, Trace.DTT*1e5)


    ax.set_ylabel('DTT (1e-5)', fontsize=15)
    ax.set_xlabel('t (ps)', fontsize=15)
    ax.grid('on')
    

    



def plot_tf_module(tf_data, ax, ylim=10e-7, xlim=(1,100), normalized=False) :
    """Plot the tf squared module

    Args:
        tf_data (dic): a dictionnary defined such as in Pap_Thermal_Treatment
    """

    # a0.plot(tf_data['freq'], tf_data['phase'])
    # f.suptitle('FFT')
    # a0.set_ylabel('Phase')
    if normalized :
        ax.plot(tf_data['freq'], tf_data['amp']**2/tf_data['amp'].max()**2)
    else :
        ax.plot(tf_data['freq'], tf_data['amp']**2)

    try :
            
        for j in range(len(tf_data['peaks'])):
            i = tf_data['peaks'][j]
            
            ax.text(tf_data['freq'][i], tf_data['amp'][i]**2+0.03*tf_data['amp'][i]**2,
                    "{0:.2f} ps".format(1000/tf_data['freq'][i]),horizontalalignment='center',
                    verticalalignment='bottom', fontsize=10)
            ax.scatter(tf_data['freq'][i], tf_data['amp'][i]**2, color='red', s=5)


            f_left = tf_data['freq'][int(tf_data['left'][j])]+tf_data['df']*np.mod(tf_data['left'][j], 1)
            f_right = tf_data['freq'][int(tf_data['right'][j])]+tf_data['df']*np.mod(tf_data['right'][j], 1)
            whm = tf_data['amp'][i]**2/2

            #a1.scatter(f_left, whm, c='r')
            #a1.scatter(f_right, whm, c='r')

    except :
        pass

    ax.set_ylim(0,ylim)
    ax.set_xlim(xlim)

    
    # def convert(x) :
    #     for i, val in enumerate(x) :
    #         if val <= 1e-30 :
    #             x[i] = 1e-30
    #     return 1/x
    
    # ax = ax.secondary_xaxis("top", functions=(convert, convert))


    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    ax.set_ylabel(r'Spectral Energy : (|DTT(w)|$^2$)', fontsize=20)


    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

def plot_tf_phase(tf_data, ax, ylim=10e-7, xlim=(0,100)) :
    """Plot the tf squared module

    Args:
        tf_data (dic): a dictionnary defined such as in Pap_Thermal_Treatment
    """

    ax.plot(tf_data['freq'], tf_data['phase'])
    ax.set_title('FFT')
    ax.set_ylabel('Phase', fontsize=20)

    
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    



### Thermal fit plotting
def Plot_thermal_fit(Trace,ax, n_exp =1) :


    
    #Choose which fits to plot
    func = getattr(P_and_P_func, 'ExpDec' + str(n_exp))
    #func = locals()['ExpDec' + str(n_exp)]
    plot_key = str(n_exp) + ' exp'

    ax.plot(Trace.delay, Trace.DTT*1e5, label='Raw signal', c='blue')
    ax.plot(Trace.delay, func(Trace.delay,*Trace.thermal_fit[plot_key]['coeffs'])*1e5, 'r-',label='fit')

    
    # plt.plot(Trace.delay, Trace.DTT*1e5, label='Raw signal', c='blue')
    # plt.plot(Trace.delay, ExpDec1(Trace.delay,*Trace.thermal_fit['1 exp']['coeffs'])*1e5, 'r-',label='fit: Raw signal')



    #Define the fitting parameters text

    text1 = r'A1 = ({:.1f} $\pm$ {:.1f}) e-5, tau1 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['1 exp']['A1']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['1 exp']['tau1']), Trace.thermal_fit['1 exp']['A1_err']*1e5, Trace.thermal_fit['1 exp']['tau1'], Trace.thermal_fit['1 exp']['tau1_err'])

    text2 = r'A1 = ({:.1f} $\pm$ {:.1f}) e-5, tau1 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['2 exp']['A1']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['2 exp']['tau1']), Trace.thermal_fit['2 exp']['A1_err']*1e5, Trace.thermal_fit['2 exp']['tau1'], Trace.thermal_fit['2 exp']['tau1_err']) + '\n' \
          + r'A2 = ({:.1f} $\pm$ {:.1f}) e-5, tau2 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['2 exp']['A2']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['2 exp']['tau2']), Trace.thermal_fit['2 exp']['A2_err']*1e5, Trace.thermal_fit['2 exp']['tau2'], Trace.thermal_fit['2 exp']['tau2_err'])

    text3 = r'A1 = ({:.1f} $\pm$ {:.1f}) e-5, tau1 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['3 exp']['A1']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['3 exp']['tau1']), Trace.thermal_fit['3 exp']['A1_err']*1e5, Trace.thermal_fit['3 exp']['tau1'], Trace.thermal_fit['3 exp']['tau1_err']) + '\n' \
          + r'A2 = ({:.1f} $\pm$ {:.1f}) e-5, tau2 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['3 exp']['A2']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['3 exp']['tau2']), Trace.thermal_fit['3 exp']['A2_err']*1e5, Trace.thermal_fit['3 exp']['tau2'], Trace.thermal_fit['3 exp']['tau2_err']) + '\n' \
          + r'A3 = ({:.1f} $\pm$ {:.1f}) e-5, tau3 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['3 exp']['A3']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['3 exp']['tau3']), Trace.thermal_fit['3 exp']['A3_err']*1e5, Trace.thermal_fit['3 exp']['tau3'], Trace.thermal_fit['3 exp']['tau3_err'])

    text4 = r'A1 = ({:.1f} $\pm$ {:.1f}) e-5, tau1 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['4 exp']['A1']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['4 exp']['tau1']), Trace.thermal_fit['4 exp']['A1_err']*1e5, Trace.thermal_fit['4 exp']['tau1'], Trace.thermal_fit['4 exp']['tau1_err']) + '\n' \
          + r'A2 = ({:.1f} $\pm$ {:.1f}) e-5, tau2 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['4 exp']['A2']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['4 exp']['tau2']), Trace.thermal_fit['4 exp']['A2_err']*1e5, Trace.thermal_fit['4 exp']['tau2'], Trace.thermal_fit['4 exp']['tau2_err']) + '\n' \
          + r'A3 = ({:.1f} $\pm$ {:.1f}) e-5, tau3 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['4 exp']['A3']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['4 exp']['tau3']), Trace.thermal_fit['4 exp']['A3_err']*1e5, Trace.thermal_fit['4 exp']['tau3'], Trace.thermal_fit['4 exp']['tau3_err']) + '\n' \
          + r'A4 = ({:.1f} $\pm$ {:.1f}) e-5, tau4 = ({:.0f} $\pm$ {:.0f}) ps'.format(Trace.thermal_fit['4 exp']['A4']*1e5*np.exp(-Trace.time_cropped/Trace.thermal_fit['4 exp']['tau4']), Trace.thermal_fit['4 exp']['A4_err']*1e5, Trace.thermal_fit['4 exp']['tau4'], Trace.thermal_fit['4 exp']['tau4_err'])

 
    if  np.min(Trace.DTT) > 0 :
        ax.text(Trace.delay[100], (np.min(Trace.DTT) + (np.max(Trace.DTT) - np.min(Trace.DTT))*0.75)*1e5, locals()['text'+str(n_exp)], fontsize = 10)

    else : #negative sign
        ax.text(Trace.delay[100], (np.min(Trace.DTT) + (np.max(Trace.DTT) - np.min(Trace.DTT))*0.25)*1e5, locals()['text'+str(n_exp)], fontsize = 10)


    ax.set_xlabel('Time (ps)', fontsize=20)
    ax.set_ylabel('DT/T', fontsize=20)

    ax.set_title(str(n_exp) + ' Exponential fit')
    ax.legend(fontsize=10)
    




def Plot_multiple_thermal_fit(Trace,ax, n_min=1, n_max=4) :

    
    #Choose which fits to plot
    
    Colors = ['green', 'blue', 'red', 'orange']


    ax.plot(Trace.delay, Trace.DTT*1e5, label='Raw signal', c='black')
    for i in range(n_min, n_max+1) :
        
        n_modes = i
        c = Colors[i-1]

        func = getattr(P_and_P_func, 'ExpDec' + str(n_modes))
        
        plot_key = str(n_modes) + ' exp'

        
        ax.plot(Trace.delay, func(Trace.delay,*Trace.thermal_fit[plot_key]['coeffs'])*1e5, color=c, label='{} exp'.format(n_modes))



    ax.set_xlabel('Time (ps)', fontsize=20)
    ax.set_ylabel('DT/T', fontsize=20)
    ax.set_title('All thermal fits')
    ax.legend(fontsize=10)
    

    



### Acoustic plots

def Plot_acoustic_raw(Trace, ax) :

    
    ax.plot(Trace.delay, Trace.DTT_acoustic*1e5)

    ax.axhline(0, color='black', alpha=0.5, linestyle='--')
    ax.set_ylabel('DTT (1e-5)')
    ax.set_xlabel('t (ps)')
    ax.grid('on')



def Plot_acoustic_fit(Trace, ax,  n_modes=1, tmax=1000, idx_delay=40, prop_ord=0.6, ) :
 
    #Choose which fits to plot
    func = getattr(P_and_P_func, 'Osc' + str(n_modes))
    
    plot_key = str(n_modes) + ' modes'
    delay_with_0 = np.linspace(0, tmax, 1000)
    ax.plot(Trace.delay, Trace.DTT_acoustic*1e5, label='Raw signal', c='blue')
    ax.plot(delay_with_0, func(delay_with_0,*Trace.acoustic_fit[plot_key]['coeffs'])*1e5, 'r-', alpha=0.5,label='fit')

    text =''
    for j in range(n_modes) :
        coeff_num = str(j+1)
        text += 'A' + coeff_num + r' = ({:.1f} $\pm$ {:.1f}) e-5'.format(np.round(Trace.acoustic_fit[str(n_modes) + ' modes']['A' + coeff_num]*1e5*np.exp(-Trace.time_cropped/Trace.acoustic_fit[str(n_modes) + ' modes']['tau' + coeff_num]), 1), Trace.acoustic_fit[str(n_modes) + ' modes']['A' + coeff_num+'_err']*1e5)
        text += ', tau' + coeff_num + r' = ({:.1f} $\pm$ {:.1f}) ps'.format(np.round(Trace.acoustic_fit[str(n_modes) + ' modes']['tau' + coeff_num], 1), Trace.acoustic_fit[str(n_modes) + ' modes']['tau' + coeff_num+'_err'])
        text += ', T' + coeff_num + r' = ({:.1f} $\pm$ {:.1f}) ps'.format(np.round(Trace.acoustic_fit[str(n_modes) + ' modes']['T' + coeff_num], 1), Trace.acoustic_fit[str(n_modes) + ' modes']['T' + coeff_num+'_err'])
        text += ', phi' + coeff_num + r' = ({:.1f} $\pm$ {:.1f}) $^\circ$'.format(np.round(Trace.acoustic_fit[str(n_modes) + ' modes']['phi' + coeff_num]*180/np.pi, 1), Trace.acoustic_fit[str(n_modes) + ' modes']['phi' + coeff_num+'_err']*180/np.pi)
        text += ', Q' + coeff_num + r' = ({:.1f} $\pm$ {:.1f})'.format(np.round(Trace.acoustic_fit[str(n_modes) + ' modes']['Q' + coeff_num], 1), Trace.acoustic_fit[str(n_modes) + ' modes']['Q' + coeff_num+'_err'])
        text += '\n'
    
 
    ax.text(Trace.delay[idx_delay], Trace.DTT_acoustic.max()*1e5*prop_ord, text, fontsize=8.5)
    ax.text(Trace.delay[idx_delay], 0.1, "Résidu : {:.1f} 1e-3".format(Trace.acoustic_fit[str(n_modes) + ' modes']['residual']*100),transform=ax.get_xaxis_transform(), fontsize=10)



    ax.set_xlabel('Time (ps)', fontsize=20)
    ax.set_ylabel('DT/T', fontsize=20)
    ax.set_xlim((-20, tmax))
    #ax.set_title(str(n_modes) + ' modes fitted')
    ax.legend(fontsize=10)
    

    

