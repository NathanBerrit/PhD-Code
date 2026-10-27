import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib

from sklearn.metrics import root_mean_squared_error

from scipy.signal import find_peaks, peak_widths





def plot_TF(a0, a1, tf_data,  color='blue', label='raw') :

    """Plot the tf of an acoustic signal

    Args:
        tf_data (dic): a dictionnary defined such as in Pap_Thermal_Treatment
    """

    
    a0.plot(tf_data['freq'], tf_data['phase'], color=color)
 
    
    a0.set_ylabel('Phase')

    a1.plot(tf_data['freq'], tf_data['amp']**2, color=color, label=label)


def plot_tf_module(tf_data, ax, color='red') :
    """Plot the tf squared module

    Args:
        tf_data (dic): a dictionnary defined such as in Pap_Thermal_Treatment
    """

    # a0.plot(tf_data['freq'], tf_data['phase'])
    # f.suptitle('FFT')
    # a0.set_ylabel('Phase')

    ax.plot(tf_data['freq'], tf_data['amp']**2, color=color)

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




def plot_tf_phase(tf_data, ax, ylim=10e-7, xlim=(0,100), color='red', label='') :
    """Plot the tf squared module

    Args:
        tf_data (dic): a dictionnary defined such as in Pap_Thermal_Treatment
    """

    ax.plot(tf_data['freq'], tf_data['phase'], color=color, label=label)
    ax.set_title('FFT')
    ax.set_ylabel('Phase', fontsize=20)

    
    ax.set_xlim(xlim)
    ax.set_xlabel('Frequency (GHz)', fontsize=20)
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    
