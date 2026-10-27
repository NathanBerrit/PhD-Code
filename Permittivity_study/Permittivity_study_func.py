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




def get_single_index_data(dir_root, fname) :

    os.chdir(dir_root)


def get_index_data(dir_root) :


    Index_data={}
    os.chdir(dir_root)
    for root, dirs, files in os.walk(".", topdown=False):
        
        
        for f in files :
            

            key = ' '.join(f.split('_')[1:])[:-4]


            Index_data[key] = {}
            
            data = np.loadtxt(f)
            Index_data[key]['type'] = f.split('_')[0]

            
            Index_data[key]['wl'] = data[:,0]*1e-6
            Index_data[key]['n'] = data[:,1]
            Index_data[key]['k'] = data[:,2]
            Index_data[key]['eps_1'] = Index_data[key]['n']**2 - Index_data[key]['k']**2
            Index_data[key]['eps_2'] = 2*Index_data[key]['k']*Index_data[key]['n']
            
    
    return(Index_data)


def make_1D_interpolation(Index_data) :

    for key in Index_data :
        data = Index_data[key]

        data['interpolation'] = {}
        data['interpolation']['n'] = sc.interpolate.interp1d(data['wl'], data['n'])
        data['interpolation']['k'] = sc.interpolate.interp1d(data['wl'], data['k'])
        data['interpolation']['eps_1'] = sc.interpolate.interp1d(data['wl'], data['eps_1'])
        data['interpolation']['eps_2'] = sc.interpolate.interp1d(data['wl'], data['eps_2'])
        


def compute_LSPR_offset(Index_data, WL, key_ref) :
    """We take the data linked to key_ref as a reference. This function computes for each data the equivalent LSPR WL according to eps_1.
    We consider that onle eps1 rules the resonance position. So we find lda such that eps1_data(lda) = eps1_ref(lda_ref). Then we obtain an offset lda-lda_ref"""
    
    
    

    for key in Index_data :
        data = Index_data[key]
        data['LSPR_offset'] = {}
        data['LSPR_offset']['WL_ref'] = WL
        data['LSPR_offset']['WL_equiv'] = []

        for wl in WL :

            eps1_ref = Index_data[key_ref]['interpolation']['eps_1'](wl)
            
            
            f = lambda x : data['interpolation']['eps_1'](x) - eps1_ref
            
            res = sc.optimize.root_scalar(f, x0=[wl], bracket=[400e-9, 1000e-9])
            data['LSPR_offset']['WL_equiv'].append(res.root)

        

        data['LSPR_offset']['WL_equiv'] = np.array(data['LSPR_offset']['WL_equiv'])