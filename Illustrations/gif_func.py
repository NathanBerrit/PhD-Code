import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *
import importlib


def ExpDec4(x,A1,t1,A2,t2,A3,t3,A4,t4):
    return  A1*np.exp(-x/t1) + A2*np.exp(-x/t2)+ A3*np.exp(-x/t3) + A4*np.exp(-x/t4)
def ExpDec3(x,A1,t1,A2,t2,A3,t3):
    return  A1*np.exp(-x/t1) + A2*np.exp(-x/t2)+ A3*np.exp(-x/t3)
def ExpDec2(x,A1,t1,A2,t2):
    return  A1*np.exp(-x/t1) + A2*np.exp(-x/t2)
def ExpDec1(t,A1,t1):
    return  A1*np.exp(-t/t1)
pass


def Osc4(t, A1,tau1, T1, phi1,A2,tau2, T2, phi2,A3,tau3, T3, phi3,A4,tau4, T4, phi4):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1) + \
            A2*np.exp(-t/tau2)*np.sin(2*np.pi/T2*t+phi2) + \
            A3*np.exp(-t/tau3)*np.sin(2*np.pi/T3*t+phi3) + \
            A4*np.exp(-t/tau4)*np.sin(2*np.pi/T4*t+phi4) 
def Osc3(t, A1,tau1, T1, phi1,A2,tau2, T2, phi2,A3,tau3, T3, phi3):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1) + \
            A2*np.exp(-t/tau2)*np.sin(2*np.pi/T2*t+phi2) + \
            A3*np.exp(-t/tau3)*np.sin(2*np.pi/T3*t+phi3)
def Osc2(t, A1,tau1, T1, phi1,A2,tau2, T2, phi2):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1) + \
            A2*np.exp(-t/tau2)*np.sin(2*np.pi/T2*t+phi2)
def Osc1(t, A1,tau1, T1, phi1):
    return   A1*np.exp(-t/tau1)*np.sin(2*np.pi/T1*t+phi1)




def Lorentz(x, xc, w, A, y0) :
    """Lorentzian resonance

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        A (_type_): height and intensity
        y0 (_type_): offset
    """
    res = y0 + A*w/(4*(x-xc)**2 + w**2)

    return res