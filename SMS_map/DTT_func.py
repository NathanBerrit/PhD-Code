import importlib
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import pandas as pd
import time

os.chdir(r'C:\Users\nberr\ENS_Lyon\Thèse\Code\SMS_map')
import SMS_map
importlib.reload(SMS_map)
from SMS_map import *



def aff(x,a,b):
    return a*x + b

def lin(x,a):
    return a*x


#########################################################################################
################ Functions to Recreate the DTT signal ###################################
######################################################################################### 


def airy(x, y, xc, yc, amp, fwhm):
    """Airy function

    Args:
        x (nD array or float): x position array
        y (nD array or float): y position array
        xc (float): x position of the center
        yc (float): y position of the center
        amp (float): amplitude
        fwhm (float): fwhm

    Returns:
        A (nD array): The Airy amplitude, same dimension as x, y. A[i,j] is at x[j] and y[i]
    """
    xc, yc = float(xc),float(yc)

    if type(x)==int or type(x)==float:
            print('check airy')
            x=np.array([float(x)])
    if type(y)==int or type(y)==float:
            print('check airy')
            y=np.array([float(y)])


    Xm = x-xc
    Ym = y-yc
    

    arg = np.sqrt(Xm**2+Ym**2)/fwhm*1.61633*2
    #print('format de Xm : {}'.format(type(Xm[0,0])))
    #print('format de Ym : {}'.format(type(Ym[0,0])))
    #print('format de arg : {}'.format(type(arg[0,0])))
    

    A = amp * (2*scipy.special.j1(arg) / (arg+1e-30) )**2 #airy    
    return A



def airy_mod(t,x,y,xc,yc,amp,d_y,fwhm, f=1500, n=2):
    """evaluate a modulated airy func in time.

    Args:
        t (nD array): t position array
        x (nD array): x position 
        y (nD array): y position 
        xc (float): x position of the center
        yc (float): y position of the center
        amp (float): amplitude
        d_y (float): modulation amplitude
        fwhm (float): fwhm
        f (float): modulation frequency
        n (int) : the harmonic we consider

    Returns:
        res (nD array): the airy func modulated in time
    """    
    
    
    y_mod= np.sin(2*np.pi*f*t)
        
    return airy(x,np.real(y+d_y*y_mod),xc,yc, amp, fwhm)


def integ(t,x,y,xc,yc,amp,d_y,fwhm, f = 1500., n=2):
    """the quantity we'll integrate

    Args:
        t (nD array): t position array
        x (nD array): x position 
        y (nD array): y position 
        xc (float): x position of the center
        yc (float): y position of the center
        amp (float): amplitude
        d_y (float): modulation amplitude
        fwhm (float): fwhm
        f (float): modulation frequency
        n (int) : the harmonic we consider

    Returns:
        res (nD array): integrande. res[i] at x, y and t[i]
    """
    if n%2==0 :
        A = airy_mod(t,x,y,xc,yc,amp,d_y,fwhm, f=f, n=n)
        B = np.cos(2*n*np.pi*f*t)
    else :
        A = airy_mod(t,x,y,xc,yc,amp,d_y,fwhm, f=f, n=n)
        B = np.sin(2*n*np.pi*f*t)
    
    T = 1/f
    res = 1/T*A*B

    return res





def DTT(x,y,xc,yc,amp,d_y,fwhm,f=1500, n=2):
    """The demodulated signal (what we measure). We integrate the theorical formula for this signal.
    CAREFULL : d_y, x, y, x_c, y_c, fwhm must be in the same units. In my code SMS_map_treatment, they are in um.

    Args:
        
        x (2D array): x position 
        y (2D array): y position
        xc (float): x position of the center
        yc (float): y position of the center
        amp (float): amplitude
        d_y (float): modulation amplitude
        fwhm (float): fwhm
        f (float): modulation frequency
        n (int) : the harmonic we consider

    Returns:
        res (2D array): The nf signal. res[i, j] at x[j], y[i]
    """
    
    T = np.linspace(-1/(2*f), 1/(2*f), 100)

    shape = (np.size(T),np.shape(x)[0], np.shape(x)[1])
    #prepare 3D array to integrate
    XXX = np.zeros(shape)
    YYY = np.zeros(shape)
    TTT = np.zeros(shape)

    for i in range(np.size(T)) :
        XXX[i, :, :] = x
        YYY[i, :, :] = y
        TTT[i, :, :] = T[i]

    integrande = integ(TTT,XXX,YYY,xc,yc,amp,d_y,fwhm, f = f, n=n) #3D array

    DTT = np.trapz(integrande, dx =T[1]-T[0], axis=0)
    return DTT




