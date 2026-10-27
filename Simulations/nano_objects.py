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




class Nano_object() : 
    def __init__(self, wl, n, k, name='', eps_ext=1) :
        self.name = name
        

        self.wl = wl
        
        self.n = n
        self.k = k 
        self.eps_ext = eps_ext

        self.eps_1 = self.n**2 - self.k**2
        self.eps_2 = 2*self.n*self.k

        self.update()
        self.wl_res = self.wl[np.argmax(self.sigma_ext)]



    def set_sigma_abs(self, eps_ext) :
            """Compute the absorbtion cross-section.

            Args:
                eps_ext (float or 1D array): dielectric constant of surronding area

            Returns:
                sigma_abs (1D array): scattering cross section
            """

            alpha = self.polarizability
            wavenumber = 2*np.pi*np.sqrt(eps_ext)/self.wl

            return wavenumber*np.imag(alpha)
        
    def set_sigma_scat(self,eps_ext) :
        """Compute the scattering cross-section.

        Args:
            eps_ext (float or 1D array): dielectric constant of surronding area

        Returns:
            sigma_scat (1D array): scattering cross section
        """
        
        alpha = self.polarizability
        wavenumber = 2*np.pi*np.sqrt(eps_ext)/self.wl

        return( wavenumber**4/(6*np.pi)*np.abs(alpha)**2)



class Sphere(Nano_object) :
    def __init__(self, r = 1, **kwargs):
        
        self.shape='Sphere'
        self.r = r

        self.L = 1/3
        super().__init__(**kwargs)


    def update(self) :

        self.polarizability = self.set_polarizability(self.eps_ext)
        self.sigma_scat = self.set_sigma_scat(self.eps_ext)
        self.sigma_abs = self.set_sigma_abs(self.eps_ext)
        self.sigma_ext = self.sigma_scat + self.sigma_abs

        self.lda_SPR = np.max(self.sigma_ext)
        



    def set_polarizability(self, eps_ext) :
        """Compute the polarizability of the Nano_objecterial (source : Romain Rouxel PhD thesis).

        Args:
            wl (1D array): wavelength
            eps_ext (float of complex): dielectric constant of surronding area
            polar (tuple): (px,py,pz) linear polar vector

        Returns:
            alpha (1D complex array): polarizability
        """

   
        V = np.pi*4/3*self.r**3
        L = 1/3

        
        eps = self.eps_1 + 1j*self.eps_2



        A = V/L
        B = eps - eps_ext
        C = eps + ((1-L)/L)*eps_ext
        alpha = A*B/C
        return alpha
    

class Ellipsoid(Nano_object) :
    def __init__(self, a, c=1, polar = (1, 0, 0), **kwargs):
        """define an ellipsoid

        Args:
            a (float): a>b
            b (float): a>b
        """
        self.polar = polar
        self.shape='ellipsoid'
        self.a = a
        self.c = c
        self.e = np.sqrt(1-(self.c/self.a)**2)

        g = np.sqrt((1-self.e**2)/self.e**2)

        L = g/(2*self.e**2)*(np.pi/2 - np.arctan(g)) - g**2/2
        if self.polar == (0,0,1) :
            L = 1 - 2*L #L for polar along z
        
        self.L = L
        
        super().__init__(**kwargs)

    def update(self) :

        self.e = np.sqrt(1-(self.c/self.a)**2)


        g = np.sqrt((1-self.e**2)/self.e**2)

        L = g/(2*self.e**2)*(np.pi/2 - np.arctan(g)) - g**2/2
        
        if self.polar == (0,0,1) :
            L = 1 - 2*L #L for polar along z
        
        self.L = L
        
        self.polarizability = self.set_polarizability(self.eps_ext)
        self.sigma_scat = self.set_sigma_scat(self.eps_ext)
        self.sigma_abs = self.set_sigma_abs(self.eps_ext)
        self.sigma_ext = self.sigma_scat + self.sigma_abs

        self.lda_SPR = self.wl[np.argmax(self.sigma_ext)]


    def get_L(self) :
        
        return(self.L)


    def set_polarizability(self, eps_ext=1) :
        """Compute the polarizability of the Nano_objecterial.

        Args:
            wl (1D array): wavelength
            eps_ext (float of complex): dielectric constant of surronding area
            polar (tuple): (px,py,pz) linear polar vector


        Returns:
           alpha (1D complex array): polarizability
        """

        self.e = np.sqrt(1-(self.c/self.a)**2)
        
        V = np.pi*4/3*self.a**2*self.c
        g = np.sqrt((1-self.e**2)/self.e**2)
        L = g/(2*self.e**2)*(np.pi/2 - np.arctan(g)) - g**2/2 #L for polar along x or y
        
        if self.polar == (0,0,1) :
            L = 1 - 2*L #L for polar along z

        eps = self.eps_1 + 1j*self.eps_2

        A = V/L
        B = eps - eps_ext
        C = eps + ((1-L)/L)*eps_ext
        return A*B/C




