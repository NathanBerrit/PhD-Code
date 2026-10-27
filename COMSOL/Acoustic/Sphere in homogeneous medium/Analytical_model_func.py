import importlib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import io
import matplotlib
from scipy.optimize import fsolve
import scipy as sc
from mpmath import findroot
import mpmath
fontsize=30
import matplotlib as mpl




class Material() :
    def __init__(self, name, E, nu, rho) :

        self.E = E
        self.nu = nu
        self.rho = rho
        self.lda = E*nu/((1-2*nu)*(1+nu))
        self.mu = E/(2*(1+nu))

        self.cL = np.sqrt((self.lda+2*self.mu)/self.rho)
        self.cT = np.sqrt(self.mu/self.rho)

        self.Z = self.rho*self.cL





def define_eigen_equation(mat1, mat2) :


    ## Constants

    alpha = mat2.cL/mat1.cL
    beta = mat2.cT/mat1.cT
    gamma = mat2.cT/mat2.cL
    delta = mat2.rho/mat1.rho

    def f(x) : 

        A = x**2/delta 
        B = 1 + 1j*x/alpha
        C = x**2 + 4*(alpha*gamma)**2 * (1/(delta*beta*beta)-1) * (1 + 1j*x/alpha)

        part1 = x * 1j * (mpmath.exp(2j*x) + 1) / (mpmath.exp(2j*x) - 1)
        part2 = 1 - A * B/C

        return part1 - part2
    return f

def Solve_eigenvalues(mat1, mat2, R_sphere, x_low=0.1, x_high = 10, n_x=100,  y_low=0, y_high = 5, n_y=5,) :


    f_to_solve = define_eigen_equation(mat1, mat2)

    racines = []
    re_vals = np.linspace(x_low, x_high, n_x)
    im_vals = np.linspace(y_low, y_high, n_y )

    for re in re_vals:
        for im in im_vals:
            try:
                r = findroot(f_to_solve, re + 1j*im, solver='muller')
                #print(r)
                if not any(abs(r - r_existante) < 1e-6 for r_existante in racines) and abs(complex(r).real) > 1e-5:
                    racines.append(r)
                    #print('added !')
            except Exception:
                pass

    racines = np.array([complex(r) for r in racines], dtype=complex)

    res = {}

    res['omega_reduced'] = np.real(racines)
    res['freq'] = res['omega_reduced'] * mat1.cL / (R_sphere*2*np.pi)
    res['gamma_reduced'] = np.imag(racines)
    res['gamma'] = res['gamma_reduced'] * mat1.cL/R_sphere
    res['Q'] = res['freq']/res['gamma'] * np.pi

    return(res)







