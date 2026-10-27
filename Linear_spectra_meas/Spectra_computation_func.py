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


def aff(x,a,b):
    return a*x + b

def lin(x,a):
    return a*x

def J1(z) :
    return jv(1, z)

def I_Airy(r, I0, d0) :
    """Intensity profile with an airy distribution

    Args:
        r (nD array): position
        I0 (float): max intensity
        d0 (float): FWHM
    """
    
    arg = 1.62*2*r/d0
    B = 2*J1(arg)
    return(I0 *B*B/(arg*arg+1e-30))



def I_Gaussian(r,I0, d0) :
    """Intensity profile with a Gaussian distribution

    Args:
        r (nD array): position
        I0 (float): max intensity
        d0 (float): FWHM
    """
    return(I0*np.exp(-4*np.log(2)*(r**2)/d0**2))

def integ_max(t, d0,f, delta_y, n=2, beam='Airy') :
    """quantity to be integrated : DTT = integrande * sigma_ext

    Args:
        t (1D array): time
        d0 (float): FWHM
        f (float): modulation frequency
        delta_y (float): position modulation amplitude
        beam (str) : kind of beam, can be either Airy or Gaussian
    """
    

    if n%2==0 :
        B = np.cos(2*n*np.pi*f*t)
    else :
        B = np.sin(2*n*np.pi*f*t)
    
    if beam == 'Airy' :
        intensity = I_Airy(delta_y*np.sin(2*np.pi*f*t), 1, d0)
        prefactor = 0.8316/(d0*d0) * f*np.pi

    if beam == 'Gaussian' :
        intensity = I_Gaussian(delta_y*np.sin(2*np.pi*f*t), 1, d0)
        prefactor = 4*np.log(2)/(d0**2)*f

    

    return(prefactor * intensity * B)


def R2F_int(d0, delta_y, beam='Airy', n_points=2000) :
    """R2F response in not-simplyfied form.
    We must use sigma_ext = DTT / R2F_int

    Args:
        d_y (float or 1D array): modulation amplitude
        d0 (float or 1D array): fwhm
        beam (str) : kind of beam, can be either Airy or Gaussian
        n_points (int): number of points of integration.
    

    Returns:
        res (nD array): The max signal. 1D if d0 or delta_y is an array, 2D if both are, float otherwise
    """

    f = 1000 #modulation frequency : it doesn't change the result.
    n=2 #we work with second harmonic.
    
    T = np.linspace(-1/(2*f), 1/(2*f), n_points)
    
    # print(type(d0))
    # print(type(delta_y))
    # print('-----------------')
    if np.size(d0)==1 and np.size(delta_y)==1 :
        
        integrande = integ_max(T, d0, f, delta_y, n=n, beam=beam)
        DTT = np.trapz(integrande, dx =T[1]-T[0])

    elif np.size(d0)==1 :
        
        TT, DDelta_y = np.meshgrid(T, delta_y)

        integrande = integ_max(TT,d0,f, DDelta_y, n=n, beam=beam)
        DTT = np.trapz(integrande, dx =T[1]-T[0], axis=1)

    elif np.size(delta_y)==1 :
        
        TT, DD0 = np.meshgrid(T, d0)
        integrande = integ_max(TT,DD0,f, delta_y, n=n, beam=beam)
        DTT = np.trapz(integrande, dx =T[1]-T[0], axis=1)

    else :
        
        shape = (np.size(T),np.size(d0), np.size(delta_y))
        DDD0 = np.zeros(shape)
        DDDelta_y = np.zeros(shape)
        TTT = np.zeros(shape)

        for i in range(np.size(T)) :
            TTT[i, :, :] = T[i]
        for j in range(np.size(delta_y)) :
            DDDelta_y[:, :, j] = delta_y[j]  
        for k in range(np.size(d0)) :
            DDD0[:, k, :] =d0[k]

        integrande = integ_max(TTT,DDD0,f, DDDelta_y, n=n, beam=beam)
        DTT = np.trapz(integrande, dx =T[1]-T[0], axis=0)


    return(DTT)


def R2F_ana(d0, delta_y=600) :
    return -0.8316*np.pi/(d0*d0)*(mpmath.hyp2f3(1.5, 1.5, 2, 2, 3, -10.45*delta_y*delta_y/(d0*d0))-mpmath.hyp2f3(0.5, 1.5, 1, 2, 3, -10.45*delta_y*delta_y/(d0*d0)))

def R2F_DL(d0, delta_y) :
    """Computes the 20 first fourier coeffs of the analytic R2F. Computed with mathematica, around x=0.

    Args:
        d0 (float or array): FWHM
        delta_y (float or array): modulation amplitude

    """
    x = -10.4501*(delta_y/d0)**2
    DL = x/8 - (5*x**2)/384 + (35*x**3)/49152 - (49*x**4)/1966080 + (77*x**5)/125829120 - (1573*x**6)/140928614400 + (20449*x**7)/129879811031040 - (347633*x**8)/196378274278932480 + (10207769*x**9)/628410477692583936000 - \
        (17631601*x**10)/142200313809293279232000 + (31194371*x**11)/39003514644834728017920000 - (55190041*x**12)/12493605811033460078700134400 + (55190041*x**13)/2611996521560062053786908098560 - \
        (1600511189*x**14)/18109842549483096906255896150016000 + (84638797583*x**15)/259622702789389677248084527206629376000 - (154341336769*x**16)/144444631006460438614388846045870161920000 + \
        (8123228251*x**17)/2593250102640661173253651178880661192704000 - (15818918173*x**18)/1914559504349562420482124184636465291984896000 + (585299972401*x**19)/29690399698220906459168941201747030989550387200000 - \
        (23997298868441*x**20)/561340399955963636396952234179245558825819412889600000
    
    res = 0.8316/(d0*d0)*DL*np.pi/2
    return res


def Response_function(d0, delta_y, beam='Airy', n_points=2000) :
    """Response function for either an Airy or a Gaussian beam.

    Args:
        d_y (float or 1D array): modulation amplitude
        d0 (float or 1D array): fwhm
        beam (str) : kind of beam, can be either Airy or Gaussian
        n_points (float): number of integration points

    Returns:
        res (nD array): The max signal. 1D if d0 or delta_y is an array, 2D if both are (with res[i,j] = res(d0[j], deltay[i]))
    """


    if beam == 'Airy' :
        res = R2F_int(d0, delta_y, beam='Airy', n_points=n_points)
    
    if beam == 'Gaussian' :

        res = R2F_int(d0, delta_y, beam='Gaussian', n_points=n_points)
        # if np.size(d0)==1 or np.size(delta_y)==1 :
        
        #     res = 4*np.log(2) / d0**2 * np.exp(-2*np.log(2)*delta_y**2 / d0**2) * i1(2*np.log(2)*delta_y**2 / d0**2)

        # else : # they are both 1D array
            
        #     DD0, DDelta_y = np.meshgrid(delta_y, d0)
        #     res = 4*np.log(2) / DD0**2 * np.exp(-2*np.log(2)*DDelta_y**2 / DD0**2) * i1(2*np.log(2)*DDelta_y**2 / DD0**2)

        

    return res




def sigma_ext(dtt, d0, delta_y, beam='Airy') :
    """Compute sigma_ext with the needed data

    Args:
        dtt (float or array): dtt measured with the photodiode
        d0 (float or array): fwhm
        delta_y (float or array): modulation amplitude.
    """

    # R2F = []
    # if not type(d0) == type([0, 1]) :
        
    #     R2F = R2F_ana(d0, delta_y)
    
    # else :
    #     print('erreur: {}'.format(d0))
    #     for i in range(len(d0)) :
    #         R2F.append(R2F_ana(d0[i], delta_y))

    R2F = Response_function(d0, delta_y, beam = beam)

    return(dtt/(1+dtt) / R2F)

def incert_sigma_ext(dtt, d0, delta_y, beam="Airy") :
    """Numerical estimation of uncertainty of sigma_ext

    Args:
        dtt (float): DTT
        d0 (float): d0
        delta_y (float): delta_y
        beam (str, optional): Beam profile considered. Defaults to "Airy".
    """

    
    D0 = np.linspace(300e-9, 1500e-9, 1000)
    pas_d0 = D0[1]- D0[0]
    erreur_d0 = 10e-9 #uncertainty on d0 estimation

    Delta_y = np.linspace(200e-9, 800e-9, 500)
    pas_delta_y = Delta_y[1]- Delta_y[0]
    erreur_delta_y = 3e-9 #uncertainty on delta_y estimation

    DTT = np.linspace(0.005e-3, 500e-3, 2000)
    pas_dtt = DTT[1]- DTT[0]
    erreur_dtt = 1e-3 #uncertainty on DTT estimation

    #Computes numerical derivative of Sigma_ext along each variable.
    sigma_grad_DTT = np.gradient(sigma_ext(DTT, d0, delta_y, beam=beam), pas_dtt)
    idx_dtt = np.searchsorted(DTT, dtt)
    sigma_grad_D0 = np.gradient(sigma_ext(dtt, D0, delta_y, beam=beam), pas_d0)
    idx_d0 = np.searchsorted(D0, d0)
    sigma_grad_Delta_y = np.gradient(sigma_ext(dtt, d0, Delta_y, beam=beam), pas_delta_y)
    idx_delta_y = np.searchsorted(Delta_y, delta_y)

    #print(idx_delta_y, idx_d0, idx_dtt)

    #Propagate uncertainty
    incert = np.sqrt((sigma_grad_DTT[idx_dtt]*erreur_dtt)**2 + (sigma_grad_D0[idx_d0]*erreur_d0)**2 + (sigma_grad_Delta_y[idx_delta_y]*erreur_delta_y)**2)
    return(incert)




def get_FWHM(lda, Lambda_calib, FWHM_calib, fit_fun='lin', method = 'fine') :
    """_summary_

    Args:
        lda (_type_): _description_
        Lambda_calib (_type_): _description_
        FWHM_calib (_type_): _description_
        fit_fun (str, optional) : fitting function. 'lin' is linear, 'aff' is affine.
        method (str, optional): 'coarse' : use a 1D fit. 'fine' : use each points of the calibration : takes into account small tendances in FWHM(lambda).

    """

    if fit_fun == 'lin' :
        f = lin
    if fit_fun == 'aff' :
        f = aff

    if method == 'coarse' :
            
            
        if fit_fun == 'lin' :
            f = lin
        if fit_fun == 'aff' :
            f = aff

        best_a, pcov = curve_fit(f, Lambda_calib, FWHM_calib)
        fwhm = f(lda, *best_a)
    if method == 'fine' :
        index = np.searchsorted(Lambda_calib, lda)

        if lda in Lambda_calib :
            # print('trouvé dans la liste')
            # print('lda calib correspondant = {}'.format(Lambda_calib[index]))
            fwhm = FWHM_calib[index]
        
        elif index==0 :
            print('sous lda min')
            fwhm = FWHM_calib[index]

        elif index==len(Lambda_calib) :
            #print('au dessus de lda max')
            fwhm = FWHM_calib[index-1]

        else :
            # print('Entre les deux !')
            # print('lda calib correspondant = {}'.format(Lambda_calib[index-1]))
            #Between 2 calibrated wavelengths : we take the convex combination of the 2 calibrated wavelengths
            try :
                lda_p = Lambda_calib[index]
                lda_m = Lambda_calib[index-1]

                fwhm_p = FWHM_calib[index]
                fwhm_m = FWHM_calib[index-1]

                #print('lda_p = {}nm, lda_m = {}'.format(lda_p, lda_m))
                #print('fwhm_p = {}nm, fwhm_m = {}'.format(fwhm_p, fwhm_m))

                
                t = (lda-lda_m)/(lda_p-lda_m)



                fwhm = ((1-t)*fwhm_m + t*fwhm_p)
            except :
                print('exctept occured')
                #out of range of calibration
                fwhm = FWHM_calib[index-1]

    #print(fwhm)
    return fwhm


def sigma_ext_processing(DTT, Lambda, Lambda_calib, FWHM_calib, delta_y, method='fine') :
    """Computes the extinction section from experimental measurement of DTT and calibration.

    Args:
        DTT (list): Signal extracted from SMS measurement
        Lambda (list): list of wavelegnths calculated
        Lambda_calib (list): list of wavelength used for calibration
        FWHM_calib (list): list of fwhm extracted from calibration
        delta_y (float): modulation amplitude
        method (str, optional): 'coarse' : use a 1D fit. 'fine' : use each points of the calibration : takes into account small tendances in FWHM(lambda).


    Return:
        Sigma (list) : list of sigma_ext
        Sigma_incert (list) : list of incertainty on sigma_ext.
    """

    Sigma = []
    Sigma_incert = []

    for i in range(len(Lambda)) :
        print('------------------------------')

        lda = Lambda[i]*1e9
        #print(lda)
        
        fwhm = get_FWHM(lda, Lambda_calib, FWHM_calib, fit_fun='lin', method=method)
        
        sigma = sigma_ext(DTT[i], fwhm*1e-9, delta_y)
        incert_sigma = incert_sigma_ext(DTT[i], fwhm*1e-9, delta_y)


        
        print('lambda = {}'.format(lda))
        print('fwhm = {}'.format(fwhm))
        print('------------------------------')

        Sigma.append(sigma)
        Sigma_incert.append(incert_sigma)

    return(np.array(Sigma), np.array(Sigma_incert))


def corr_DTT(dtt, Voff, V) :
    """Corrects DTT according to the photodiode dark current.

    Args:
        dtt (float or array): raw dtt observed
        Voff (float or array): photodiode tension without light.
        V (float or array): photodiode tension during measurments.
    """
    return(dtt*V/(V-Voff))

def read_data(dir_name,Source_list,other_title='no', force_recomputation=False) :
    """read data from an excel file

    Args:
        dir_name (str): directory containing a .ods with experimental data.
        

    Returns:
        Lambda (dict) : wavelength with a key for every page of the excel
        DTT (dict) : measured DTT with a key for every page of the excel
        Voff (dict) : photodiode offset tension with a key for every page of the excel
        V (dict) : photodiode tension with a key for every page of the excel
    """


    Lambda = {}
    DTT = {}
    Voff = {}
    V = {}

    DTT_corr = {}
    Sigma = {}
    Sigma_err = {}

    if other_title == 'no' :
        sheet_name = dir_name
    else :
        sheet_name = other_title

    if not (force_recomputation) :
        try :
            
            df_sheet_all = pd.read_excel(sheet_name + '_treated.ods', sheet_name=None)

            is_treated = True

            for key in Source_list :
                try :
                    data = df_sheet_all[key].__array__()

                    Lambda[key] = np.array(data[:,0])*1e-9
                    DTT[key] = np.array(data[:,1])*1e-3
                    Voff[key] = np.array(data[:,2])*1e-3
                    V[key] = np.array(data[:,3])*1e-3

                    DTT_corr[key] = np.array(data[:,4])*1e-3

                    Sigma[key] = np.array(data[:,5])*1e-18
                    Sigma_err[key] = np.array(data[:,6])*1e-18


        
                except :
                    print('erreur importation sur' + sheet_name)
                    pass

        except :
            is_treated = False

            df_sheet_all = pd.read_excel(sheet_name + '.ods', sheet_name=None)
            for key in Source_list :
                try :
                    data = df_sheet_all[key].__array__()

                    Lambda[key] = np.array(data[:,0])*1e-9
                    DTT[key] = np.array(data[:,1])*1e-3
                    Voff[key] = np.array(data[:,2])*1e-3
                    V[key] = np.array(data[:,3])*1e-3
                except :
                    print('erreur importation sur ' + dir_name)
                    pass

    else :
        is_treated = False

        df_sheet_all = pd.read_excel(sheet_name + '.ods', sheet_name=None)
        for key in ['OPO', 'SHG'] :
            try :
                data = df_sheet_all[key].__array__()

                Lambda[key] = np.array(data[:,0])*1e-9
                DTT[key] = np.array(data[:,1])*1e-3
                Voff[key] = np.array(data[:,2])*1e-3
                V[key] = np.array(data[:,3])*1e-3
            except :
                print('erreur importation sur' + dir_name)
                pass


    


    return df_sheet_all, Lambda, DTT, Voff, V, DTT_corr, Sigma, Sigma_err, is_treated
    

def save_treated_data_excel(fname, data, Source_list) :

    file = data['file']

    for source_key in Source_list :

        file[source_key]['DTT_corr (1e-3)'] = data['DTT'][source_key]*1e3
        file[source_key]['Sigma (nm2)'] = data['Sigma'][source_key]*1e18
        file[source_key]['Sigma_err (nm2)'] = data['Sigma_err'][source_key]*1e18

 

        
    with pd.ExcelWriter(fname) as writer:

        for source_key in Source_list :
                
            # writing to the 'Employee' sheet
            file[source_key].to_excel(writer, sheet_name=source_key, index=False)



def Spectra_computation(dir_root, dir_name, Lambda_calib, FWHM_calib, delta_y, Source_list,other_title='no', force_recomputation=False, spectra_title = '', method='fine') :
    """_summary_

    Args:
        dir_root (_type_): _description_
        dir_name (_type_): _description_
        Lambda_calib (_type_): _description_
        FWHM_calib (_type_): _description_
        delta_y (_type_): _description_
        method (str, optional): 'coarse' : use a 1D fit. 'fine' : use each points of the calibration : takes into account small tendances in FWHM(lambda).


    Returns:
        _type_: _description_
    """

    os.chdir('\\'.join((dir_root, dir_name)))
    
    f, Lambda, DTT, Voff, V, DTT_corr, Sigma, Sigma_err, is_treated = read_data(dir_name,Source_list, other_title=other_title, force_recomputation = force_recomputation)
    
    if is_treated and not(force_recomputation):
        print(dir_name + ' already treated')

    if not(is_treated) or force_recomputation :

        for key in Lambda :
            print(key)
            
            DTT_corr[key] = corr_DTT(DTT[key], Voff[key], V[key])
            Sigma[key], Sigma_err[key] = sigma_ext_processing(DTT_corr[key], Lambda[key], Lambda_calib[key], FWHM_calib[key], delta_y, method=method)

    data = {}



    if other_title == 'no' :
        sheet_name = dir_name
    else :
        sheet_name = other_title
    data['file'] = f

    if spectra_title != '' :
        data['title'] = spectra_title
    else :
        data['title'] = sheet_name
    
    data['dir_root'] = dir_root
    data['dir_name'] = dir_name
    data['fname'] = sheet_name
    data['DTT'] = DTT_corr
    data['Lambda'] = Lambda
    data['Voff'] = Voff
    data['V'] = V
    data['Sigma'] = Sigma
    data['Sigma_err'] = Sigma_err
    data['path'] = '\\'.join((dir_root, dir_name))
    data['Source_list']  = Source_list


    
    if not(is_treated) :
        save_treated_data_excel(sheet_name + '_treated.ods', data, Source_list)

    

    return data






#################################################################################
####### Data fitting ############################################################
#################################################################################

def Lorentz(x, xc, w, A, y0) :
    """Lorentzian resonance

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        A (_type_): height and intensity
        y0 (_type_): offset
    """
    res = y0 + A*w*w/(4*(x-xc)**2 + w**2)

    return res

def Lorentz_no_offset(x, xc, w, A) :
    """Lorentzian resonance without offset. A is the maximum of the function.

    Args:
        x (_type_): energy
        xc (_type_): central energy
        w (_type_): FWHM
        A (_type_): height and intensity
    """
    res = A*w*w*(1/(4*(x-xc)**2 + w**2))

    
    return res


def get_energy_spectra(data) :
    """organise the DTT treated data in a simple form, returning energy, sigma_ext and its incertainty.

    Args:
        data (dic): dictionnary of treated DTT experimental data.

    Return:
        E_sorted (1d array) : energy in eV, sorted
        Sigma_sorted (1d array) : Extinction cross_section in nm2, sorted with E,
        Sigma_err_sorted (1d array) : error on Sigma in nm2, sorted with E
    """
    
    E = 1240*1e-9/(np.concatenate([data['Lambda'][key] for key in data['Lambda']]) + 1e-30)
    

    Sigma = np.concatenate([data['Sigma'][key] for key in data['Sigma']])
    Sigma_err = np.concatenate([data['Sigma_err'][key] for key in data['Sigma_err']])

    
    ##Sorting data by energy

    index_sorted = E.argsort()

    E_sorted = E[index_sorted]
    Sigma_sorted = Sigma[index_sorted]
    Sigma_err_sorted = Sigma_err[index_sorted]

    return E_sorted.astype(float), Sigma_sorted.astype(float), Sigma_err_sorted.astype(float)




def Lorentz_fit_energy_spectra(data) :
    """Makes a Lorentzian fit on the energy spectra of data

    Args:
        data (dic): Spectra data
        p0 (list): initial guess parameters
    
    Returns :
        res_fit (dic): results of fit with all the fitted parameters. 'converged' is set to False if the fit was impossible.
    """

    E, S_ext, S_ext_err = get_energy_spectra(data)
    E, S_ext, S_ext_err = E.astype(float), S_ext.astype(float), S_ext_err.astype(float) #fix a type compatibility for curve_fit
    

    
    res_fit = {}

    #initial guess [E_c, w, A]
    p0 = np.array([E[S_ext.argmax()], 0.1, 30]) 

   
    try :

        #res, pcov = curve_fit(Lorentz_no_offset, E, S_ext*1e15, p0 = p0, sigma = np.array(S_ext_err*1e15), absolute_sigma=True)
        res, pcov = curve_fit(Lorentz_no_offset, E, S_ext*1e15, p0 = p0)
        p_sigma = np.sqrt(np.diag(pcov))


        res_fit['E_c'] = res[0]
        res_fit['w'] = res[1]
        res_fit['A'] = res[2]*1e-15

        res_fit['E_c_err'] = p_sigma[0]
        res_fit['w_err'] = p_sigma[1]
        res_fit['A_err'] = p_sigma[2]*1e-15
        res_fit['converged'] = True
        

    except :
        res_fit['E_c'] = p0[0]
        res_fit['w'] = p0[1]
        res_fit['A'] = p0[2]*1e-15

        res_fit['E_c_err'] = 0
        res_fit['w_err'] = 0
        res_fit['A_err'] = 0
        res_fit['converged'] = False
        


    E_left = res_fit['E_c'] - res_fit['w']/2
    E_right = res_fit['E_c'] + res_fit['w']/2

    Lda_left = 1240/E_left
    Lda_right = 1240/E_right

    res_fit["E_left"] = res_fit['E_c'] - res_fit['w']/2
    res_fit["E_right"] = res_fit['E_c'] + res_fit['w']/2

    res_fit["Lda_left"] = 1240/E_left
    res_fit["Lda_right"] = 1240/E_right



    res_fit['Delta_lda'] = Lda_left-Lda_right
    res_fit['Lda_c'] = 1240/res_fit['E_c']


    res_fit['integral'] = res_fit['A']*res_fit['w'] * np.pi/2
    res_fit['integral_err'] = res_fit['integral'] * np.sqrt((res_fit['A_err']/res_fit['A'])**2 + (res_fit['w_err']/res_fit['w'])**2 )

    
    res_fit['Delta_lda_err'] = res_fit['Delta_lda']*res_fit['w_err']/res_fit['w']
    res_fit['Lda_c_err'] = res_fit['Lda_c']*res_fit['E_c_err']/res_fit['E_c']


    return(res_fit)



                