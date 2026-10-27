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

class P_and_P_dataset() :
    def __init__(self, dir_root, dir_name) :

        self.name = dir_root + '\\' + dir_name
        self.dir_root = dir_root
        self.dir_name = dir_name

        self.create_all_traces_dataset()


        

    def create_all_traces_dataset(self) :
        """Creates a dictionnary of the dataset, each key is the single trace file name, and is linked to a dictionnary 
        with t_i, t_f, t_step, t and DTT array.

        Time is in picosecond
        """

        files = os.listdir(self.dir_root + '\\' + self.dir_name)


        self.All_traces_data = {}

        compt=0

        for i in range(len(files)) :
            
            key = files[i]
            #print(key.split('_')[-2][0:4])
            if key.split('_')[-2][0:4] == 'Scan' : #remove the mean files
                #print('ok')
                compt +=1
                self.All_traces_data[key] = {}

                lines = open( self.dir_root + '\\' + self.dir_name + '\\' + files[i],'r').readlines()

                self.All_traces_data[key]["t_i"] = float(lines[2].strip())
                self.All_traces_data[key]["t_f"] = float(lines[3].strip())
                self.All_traces_data[key]["t_step"] = float(lines[4].strip())

                
                nb_t_values = int(lines[5].strip()) 

                self.All_traces_data[key]["t"] = []
                self.All_traces_data[key]["DTT"] = []

                for j in range(nb_t_values):
                    a,b = lines[10+j].strip().split(',')
                    self.All_traces_data[key]["t"].append(float(a))
                    self.All_traces_data[key]["DTT"].append(float(b))

                self.All_traces_data[key]["DTT"]  = np.array(self.All_traces_data[key]["DTT"])
                self.All_traces_data[key]["t"]  = np.array(self.All_traces_data[key]["t"])
        self.n_traces = compt
        self.t_step = float(lines[4].strip())

    def synchronize_data(self) :
        #print([self.All_traces_data[key]['t_step'] for key in self.All_traces_data])
        #Check if all time steps are the same
        e = {self.All_traces_data[key]['t_step'] for key in self.All_traces_data} 

        if len(e)==1:
            key0 = next(iter(self.All_traces_data))
            self.dt = self.All_traces_data[key0]['t_step']
        else:
            self.dt = 1
            raise ValueError('The time steps are different between time traces')
        
        #Get the first and last time step

        t_min = np.min([self.All_traces_data[key]['t_i'] for key in self.All_traces_data])
        t_max = np.max([self.All_traces_data[key]['t_f'] for key in self.All_traces_data])

        n = int((t_max-t_min)/self.dt) +1

        #Make the mean trace

        self.t = np.linspace(t_min, t_max, n)
        DTT_all = [[] for i in range(n)]
        

        for key in self.All_traces_data :
            trace = self.All_traces_data[key]

            t = trace['t']
            DTT = trace['DTT']
            
            for j in range(len(t)) :
                t_index = np.searchsorted(self.t, t[j])
                DTT_all[t_index].append(DTT[j])

        
        self.DTT_tot = np.array([np.mean(x) for x in DTT_all])




    def find_delay_zero(self, n_start=0, n_stop=-1, force_n=-1) :
        """Force n forces the choice of delay 0

        Args:
            n_start (int, optional): _description_. Defaults to 0.
            n_stop (int, optional): _description_. Defaults to -1.
            force_n (int, optional): _description_. Defaults to -1.
        """
        
        max_found_value = max(np.abs(self.DTT_tot[n_start:n_stop]))
        n_0 = int(np.where(np.abs(self.DTT_tot)==max_found_value)[0])

        if force_n != -1 :
            n_0 = force_n
        print(n_0)
        print(self.DTT_tot[n_0])
        print(max_found_value)
        self.t0 = self.t[n_0]
        self.t0_index = n_0

        self.delay_tot = self.t - self.t0

    def crop_delay_and_remove_offset(self, idx_crop) :

        self.delay = self.delay_tot[idx_crop:]
        self.DTT = self.DTT_tot[idx_crop:] - self.mean_start

        
        
        
    def fit_thermal_signal(self, p0, Bounds_high, Bounds_low) :

        fit_func = [ExpDec1, ExpDec2, ExpDec3, ExpDec4]
        res1, res2, res3, res4 = fit_signal_exp(self.DTT, self.delay, p0, Bounds_low, Bounds_high, fit_func)

        popt1, incert_1, err_1 = res1
        popt2, incert_2, err_2 = res2
        popt3, incert_3, err_3 = res3
        popt4, incert_4, err_4 = res4

        self.thermal_fit = {
            '1 exp' : {},
            '2 exp' : {},
            '3 exp' : {},
            '4 exp' : {}
        }

        self.thermal_fit['1 exp'] = {
            'coeffs' : popt1,
            'coeffs_err' : incert_1,
            'residual' : err_1 
        }

        self.thermal_fit['2 exp'] = {
            'coeffs' : popt2,
            'coeffs_err' : incert_2,
            'residual' : err_2 
        }

        self.thermal_fit['3 exp'] = {
            'coeffs' : popt3,
            'coeffs_err' : incert_3,
            'residual' : err_3 
        }

        self.thermal_fit['4 exp'] = {
            'coeffs' : popt4,
            'coeffs_err' : incert_4,
            'residual' : err_4 
        }

        

        self.thermal_fit['1 exp']['A1'] = self.thermal_fit['1 exp']['coeffs'][0]
        self.thermal_fit['1 exp']['tau1'] = self.thermal_fit['1 exp']['coeffs'][1]
        self.thermal_fit['1 exp']['A1_err'] = self.thermal_fit['1 exp']['coeffs_err'][0]
        self.thermal_fit['1 exp']['tau1_err'] = self.thermal_fit['1 exp']['coeffs_err'][1]


        self.thermal_fit['2 exp']['A1'] = self.thermal_fit['2 exp']['coeffs'][0]
        self.thermal_fit['2 exp']['tau1'] = self.thermal_fit['2 exp']['coeffs'][1]
        self.thermal_fit['2 exp']['A1_err'] = self.thermal_fit['2 exp']['coeffs_err'][0]
        self.thermal_fit['2 exp']['tau1_err'] = self.thermal_fit['2 exp']['coeffs_err'][1]
        self.thermal_fit['2 exp']['A2'] = self.thermal_fit['2 exp']['coeffs'][2]
        self.thermal_fit['2 exp']['tau2'] = self.thermal_fit['2 exp']['coeffs'][3]
        self.thermal_fit['2 exp']['A2_err'] = self.thermal_fit['2 exp']['coeffs_err'][2]
        self.thermal_fit['2 exp']['tau2_err'] = self.thermal_fit['2 exp']['coeffs_err'][3]

        self.thermal_fit['3 exp']['A1'] = self.thermal_fit['3 exp']['coeffs'][0]
        self.thermal_fit['3 exp']['tau1'] = self.thermal_fit['3 exp']['coeffs'][1]
        self.thermal_fit['3 exp']['A1_err'] = self.thermal_fit['3 exp']['coeffs_err'][0]
        self.thermal_fit['3 exp']['tau1_err'] = self.thermal_fit['3 exp']['coeffs_err'][1]
        self.thermal_fit['3 exp']['A2'] = self.thermal_fit['3 exp']['coeffs'][2]
        self.thermal_fit['3 exp']['tau2'] = self.thermal_fit['3 exp']['coeffs'][3]
        self.thermal_fit['3 exp']['A2_err'] = self.thermal_fit['3 exp']['coeffs_err'][2]
        self.thermal_fit['3 exp']['tau2_err'] = self.thermal_fit['3 exp']['coeffs_err'][3]
        self.thermal_fit['3 exp']['A3'] = self.thermal_fit['3 exp']['coeffs'][4]
        self.thermal_fit['3 exp']['tau3'] = self.thermal_fit['3 exp']['coeffs'][5]
        self.thermal_fit['3 exp']['A3_err'] = self.thermal_fit['3 exp']['coeffs_err'][4]
        self.thermal_fit['3 exp']['tau3_err'] = self.thermal_fit['3 exp']['coeffs_err'][5]


        self.thermal_fit['4 exp']['A1'] = self.thermal_fit['4 exp']['coeffs'][0]
        self.thermal_fit['4 exp']['tau1'] = self.thermal_fit['4 exp']['coeffs'][1]
        self.thermal_fit['4 exp']['A1_err'] = self.thermal_fit['4 exp']['coeffs_err'][0]
        self.thermal_fit['4 exp']['tau1_err'] = self.thermal_fit['4 exp']['coeffs_err'][1]
        self.thermal_fit['4 exp']['A2'] = self.thermal_fit['4 exp']['coeffs'][2]
        self.thermal_fit['4 exp']['tau2'] = self.thermal_fit['4 exp']['coeffs'][3]
        self.thermal_fit['4 exp']['A2_err'] = self.thermal_fit['4 exp']['coeffs_err'][2]
        self.thermal_fit['4 exp']['tau2_err'] = self.thermal_fit['4 exp']['coeffs_err'][3]
        self.thermal_fit['4 exp']['A3'] = self.thermal_fit['4 exp']['coeffs'][4]
        self.thermal_fit['4 exp']['tau3'] = self.thermal_fit['4 exp']['coeffs'][5]
        self.thermal_fit['4 exp']['A3_err'] = self.thermal_fit['4 exp']['coeffs_err'][4]
        self.thermal_fit['4 exp']['tau3_err'] = self.thermal_fit['4 exp']['coeffs_err'][5]
        self.thermal_fit['4 exp']['A4'] = self.thermal_fit['4 exp']['coeffs'][6]
        self.thermal_fit['4 exp']['tau4'] = self.thermal_fit['4 exp']['coeffs'][7]
        self.thermal_fit['4 exp']['A4_err'] = self.thermal_fit['4 exp']['coeffs_err'][6]
        self.thermal_fit['4 exp']['tau4_err'] = self.thermal_fit['4 exp']['coeffs_err'][7]

        



    def remove_thermic_part(self, Trace_thermic, n_exp, n_start_alt=0, n_stop_alt=-1) :
        """Remove the thermal fit to the DTT data.

        Args:
            Trace_thermic (PaP_dataset): PaP_dataset used to do the thermal fit.
            n_exp (int): number of exponential used for the fit.
        """

        func = globals()['ExpDec' + str(n_exp)]
        
        exp_key = str(n_exp) + ' exp'

        self.DTT_acoustic = self.DTT - func(self.delay,*Trace_thermic.thermal_fit[exp_key]['coeffs'])
        pass

    def study_tf(self, signal, time, dt, height, threshold) :
        """make a dictionnary containing TF information.

        Args:
            signal (_type_): _description_
            time (_type_): _description_
            dt (_type_): _description_
            height (_type_): _description_
            threshold (_type_): _description_
        """

        dic = {}

        dic['freq'], dic['df'], dic['amp'], dic['phase'] =  make_tf(signal, time, dt)
        dic['peaks'], dic['width'], dic['left'], dic['right'] = compute_peaks(dic['amp']**2, dic['df'], height=height, threshold=threshold)
        return(dic)

    def fit_acoustic(self, p0, Bounds_high, Bounds_low) :
        
        fit_func = [Osc1, Osc2, Osc3, Osc4]
        res1, res2, res3, res4 = fit_signal_osc(self.DTT_acoustic, self.delay, p0, Bounds_low, Bounds_high, fit_func)
        
        popt1, incert_1, err_1 = res1
        popt2, incert_2, err_2 = res2
        popt3, incert_3, err_3 = res3
        popt4, incert_4, err_4 = res4


        self.acoustic_fit = {
            '1 modes' : {},
            '2 modes' : {},
            '3 modes' : {},
            '4 modes' : {}
        }

        Numbers = ['1', '2', '3', '4']

        

        for n in Numbers :

            #print(locals()['popt' + n])
            self.acoustic_fit[n + ' modes']['coeffs'] = locals()['popt' + n]
            self.acoustic_fit[n + ' modes']['coeffs_err'] = locals()['incert_' + n]
            self.acoustic_fit[n + ' modes']['residual'] = locals()['err_' + n]


            for j in range(int(n)) :

                

                # Takes into account the offset from the electron peak to compute the phase phi.
                index_coeff = str(j+1)


                self.acoustic_fit[n + ' modes']['A' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 0]
                self.acoustic_fit[n + ' modes']['tau' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 1]
                self.acoustic_fit[n + ' modes']['T' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 2]
                self.acoustic_fit[n + ' modes']['phi' + index_coeff] = self.acoustic_fit[n + ' modes']['coeffs'][4*j + 3]
                self.acoustic_fit[n + ' modes']['Q' + index_coeff] = np.pi*self.acoustic_fit[n + ' modes']['tau' + index_coeff]/self.acoustic_fit[n + ' modes']['T' + index_coeff]
                
                self.acoustic_fit[n + ' modes']['A' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 0]
                self.acoustic_fit[n + ' modes']['tau' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 1]
                self.acoustic_fit[n + ' modes']['T' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 2]
                self.acoustic_fit[n + ' modes']['phi' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['coeffs_err'][4*j + 3]
                self.acoustic_fit[n + ' modes']['Q' + index_coeff + '_err'] = self.acoustic_fit[n + ' modes']['Q' + index_coeff]*np.sqrt((self.acoustic_fit[n + ' modes']['tau' + index_coeff+'_err']/self.acoustic_fit[n + ' modes']['tau' + index_coeff])**2 + (self.acoustic_fit[n + ' modes']['T' + index_coeff+'_err']/self.acoustic_fit[n + ' modes']['T' + index_coeff])**2)


# Saving and loading data
def save_PaP_dataset(Trace, dir_save_treated_data, filename) :
    """Save the Trace object with name fname in the treated data directory.

    Args:
        Trace (P_and_P_dataset): The P_and_P_dataset object that we want to saave
        dir_save_treated_data (str) : the path where we want to save data. If the directory doesn't exist, this function will create it.
        filename (str): The name we want to give to it
    """


    if not(os.path.exists(dir_save_treated_data)) :
        os.mkdir(dir_save_treated_data)

    os.chdir(dir_save_treated_data)

    with open(filename + '.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(Trace, outp, pickle.HIGHEST_PROTOCOL)

    
    pass

def get_PaP_dataset(dir_path, filename) :
    """Get a PaP_dataset object from the path.

    Args:
        dir_path (str): path
        filename (str): filename

    Returns:
        Trace: The PaP_dataset_object.
    """

    try :
        os.chdir(dir_path)
        with open(filename + '.pkl', 'rb') as file:
            Trace = pickle.load(file)
    
            return Trace
       
    except :
        print("Error : there is no treated data")
        pass
    
    
    
   

### Thermal fitting

def ExpDec4(x,A1,t1,A2,t2,A3,t3,A4,t4):
    return  A1*np.exp(-x/t1) + A2*np.exp(-x/t2)+ A3*np.exp(-x/t3) + A4*np.exp(-x/t4)
def ExpDec3(x,A1,t1,A2,t2,A3,t3):
    return  A1*np.exp(-x/t1) + A2*np.exp(-x/t2)+ A3*np.exp(-x/t3)
def ExpDec2(x,A1,t1,A2,t2):
    return  A1*np.exp(-x/t1) + A2*np.exp(-x/t2)
def ExpDec1(t,A1,t1):
    return  A1*np.exp(-t/t1)
pass


def fit_signal_exp(signal_to_fit, delay, p0, Bounds_low, Bounds_high, fit_funcs) :
    """make the exponential fit for the function

    Args:
        signal_to_fit (1D array): signal
        delay (1D array): time
        p01 (tuple): fitting parameter for the 1 exp fit
        p02 (tuple): fitting parameter for the 2 exp fit
        p03 (tuple): fitting parameter for the 3 exp fit
        p04 (tuple): fitting parameter for the 4 exp fit
        Bounds_low (list): list of the 6 bounds
        Bounds_high (list): list of the 6 bounds
        fit_funcs (list) : list of the 4 fitting functions.

    Returns:
        resi (tuple): each res is a tuple (popt, pcov, err)
    """

    ExpDec1, ExpDec2, ExpDec3, ExpDec4 = fit_funcs[0], fit_funcs[1], fit_funcs[2], fit_funcs[3]

    # Curve fitting
    popt1, pcov1 = curve_fit(ExpDec1, delay, signal_to_fit, p0=p0[0:2])
    popt2, pcov2 = curve_fit(ExpDec2, delay, signal_to_fit, p0=p0[0:4], bounds=(Bounds_low[0:4], Bounds_high[0:4]))                         
    popt3, pcov3 = curve_fit(ExpDec3, delay, signal_to_fit, p0=p0[0:6], bounds=(Bounds_low[0:6], Bounds_high[0:6]))
    popt4, pcov4 = curve_fit(ExpDec4, delay, signal_to_fit, p0=p0, bounds=(Bounds_low, Bounds_high))

    incert_1 = np.sqrt(np.diag(pcov1))
    incert_2 = np.sqrt(np.diag(pcov2))
    incert_3 = np.sqrt(np.diag(pcov3))
    incert_4 = np.sqrt(np.diag(pcov4))

    # Residual errors
    err_1 = (root_mean_squared_error(signal_to_fit,ExpDec1(delay,*popt1))/np.abs(np.max(signal_to_fit)))**2
    err_2 = (root_mean_squared_error(signal_to_fit,ExpDec2(delay,*popt2))/np.abs(np.max(signal_to_fit)))**2
    err_3 = (root_mean_squared_error(signal_to_fit,ExpDec3(delay,*popt3))/np.abs(np.max(signal_to_fit)))**2
    err_4 = (root_mean_squared_error(signal_to_fit,ExpDec4(delay,*popt4))/np.abs(np.max(signal_to_fit)))**2

    res1 = (popt1, incert_1, err_1)
    res2 = (popt2, incert_2, err_2)
    res3 = (popt3, incert_3, err_3)
    res4 = (popt4, incert_4, err_4)


    return (res1, res2, res3, res4)



### Acoustic fitting and treatment


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


def fit_signal_osc(signal_to_fit, delay, p0, Bounds_low, Bounds_high, fit_funcs) :
    """make the damped oscillation fit for the function

    Args:
        signal_to_fit (1D array): signal
        delay (1D array): time
        p0 (tuple): initial guess for every mode.
        
        Bounds_low (list): list of the 6 bounds
        Bounds_high (list): list of the 6 bounds
        fit_funcs (list) : list of the 4 fitting functions.

    Returns:
        res (tuple): each res is a tuple (popt, pcov, err)
    """

    Osc1, Osc2, Osc3, Osc4 = fit_funcs[0], fit_funcs[1], fit_funcs[2], fit_funcs[3]

    # Curve fitting
    popt1, pcov1 = curve_fit(Osc1, delay, signal_to_fit, p0=p0[0:4])
    popt2, pcov2 = curve_fit(Osc2, delay, signal_to_fit, p0=p0[0:8], bounds=(Bounds_low[0:8], Bounds_high[0:8]))                         
    popt3, pcov3 = curve_fit(Osc3, delay, signal_to_fit, p0=p0[0:12], bounds=(Bounds_low[0:12], Bounds_high[0:12]))
    popt4, pcov4 = curve_fit(Osc4, delay, signal_to_fit, p0=p0, bounds=(Bounds_low, Bounds_high))

    incert_1 = np.sqrt(np.diag(pcov1))
    incert_2 = np.sqrt(np.diag(pcov2))
    incert_3 = np.sqrt(np.diag(pcov3))
    incert_4 = np.sqrt(np.diag(pcov4))

    # Residual errors
    err_1 = (root_mean_squared_error(signal_to_fit,Osc1(delay,*popt1))/np.abs(np.max(signal_to_fit)))**2
    err_2 = (root_mean_squared_error(signal_to_fit,Osc2(delay,*popt2))/np.abs(np.max(signal_to_fit)))**2
    err_3 = (root_mean_squared_error(signal_to_fit,Osc3(delay,*popt3))/np.abs(np.max(signal_to_fit)))**2
    err_4 = (root_mean_squared_error(signal_to_fit,Osc4(delay,*popt4))/np.abs(np.max(signal_to_fit)))**2

    res1 = (popt1, incert_1, err_1)
    res2 = (popt2, incert_2, err_2)
    res3 = (popt3, incert_3, err_3)
    res4 = (popt4, incert_4, err_4)


    return (res1, res2, res3, res4)





### Spectral studies


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

def make_Lorentzian_fit(tf_data, f_low, f_high, p0) :

    x_data = tf_data['freq'][np.searchsorted(tf_data['freq'], f_low):np.searchsorted(tf_data['freq'], f_high)]
    y_data = tf_data['amp'][np.searchsorted(tf_data['freq'], f_low):np.searchsorted(tf_data['freq'], f_high)]**2

    popt, pcov = curve_fit(Lorentz_no_offset, x_data, y_data, p0=p0)
    incert = np.sqrt(np.diag(pcov))
    err = (root_mean_squared_error(y_data,Lorentz_no_offset(x_data,*popt))/np.abs(np.max(y_data)))**2


    res_dic = {}

    pass

def make_tf(signal, time, dt) :
    
    sp = np.fft.rfft(signal)
    freq = 1000*np.fft.rfftfreq(len(time),dt)  #frequency in GHz
    df = freq[1]-freq[0]


    sp_amp = np.absolute(sp)
    sp_phase = np.angle(sp)

    return(freq, df, sp_amp, sp_phase)


def compute_peaks(signal,df, height=0.3e-3, threshold=0.1e-3) :

    peaks, peak_dic = find_peaks(signal, height=height, threshold=threshold)

    width, width_height, left, right = peak_widths(signal, peaks, rel_height=0.5 )


    try :
        width = np.array(width)*df
        i_peak = peaks[0]
        

    except :
        pass

    return(peaks, width, left, right)




def padd_signal(n_prolong, signal, time) :
    """Created a prolongated signal with 0 values in order to have a better resolution on TF signal

    Args:
        n_prolong (int): number of points to add
        signal (1D array): the signal to be prolongated
        delay (1D array): the time axis

    Returns:
        signal_prolongated : prolongtaed signal array
        time_prolongated : prolongated time array
    """

    n_prolong
    dt = time[1] - time[0]

    # t_ini = np.arange(0, n_prolong//2)*dt
    # t_mid = time + t_ini[-1] + dt
    # t_end = t_mid[-1] + np.arange(0, n_prolong//2)*dt + dt

    # signal_prolongated = np.concatenate((np.zeros(n_prolong//2), signal, np.zeros(n_prolong//2)))
    # time_prolongated = np.concatenate((t_ini, t_mid, t_end))

    signal_prolongated = np.concatenate(( signal, np.zeros(n_prolong)))
    time_prolongated = np.concatenate(( time, time[-1] + np.arange(0, n_prolong)*dt + dt))



    return signal_prolongated, time_prolongated





def Lorentz_fit_tf(tf_data, f_low, f_high) :
    """Makes a Lorentzian fit on the energy spectra of data

    Args:
        tf_data (dic): tf_data to be fitted
    
    Returns :
        res_fit (dic): results of fit with all the fitted parameters. 'converged' is set to False if the fit was impossible.
    """

    
    res_fit = {}
    
    freq_raw = tf_data['freq']

    idx_low = np.searchsorted(freq_raw, f_low)
    idx_high = np.searchsorted(freq_raw, f_high)

    Energy = tf_data['amp'][idx_low:idx_high]**2
    freq = tf_data['freq'][idx_low:idx_high]

    

    #initial guess [f_c, w, A]
    p0 = np.array([freq[Energy.argmax()], 2, Energy.max()]) 

    

    try :

        res, pcov = curve_fit(Lorentz_no_offset, freq, Energy, p0= p0,)
        p_sigma = np.sqrt(np.diag(pcov))

        res_fit['coeffs'] = res
        res_fit['f_c'] = res[0]
        res_fit['w'] = res[1]
        res_fit['A'] = res[2]

        res_fit['f_c_err'] = p_sigma[0]
        res_fit['w_err'] = p_sigma[1]
        res_fit['A_err'] = p_sigma[2]
        res_fit['converged'] = True
        

    except :
        res_fit['coeffs'] = p0
        res_fit['f_c'] = p0[0]
        res_fit['w'] = p0[1]
        res_fit['A'] = p0[2]

        res_fit['f_c_err'] = 0
        res_fit['w_err'] = 0
        res_fit['A_err'] = 0
        res_fit['converged'] = False
        


    f_left = res_fit['f_c'] - res_fit['w']/2
    f_right = res_fit['f_c'] + res_fit['w']/2



    res_fit["f_left"] = f_left
    res_fit["f_right"] = f_right

    res_fit['Q'] = res_fit['f_c']/(res_fit['w'])

    res_fit['Q_err'] = res_fit['Q']* np.sqrt((res_fit['f_c_err']/res_fit['f_c'])**2 + (res_fit['w_err']/res_fit['w'])**2 )

    res_fit['integral'] = res_fit['A']*res_fit['w'] * np.pi/2
    res_fit['integral_err'] = res_fit['integral'] * np.sqrt((res_fit['A_err']/res_fit['A'])**2 + (res_fit['w_err']/res_fit['w'])**2 )

    return(res_fit)





###### Saving treated data


def save_all_raw_signal(Trace) :
    """Save curves datas in a .dat The data are an array with [delay (ps), Mean raw signal, Thermic fit, Raw acousit signal, Acoustic fit]

    Args:
        Trace (class PaP_dataset): A Pump_probe dataset which has been treated thermically and acoustically.
    """





    N_big = np.size(Trace.delay_tot)
    data = np.zeros(shape=(N_big, 2))



    data[:, 0] = Trace.delay_tot

    data[:, 1] = Trace.DTT_tot


    

    header = 'delay_tot_(ps) DTT_tot_(1e-5)'

    os.chdir(Trace.name + '_treated')
    filename = 'Total_raw_signal_data.dat'
    with open(filename, 'w') as file:
        file.write(header + '\n')  # Write the header line
        np.savetxt(file, data)  # Append the data




def save_signal_and_fits_curves(Trace) :
    """Save curves datas in a .dat The data are an array with [delay (ps), Mean raw signal, Thermic fit, Raw acousit signal, Acoustic fit]

    Args:
        Trace (class PaP_dataset): A Pump_probe dataset which has been treated thermically and acoustically.
    """




    delay = Trace.delay




    N = np.size(Trace.delay)
    N_big = np.size(Trace.delay_tot)
    data = np.zeros(shape=(N_big, 7))
    data[:N, 0] = delay
    data[:N, 1] = Trace.DTT


    func = globals()['ExpDec' + str(Trace.n_exp_thermic)]
        
    exp_key = str(Trace.n_exp_thermic) + ' exp'

    Trace.DTT_thermic_fit = func(Trace.delay,*Trace.thermal_fit[exp_key]['coeffs'])


    data[:N, 2] = Trace.DTT_thermic_fit
    data[:N, 3] = Trace.DTT_acoustic


    func_acoustic = globals()['Osc' + str(Trace.n_modes_acoustic)]
    plot_key = str(Trace.n_modes_acoustic) + ' modes'
    Trace.DTT_acoustic_fit = func_acoustic(Trace.delay,*Trace.acoustic_fit[plot_key]['coeffs'])

    data[:N, 4] = Trace.DTT_acoustic_fit


    data[:, 5] = Trace.delay_tot

    data[:, 6] = Trace.DTT_tot


    

    header = ' delay_(ps) global_raw_signal thermic_fit Acoustic_raw_signal Acoustic_fit delay_tot_(ps) DTT_tot_(1e-5)'


    os.chdir(Trace.name + '_treated')
    filename = 'signal_and_fit_curves.dat'
    with open(filename, 'w') as file:
        file.write(header + '\n')  # Write the header line
        np.savetxt(file, data)  # Append the data





def save_fits_coeffs_acoustic(Trace) :
    """Save curves datas in a .dat The data are an array with the fitting coefficients and the number of acoustic modes used for fit.

    Args:
        Trace (class PaP_dataset): A Pump_probe dataset which has been treated thermically and acoustically.
    """

    n_modes = Trace.n_modes_acoustic



    header_acoustic = 'Acousitc fitting coefficients ' + '\n' + \
         'Number of modes / A1 / A1_err / tau1 / tau1_err / T1 / T1_err / phi1/ phi1_err' + \
         '/ A2 / A2_err / tau2 / tau2_err / T2 / T2_err / phi2/ phi2_err' + \
         '/ A3 / A3_err / tau3 / tau3_err / T3 / T3_err / phi3/ phi3_err' + \
         '/ A4 / A4_err / tau4 / tau4_err / T4 / T4_err / phi4/ phi4_err'
    
    
    Data_acoustic = []
    Data_acoustic.append(n_modes)

    for i in range(n_modes) :
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['A' + str(i+1)])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['A' + str(i+1) + '_err'])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['tau' + str(i+1)])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['tau' + str(i+1) + '_err'])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['T' + str(i+1)])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['T' + str(i+1) + '_err'])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['phi' + str(i+1)])
        Data_acoustic.append(Trace.acoustic_fit[str(n_modes) + ' modes']['phi' + str(i+1) + '_err'])


    os.chdir(Trace.name + '_treated')

    filename = 'Acoustic fits coefficients.dat'
    with open(filename, 'w') as file:
        file.write(header_acoustic + '\n')  # Write the header line

        # Convertir la liste en une seule ligne séparée par des virgules
        file.write(','.join(map(str, Data_acoustic)) + '\n')


def save_fits_coeffs_thermic(Trace) :
    """Save curves datas in a .dat The data are an array with the fitting coefficients and the number of exponentials used for thermal fit.

    Args:
        Trace (class PaP_dataset): A Pump_probe dataset which has been treated thermically and acoustically.
    """

    n_exp = Trace.n_exp_thermic



    header_thermic = 'Themic fitting coefficients \n Number of modes / A1 / A1_err / tau1 / tau1_err / A2 / A2_err / tau2 / tau2_err / A3 / A3_err / tau3 / tau3_err / A4 /  A4_err / tau4 / tau4_err'
    
    Data_thermic = []
    Data_thermic.append(n_exp)

    for i in range(n_exp) :
        Data_thermic.append(Trace.thermal_fit[str(n_exp) + ' exp']['A' + str(i+1)])
        Data_thermic.append(Trace.thermal_fit[str(n_exp) + ' exp']['A' + str(i+1) + '_err'])
        Data_thermic.append(Trace.thermal_fit[str(n_exp) + ' exp']['tau' + str(i+1)])
        Data_thermic.append(Trace.thermal_fit[str(n_exp) + ' exp']['tau' + str(i+1) + '_err'])


    os.chdir(Trace.name + '_treated')

    filename = 'Thermic fits coefficients.dat'
    with open(filename, 'w') as file:
        file.write(header_thermic + '\n')  # Write the header line

        
        # Convertir la liste en une seule ligne séparée par des virgules
        file.write(','.join(map(str, Data_thermic)) + '\n')
       