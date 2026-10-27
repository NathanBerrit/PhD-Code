import os
import numpy as np
#import pickle
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpmath import *




def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def line_fit(x, amp, cen, wid): 
    #return amp * np.exp(-4*np.log(2)*(x-cen)**2/wid**2) #gaussian ; 'wid' is FWHM
    return amp * (2*scipy.special.j1((x-cen)/wid*1.61633*2) / ((x-cen)/wid*1.61633*2) )**2 #airy


def create_dataset(dir, V_lia, c, freq, coeff_sms) :
    """Generates a dictionnary which keys are all the file names in the directory dir.

    Args:
        dir (str): path to the directory to scan

    Return:
        Data (dict) : a dictionnary with an SMS class associated to every location.
    """
    Data = {}
    
    root, dirs, files = list(os.walk(dir))[0]

    for name in files:
        if name.endswith('.dat'):

            # Getting data
            #wl = float(name[-15:-12])
            wl = float(name.split('_')[-1][:-6])
            source = name.split('_')[-2]
            Data[name] = SMS(root = root, filename = name, wl = wl, V_lia = V_lia, c=c, freq=freq, coeff_sms = coeff_sms)
            Data[name].source = source

    return(Data)



def save_treated_data(dir_root, Data) :
    """Save all treated SMS map in a pickle object (as SMS classes). Each pkl contains the fitting information on the associated sms map.

    Args:
        dir_root (str): Root directory of studied sms maps.
        Data (Dictionnary): dictionnary containing each sms map.
    """

    ### Saving each fit result for every map
    dir_save_treated_data = dir_root + r'/Treated_SMS'
    if not(os.path.exists(dir_save_treated_data)) :
        os.mkdir(dir_save_treated_data)

    os.chdir(dir_save_treated_data)

    for key in Data :
        sms = Data[key]
        save_object(sms, sms.fname[:-4] + '.pkl')
    
    pass

class SMS() :
    def __init__(self, root, filename, wl = None, V_lia = 0.5, c=1.2, freq=1500, coeff_sms = 2.24) :

        self.fname = filename
        self.root = root
        self.path = self.root + '/' + self.fname
        
        
        date = self.fname.split('_')[0][1:]
        self.date = date
        
        tmp = np.genfromtxt(open(self.path),skip_header=2,usecols=0,unpack=True) #load header to extract metadata

        self.xmin = tmp[0]
        self.xmax = tmp[1]
        self.xstep = tmp[2]
        self.xnumber = int(tmp[3])
        self.ymin = tmp[4]
        self.ymax = tmp[5]
        self.ystep = tmp[6]
        self.ynumber = int(tmp[7])
        
        self.n_mean = tmp[8] #nimber of points used for averaging.
        self.n_scan = tmp[9] #number of scans done
        self.t_int = tmp[10] #Integration time in ms
        self.sensi_lockin = tmp[11] #Lock in sensitivity (check unit either in file or in tmp[12])

        self.wl = wl #wavelength (in nm)
        self.c = c #sensitivity of piezo modulation (in nm/mV)
        self.V_lia = V_lia # Tension of command for modulation (in V)
        self.freq = freq #modulation frequency (in Hz)
        self.coeff_sms = coeff_sms #The coefficient input in nanoscope. Theorical value is pi/sqrt(2) ~ 2.22, and each setup uses its own value.

        self.x = np.linspace(self.xmin, self.xmax, self.xnumber)
        self.y = np.linspace(self.ymin, self.ymax, self.ynumber)


        try:
            nx,ny,DTTmap = np.genfromtxt(open(self.path),skip_header=16,skip_footer=1,unpack=True,delimiter=",") #load actual map
        except ValueError:
            nx,ny,DTTmap = np.genfromtxt(open(self.path),skip_header=16,skip_footer=2,unpack=True,delimiter=",") #load actual map



        self.DTT = np.reshape(DTTmap,(self.xnumber,self.ynumber)).transpose()[::-1,:] #dt values as a 2d array ; still uncropped
        #Carefull : DTT[j, i] = DTT(x[i],y[j]). The map is horizontal

        

        self.DTT_center_cropped = self.DTT[self.ynumber//2-7:self.ynumber//2+7, :] #to avoid maximum that are not at the center of studied object
        use_crop = True

        if use_crop :
            self.amp = self.DTT_center_cropped.max() # dT/T
            self.j_max, self.i_max = np.argwhere(self.DTT==self.amp)[0]
            
            

        else :
            self.amp = self.DTT.max() # dT/T
            self.i_max,self.j_max = np.unravel_index(self.DTT.argmax(),self.DTT.shape) # the max is in x[i_max], y[j_max]
        
        

    def crop(self, dn_low=-1, dn_high=-1, dn_left=-1, dn_right=-1) :
        """Create and set a DTT_cropped around the maximum.

        Args:
            dn_low (int, optional): the number of line to keep below max Defaults to 0.
            dn_high (int, optional): the number of line to keep above max Defaults to 0.
            dn_left (int, optional): the number of columns to keep below max Defaults to 0.
            dn_right (int, optional): the number of columns to keep above max Defaults to 0.
        
        Returns:
            DTT_cropped (2D array): cropped data
            x_cropped (1D array): cropped x
            y_cropped (1D array): cropped y
        """
        
        i_max, j_max = self.i_max, self.j_max

        if dn_low==-1 :
            dn_low = j_max

        if dn_left==-1 :
            dn_left = i_max

        if dn_high==-1 :
            dn_high = np.shape(self.DTT)[0]-j_max

        if dn_right==-1 :
            dn_right = np.shape(self.DTT)[1]-i_max


        DTT_cropped = self.DTT[j_max-dn_low:j_max+dn_high,i_max-dn_left:i_max+dn_right]   #cropping the map around the maximum
        

        x_cropped = self.x[i_max-dn_left:i_max+dn_right]
        y_cropped = self.y[j_max-dn_low:j_max+dn_high]
    
        self.DTT_cropped = DTT_cropped
        self.x_cropped = x_cropped
        self.y_cropped = y_cropped

        return(DTT_cropped, x_cropped, y_cropped)
    

    def fit_FWHM_1D(self, line_fit) :
        """for every line in the cropped data, we make a fit with line_fit, and stock the datas in SMS.

        Args:
            line_fit (func): the function used to do the fit.

        Returns:
            FWHM : The largest FWHM int the selection
            max : The largest amplitude *1000.
        """

        
        n_lines = np.shape(self.DTT_cropped)[0]
        #get regression coefficient
        std_vals = np.zeros(shape=(n_lines,3)) # create array to store the std deviations of all parameters
        best_wid = np.zeros(n_lines) # create array to store the fitted values of width 
        std_wid = np.zeros(n_lines) # create array to store the std deviations of width 
        best_amp = np.zeros(n_lines) # create array to store the std deviations of amplitude
        
        best_vals = np.zeros(shape=(n_lines,3))

        for line_index in range(n_lines) :
            
            
            line_DTT = self.DTT_cropped[line_index,:]
            # print(line_index)
            # print(line_DTT)
            # print(np.shape(self.DTT_cropped))

            
            init_vals = [np.max(line_DTT), (self.xmin+self.xmax)/2+0.05, (self.xmax-self.xmin)/3]     # initialisation for [amp, cen, wid]
            res, pcov = curve_fit(line_fit, self.x_cropped, line_DTT, p0=init_vals)  #actual fitting

            best_vals[line_index,0] = res[0]
            best_vals[line_index,1] = res[1]
            best_vals[line_index,2] = res[2]
            best_wid[line_index] = res[2]
            best_amp[line_index] = res[0]
            std_vals[line_index] = np.sqrt(np.diag(pcov)) # compute 1 std deviation of [amp, cen, wid] 
            std_wid[line_index] = np.sqrt(np.diag(pcov))[2]
            
        idx_selected = np.argmax(best_amp) #select the one with the largest signal
        FWHM = best_wid[idx_selected] 
        val_max = float(np.max(best_amp))

        #put results in class
        self.FWHM_1D = FWHM*1e3 #in nm
        self.amp_1D = val_max
        
        self.idx_selected = idx_selected
        self.best_wid = best_wid[idx_selected]*1e3 #in nm
        self.all_best_wid = np.array(best_wid)*1e3
        self.best_amp = best_amp[idx_selected]
        self.all_best_amp = best_amp
        self.std_vals = std_vals[idx_selected]
        self.std_wid = std_wid[idx_selected]*1e3 #in nm
        self.all_std_wid = np.array(std_wid )*1e3 #in nm
        self.best_vals = best_vals[idx_selected,:]
        self.all_vals = best_vals

        return(FWHM, val_max)


    def fit_2D(self, func, par_lists) :

        X, Y =np.meshgrid(self.x, self.y)

        #create a dictionnary of initial value
        init_vals_dict ={'x0':self.x[self.i_max],'y0':self.y[self.j_max], 'amp' : self.amp, 'dy': self.c*self.V_lia, 'fwhm':self.FWHM_1D*1e-3}

        #Among this list, we select the free parameters to be fitted
        init_vals = [init_vals_dict[param] for param in par_lists]
        
        #We make the fit and save the errors
        best_vals, pcov = curve_fit(func, (X,Y), self.DTT.ravel(), p0=init_vals)  

        sigma_err = np.sqrt(np.diag(pcov))
        chi_sq=((self.DTT.ravel()-func((X,Y),*best_vals))**2).sum()
        
        SStot=((self.DTT.ravel()-np.average(self.DTT.ravel()))**2).sum()
        R2 = 1-chi_sq/SStot



        #Now we save the results of the fit in the sms object

        fitted_vals = {}
        fitted_vals_err = {}
        c=0
        for param in par_lists :
            fitted_vals[param] = best_vals[c]
            fitted_vals_err[param] = sigma_err[c]
            if param == 'fwhm' :
                self.FWHM_err_2D = sigma_err[c]*1e3 #in nm
            c+=1
            
        
        # Saving initial and fitted parameters
        self.init_vals_dict = init_vals_dict
        self.fit_2D_vals = fitted_vals
        self.fit_2D_err = fitted_vals_err
        self.R2_2D = R2

        # Specifically saving amp and fwhm
        try :
            self.FWHM_2D = self.fit_2D_vals['fwhm']*1e3 #in nm
        except :
            self.FWHM_2D = self.init_vals_dict['fwhm']*1e3 #in nm
    
        try :
            self.amp_2D = self.fit_2D_vals['amp']
        except :
            self.amp_2D = self.init_vals_dict['amp']

        return(init_vals_dict, fitted_vals, fitted_vals_err, R2)


    def fit_peaks(self) :
        """Make an interpolation of FWHM using the scipy find_peak method.
        """
        line_DTT = self.DTT[self.j_max,:]/self.DTT.max()

        d_wl = self.xstep

        
        
        width, width_height, left, right = scipy.signal.peak_widths(line_DTT, np.array([line_DTT.argmax()]), rel_height=0.5 )



        width = np.array(width)*d_wl

        self.FWHM_peak = width[0]*1e3