# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:40:01 2019

@author: femto-nano
"""

import numpy as np

# gold heat capacity
c_m = 128.
# density
rho = 19.3e3
# volumic heat capacity
c_v = rho*c_m
#c_v = 2.492e6
print('Volumic heat capacity : {:.2e}'.format(c_v))

# pump absorption cross-section (m²)
#sigma_abs_pp = 5200e-18
sigma_abs_pp = 11417.89e-18 #value from old Romain code


# disk dimensions (m)
D, h = 130e-9, 20e-9 #old values
#D, h = 91e-9, 23e-9


V =np.pi*h*(D/2.)**2

# pump power before the objective with the chopper on (W)

P_pp_bef = 10.e-6 #old values
#P_pp_bef = 6.e-6



# pump power at the particle (W) (effects of chopper and objective are corrected)

t_obj, alpha_chopper = 0.173, 2.58 # old values
#t_obj, alpha_chopper = 0.1983, 2.55
P_pp = P_pp_bef*alpha_chopper*t_obj



# pump linewidth (m)

FWHM_pp = 291.6e-9 # old values
#FWHM_pp = 248.8e-9

print('Pump FWHM : {:.2e}'.format(FWHM_pp))
# pump surface (m²)
S_pp = np.pi*(FWHM_pp/2.)**2/np.log(2)

print('Pump surface : {:.2f} nm2'.format(S_pp*1e18))

# pump fluence (J/m²)
f_rep = 80e6
T_rep = 1/f_rep
f = P_pp*T_rep/(S_pp)

print('Fluence : {:.2e}'.format(f))


# particle temperature increase (K)
DT = sigma_abs_pp*f/(V*c_v)
print('DT : {:.2e}'.format(DT))

# probe linewidth (m)
FWHM_pr = 560e-9
print('FWHM_pr : {:.2e}'.format(FWHM_pr))


# probe surface (m²)
S_pr = np.pi*(FWHM_pr/2.)**2/np.log(2)
print('S_pr : {:.2f} nm2'.format(S_pr*1e18))

# pump-probe signal thermal fit amplitude (=A1+A2)
amp_th =  18e-5

# maximum (or minimum) of short time signal
amp_peak =  -1.1005E-4

# pump-probe signal after calibration (divide by c=V_LIA(pr,f)/V_DMM(pr,DC)=0.400)
#DTT = amp/0.400

# extinction cross-section variation per kelvin (in nm^2!!)

coeff = 0.4 # old value used
#coeff =1
A_p = -amp_th/coeff*S_pr/DT*1e18
A_peak = -amp_peak/coeff*S_pr/DT*1e18  

print('Delta_sigma / Delta_T = {:.1e}'.format(A_p))
print('Delta_sigma = {:1f} nm2'.format(A_p*DT))
print('Delta T = {:.1f} K'.format(DT))
             
       
       

       
       