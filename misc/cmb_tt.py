### code written by David Valcin

from classy import Class
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import pi

########################################################################
########################################################################
### Compute the non linear power spectrum
########################################################################
########################################################################

# create instance of the class "Class"
LambdaCDM = Class()
# pass input parameters
LambdaCDM.set({'omega_b':0.022032,'omega_cdm':0.12038,'h':0.67556,'A_s':2.215e-9,'tau_reio':0.0925,'ic':'niv'})
LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
# run class
LambdaCDM.compute()
# get all C_l output
cls = LambdaCDM.raw_cl(2500)
# To check the format of cls
# ~cls.viewkeys()

ll = cls['ell'][3:]
clTT = cls['tt'][3:]
clEE = cls['ee'][3:]
clPP = cls['pp'][3:]


# create instance of the class "Class"
LambdaCDM1 = Class()
# pass input parameters
LambdaCDM1.set({'omega_b':0.022032,'omega_cdm':0.12038,'h':0.67556,'A_s':2.215e-9,'tau_reio':0.0925,'ic':'cdi'})
LambdaCDM1.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
# run class
LambdaCDM1.compute()
# get all C_l output
cls1 = LambdaCDM1.lensed_cl(2500)
# To check the format of cls
# ~cls.viewkeys()

ll1 = cls1['ell'][2:]
clTT1 = cls1['tt'][2:]
clEE1 = cls1['ee'][2:]
clPP1 = cls1['pp'][2:]


# create instance of the class "Class"
LambdaCDM2 = Class()
# pass input parameters
LambdaCDM2.set({'omega_b':0.022032,'omega_cdm':0.12038,'h':0.67556,'A_s':2.215e-9,'n_s':0.9619,'tau_reio':0.0925})
LambdaCDM2.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
# run class
LambdaCDM2.compute()
# get all C_l output
cls2 = LambdaCDM2.lensed_cl(2500)
# To check the format of cls
# ~cls.viewkeys()

ll2 = cls2['ell'][2:]
clTT2 = cls2['tt'][2:]
clEE2 = cls2['ee'][2:]
clPP2 = cls2['pp'][2:]

# create instance of the class "Class"
LambdaCDM3 = Class()
# pass input parameters
LambdaCDM3.set({'omega_b':0.022032,'omega_cdm':0.12038,'h':0.67556,'A_s':2.215e-9,'tau_reio':0.0925,'ic':'nid'})
LambdaCDM3.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
# run class
LambdaCDM3.compute()
# get all C_l output
cls3 = LambdaCDM3.lensed_cl(2500)
# To check the format of cls
# ~cls.viewkeys()

ll3 = cls3['ell'][2:]
clTT3 = cls3['tt'][2:]
clEE3 = cls3['ee'][2:]
clPP3 = cls3['pp'][2:]

# plot C_l^TT
plt.figure(1)
plt.xscale('log');plt.yscale('log');plt.xlim(2,2500)
plt.xlabel(r'$\ell$', fontsize=14)
plt.ylabel(r'$[\ell(\ell+1)/2\pi]\:  C_\ell^\mathrm{TT}$', fontsize=14)
# ~plt.plot(ll,clTT*ll*(ll+1)/2./pi,'r-', label='neutrino velocity isocurvature')
plt.plot(ll1,clTT1*ll1*(ll1+1)/2./pi,'r-', label='CDM isocurvature')
plt.plot(ll2,clTT2*ll2*(ll2+1)/2./pi,'b-', label='Adiabatic')
plt.plot(ll3,clTT3*ll3*(ll3+1)/2./pi,'c-', label='Neutrino density isocurvature')
plt.legend(loc='best', fontsize=16)

plt.show()
