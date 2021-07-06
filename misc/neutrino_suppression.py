### code written by David Valcin
### for calibration details see arXiv:1901.06045

from classy import Class
import numpy as np
import sys
import matplotlib.pyplot as plt

########################################################################
########################################################################
### BE_HaPPY configuration
########################################################################
########################################################################

### if you want to compare several spectra define quantities as list (e.g. coord = [0, 1]) and compute for the desired index

# integration boundaries for k (in h/Mpc).
# It has to be within the boundaries of the k model above, respectively 0.15, 0.2, 0.4
kmin = 0.0001
kmax = 0.5
kbins = 261

# desired redshift must be between [0-2]
z = 0.0


########################################################################
########################################################################
### Here an example
### Import a linear power spectrum. Make sure to use values close the paper reference cosmology
########################################################################
########################################################################
znumber = 1
zmax = 2 # needed for transfer
karray = np.logspace(np.log10(kmin), np.log10(kmax), kbins)
h = 0.67810

# ~params_def = {'output': 'mPk', 'non linear': 'halofit'}
m_0 = 0.0
ocdm_0 = 0.261205 - m_0/(93.14*h**2) 
params_def = {'output': 'mPk', 'non linear': 'halofit', 'Omega_cdm':ocdm_0,'N_ncdm':3,'N_ur':0.00441,'m_ncdm':str(m_0)+','+str(m_0)+','+str(m_0)}



# Create an instance of the CLASS wrapper
cosmo = Class()
# Set the parameters to the cosmological code
cosmo.set(params_def)
cosmo.compute()
#### Define the linear growth factor and growth rate (growth factor f in class)
h = cosmo.h()
# ~mcu = cosmo.N_ur()
# ~print(mcu)

Plin= np.zeros(len(karray))
Pnonlin= np.zeros(len(karray))
for i,k in enumerate(karray) :
	Plin[i] = (cosmo.pk_lin(k ,z)) # function .pk(k,z)
	Pnonlin[i] = (cosmo.pk(k ,z)) # function .pk(k,z)


Plin *= h**3
Pnonlin *= h**3

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
m_1 = 0.02
ocdm_1 = 0.261205 - m_1*3/(93.14*h**2) 
params_1 = {'output': 'mPk', 'non linear': 'halofit', 'Omega_cdm':ocdm_1,'N_ncdm':3,'N_ur':0.00441,'m_ncdm':str(m_1)+','+str(m_1)+','+str(m_1)}
# ~params_1 = {'output': 'mPk', 'non linear': 'halofit', 'Omega_cdm':ocdm_1,'N_ncdm':1,'N_ur':2.0308,'m_ncdm':0.06}

# Create an instance of the CLASS wrapper
cosmo_1 = Class()
# Set the parameters to the cosmological code
cosmo_1.set(params_1)
cosmo_1.compute()
#### Define the linear growth factor and growth rate (growth factor f in class)


Plin_1= np.zeros(len(karray))
Pnonlin_1= np.zeros(len(karray))
for i,k in enumerate(karray) :
	Plin_1[i] = (cosmo_1.pk_lin(k ,z)) # function .pk(k,z)
	Pnonlin_1[i] = (cosmo_1.pk(k ,z)) # function .pk(k,z)

Plin_1 *= h**3
Pnonlin_1 *= h**3
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
m_2 = 0.04
ocdm_2 = 0.261205 - m_2*3/(93.14*h**2) 
params_2 = {'output': 'mPk', 'non linear': 'halofit', 'Omega_cdm':ocdm_2,'N_ncdm':3,'N_ur':0.00441,'m_ncdm':str(m_2)+','+str(m_2)+','+str(m_2)}

# Create an instance of the CLASS wrapper
cosmo_2 = Class()
# Set the parameters to the cosmological code
cosmo_2.set(params_2)
cosmo_2.compute()
#### Define the linear growth factor and growth rate (growth factor f in class)


Plin_2= np.zeros(len(karray))
Pnonlin_2= np.zeros(len(karray))
for i,k in enumerate(karray) :
	Plin_2[i] = (cosmo_2.pk_lin(k ,z)) # function .pk(k,z)
	Pnonlin_2[i] = (cosmo_2.pk(k ,z)) # function .pk(k,z)

Plin_2 *= h**3
Pnonlin_2 *= h**3
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
m_3 = 0.06
ocdm_3 = 0.261205 - m_3*3/(93.14*h**2) 
params_3 = {'output': 'mPk', 'non linear': 'halofit', 'Omega_cdm':ocdm_3,'N_ncdm':3,'N_ur':0.00441,'m_ncdm':str(m_3)+','+str(m_3)+','+str(m_3)}

# Create an instance of the CLASS wrapper
cosmo_3 = Class()
# Set the parameters to the cosmological code
cosmo_3.set(params_3)
cosmo_3.compute()
#### Define the linear growth factor and growth rate (growth factor f in class)


Plin_3= np.zeros(len(karray))
Pnonlin_3= np.zeros(len(karray))
for i,k in enumerate(karray) :
	Plin_3[i] = (cosmo_3.pk_lin(k ,z)) # function .pk(k,z)
	Pnonlin_3[i] = (cosmo_3.pk(k ,z)) # function .pk(k,z)

Plin_3 *= h**3
Pnonlin_3 *= h**3
########################################################################
########################################################################
### Compute the non linear power spectrum
########################################################################
########################################################################
karray /= h


plt.figure()
plt.plot(karray, (Plin_1 - Plin)/Plin, label =r'$\Sigma_{m_{\nu}} = 0.06 \:eV$', c='r')
plt.plot(karray, (Plin_2 - Plin)/Plin, label =r'$\Sigma_{m_{\nu}} = 0.12 \:eV$', c='b')
plt.plot(karray, (Plin_3 - Plin)/Plin, label =r'$\Sigma_{m_{\nu}} = 0.18 \:eV$',c='c')
plt.legend(loc='best', fontsize = 16)
plt.xlabel('k [h/Mpc] ', fontsize = 14)
plt.ylabel(r' $(P_{m_{\nu}} - P_{m_{\nu}= 0} )/P_{m_{\nu}= 0} \:[h^3/ Mpc^{-3}]$', fontsize = 14)
plt.xscale('log')
# ~plt.yscale('log')
plt.show()
plt.close()
