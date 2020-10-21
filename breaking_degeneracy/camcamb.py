import numpy as np
import camb
import sys,os
import time
import gc

@profile
# This functions takes the value of the cosmological parameters and returns the 
# linear matter Pk from CAMB
def Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=None, 
				 hierarchy='normal', Mnu=0.00, Nnu=0, Neff=3.046, tau=None,
				 pivot_scalar=0.05, pivot_tensor=0.05, kmax=10.0, k_per_logint=30, cdm=0):

	Omega_cb = Omega_c + Omega_b
	pars = camb.CAMBparams()

	# set accuracy of the calculation
	pars.set_accuracy(AccuracyBoost=5.0, lSampleBoost=5.0, 
					  lAccuracyBoost=5.0, 
					  DoLateRadTruncation=True)

	# set value of the cosmological parameters
	pars.set_cosmology(H0=h*100.0, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, 
					   mnu=Mnu, omk=Omega_k, neutrino_hierarchy=hierarchy, 
					   num_massive_neutrinos=Nnu, nnu=Neff, tau=tau)
				   
	# set the value of the primordial power spectrum parameters
	pars.InitPower.set_params(As=As, ns=ns, 
							  pivot_scalar=pivot_scalar, 
							  pivot_tensor=pivot_tensor)

	# set redshifts, k-range and k-sampling
	pars.set_matter_power(redshifts=redshifts, kmax=kmax, k_per_logint=k_per_logint)
	
	# compute results
	results = camb.get_results(pars)
	pars = None
	
	if cdm == 0:
		# interpolate to get Pmm, Pcc...etc
		k, zs, Pkmm = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
														npoints=400, var1=7, var2=7, 
														have_power_spectra=True, 
														params=None)


	
	elif cdm == 1: #Pcb
											
		k, zs, Pkcc = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
												npoints=400, var1=2, var2=2, 
												have_power_spectra=True, 
												params=None)

		k, zs, Pkbb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
												npoints=400, var1=3, var2=3, 
												have_power_spectra=True, 
												params=None)

		k, zs, Pkcb = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
												npoints=400, var1=2, var2=3, 
												have_power_spectra=True, 
												params=None)

		Pkmm = (Omega_c**2*Pkcc + Omega_b**2*Pkbb +\
		2.0*Omega_b*Omega_c*Pkcb)/Omega_cb**2

	# get sigma_8 and Hz in km/s/(kpc/h)
	s8_CAMB = np.array(results.get_sigma8())
	print('sigma_8(z=0) = %.4f'%s8_CAMB[-1])

	if s8 is not None:
		Pkmm = Pkmm*(s8/s8_CAMB[-1])**2

	"""
	# do a loop over all redshifts
	for i,z in enumerate(zs):
		fout1 = 'Pk_m_z=%s.txt'%z
		np.savetxt(fout1,np.transpose([k,Pkmm[i,:]]))
	"""
	
	results = None
	
	gc.collect()
	
	time.sleep(1)
	return zs, k, Pkmm
	



@profile
def derivative(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=None, 
			   hierarchy='normal', Mnu=0.00, Nnu=0, Neff=3.046, tau=None,
			   pivot_scalar=0.05, pivot_tensor=0.05, 
			   kmax=10.0, k_per_logint=30, parameter=None, diff=0.01, cdm=0):

	if parameter not in ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']:
		print('Specify the parameter used to compute the derivative!!!')
		print( 'Om, Ob, h, ns or s8')
		return 0


	elif parameter=='Om':
		variation = (Omega_c + Omega_b)*diff
		Omega_cp = Omega_c + variation
		Omega_cm = Omega_c - variation

		zs1, k1, Pk1 = Pk_cosmology(Omega_cp, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		zs2, k2, Pk2 = Pk_cosmology(Omega_cm, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='Ob':
		variation = Omega_b*diff
		Omega_bp = Omega_b + variation;  Omega_cp = Omega_c - variation
		Omega_bm = Omega_b - variation;  Omega_cm = Omega_c + variation

		zs1, k1, Pk1 = Pk_cosmology(Omega_cp, Omega_bp, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		zs2, k2, Pk2 = Pk_cosmology(Omega_cm, Omega_bm, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='ns':
		variation = ns*diff
		ns_p = ns + variation
		ns_m = ns - variation

		zs1, k1, Pk1 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns_p, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		zs2, k2, Pk2 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns_m, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='h':
		variation = h*diff
		h_p = h + variation
		h_m = h - variation

		zs1, k1, Pk1 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h_p, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 
									
		np.savetxt('h1.txt', np.transpose([k1,Pk1[0,:]]))

		zs2, k2, Pk2 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h_m, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
									
		np.savetxt('h2.txt', np.transpose([k1,Pk1[0,:]]))

	elif parameter=='s8':
		variation = s8*diff
		s8_p = s8 + variation
		s8_m = s8 - variation

		zs1, k1, Pk1 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8_p, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		zs2, k2, Pk2 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8_m, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='Mnu':
		variation = 0.05 # we are assuming 0.0eV and 0.10eV cosmologies!!! 
						 # (variation/2 because we only have Mnu_p)
		Omega_cp = Omega_c - 2.0*variation/(93.14*h**2)

		Mnu, Nnu, s8p, Neff = 0.10, 3, s8, 3.046 #*0.9996
		zs1, k1, Pk1 = Pk_cosmology(Omega_cp, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8p, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 
		np.savetxt('borrar1.txt', np.transpose([k1,Pk1[0,:]]))

		Mnu, Nnu, Neff = 0.00, 0, 3.046
		zs2, k2, Pk2 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
		np.savetxt('borrar2.txt', np.transpose([k2,Pk2[0,:]]))

	# compute Pk of the fiducial model
	zs0, k0, Pk0 = Pk_cosmology(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
								s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
								pivot_tensor, kmax, k_per_logint, cdm)    

	gc.collect()

	if np.any(zs1!=zs2):  raise Exception('redshifts are different!!')
	if np.any(k1!=k2):    raise Exception('wavenumbers are different!!')

	if not(np.allclose(zs0,zs1)):  raise Exception('redshifts are different!!')
	if not(np.allclose(k0,k1)):    raise Exception('wavenumbers are different!!')

	# do a loop over all redshifts
	if cdm == 0:
		root_derv = 'mm'
	elif cdm == 1:
		root_derv = 'cb'
	for i,z in enumerate(zs1):
		fout = '%s/derivative_%s_z=%s.txt'%(root_derv, parameter,z)
		np.savetxt(fout, np.transpose([k1, (Pk1[i,:]-Pk2[i,:])/(2.0*variation)]))
		fout = '%s/log_derivative_%s_z=%s.txt'%(root_derv,parameter,z)
		np.savetxt(fout, np.transpose([k1, (Pk1[i,:]-Pk2[i,:])/(2.0*variation)/Pk0[i,:]]))
		
	del zs1, k1, Pk1
	del zs2, k2, Pk2	
	del zs0, k0, Pk0	
	time.sleep(1)
		
################################## INPUT ######################################
# neutrino parameters
hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
Mnu       = 0.00  #eV
Nnu       = 0  #number of massive neutrinos
Neff      = 3.046

# cosmological parameters
h       = 0.6711
Omega_c = 0.2685 #- Mnu/(93.14*h**2)
Omega_b = 0.049
Omega_k = 0.0
tau     = None

# initial P(k) parameters
ns           = 0.9624
As           = 2.13e-9
pivot_scalar = 0.05
pivot_tensor = 0.05

# redshifts and k-range
redshifts    = [0.0] 
kmax         = 10.0
k_per_logint = 30
###############################################################################


# compute derivatives
for parameter in ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']:
#~ for parameter in ['h']:
	print('pop')
	derivative(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=0.834, 
			   parameter=parameter, diff=0.01, cdm=1) #diff does not apply to Mnu

