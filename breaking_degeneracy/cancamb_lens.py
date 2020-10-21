import numpy as np
import camb
from camb import model, initialpower
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import sys,os
import time
import matplotlib.pyplot as plt
from memory_profiler import profile
from scipy import interpolate
from scipy import integrate


@profile
def Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=None, 
				 hierarchy='normal', Mnu=0.00, Nnu=0, Neff=3.046, tau=None,
				 pivot_scalar=0.05, pivot_tensor=0.05, kmax=10.0, k_per_logint=30, cdm=0):

	Omega_cb = Omega_c + Omega_b
	pars = camb.CAMBparams()
	
	print(h)
	# set accuracy of the calculation
	#~ pars.set_accuracy(AccuracyBoost=5.0, lSampleBoost=5.0, 
					  #~ lAccuracyBoost=5.0, 
					  #~ DoLateRadTruncation=True)
	#~ # set value of the cosmological parameters
	pars.set_cosmology(H0=h*100.0, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, 
					   mnu=Mnu, omk=Omega_k, neutrino_hierarchy=hierarchy, 
					   num_massive_neutrinos=Nnu, nnu=Neff, tau=tau)
				   
	#~ # set the value of the primordial power spectrum parameters
	pars.InitPower.set_params(As=As, ns=ns, 
							  pivot_scalar=pivot_scalar, 
							  pivot_tensor=pivot_tensor)

	
	lmax = 2500
	lred = [0.0, 0.4, 0.55, 0.70, 0.85, 1.0, 1.15]
	sred = [1.00, 0.85, 1.00, 1.15, 1.3, 1.45, 1.60]
	#~ pars.set_for_lmax(lmax, lens_potential_accuracy=1)
	# set accuracy of the calculation
	#~ pars.set_accuracy(AccuracyBoost=5.0, lSampleBoost=5.0, 
					  #~ lAccuracyBoost=5.0, 
					  #~ DoLateRadTruncation=True)
	pars.set_matter_power(redshifts=lred, kmax=kmax, k_per_logint=k_per_logint)
	
	
	
	#----------------------------------------------------------------------------------------
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------
	#~ #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
	#so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
	results= camb.get_background(pars)
	results2 = camb.get_results(pars)
	# interpolate to get Pmm, Pcc...etc
	#~ k, zs, Pkmm = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax, 
													#~ npoints=400, var1=7, var2=7, 
													#~ have_power_spectra=True, 
													#~ params=None)

	Om = 0.3175
	H0 = h*100.0
	c = 3e5
	nz = 100
	ls = np.arange(2,2500+1, dtype=np.float64)
	cl11 = np.zeros(ls.shape)
	cl12 = np.zeros(ls.shape)
	cl22 = np.zeros(ls.shape)
	
	i = redshifts
	#~ for i in range(1,5):
	z_s= sred[i]
	z_l = lred[i]

	deltz = 0.1
	#~ # here chibin1 > chibin2
	chibin1= results.comoving_radial_distance(z_s-deltz)
	chibin2 = results.comoving_radial_distance(z_s+deltz)
	if z_l >= 0.1:
		chibin3= results.comoving_radial_distance(z_l-deltz)
		chibin4 = results.comoving_radial_distance(z_l+deltz)
		print(chibin3, chibin4)
	else:
		chibin3= results.comoving_radial_distance(z_l)
		chibin4 = results.comoving_radial_distance(z_l+deltz)
		print(chibin3, chibin4)
	
	chistar = results.conformal_time(0)- results.tau_maxvis
	print(chistar)
	chis = np.linspace(0,chistar,nz)
	zs=results.redshift_at_comoving_radial_distance(chis)
	#~ #Calculate array of delta_chi, and drop first and last points where things go singular
	dchis = (chis[2:]-chis[:-2])/2
	chis = chis[1:-1]
	zs = zs[1:-1]
	
	#Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
	#Here for lensing we want the power spectrum of the Weyl potential.
	PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
		hubble_units=False, k_hunit=False, kmax=kmax,
		var1=7,var2=7, zmax=zs[-1])
	
	def gal_dist(chisource):
		zso=results.redshift_at_comoving_radial_distance(chisource)
		z0 = 1
		alpha = 2
		beta = 1.5
		nz = (zso /z0)**alpha *np.exp(-(zso/z0))**beta
		return  nz
	
	#get normalization of the distribution
	ni_gal, _ = integrate.quad(gal_dist, chibin3, chibin4)
	ni_lens, _ = integrate.quad(gal_dist, chibin1, chibin2)
	print(ni_gal)
	print(ni_lens)

	# galaxy window function
	def win_gal(chisource):
		zchi=results.redshift_at_comoving_radial_distance(selchi)
		zso =results.redshift_at_comoving_radial_distance(chisource)
		z0 = 1
		alpha = 2
		beta = 1.5
		nz = (zso /z0)**alpha *np.exp(-(zso/z0))**beta
		Hz = results.hubble_parameter(zso)
		#~ return (nz/ni)
		if chibin3 <= chisource <= chibin4:
			return (nz/ni_gal)
		else:
			return 0
		
	#Get lensing window function (flat universe)
	def win_lens(chisource, selchi):
		zchi=results.redshift_at_comoving_radial_distance(selchi)
		zso =results.redshift_at_comoving_radial_distance(chisource)
		z0 = 1
		alpha = 2
		beta = 1.5
		nz = (zso /z0)**alpha *np.exp(-(zso/z0))**beta
		win = ((chisource-selchi)/(chisource))
		azs = 1/(1+zchi)
		if selchi <= chibin2:
			return 3/2.*(H0/c)**2 * Om /azs * (nz/ni_lens) * win * selchi
		else:
			return 0
	
	W = np.zeros(chis.shape)
	F = np.zeros(chis.shape)
	for j, selchi in enumerate(chis):
		W[j],_ = integrate.quad(win_lens, chibin1, chibin2, args=(selchi,))
		F[j] = win_gal(selchi)

	#Do integral over chi
	w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
	for i, l in enumerate(ls):
		k=(l+0.5)/chis
		w[:]=1
		w[k<1e-4]=0
		w[k>=kmax]=0
		cl11[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*F**2/chis**2)
		cl12[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*F*W /chis**2)
		cl22[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*W**2/chis**2)

	
	#~ plt.loglog(ls,cl11, label = 'z = ' + str(z_l))
	#~ plt.loglog(ls,cl12, label = 'z = ' + str(z_s)+', '+str(z_l))
	#~ plt.loglog(ls,cl22, label = 'z = ' + str(z_s))
	#~ plt.ylim([1e-9,1e-3])
	#~ plt.xlim([2,1e3])
	#~ plt.legend(loc = 'lower left')
	#~ plt.ylabel('$C_l^{gg}$')
	#~ plt.ylabel('$C_l^{\kappa \kappa}$')
	#~ plt.xlabel('$l$')
	#~ plt.show()
	#~ kill
	
	
	
	# get sigma_8 and Hz in km/s/(kpc/h)
	print(s8)
	s8_CAMB = np.array(results2.get_sigma8())
	print('sigma_8(z=0) = %.4f'%s8_CAMB[-1])

	if s8 is not None:
		cl11 = cl11*(s8/s8_CAMB[-1])**2
		cl12 = cl12*(s8/s8_CAMB[-1])**2
		cl22 = cl22*(s8/s8_CAMB[-1])**2

	"""
	# do a loop over all redshifts
	for i,z in enumerate(zs):
		fout1 = 'Pk_m_z=%s.txt'%z
		np.savetxt(fout1,np.transpose([k,Pkmm[i,:]]))
	"""
	
	results = None
	
	return ls, cl11, cl12, cl22


@profile
def derivative(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=None, 
			   hierarchy='normal', Mnu=0.00, Nnu=0, Neff=3.046, tau=None,
			   pivot_scalar=0.05, pivot_tensor=0.05, 
			   kmax=10.0, k_per_logint=30, parameter=None, diff=0.01, cdm=0):

	if parameter not in ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']:
		print('Specify the parameter used to compute the derivative!!!')
		print('Om, Ob, h, ns or s8')
		return 0


	elif parameter=='Om':
		variation = (Omega_c + Omega_b)*diff
		Omega_cp = Omega_c + variation
		Omega_cm = Omega_c - variation

		k1, Pk1a, Pk1b, Pk1c = Pk_cosmology_lensed(Omega_cp, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c = Pk_cosmology_lensed(Omega_cm, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='Ob':
		variation = Omega_b*diff
		Omega_bp = Omega_b + variation;  Omega_cp = Omega_c - variation
		Omega_bm = Omega_b - variation;  Omega_cm = Omega_c + variation

		k1, Pk1a, Pk1b, Pk1c  = Pk_cosmology_lensed(Omega_cp, Omega_bp, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c = Pk_cosmology_lensed(Omega_cm, Omega_bm, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='ns':
		variation = ns*diff
		ns_p = ns + variation
		ns_m = ns - variation

		k1, Pk1a, Pk1b, Pk1c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns_p, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns_m, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='h':
		variation = h*diff
		h_p = h + variation
		h_m = h - variation

		k1, Pk1a, Pk1b, Pk1c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h_p, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 
									

		k2, Pk2a, Pk2b, Pk2c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h_m, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
									

	elif parameter=='s8':
		variation = s8*diff
		s8_p = s8 + variation
		s8_m = s8 - variation
		
		
		

		k1, Pk1a, Pk1b, Pk1c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8_p, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8_m, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
									
									
		

	elif parameter=='Mnu':
		variation = 0.05 # we are assuming 0.0eV and 0.10eV cosmologies!!! 
						 # (variation/2 because we only have Mnu_p)
		Omega_cp = Omega_c - 2.0*variation/(93.14*h**2)

		Mnu, Nnu, s8p, Neff = 0.10, 3, s8, 3.046 #*0.9996
		k1, Pk1a, Pk1b, Pk1c = Pk_cosmology_lensed(Omega_cp, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8p, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 
		#~ np.savetxt('borrar1.txt', np.transpose([k1,Pk1[0,:]]))

		Mnu, Nnu, Neff = 0.00, 0, 3.046
		k2, Pk2a, Pk2b, Pk2c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
		#~ np.savetxt('borrar2.txt', np.transpose([k2,Pk2[0,:]]))

	# compute Pk of the fiducial model
	k0, Pk0a, Pk0b, Pk0c = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
								s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
								pivot_tensor, kmax, k_per_logint, cdm)    

	#~ plt.figure()
	#~ plt.loglog(k1, Pk0a)
	#~ plt.loglog(k1, Pk1a)
	#~ plt.loglog(k1, Pk2a)
	#~ plt.show()

	#~ if np.any(zs1!=zs2):  raise Exception('redshifts are different!!')
	if np.any(k1!=k2):    raise Exception('wavenumbers are different!!')

	#~ if not(np.allclose(zs0,zs1)):  raise Exception('redshifts are different!!')
	if not(np.allclose(k0,k1)):    raise Exception('wavenumbers are different!!')

	# do a loop over all redshifts
	root_derv = 'lens'
	lred = [0.0, 0.4, 0.55, 0.70, 0.85, 1.0, 1.15]
	#~ for i,z in enumerate(lred[redshifts]):
	z = lred[redshifts]
	#~ fout = '%s/pk_%s_z=%s.txt'%(root_derv, parameter,z)
	fout = '%s/pk_z=%s.txt'%(root_derv,z)
	np.savetxt(fout, np.transpose([k0, Pk0a, Pk0b, Pk0c]))
	fout = '%s/derivative_%s_z=%s.txt'%(root_derv, parameter,z)
	np.savetxt(fout, np.transpose([k1, (Pk1a-Pk2a)/(2.0*variation), (Pk1b-Pk2b)/(2.0*variation), 
	(Pk1c-Pk2c)/(2.0*variation)]))
	fout = '%s/log_derivative_%s_z=%s.txt'%(root_derv,parameter,z)
	np.savetxt(fout, np.transpose([k1, (Pk1a-Pk2a)/(2.0*variation)/Pk0a, 
	(Pk1b-Pk2b)/(2.0*variation)/Pk0b, (Pk1c-Pk2c)/(2.0*variation)/Pk0c]))
		

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
redshifts    = 0
kmax         = 10.0
k_per_logint = 30
###############################################################################


# compute derivatives
for parameter in ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']:
#~ for parameter in ['h']:
	derivative(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=0.834, 
			   parameter=parameter, diff=0.01, cdm=0) #diff does not apply to Mnu


 

