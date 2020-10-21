@profile
def Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=None, 
				 hierarchy='normal', Mnu=0.00, Nnu=0, Neff=3.046, tau=None,
				 pivot_scalar=0.05, pivot_tensor=0.05, kmax=10.0, k_per_logint=30, cdm=0):

	Omega_cb = Omega_c + Omega_b
	pars = camb.CAMBparams()

	#~ # set accuracy of the calculation
	#~ pars.set_accuracy(AccuracyBoost=5.0, lSampleBoost=5.0, 
					  #~ lAccuracyBoost=5.0, 
					  #~ DoLateRadTruncation=True)

	# set value of the cosmological parameters
	pars.set_cosmology(H0=h*100.0, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, 
					   mnu=Mnu, omk=Omega_k, neutrino_hierarchy=hierarchy, 
					   num_massive_neutrinos=Nnu, nnu=Neff, tau=tau)
				   
	# set the value of the primordial power spectrum parameters
	pars.InitPower.set_params(As=As, ns=ns, 
							  pivot_scalar=pivot_scalar, 
							  pivot_tensor=pivot_tensor)
	
	#~ pars = camb.CAMBparams()
	#~ pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
	#~ pars.InitPower.set_params(As=2e-9, ns=0.965)

	lmax = 2000
	pars.set_for_lmax(lmax, lens_potential_accuracy=1)
	
	
	
	#set Want_CMB to true if you also want CMB spectra or correlations
	pars.Want_CMB = False 
	#NonLinear_both or NonLinear_lens will use non-linear corrections
	pars.NonLinear = model.NonLinear_both

	#Set up W(z) window functions, later labelled W1, W2. Gaussian here.
	pars.SourceWindows = [
		GaussianSourceWindow(redshift=redshifts, source_type='counts', bias=1.2, sigma=0.04, dlog10Ndm=-0.2),
		GaussianSourceWindow(redshift=redshifts, source_type='lensing', sigma=0.07)]
	
	results = camb.get_results(pars)
	kill
	
	DA = results.angular_diameter_distance(redshifts)
	Hz = results.hubble_parameter(redshifts)
	print('H(z=99)      = %.4f km/s/(kpc/h)'%(Hz/1e3/h))
	cls = results.get_source_cls_dict()
	ls=  np.arange(2, lmax+1)
	
	
	time.sleep(1)
	return ls, cls["W1xW1"][2:lmax+1], cls["W2xW2"][2:lmax+1], cls["W1xW2"][2:lmax+1], DA, Hz



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

		k1, Pk1a, Pk1b, Pk1c, DA, Hz = Pk_cosmology_lensed(Omega_cp, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c, DA, Hz = Pk_cosmology_lensed(Omega_cm, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='Ob':
		variation = Omega_b*diff
		Omega_bp = Omega_b + variation;  Omega_cp = Omega_c - variation
		Omega_bm = Omega_b - variation;  Omega_cm = Omega_c + variation

		k1, Pk1a, Pk1b, Pk1c, DA, Hz  = Pk_cosmology_lensed(Omega_cp, Omega_bp, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c, DA, Hz = Pk_cosmology_lensed(Omega_cm, Omega_bm, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='ns':
		variation = ns*diff
		ns_p = ns + variation
		ns_m = ns - variation

		k1, Pk1a, Pk1b, Pk1c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns_p, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns_m, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='h':
		variation = h*diff
		h_p = h + variation
		h_m = h - variation

		k1, Pk1a, Pk1b, Pk1c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h_p, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 
									

		k2, Pk2a, Pk2b, Pk2c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h_m, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
									

	elif parameter=='s8':
		variation = s8*diff
		s8_p = s8 + variation
		s8_m = s8 - variation

		k1, Pk1a, Pk1b, Pk1c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8_p, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 

		k2, Pk2a, Pk2b, Pk2c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8_m, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)

	elif parameter=='Mnu':
		variation = 0.05 # we are assuming 0.0eV and 0.10eV cosmologies!!! 
						 # (variation/2 because we only have Mnu_p)
		Omega_cp = Omega_c - 2.0*variation/(93.14*h**2)

		Mnu, Nnu, s8p, Neff = 0.10, 3, s8, 3.046 #*0.9996
		k1, Pk1a, Pk1b, Pk1c, DA, Hz = Pk_cosmology_lensed(Omega_cp, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8p, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm) 
		#~ np.savetxt('borrar1.txt', np.transpose([k1,Pk1[0,:]]))

		Mnu, Nnu, Neff = 0.00, 0, 3.046
		k2, Pk2a, Pk2b, Pk2c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
									s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
									pivot_tensor, kmax, k_per_logint, cdm)
		#~ np.savetxt('borrar2.txt', np.transpose([k2,Pk2[0,:]]))

	# compute Pk of the fiducial model
	k0, Pk0a, Pk0b, Pk0c, DA, Hz = Pk_cosmology_lensed(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, 
								s8, hierarchy, Mnu, Nnu, Neff, tau, pivot_scalar, 
								pivot_tensor, kmax, k_per_logint, cdm)    

	

	if np.any(zs1!=zs2):  raise Exception('redshifts are different!!')
	if np.any(k1!=k2):    raise Exception('wavenumbers are different!!')

	if not(np.allclose(zs0,zs1)):  raise Exception('redshifts are different!!')
	if not(np.allclose(k0,k1)):    raise Exception('wavenumbers are different!!')

	# do a loop over all redshifts
	root_derv = 'lens'
	for i,z in enumerate(zs1):
		fout = '%s/pk_z=%s.txt'%(root_derv, parameter,z)
		np.savetxt(fout, np.transpose([k0, Pk0a[i,:], Pk0b[i,:], Pk0c[i,:], DA, H]))
		fout = '%s/derivative_%s_z=%s.txt'%(root_derv, parameter,z)
		np.savetxt(fout, np.transpose([k1, (Pk1a[i,:]-Pk2a[i,:])/(2.0*variation), (Pk1b[i,:]-Pk2b[i,:])/(2.0*variation), 
		(Pk1c[i,:]-Pk2c[i,:])/(2.0*variation)]))
		fout = '%s/log_derivative_%s_z=%s.txt'%(root_derv,parameter,z)
		np.savetxt(fout, np.transpose([k1, (Pk1a[i,:]-Pk2a[i,:])/(2.0*variation)/Pk0a[i,:], 
		(Pk1b[i,:]-Pk2b[i,:])/(2.0*variation)/Pk0b[i,:], (Pk1c[i,:]-Pk2c[i,:])/(2.0*variation)/Pk0c[i,:]]))
		
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
redshifts    = 0.0 
kmax         = 10.0
k_per_logint = 30
###############################################################################


# compute derivatives
for parameter in ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']:
#~ for parameter in ['h']:
	derivative(Omega_c, Omega_b, Omega_k, h, ns, As, redshifts, s8=0.834, 
			   parameter=parameter, diff=0.01, cdm=1) #diff does not apply to Mnu


 

