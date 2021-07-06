def signal(kmiddle, fid, cdm):
	# Define your cosmology (what is not specified will be set to CLASS default parameters)
	params = {
	'output': 'mPk',
	'z_max_pk': 100,
	'P_k_max_h/Mpc': 10,
	'non linear': 'halofit',
	'Omega_cdm': fid[0],
	#~ 'Omega_cdm': fid[0] - fid[5]/(93.14*fid[2]**2),
	'omega_b': fid[1],
	'h': fid[2],
	'n_s': fid[3], 
	'sigma8': fid[4],
	#~ 'N_ur': 2.0328,
	#~ 'N_ncdm' : 1,
	#~ 'm_ncdm' : fid[5]}
	'N_ur': 3.046}

	# Create an instance of the CLASS wrapper
	cosmo = Class()

	# Create an instance of the CLASS wrapper
	cosmo = Class()

	# Set the parameters to the cosmological code
	cosmo.set(params)

	# Run the whole code. Depending on your output, it will call the
	# CLASS modules more or less fast. For instance, without any
	# output asked, CLASS will only compute background quantities,
	# thus running almost instantaneously.
	# This is equivalent to the beginning of the `main` routine of CLASS,
	# with all the struct_init() methods called.
	cosmo.compute()


	#~ print(kmiddle)
	h = cosmo.h()
	#~ kmiddle /= h # rescale for 1/mpc units
	



	#### Store the selected redshifts in a array and deduce its length for the loops
	#### array manipulation because len() and size only work for znumber >1
	redshift = 0.0 #!!!!!!!
	red = np.array(redshift)
	znumber = red.size 
	redshift = np.zeros(znumber,'float64') 
	redshift[:] = red
	
	lkb = len(kmiddle)

	#### get the linear power spectrum from class. here multiply input k array by h because get_pk uses 1/mpc 
	if cdm == 0:
		pk_lin = cosmo.get_pk_array(kmiddle, redshift, lkb, znumber, 0) #if we want Pmm

	elif cdm == 1:
		pk_lin = cosmo.get_pk_cb_array(kmiddle, redshift, lkb, znumber, 0) # if we want Pcb
		

	### rescale the amplitude of pk_lin accoridngly to give it in h/mpc
	pk_lin *= h**3
	lpk = len(pk_lin)
	
	# Clean CLASS (the equivalent of the struct_free() in the `main`
	# of CLASS. This step is primordial when running in a loop over different
	# cosmologies, as you will saturate your memory very fast if you ommit
	# it.
	cosmo.struct_cleanup()
	

	# If you want to change completely the cosmology, you should also
	# clean the arguments, otherwise, if you are simply running on a loop
	# of different values for the same parameters, this step is not needed
	cosmo.empty()


	return pk_lin, lpk
