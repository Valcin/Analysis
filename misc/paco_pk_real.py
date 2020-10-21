
####################################################################
import numpy as np
import h5py
import readsnap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import tempfile
import pyximport
pyximport.install()
import redshift_space_library as RSL
from readfof import FoF_catalog
import MAS_library as MASL
import Pk_library as PKL
import mass_function_library as MFL
from time import time
from bias_library import halo_bias
from scipy.optimize import curve_fit
#~ import FASTPT_simple as FASTPT
######## INPUT ########
#~ snapshot_fname = '/home/david/codes/Paco/data/snapdir_004/snap_004'

nu = 0.0
#~ z = 0.0
dims = 1200 #the number of cells in the density field is dims^3
axis = 0 #in redshift-space distortion axis
cores = 16 #number of openmp threads used to compute Pk

########################################################################

redshift = [0.0, 0.5, 1.0, 2.0]
for z in redshift:
########################################################################
	# neutrino parameters
	hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
	Mnu       = 0.0  #eV
	Nnu       = 3  #number of massive neutrinos
	Neff      = 3.046

	# cosmological parameters
	h       = 0.6711
	Omega_c = 0.2685 - Mnu/(93.14*h**2)
	Omega_b = 0.049
	Omega_k = 0.0
	tau     = None
	#~ Omega_m = Omega_b + Omega_c
	#~ # read snapshot properties
	#~ head = readsnap.snapshot_header(snapshot_fname)
	BoxSize = 1000.0 #Mpc/h                                         
	#~ redshift = head.redshift
	#~ Hubble = 100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l)#km/s/(Mpc/h)
	#~ h = head.hubble

	#~ Class = np.loadtxt('/home/dvalcin/codes/Paco/data2/0.0eV/expected_Pk_z='+str(z)+'.txt')
	#~ Class_trans = np.loadtxt('/home/dvalcin/codes/Paco/data2/0.0eV/test_'+str(z)+'_tk.dat')
	#~ kclass = Class[:,0]
	#~ Pclass = Class[:,1]
	#kclass_nl = Class_nl[:,0]
	#Pclass_nl = Class_nl[:,1]
	#~ ktrans = Class_trans[:,0]
	#~ Tb = Class_trans[:,2]
	#~ Tcdm = Class_trans[:,3]
	#~ Tm = Class_trans[:,8]

	#~ k = np.logspace(np.min(np.log10(kclass)), np.max(np.log10(kclass)), 150)
	#~ Tb = np.interp(k,ktrans,Tb)
	#~ Tcdm =  np.interp(k,ktrans,Tcdm)
	#~ Tm =  np.interp(k,ktrans,Tm)
	#~ Pclass  = np.interp(k,kclass,Pclass)
	#Pclass_nl = np.interp(k,kclass_nl,Pclass_nl)

	#-----------------------------------------------------------------------
	#-------- get the transfer function and Pcc ----------------------------
	#-----------------------------------------------------------------------
	#~ Tcb = (Omega_c * Tcdm + Omega_b * Tb)/(Omega_c + Omega_b)
	#~ Pcc = Pclass * (Tcb/Tm)**2
	#~ #Pcc_nl = Pclass_nl * (Tcb/Tm)**2
	#~ Plin = Pcc

	#~ cname = '/home/dvalcin/codes/Paco/data2/0.0eV/Plin_z='+str(z)+'.txt'
	#~ with open(cname, 'w') as fid_file:
		#~ for index_k in xrange(len(k)):
			#~ fid_file.write('%.8g %.8g\n' % (k[index_k], Plin[index_k]))
	#~ print '\n'

	#-----------------------------------------------------------------------
	#---------------- matter neutrino Real space ---------------------------
	#-----------------------------------------------------------------------
	for i in xrange(1,11):
#~ for i in dispo:
		myfile1 = '/home/dvalcin/codes/Paco/data2/0.0eV/NCV'+str(i)+'/density_field_c_z='+str(z)+'00.hdf5'
		f1 = h5py.File(myfile1, 'r')
		delta1 = f1[u'df_cr'][ : ]
		f1.close()
		del f1



		#~ # move particles to redshift-space
		#~ RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

		#~ # compute density field in redshift-space
		##~ delta = np.zeros((dims, dims, dims), dtype=np.float32) #this should be your density field
		##~ MASL.MA(pos, delta, BoxSize, MAS='CIC') # computes the density in each cell of the grid
		delta1 = delta1/np.mean(delta1) - 1.0

		# compute power spectra
		Pk1 = PKL.Pk(delta1, BoxSize, axis, MAS='CIC', threads=cores) #Pk here is a class with all power spectra

		# 3D Pk
		k = Pk1.k3D
		#~ Pk0 = (Pk1.Pk[:,0] + Pk2.Pk[:,0])/2 #monopole
		#~ Pk2_av= (Pk1.Pk[:,1] + Pk2.Pk[:,1])/2 #quadrupole
		#~ Pk4 = (Pk1.Pk[:,2] + Pk2.Pk[:,2])/2 #hexadecapole
		#~ Nmodes = (Pk1.Nmodes3D + Pk2.Nmodes3D)/2 #number of modes in each Pk bin


		#~ temp = np.array([Pk1.Pk[:,0], Pk2.Pk[:,0]])
		#~ std = np.std(temp, axis=0)
		cname = '/home/dvalcin/plots/Pcc_realisation_'+str(nu)+'_z='+str(z)+'.txt'
		if i ==1:
			with open(cname, 'w') as fid_file:
				for index_k in xrange(len(k)):
					fid_file.write('%.8g %.8g\n' % ( Pk1.Pk[index_k,0],k[index_k]))
			fid_file.close()
		else:
			#Create temporary file read/write
			t = tempfile.NamedTemporaryFile(mode="r+")
			#Open input file read-only
			i = open(cname, 'r')
			#Copy input file to temporary file, modifying as we go
			for line in i:
				#~ print line
				t.write(line.rstrip()+"\n")
			i.close() #Close input file
			t.seek(0) #Rewind temporary file to beginning
			o = open(cname, "w")  #Reopen input file writable
			#Overwriting original file with temporary file contents          
			for i,line in enumerate(t):
				o.write('%.8g' %(Pk1.Pk[i,0])+' '+line)   
			t.close() #Close temporary file, will cause it to be deleted
			o.close()


		#~ # clear variable for memory
		del delta1
		del Pk1


	########################################################################
	############# scale factor 
	########################################################################
	#~ red = ['0.0','0.5','1.0','2.0','3.0']
	#~ ind = red.index(str(z))
	#~ f = [0.518,0.754,0.872,0.956,0.98]
	#~ mono = (1 + 2/3.*(f[ind]) + 1/5.*(f[ind])**2) 
	#~ quadru = (4/3.*(f[ind]) + 4/7.*(f[ind])**2)

	########################################################################
	############# save matter spectrum 
	########################################################################

	#~ cname = '/home/dvalcin/plots/Pk_z='+str(z)+'.txt'
	#~ with open(cname, 'w') as fid_file:
		#~ for index_k in xrange(len(k)):
			#~ fid_file.write('%.8g %.8g %.8g\n' % (k[index_k], Pk0[index_k], Pk2_av[index_k]))
	#~ print '\n'

	########################################################################
	########################################################################





	#~ plt.figure()
	#~ #plt.plot(k1,Pk0_1, label='monopole NCV1', color='b')
	#~ #plt.plot(k2,Pk0_2, label='monopole NCV2', color='g')
	#~ plt.plot(k,Pk0 , label='monopole (NCV1 + NCV2)/2', color='r')
	#~ #plt.plot(kclass,pclass, label='monopole class + linear kaiser',linestyle ='--', color='b')
	#~ plt.plot(kcamb,pcamb, label='monopole camb',linestyle ='--', color='g')
	#~ #plt.plot(kclass,qnk, label='monopole class + quasi linear kaiser',linestyle =':', color='b')
	#~ #plt.plot(k,Pk2, label='quadrupole simu', color='r')
	#~ #plt.plot(kclass,kpk*quadru, label='quadrupole class + linear kaiser',linestyle ='--', color='r')
	#~ plt.title('3D power spectrum at z = '+str(z) )
	#~ plt.legend(loc='upper right')
	#~ plt.xscale('log')
	#~ plt.xlabel('k')
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.xlim(8e-3,1)
	#~ plt.ylim(1e1,1e5)
	#~ plt.yscale('log')
	#~ plt.ylabel('P(k)')
	#~ plt.savefig('/home/dvalcin/plots/1.png')  

	#~ plt.figure()
	#~ plt.plot(k,Pk0/pcamb, label='Pk simu / Camb', color='b')
	#~ plt.title('3D power spectrum at z = '+str(z) )
	#~ plt.legend(loc='upper right')
	#~ plt.xscale('log')
	#~ plt.xlabel('k')
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.xlim(8e-3,0.1)
	#~ plt.ylim(0.9,1.1)
	#~ plt.ylabel('P(k) ratio')
	#~ plt.savefig('/home/dvalcin/plots/2.png')  

	#####################################################################
	########## Halo power spectra 
	#####################################################################


	# halo positions are in kpc/h but we want them in Mpc/h
	# pos_h is the halo position
	# BoxSize is the box size of the simulation, in units of Mpc/h
	# MAS=‘CIC’ is the scheme we use to interpolate 
	# what the below line returns is an array with the interpolated positions of the halos in the grid
	# i.e. it returns n_h, where n_h is the halo number density
	# since we want the power spectrum we need delta_h = n_h / <n_h> -1



	# create a list to match z with the snapshot denomination
	snaplist = ['3.0','2.8','2.6','2.4','2.2','2.0','1.9','1.8','1.7','1.6','1.5','1.4','1.3','1.2','1.1','1.0','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0.0']
	snapz = snaplist.index(str(z))
	snaplist2 = ['2.0','1.0','0.5','0.0']
	snapz2 = snaplist2.index(str(z))

	dispo = [1,2,3,5,6,7,8]
	for i in xrange(1,11):
		base = '/home/dvalcin/codes/Paco/data2/0.0eV/NCV'+str(i)
		
		if i == 1 or i == 2:
			fof = FoF_catalog(base, snapz,read_IDs=False)
		else:
			fof = FoF_catalog(base, snapz2,read_IDs=False)

		Masses=fof.GroupMass * 1e10 # for masses 10^10 Msun/h
		pos = fof.GroupPos/1e3 #Mpc/h

	


	########################################################################
	############ compute the halo mass ranges 
	########################################################################

		limM = [5e11,1e12,3e12,1e13]
		#~ Hmass_ind_a = np.where((limM[1] >= Masses) & (Masses >= limM[0]) )[0]
		#~ #----------------------------------------------------------------------*
		#~ Hmass_ind_b = np.where((limM[2] >= Masses) & (Masses >= limM[1]) )[0]
		#~ #-----------------------------------------------------------------------
		#~ Hmass_ind_c = np.where((limM[3] >= Masses) & (Masses >= limM[2]) )[0]
		Hmass_ind_a = np.where(Masses >= limM[0])[0]
		#----------------------------------------------------------------------*
		Hmass_ind_b = np.where(Masses >= limM[1])[0]
		#-----------------------------------------------------------------------
		Hmass_ind_c = np.where(Masses >= limM[2])[0]
		#-----------------------------------------------------------------------
		Hmass_ind_d = np.where(Masses >= limM[3])[0]



		Hmass_a = Masses[Hmass_ind_a]
		#-------------------------------
		Hmass_b = Masses[Hmass_ind_b]
		#-------------------------------
		Hmass_c = Masses[Hmass_ind_c]
		#-------------------------------
		Hmass_d = Masses[Hmass_ind_d]


		#~ Hmass_a = (Hmass_1a + Hmass_2a)/2
		#~ Hmass_b = (Hmass_1b + Hmass_2b)/2
		#~ Hmass_c = (Hmass_1c + Hmass_2c)/2
		#~ Hmass_d = (Hmass_1d + Hmass_2d)/2




		########################################################################
		##### compute the velocity dispersion of the halos
		########################################################################
		#~ Vel1 = fof1.GroupVel * (1+z) # for physical velocities in km/s
		#~ Vel2 = fof2.GroupVel * (1+z) # for physical velocities in km/s
		#~ Vel1a = Vel1[Hmass_ind_1a]
		#~ Vel2a = Vel2[Hmass_ind_2a]
		#~ Vel1b = Vel1[Hmass_ind_1b]
		#~ Vel2b = Vel2[Hmass_ind_2b]
		#~ Vel1c = Vel1[Hmass_ind_1c]
		#~ Vel2c = Vel2[Hmass_ind_2c]
		#~ Vel1d = Vel1[Hmass_ind_1d]
		#~ Vel2d = Vel2[Hmass_ind_2d]

		#~ sigvel1a = np.std(Vel1a, dtype=np.float64)
		#~ sigvel2a = np.std(Vel2a, dtype=np.float64)
		#~ sigvel1b = np.std(Vel1b, dtype=np.float64)
		#~ sigvel2b = np.std(Vel2b, dtype=np.float64)
		#~ sigvel1c = np.std(Vel1c, dtype=np.float64)
		#~ sigvel2c = np.std(Vel2c, dtype=np.float64)
		#~ sigvel1d = np.std(Vel1d, dtype=np.float64)
		#~ sigvel2d = np.std(Vel2d, dtype=np.float64)

		#~ sigvela = (sigvel1a + sigvel2a)/2
		#~ sigvelb = (sigvel1b + sigvel2b)/2
		#~ sigvelc = (sigvel1c + sigvel2c)/2
		#~ sigveld = (sigvel1d + sigvel2d)/2


		#####################################################################
		########## compute the halo mass function
		#####################################################################
		#~ print np.max(Masses1), np.min(Masses1), np.log10(np.max(Masses1))
		#~ bins = np.logspace(11,16,32)
		#~ bins1 = np.logspace(np.log10(np.min(Masses1)),np.log10(np.max(Masses1)),32)
		#~ bins2 = np.logspace(np.log10(np.min(Masses2)),np.log10(np.max(Masses2)),32)
		#~ cmf = MFL.Crocce_mass_function(kcamb,pcamb,Omega_b + Omega_c,z,1e11,1e16,32,Masses=None)
		#~ tmf = MFL.Tinker_mass_function(kcamb,pcamb,Omega_b + Omega_c,z,1e11,1e16,32,Masses=None)
		#~ wmf = MFL.Watson_mass_function_FoF(kcamb,pcamb,Omega_b + Omega_c,1e11,1e16,32,Masses=None)
		#~ smf = MFL.ST_mass_function(kcamb,pcamb,Omega_b + Omega_c,1e11,1e16,32,Masses=None)
		#~ print np.shape(cmf)

		# halo mass function
		#~ plt.figure()
		#~ hist1, binedge1 = np.histogram(Masses1, bins1)
		#~ hist2, binedge2 = np.histogram(Masses2, bins2)
		#~ deltaM1=binedge1[1:]-binedge1[:-1] #size of the bin
		#~ deltaM2=binedge1[1:]-binedge2[:-1] #size of the bin
		#~ M_middle1=10**(0.5*(np.log10(binedge1[1:])+np.log10(binedge1[:-1]))) #center of the bin
		#~ M_middle2=10**(0.5*(np.log10(binedge2[1:])+np.log10(binedge2[:-1]))) #center of the bin


		#~ hist = (hist1 + hist2)/2
		#~ M_middle = (M_middle1 + M_middle2)/2
		#~ deltaM = (deltaM1+deltaM2)/2
		#~ dndm = hist/1e9/deltaM

		#~ print dndm[0:10]

		#~ plt.plot(bins,cmf[1], color='r', label='Crocce mass function')
		#~ plt.plot(bins,tmf[1], color='g', label='Tinker mass function')
		#~ plt.plot(bins,wmf[1], color='b', label='Watson mass function')
		#~ plt.plot(bins,smf[1], color='gold', label='SMT mass function')
		#~ plt.scatter(M_middle, dndm, color='k', label='N-body')
		#~ plt.title('halo mass function at z = '+str(z))
		#~ plt.xlabel('Mass Msun/h')
		#~ plt.ylabel('dn/dM')
		#~ plt.legend(loc='lower left')
		#~ plt.yscale('log')
		#~ plt.xscale('log')
		#~ plt.ylim(1e-28,1e-12)
		#~ plt.show()
		#~ plt.savefig('/home/dvalcin/plots/halo_mass_funct_at_z= '+str(z)+'.png', dpi = 500)



		# halo mass function residuals
		#~ CMF = np.interp(M_middle, bins, cmf[1])
		#~ TMF = np.interp(M_middle, bins, tmf[1])
		#~ WMF = np.interp(M_middle, bins, wmf[1])
		#~ SMF = np.interp(M_middle, bins, smf[1])
		#~ plt.figure()
		#~ plt. plot(M_middle, dndm/CMF, color='r', label='Crocce mass function')
		#~ plt. plot(M_middle, dndm/TMF, color='g', label='Tinker mass function')
		#~ plt. plot(M_middle, dndm/WMF, color='b', label='Watson mass function')
		#~ plt. plot(M_middle, dndm/SMF, color='gold', label='SMT mass function')
		#~ plt.xlabel('Mass Msun/h')
		#~ plt.ylabel('residuals')
		#~ plt.legend(loc='upper left')
		#~ plt.title('residuals at z = '+str(z))
		#~ plt.xscale('log')
		#~ plt.axhline(1, color='k')
		#~ plt.ylim(0,2)
		#~ plt.show()
		#~ plt.savefig('/home/dvalcin/plots/residual_halo_mass_funct_at_z= '+str(z)+'.png', dpi = 500)






		#~ ##################################################################
		#~ ########## First mass range 
		#~ ##################################################################


		delta1a = np.zeros((dims,dims,dims), dtype=np.float32)
		MASL.MA(pos[Hmass_ind_a], delta1a, BoxSize, MAS='CIC', W=None)  
		delta1a = delta1a/np.mean(delta1a, dtype=np.float64) - 1.0

		# compute power spectra
		Pk1a = PKL.Pk(delta1a, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra

		#shot noise
		Pshot_m1 = 1/(len(Hmass_a)/BoxSize**3)

		# 3D Pk
		k_m1= Pk1a.k3D
		#~ Pk0_m1 = (Pk1a.Pk[:,0] + Pk2a.Pk[:,0] + Pk3a.Pk[:,0] + Pk5a.Pk[:,0] + Pk6a.Pk[:,0] + Pk7a.Pk[:,0] + Pk8a.Pk[:,0])/7 #monopole
		#~ Pk2_m1 = (Pk1a.Pk[:,1] + Pk2a.Pk[:,1] + Pk3a.Pk[:,1] + Pk5a.Pk[:,1] + Pk6a.Pk[:,1] + Pk7a.Pk[:,1] + Pk8a.Pk[:,1])/7 #quadrupole
		#~ Pk4_m1 = (Pk1a.Pk[:,2] + Pk2a.Pk[:,2] + Pk3a.Pk[:,2] + Pk5a.Pk[:,2] + Pk6a.Pk[:,2] + Pk7a.Pk[:,2] + Pk8a.Pk[:,2])/7 #hexadecapole
		#~ Nmodes_m1 = (Pk1a.Nmodes3D + Pk2a.Nmodes3D + Pk3a.Nmodes3D + Pk5a.Nmodes3D + Pk6a.Nmodes3D + Pk7a.Nmodes3D + Pk8a.Nmodes3D)/7 #number of modes in each Pk bin

		#~ temp1 = np.array([Pk1a.Pk[:,0], Pk2a.Pk[:,0], Pk3a.Pk[:,0], Pk5a.Pk[:,0], Pk6a.Pk[:,0], Pk7a.Pk[:,0], Pk8a.Pk[:,0]])
		#~ std1 = np.std(temp1, axis=0)

		cname = '/home/dvalcin/plots/Phh1_realisation_'+str(nu)+'_z='+str(z)+'.txt'
		if i == 1:
			with open(cname, 'w') as fid_file:
				for index_k in xrange(len(k_m1)):
					fid_file.write('%.8g %.8g %.8g\n' % (Pk1a.Pk[index_k,0],k_m1[index_k],Pshot_m1))
			fid_file.close()
		else:
			#Create temporary file read/write
			t = tempfile.NamedTemporaryFile(mode="r+")
			#Open input file read-only
			i = open(cname, 'r')
			#Copy input file to temporary file, modifying as we go
			for line in i:
				t.write(line.rstrip()+"\n")
			i.close() #Close input file
			t.seek(0) #Rewind temporary file to beginning
			o = open(cname, "w")  #Reopen input file writable
			#Overwriting original file with temporary file contents          
			for i,line in enumerate(t):
				o.write('%.8g' %(Pk1a.Pk[i,0])+' '+'%.8g' %(Pshot_m1)+' '+line)   
			t.close() #Close temporary file, will cause it to be deleted
			o.close()

		#~ # 2D Pk
		#~ kpar_m1 = (Pk1a.kpar + Pk2a.kpar)/2
		#~ kperp_m1 = (Pk1a.kper + Pk2a.kper)/2
		#~ Pk2D_m1 = (Pk1a.Pk2D + Pk2a.Pk2D)/2
		#~ Nmodes2D_m1 = (Pk1a.Nmodes2D + Pk2a.Nmodes2D)/2

		#~ # clear variable for memory
		del delta1a
		del Pk1a

		#~ ####################################################################
		#~ ########## second mass range
		#~ ####################################################################


		delta1b = np.zeros((dims,dims,dims), dtype=np.float32)
		MASL.MA(pos[Hmass_ind_b], delta1b, BoxSize, MAS='CIC', W=None)  
		delta1b = delta1b/np.mean(delta1b, dtype=np.float64) - 1.0


		# compute power spectra
		Pk1b = PKL.Pk(delta1b, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra

		#shot noise
		Pshot_m2 = 1/(len(Hmass_b)/BoxSize**3)

		# 3D Pk
		k_m2= Pk1b.k3D
		#~ Pk0_m2 = (Pk1b.Pk[:,0] + Pk2b.Pk[:,0] + Pk3b.Pk[:,0] + Pk5b.Pk[:,0] + Pk6b.Pk[:,0] + Pk7b.Pk[:,0] + Pk8b.Pk[:,0])/7 #monopole
		#~ Pk2_m2 = (Pk1b.Pk[:,1] + Pk2b.Pk[:,1] + Pk3b.Pk[:,1] + Pk5b.Pk[:,1] + Pk6b.Pk[:,1] + Pk7b.Pk[:,1] + Pk8b.Pk[:,1])/7 #quadrupole
		#~ Pk4_m2 = (Pk1b.Pk[:,2] + Pk2b.Pk[:,2] + Pk3b.Pk[:,2] + Pk5b.Pk[:,2] + Pk6b.Pk[:,2] + Pk7b.Pk[:,2] + Pk8b.Pk[:,2])/7 #hexadecapole
		#~ Nmodes_m2 = (Pk1b.Nmodes3D + Pk2b.Nmodes3D + Pk3b.Nmodes3D + Pk5b.Nmodes3D + Pk6b.Nmodes3D + Pk7b.Nmodes3D + Pk8b.Nmodes3D)/7 #number of modes in each Pk bin

		#~ temp2 = np.array([Pk1b.Pk[:,0], Pk2b.Pk[:,0], Pk3b.Pk[:,0], Pk5b.Pk[:,0], Pk6b.Pk[:,0], Pk7b.Pk[:,0], Pk8b.Pk[:,0]])
		#~ std2 = np.std(temp2, axis=0)
		cname = '/home/dvalcin/plots/Phh2_realisation_'+str(nu)+'_z='+str(z)+'.txt'
		if i == 1:
			with open(cname, 'w') as fid_file:
				for index_k in xrange(len(k_m2)):
					fid_file.write('%.8g %.8g %.8g\n' % (Pk1b.Pk[index_k,0],k_m2[index_k],Pshot_m2))
			fid_file.close()
		else:
			#Create temporary file read/write
			t = tempfile.NamedTemporaryFile(mode="r+")
			#Open input file read-only
			i = open(cname, 'r')
			#Copy input file to temporary file, modifying as we go
			for line in i:
				t.write(line.rstrip()+"\n")
			i.close() #Close input file
			t.seek(0) #Rewind temporary file to beginning
			o = open(cname, "w")  #Reopen input file writable
			#Overwriting original file with temporary file contents          
			for i,line in enumerate(t):
				o.write('%.8g' %(Pk1b.Pk[i,0])+' '+'%.8g' %(Pshot_m2)+' '+line)   
			t.close() #Close temporary file, will cause it to be deleted
			o.close()

		#~ # 2D Pk
		#~ kpar_m1 = (Pk1a.kpar + Pk2a.kpar)/2
		#~ kperp_m1 = (Pk1a.kper + Pk2a.kper)/2
		#~ Pk2D_m1 = (Pk1a.Pk2D + Pk2a.Pk2D)/2
		#~ Nmodes2D_m1 = (Pk1a.Nmodes2D + Pk2a.Nmodes2D)/2

		#~ # clear variable for memory
		del delta1b
		del Pk1b


		#~ ###############################################################
		#~ ######## third mass range
		#~ ###############################################################


		delta1c = np.zeros((dims,dims,dims), dtype=np.float32)
		MASL.MA(pos[Hmass_ind_c], delta1c, BoxSize, MAS='CIC', W=None)  
		delta1c = delta1c/np.mean(delta1c, dtype=np.float64) - 1.0

		# compute power spectra
		Pk1c = PKL.Pk(delta1c, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra

		#shot noise
		Pshot_m3 = 1/(len(Hmass_c)/BoxSize**3)

		# 3D Pk
		k_m3= Pk1c.k3D
		#~ Pk0_m3 = (Pk1c.Pk[:,0] + Pk2c.Pk[:,0] + Pk3c.Pk[:,0] + Pk5c.Pk[:,0] + Pk6c.Pk[:,0] + Pk7c.Pk[:,0] + Pk8c.Pk[:,0])/7 #monopole
		#~ Pk2_m3 = (Pk1c.Pk[:,1] + Pk2c.Pk[:,1] + Pk3c.Pk[:,1] + Pk5c.Pk[:,1] + Pk6c.Pk[:,1] + Pk7c.Pk[:,1] + Pk8c.Pk[:,1])/7 #quadrupole
		#~ Pk4_m3 = (Pk1c.Pk[:,2] + Pk2c.Pk[:,2] + Pk3c.Pk[:,2] + Pk5c.Pk[:,2] + Pk6c.Pk[:,2] + Pk7c.Pk[:,2] + Pk8c.Pk[:,2])/7 #hexadecapole
		#~ Nmodes_m3 = (Pk1c.Nmodes3D + Pk2c.Nmodes3D + Pk3c.Nmodes3D + Pk5c.Nmodes3D + Pk6c.Nmodes3D + Pk7c.Nmodes3D + Pk8c.Nmodes3D)/7 #number of modes in each Pk bin

		#~ temp3 = np.array([Pk1c.Pk[:,0], Pk2c.Pk[:,0], Pk3c.Pk[:,0], Pk5c.Pk[:,0], Pk6c.Pk[:,0], Pk7c.Pk[:,0], Pk8c.Pk[:,0]])
		#~ std3 = np.std(temp3, axis=0)
		cname = '/home/dvalcin/plots/Phh3_realisation_'+str(nu)+'_z='+str(z)+'.txt'
		if i == 1:
			with open(cname, 'w') as fid_file:
				for index_k in xrange(len(k_m3)):
					fid_file.write('%.8g %.8g %.8g\n' % (Pk1c.Pk[index_k,0],k_m3[index_k],Pshot_m3))
			fid_file.close()
		else:
			#Create temporary file read/write
			t = tempfile.NamedTemporaryFile(mode="r+")
			#Open input file read-only
			i = open(cname, 'r')
			#Copy input file to temporary file, modifying as we go
			for line in i:
				t.write(line.rstrip()+"\n")
			i.close() #Close input file
			t.seek(0) #Rewind temporary file to beginning
			o = open(cname, "w")  #Reopen input file writable
			#Overwriting original file with temporary file contents          
			for i,line in enumerate(t):
				o.write('%.8g' %(Pk1c.Pk[i,0])+' '+'%.8g' %(Pshot_m3)+' '+line)   
			t.close() #Close temporary file, will cause it to be deleted
			o.close()

		#~ # 2D Pk
		#~ kpar_m1 = (Pk1a.kpar + Pk2a.kpar)/2
		#~ kperp_m1 = (Pk1a.kper + Pk2a.kper)/2
		#~ Pk2D_m1 = (Pk1a.Pk2D + Pk2a.Pk2D)/2
		#~ Nmodes2D_m1 = (Pk1a.Nmodes2D + Pk2a.Nmodes2D)/2

		#~ # clear variable for memory
		del delta1c
		del Pk1c


		###############################################################
		######## fourth mass range
		###############################################################


		delta1d = np.zeros((dims,dims,dims), dtype=np.float32)
		MASL.MA(pos[Hmass_ind_d], delta1d, BoxSize, MAS='CIC', W=None)  
		delta1d = delta1d/np.mean(delta1d, dtype=np.float64) - 1.0

		# compute power spectra
		Pk1d = PKL.Pk(delta1d, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra

		#shot noise
		Pshot_m4 = 1/(len(Hmass_d)/BoxSize**3)
		
		# 3D Pk
		k_m4= Pk1d.k3D
		#~ Pk0_m4 = (Pk1d.Pk[:,0] + Pk2d.Pk[:,0] + Pk3d.Pk[:,0] + Pk5d.Pk[:,0] + Pk6d.Pk[:,0] + Pk7d.Pk[:,0] + Pk8d.Pk[:,0])/7 #monopole
		#~ Pk2_m4 = (Pk1d.Pk[:,1] + Pk2d.Pk[:,1] + Pk3d.Pk[:,1] + Pk5d.Pk[:,1] + Pk6d.Pk[:,1] + Pk7d.Pk[:,1] + Pk8d.Pk[:,1])/7 #quadrupole
		#~ Pk4_m4 = (Pk1d.Pk[:,2] + Pk2d.Pk[:,2] + Pk3d.Pk[:,2] + Pk5d.Pk[:,2] + Pk6d.Pk[:,2] + Pk7d.Pk[:,2] + Pk8d.Pk[:,2])/7 #hexadecapole
		#~ Nmodes_m4 = (Pk1d.Nmodes3D + Pk2d.Nmodes3D + Pk3d.Nmodes3D + Pk5d.Nmodes3D + Pk6d.Nmodes3D + Pk7d.Nmodes3D + Pk8d.Nmodes3D)/7 #number of modes in each Pk bin

		#~ temp4 = np.array([Pk1d.Pk[:,0], Pk2d.Pk[:,0], Pk3d.Pk[:,0], Pk5d.Pk[:,0], Pk6d.Pk[:,0], Pk7d.Pk[:,0], Pk8d.Pk[:,0]])
		#~ std4 = np.std(temp4, axis=0)
		cname = '/home/dvalcin/plots/Phh4_realisation_'+str(nu)+'_z='+str(z)+'.txt'
		if i == 1:
			with open(cname, 'w') as fid_file:
				for index_k in xrange(len(k_m4)):
					fid_file.write('%.8g %.8g %.8g\n' % (Pk1d.Pk[index_k,0],k_m4[index_k],Pshot_m4))
			fid_file.close()
		else:
			#Create temporary file read/write
			t = tempfile.NamedTemporaryFile(mode="r+")
			#Open input file read-only
			i = open(cname, 'r')
			#Copy input file to temporary file, modifying as we go
			for line in i:
				t.write(line.rstrip()+"\n")
			i.close() #Close input file
			t.seek(0) #Rewind temporary file to beginning
			o = open(cname, "w")  #Reopen input file writable
			#Overwriting original file with temporary file contents          
			for i,line in enumerate(t):
				o.write('%.8g' %(Pk1d.Pk[i,0])+' '+'%.8g' %(Pshot_m4)+' '+line)  
			t.close() #Close temporary file, will cause it to be deleted
			o.close()


		#~ # 2D Pk
		#~ kpar_m1 = (Pk1a.kpar + Pk2a.kpar)/2
		#~ kperp_m1 = (Pk1a.kper + Pk2a.kper)/2
		#~ Pk2D_m1 = (Pk1a.Pk2D + Pk2a.Pk2D)/2
		#~ Nmodes2D_m1 = (Pk1a.Nmodes2D + Pk2a.Nmodes2D)/2

		#~ # clear variable for memory
		del delta1d
		del Pk1d


	#~ #####################################################################
	#~ ################# Shoit noise removal ###############################
	#~ #####################################################################

	#~ Pshot_m1 = (1/(len(Hmass_1a)/BoxSize**3) + 1/(len(Hmass_2a)/BoxSize**3))/2
	#~ Pshot_m2 = (1/(len(Hmass_1b)/BoxSize**3) + 1/(len(Hmass_2b)/BoxSize**3))/2
	#~ Pshot_m3 = (1/(len(Hmass_1c)/BoxSize**3) + 1/(len(Hmass_2c)/BoxSize**3))/2
	#~ Pshot_m4 = (1/(len(Hmass_1d)/BoxSize**3) + 1/(len(Hmass_2d)/BoxSize**3))/2

	#~ plt.figure()
	#~ plt.plot(k_m1, Pk0_m1, label='first mass bin', color='b')
	#~ plt.axhline(Pshot_m1, color='b', linestyle='--')
	#~ plt.plot(k_m2, Pk0_m2, label='second mass bin', color='r')
	#~ plt.axhline(Pshot_m2, color='r', linestyle='--')
	#~ plt.plot(k_m3, Pk0_m3, label='third mass bin', color='g')
	#~ plt.axhline(Pshot_m3, color='g', linestyle='--')
	#~ plt.plot(k_m4, Pk0_m4, label='fourth mass bin', color='k')
	#~ plt.axhline(Pshot_m4, color='k', linestyle='--')
	#~ plt.title('3d power spectrum at z = '+str(z))
	#~ plt.xlabel('P(k)')
	#~ plt.ylabel('k')
	#~ plt.legend(loc='upper right')
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(8e-3,1.5)
	#~ plt.ylim(1e3,2e6)
	#~ plt.show()
	#~ plt.savefig('/home/dvalcin/plots/shot_noise_at_z= '+str(z)+'.png', dpi = 500)


	#~ Pk0_m1 = Pk0_m1 - Pshot_m1
	#~ Pk0_m2 = Pk0_m2 - Pshot_m2
	#~ Pk0_m3 = Pk0_m3 - Pshot_m3
	#~ Pk0_m4 = Pk0_m4 - Pshot_m4

	#~ Pkshot1 = Pk0_m1 + Pshot_m1
	#~ Pkshot2 = Pk0_m2 + Pshot_m2
	#~ Pkshot3 = Pk0_m3 + Pshot_m3
	#~ Pkshot4 = Pk0_m4 + Pshot_m4

	#~ ########################################################################
	#~ ############ compute the halo bias
	#~ ########################################################################


	#~ bhh1 = np.sqrt((Pk0_m1)/Pk0)
	#~ bhh2 = np.sqrt((Pk0_m2)/Pk0)
	#~ bhh3 = np.sqrt((Pk0_m3)/Pk0)
	# since for m4 the shot noise is sometimes higher than the power spectrum, we put bias to 0 at these scales
	#~ nul = np.where(Pk0_m4 < 0)[0]
	#~ Pk0_m4[nul] = 0
	#~ bhh4 = np.sqrt((Pk0_m4)/Pk0)


	#~ plt.figure()
	#~ plt.plot(k_m1, bhh1, label='first mass bin', color='b')
	#~ plt.plot(k_m2, bhh2, label='second mass bin', color='r')
	#~ plt.plot(k_m3, bhh3, label='third mass bin', color='g')
	#~ plt.plot(k_m4, bhh4, label='fourth mass bin', color='k')
	#~ plt.title('bias at z = '+str(z))
	#~ plt.xlabel('P(k)')
	#~ plt.ylabel('k')
	#~ plt.legend(loc='lower left')
	#~ plt.xscale('log')
	#~ plt.xlim(8e-3,1.5)
	#~ plt.ylim(0,2)
	#~ plt.ylim(0,7)
	#~ plt.show()
	#~ plt.savefig('/home/dvalcin/plots/bias_at_z= '+str(z)+'.png', dpi = 500)

	camb = np.loadtxt('/home/dvalcin/codes/Paco/data2/0.0eV/CAMB/expected_Pk_z='+str(z)+'.txt')
	cname = '/home/dvalcin/codes/Paco/data2/0.0eV/CAMB/expected_Pk_z='+str(z)+'.txt'
	kcamb = camb[:,0]
	pcamb = camb[:,1]

	#~ # TINKER
	Tb1 = halo_bias('Tinker', z, limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
	Tb2 = halo_bias('Tinker', z, limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
	Tb3 = halo_bias('Tinker', z, limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
	Tb4 = halo_bias('Tinker', z, limM[3],np.max(Masses), cname,Omega_c, Omega_b, do_DM=True )


	#~ # TORMEN
	#~ Sb1 = halo_bias('SMTT', z, limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb2 = halo_bias('SMTT', z, limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb3 = halo_bias('SMTT', z, limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb4 = halo_bias('SMTT', z, limM[3],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb1 = halo_bias('SMT01', z, limM[0],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb2 = halo_bias('SMT01', z, limM[1],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb3 = halo_bias('SMT01', z, limM[2],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )
	#~ Sb4 = halo_bias('SMT01', z, limM[3],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )

	#~ # CROCCE
	Cb1 = halo_bias('Crocce', z, limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
	Cb2 = halo_bias('Crocce', z, limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
	Cb3 = halo_bias('Crocce', z, limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
	Cb4 = halo_bias('Crocce', z, limM[3],np.max(Masses), cname,Omega_c, Omega_b, do_DM=True )

	#~ # WATSON
	#~ Wb1 = halo_bias('Watson', z, limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
	#~ Wb2 = halo_bias('Watson', z, limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
	#~ Wb3 = halo_bias('Watson', z, limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
	#~ Wb4 = halo_bias('Watson', z, limM[3],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )


	#~ print Tb1, Tb2, Tb3, Tb4
	#~ print Sb1, Sb2, Sb3, Sb4
	#~ print Cb1, Cb2, Cb3, Cb4

	########################################################################
	####### compute the fitting coefficient from Raccanelli et al.
	########################################################################
	#~ kcoeff = [0.12,0.16,0.2,0.2 ]
	#~ red = ['0.0','0.5','1.0','2.0','3.0']
	#~ ind = red.index(str(z))
	#~ klim = np.where(k <= kcoeff[ind])[0]# to cut negative values from shot noise removal, k, k_m1, k_m2, etc. are all the same
	#~ print klim

	#--- Tinker as large scale bias
	#~ def funcTb1(k, b2, b3, b4):
			#~ return Tb1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcTb2(k, b2, b3, b4):
			#~ return Tb2 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcTb3(k, b2, b3, b4):
			#~ return Tb3 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcTb4(k, b2, b3, b4):
			#~ return Tb4 + b2 * k**2 + b3 * k**3 + b4 * k**4 



	#~ popT1, pcovT1 = curve_fit(funcTb1, k_m1[klim], bhh1[klim], check_finite=True)
	#~ popT2, pcovT2 = curve_fit(funcTb2, k_m2[klim], bhh2[klim], check_finite=True)
	#~ popT3, pcovT3 = curve_fit(funcTb3, k_m3[klim], bhh3[klim], check_finite=True)
	#~ popT4, pcovT4 = curve_fit(funcTb4, k_m4[klim], bhh4[klim], check_finite=True)

	#~ perrT1 = np.sqrt(np.diag(pcovT1))
	#~ perrT2 = np.sqrt(np.diag(pcovT2))
	#~ perrT3 = np.sqrt(np.diag(pcovT3))
	#~ perrT4 = np.sqrt(np.diag(pcovT4))

	#~ #--- Crocce as large scale bias
	#~ def funcCb1(k, b2, b3, b4):
			#~ return Cb1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcCb2(k, b2, b3, b4):
			#~ return Cb2 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcCb3(k, b2, b3, b4):
			#~ return Cb3 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcCb4(k, b2, b3, b4):
			#~ return Cb4 + b2 * k**2 + b3 * k**3 + b4 * k**4 
		

	#~ popC1, pcovC1 = curve_fit(funcCb1, k_m1[klim], bhh1[klim], check_finite=True)
	#~ popC2, pcovC2 = curve_fit(funcCb2, k_m2[klim], bhh2[klim], check_finite=True)
	#~ popC3, pcovC3 = curve_fit(funcCb3, k_m3[klim], bhh3[klim], check_finite=True)
	#~ popC4, pcovC4 = curve_fit(funcCb4, k_m4[klim], bhh4[klim], check_finite=True)


	#~ #--- Watson as large scale bias
	#~ def funcWb1(k, b2, b3, b4):
			#~ return Wb1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcWb2(k, b2, b3, b4):
			#~ return Wb2 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcWb3(k, b2, b3, b4):
			#~ return Wb3 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcWb4(k, b2, b3, b4):
			#~ return Wb4 + b2 * k**2 + b3 * k**3 + b4 * k**4 
		

	#~ popW1, pcovW1 = curve_fit(funcWb1, k_m1[klim], bhh1[klim], check_finite=True)
	#~ popW2, pcovW2 = curve_fit(funcWb2, k_m2[klim], bhh2[klim], check_finite=True)
	#~ popW3, pcovW3 = curve_fit(funcWb3, k_m3[klim], bhh3[klim], check_finite=True)
	#~ popW4, pcovW4 = curve_fit(funcWb4, k_m4[klim], bhh4[klim], check_finite=True)



	#~ #--- SMT as large scale bias
	#~ def funcSb1(k, b2, b3, b4):
			#~ return Sb1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	#~ def funcSb2(k, b2, b3, b4):
			#~ return Sb2 + b2 * k**2 + b3 * k**3 + b4 * k**4
	#~ def funcSb3(k, b2, b3, b4):
			#~ return Sb3 + b2 * k**2 + b3 * k**3 + b4 * k**4
	#~ def funcSb4(k, b2, b3, b4):
			#~ return Sb4 + b2 * k**2 + b3 * k**3 + b4 * k**4
		

	#~ popS1, pcovS1 = curve_fit(funcSb1, k_m1[klim], bhh1[klim],  check_finite=True)
	#~ popS2, pcovS2 = curve_fit(funcSb2, k_m2[klim], bhh2[klim],  check_finite=True)
	#~ popS3, pcovS3 = curve_fit(funcSb3, k_m3[klim], bhh3[klim],  check_finite=True)
	#~ popS4, pcovS4 = curve_fit(funcSb4, k_m4[klim], bhh4[klim],  check_finite=True)

	#~ #--- free parameter as large scale bias
	#~ def funcb(k, b1, b2, b3, b4):
			#~ return b1 + b2 * k**2 + b3 * k**3 + b4 * k**4 

		
	#~ pop1, pcov1 = curve_fit(funcb, k_m1[klim], bhh1[klim],  check_finite=True)
	#~ pop2, pcov2 = curve_fit(funcb, k_m2[klim], bhh2[klim],  check_finite=True)
	#~ pop3, pcov3 = curve_fit(funcb, k_m3[klim], bhh3[klim],  check_finite=True)
	#~ pop4, pcov4 = curve_fit(funcb, k_m4[klim], bhh4[klim],  check_finite=True)

	#~ perr1 = np.sqrt(np.diag(pcov1))
	#~ perr2 = np.sqrt(np.diag(pcov2))
	#~ perr3 = np.sqrt(np.diag(pcov3))
	#~ perr4 = np.sqrt(np.diag(pcov4))

	########################################################################
	############# save halo spectrum 
	########################################################################

	cname = '/home/dvalcin/plots/coeff_large-scale_'+str(nu)+'_z='+str(z)+'.txt'
	with open(cname, 'w') as fid_file:
		fid_file.write('%s %s %s %s %s %s %s %s\n' % ('Tinker-m1', 'tinker-m2', 'tinker-m3', 'tinker-m4',\
		'Crocce-m1', 'Crocce-m2', 'Crocce-m3', 'Crocce-m4'))
		fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (Tb1, Cb1, Tb2, Cb2, Tb3, Cb3, Tb4, Cb4,))
	print '\n'

	#~ #-----------------------------------------------------------------------
	#~ cname = '/home/dvalcin/plots/Phh_bias_'+str(nu)+'_m1_z='+str(z)+'.txt'
	#~ with open(cname, 'w') as fid_file:
		#~ fid_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ('Tinker', \
		#~ 'tinker-fit-b2', 'tinker-fit-b3', 'tinker-fit-b4', 'free-fit-b1', 'free-fit-b2','free-fit-b3', 'free-fit-b4', \
		 #~ 'Crocce', 'Crocce-fit-b2', 'Crocce-fit-b3', 'Crocce-fit-b4', 'perrT1', 'perrT1', 'perrT1',\
		 #~ 'perr1', 'perr1', 'perr1', 'perr1','velocity-disp'))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' \
		#~ % (Tb1, popT1[0], popT1[1], popT1[2], pop1[0], pop1[1], pop1[2], pop1[3], Cb1, popC1[0], popC1[1], popC1[2],\
		#~ perrT1[0], perrT1[1], perrT1[2], perr1[0], perr1[1], perr1[2], perr1[3], sigvela))
	#~ print '\n'

	#~ cname = '/home/dvalcin/plots/Phh_bias_'+str(nu)+'_m2_z='+str(z)+'.txt'
	#~ with open(cname, 'w') as fid_file:
		#~ fid_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ('Tinker', \
		#~ 'tinker-fit-b2', 'tinker-fit-b3', 'tinker-fit-b4', 'free-fit-b1', 'free-fit-b2','free-fit-b3', 'free-fit-b4', \
		 #~ 'Crocce', 'Crocce-fit-b2', 'Crocce-fit-b3', 'Crocce-fit-b4', 'perrT1', 'perrT1', 'perrT1',\
		 #~ 'perr1', 'perr1', 'perr1', 'perr1','velocity-disp'))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' \
		#~ % (Tb2, popT2[0], popT2[1], popT2[2], pop2[0], pop2[1], pop2[2], pop2[3], Cb2, popC2[0], popC2[1], popC2[2],\
		#~ perrT2[0], perrT2[1], perrT2[2], perr2[0], perr2[1], perr2[2], perr2[3], sigvelb))
	#~ print '\n'

	#~ cname = '/home/dvalcin/plots/Phh_bias_'+str(nu)+'_m3_z='+str(z)+'.txt'
	#~ with open(cname, 'w') as fid_file:
		#~ fid_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ('Tinker', \
		#~ 'tinker-fit-b2', 'tinker-fit-b3', 'tinker-fit-b4', 'free-fit-b1', 'free-fit-b2','free-fit-b3', 'free-fit-b4', \
		 #~ 'Crocce', 'Crocce-fit-b2', 'Crocce-fit-b3', 'Crocce-fit-b4', 'perrT1', 'perrT1', 'perrT1',\
		 #~ 'perr1', 'perr1', 'perr1', 'perr1','velocity-disp'))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' \
		#~ % (Tb3, popT3[0], popT3[1], popT3[2], pop3[0], pop3[1], pop3[2], pop3[3], Cb3, popC3[0], popC3[1], popC3[2],\
		#~ perrT3[0], perrT3[1], perrT3[2], perr3[0], perr3[1], perr3[2], perr3[3], sigvelc))
	#~ print '\n'

	#~ cname = '/home/dvalcin/plots/Phh_bias_'+str(nu)+'_m4_z='+str(z)+'.txt'
	#~ with open(cname, 'w') as fid_file:
		#~ fid_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ('Tinker', \
		#~ 'tinker-fit-b2', 'tinker-fit-b3', 'tinker-fit-b4', 'free-fit-b1', 'free-fit-b2','free-fit-b3', 'free-fit-b4', \
		 #~ 'Crocce', 'Crocce-fit-b2', 'Crocce-fit-b3', 'Crocce-fit-b4', 'perrT1', 'perrT1', 'perrT1',\
		 #~ 'perr1', 'perr1', 'perr1', 'perr1','velocity-disp'))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' \
		#~ % (Tb4, popT4[0], popT4[1], popT4[2], pop4[0], pop4[1], pop4[2], pop4[3], Cb4, popC4[0], popC4[1], popC4[2],\
		#~ perrT4[0], perrT4[1], perrT4[2], perr4[0], perr4[1], perr4[2], perr4[3], sigveld))
	#~ print '\n'








	########################################################################
	########### Plots
	#######################################################################

	#~ plt.figure()
	#~ plt.plot(k_m1,Pk0_m1, label=str(limM[0]) + ' =< M =< ' + str(limM[1]), color='b')
	#~ plt.plot(k_m2,Pk0_m2, label=str(limM[1]) + ' =< M =< ' + str(limM[2]), color='r')
	#~ plt.plot(k_m3,Pk0_m3, label=str(limM[2]) + ' =< M =< ' + str(limM[3]), color='g')
	#~ plt.plot(k_m4,Pk0_m4, label=str(limM[3]) + ' =< M ', color='y')
	#~ plt.plot(k,Pk0 , label='monopole (NCV1 + NCV2)/2', color='k')
	#~ plt.title('3D halo spectrum minus shot noise vs matter power spectrum at z = ' + str(z) )
	#~ plt.legend(loc='lower left')
	#~ plt.xscale('log')
	#~ plt.xlabel('k')
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.xlim(8e-3,1)
	#~ plt.ylim(1e1)
	#~ plt.yscale('log')
	#~ plt.ylabel('P(k)')
	#~ plt.savefig('/home/dvalcin/plots/matter_vs_halo_ at_z= '+str(z)+'.png')  

	#~ plt.figure()
	#~ plt.plot(k_m1,bhh1, label=str(limM[0]) + ' =< M =< ' + str(limM[1]), color='b')
	#~ plt.plot(k_m2,bhh2, label=str(limM[1]) + ' =< M =< ' + str(limM[2]), color='r')
	#~ plt.plot(k_m3,bhh3, label=str(limM[2]) + ' =< M =< ' + str(limM[3]), color='g')
	#~ plt.plot(k_m4,bhh4, label=str(limM[3]) + ' =< M ', color='y')
	#~ plt.axhline(Tb1, label= 'Tinker, '+str(limM[0]) + ' =< M =< ' + str(limM[1]),linestyle ='--', color='b')
	#~ plt.axhline(Tb2, color='r',linestyle ='--')
	#~ plt.axhline(Tb3, color='g',linestyle ='--')
	#~ plt.axhline(Tb4, color='y',linestyle ='--')
	#~ plt.axhline(Sb1, label= 'SMT, '+str(limM[0]) + ' =< M =< ' + str(limM[1]),linestyle =':', color='b')
	#~ plt.axhline(Sb2, color='r',linestyle =':')
	#~ plt.axhline(Sb3, color='g',linestyle =':')
	#~ plt.axhline(Sb4, color='y',linestyle =':')
	#~ plt.title('halo bias for different mass range at z = ' + str(z) )
	#~ plt.legend(loc='upper right',prop={'size': 6})
	#~ plt.xscale('log')
	#~ plt.xlabel('k')
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.xlim(8e-3,1)
	#~ plt.ylim(0.2,1.5)
	#~ ##~ plt.yscale('log')
	#~ plt.ylabel('bias')
	#~ plt.savefig('/home/dvalcin/plots/halo_bias_at_z='+str(z)+'.png', dpi=1200)  
	##~ plt.show()



	#~ plt.figure()
	#~ plt.scatter(kpar_m1, kperp_m1,norm=colors.LogNorm(vmin=100, vmax=2e5), c=Pk2D_m1-Pshot_m1, cmap='jet')
	#~ plt.title(' All haloes with min mass = '+str(minm))
	#~ plt.xlim(0,0.5)
	#~ plt.ylim(0,0.5)
	#~ plt.colorbar()
	#~ plt.show()

	#~ plt.figure()
	#~ plt.scatter(kpar_m2, kperp_m2, norm=colors.LogNorm(vmin=100,vmax=2e5), c=Pk2D_m2-Pshot_m2, cmap='jet')
	#~ plt.title('haloes with M >= '+str(limM[0]))
	#~ plt.xlim(0,0.5)
	#~ plt.ylim(0,0.5)
	#~ plt.colorbar()
	#~ plt.show()

	#~ plt.figure()
	#~ plt.scatter(kpar_m3, kperp_m3,norm=colors.LogNorm(vmin=100,vmax=2e5), c=Pk2D_m3-Pshot_m3, cmap='jet')
	#~ plt.title('haloes with M >= '+str(limM[1]))
	#~ plt.xlim(0,0.5)
	#~ plt.ylim(0,0.5)
	#~ plt.colorbar()
	#~ plt.show()

