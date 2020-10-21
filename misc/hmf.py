
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
from bias_library import halo_bias, bias
from time import time
from scipy import stats
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import multiprocessing
#~ import FASTPT_simple as FASTPT
######## INPUT ########
#~ snapshot_fname = '/home/david/codes/Paco/data/snapdir_004/snap_004'


#~ z = 0.0
dims = 1200 #the number of cells in the density field is dims^3
axis = 0 #in redshift-space distortion axis
cores = 16 #number of openmp threads used to compute Pk

########################################################################

redshift = [0.0, 0.5, 1.0, 2.0]
#~ redshift = [0.0]
for z in redshift:
########################################################################
	# neutrino parameters
	hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
	#~ Mnu       = 0.0 #eV
	Mnu       = 0.0  #eV
	Nnu       = 0  #number of massive neutrinos
	Neff      = 3.046

	# cosmological parameters
	h       = 0.6711
	Omega_c = 0.2685 - Mnu/(93.14*h**2)
	Omega_b = 0.049
	Omega_k = 0.0
	Omega_m = Omega_b + Omega_c
	tau     = None
	
	#~ # read snapshot properties
	#~ head = readsnap.snapshot_header(snapshot_fname)
	BoxSize = 1000.0 #Mpc/h                                         
	#~ redshift = head.redshift
	#~ Hubble = 100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l)#km/s/(Mpc/h)
	#~ h = head.hubble

	
	#####################################################################
	########## Halo power spectra 
	#####################################################################
	#~ myfile1 = '/home/dvalcin/codes/Paco/data2/0.0eV/NCV'+str(i)+'/density_field_c_z='+str(z)+'00.hdf5'
	#~ f1 = h5py.File(myfile1, 'r')
	#~ delta1 = f1['MassTable']
	#~ f1.close()
	#~ print f1
	#~ kill

	# halo positions are in kpc/h but we want them in Mpc/h
	# pos_h is the halo position
	# BoxSize is the box size of the simulation, in units of Mpc/h
	# MAS=‘CIC’ is the scheme we use to interpolate 
	# what the below line returns is an array with the interpolated positions of the halos in the grid
	# i.e. it returns n_h, where n_h is the halo number density
	# since we want the power spectrum we need delta_h = n_h / <n_h> -1

	camb = np.loadtxt('/home/dvalcin/codes/Paco/data2/0.0eV/CAMB/Pk_cb_z='+str(z)+'00.txt')
	kcamb = camb[:,0]
	Pcamb = camb[:,1]

	# create a list to match z with the snapshot denomination
	snaplist = ['3.0','2.8','2.6','2.4','2.2','2.0','1.9','1.8','1.7','1.6','1.5','1.4','1.3','1.2','1.1','1.0','0.9','0.8','0.7','0.6','0.5','0.4','0.3','0.2','0.1','0.0']
	snapz = snaplist.index(str(z))
	snaplist2 = ['2.0','1.0','0.5','0.0']
	snapz2 = snaplist2.index(str(z))

	#~ dispo = [1,2,3,5,6,7,8]
	for i in xrange(1,11):
		#~ base = '/home/dvalcin/codes/Paco/data2/0.15eV/NCV'+str(i)
		base = '/home/dvalcin/codes/Paco/data2/0.0eV/NCV'+str(i)
		
		if i == 1 or i == 2:
			fof = FoF_catalog(base, snapz,read_IDs=False)
		else:
			fof = FoF_catalog(base, snapz2,read_IDs=False)

		Masses=fof.GroupMass * 1e10 # for masses 10^10 Msun/h
		pos = fof.GroupPos/1e3 #Mpc/h
		vel = fof.GroupVel
		ids = fof.Nids
		
		#~ a = 1./(1.+z)
		#~ with open('/home/dvalcin/codes/particle-rescaling/file'+str(i)+'.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (4.096e9, 2.15142, a, z,\
			#~ 0, 0, 4.096e9, 0, 0, 1e6, 0.3175, 0.6825 ))
			#~ for index_k in xrange(len(pos)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % ( pos[index_k,0], pos[index_k,1], pos[index_k,2],\
				#~ vel[index_k,0], vel[index_k,1], vel[index_k,2]))
		#~ fid_file.close()
		
	
	########################################################################
	############ compute the halo mass ranges 
	########################################################################

		limM = [5e11,1e12,3e12,1e13]
		limMcosmotest = [5e11,1e12,3e12,5e13]
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
		#~ Hmass_ind_d = np.where(Masses >= limM[3])[0]
		Hmass_ind_d = np.where(Masses >= limMcosmotest[3])[0]



		Hmass_a = Masses[Hmass_ind_a]
		#-------------------------------
		Hmass_b = Masses[Hmass_ind_b]
		#-------------------------------
		Hmass_c = Masses[Hmass_ind_c]
		#-------------------------------
		Hmass_d = Masses[Hmass_ind_d]


		#~ print np.mean(Hmass_a), np.mean(Hmass_b), np.mean(Hmass_c), np.mean(Hmass_d)
		#~ Hmass_a = (Hmass_1a + Hmass_2a)/2
		#~ Hmass_b = (Hmass_1b + Hmass_2b)/2
		#~ Hmass_c = (Hmass_1c + Hmass_2c)/2
		#~ Hmass_d = (Hmass_1d + Hmass_2d)/2

		#~ cname = '/home/dvalcin/plots/hsize_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname, 'w') as fid_file:
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( len(Hmass_a),len(Hmass_b), len(Hmass_c), len(Hmass_d)))
			#~ fid_file.close()
		#~ else :
			#~ with open(cname, 'a') as fid_file:
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( len(Hmass_a),len(Hmass_b), len(Hmass_c), len(Hmass_d)))
			#~ fid_file.close()

		#~ cname = '/home/dvalcin/plots/mostmass_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname, 'w') as fid_file:
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( np.sum(Hmass_a),np.sum(Hmass_b), np.sum(Hmass_c), np.sum(Hmass_d)))
			#~ fid_file.close()
		#~ else :
			#~ with open(cname, 'a') as fid_file:
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( np.sum(Hmass_a),np.sum(Hmass_b), np.sum(Hmass_c), np.sum(Hmass_d)))
			#~ fid_file.close()
			
			
			
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
		cname = '/home/dvalcin/plots/Phh4_realisation_'+str(Mnu)+'_z='+str(z)+'.txt'
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


#########################################################################################
########## compute the halo mass function for whole mass range
#########################################################################################
		#~ print np.max(Masses), np.min(Masses), np.log10(np.max(Masses))
		#~ bins = np.logspace(np.log10(4.2e11),np.log10(5e15),21)
	
		#~ hist, binedge = np.histogram(Masses, bins)
		#~ deltaM=binedge[1:]-binedge[:-1] #size of the bin
		#~ M_middle=10**(0.5*(np.log10(binedge[1:])+np.log10(binedge[:-1]))) #center of the bin
		
		#~ print hist
		#~ dndm = hist/1e9/deltaM
		
		#~ cname = '/home/dvalcin/plots/hmf_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname, 'w') as fid_file:
				#~ for index_k in xrange(len(dndm)):
					#~ fid_file.write('%.8g %.8g %.8g\n' % ( dndm[index_k],M_middle[index_k], deltaM[index_k]))
			#~ fid_file.close()
		#~ else:
			#~ #Create temporary file read/write
			#~ t = tempfile.NamedTemporaryFile(mode="r+")
			#~ #Open input file read-only
			#~ i = open(cname, 'r')
			#~ #Copy input file to temporary file, modifying as we go
			#~ for line in i:
				#~ t.write(line.rstrip()+"\n")
			#~ i.close() #Close input file
			#~ t.seek(0) #Rewind temporary file to beginning
			#~ o = open(cname, "w")  #Reopen input file writable
			#~ #Overwriting original file with temporary file contents          
			#~ for i,line in enumerate(t):
				#~ o.write('%.8g' %(dndm[i])+' '+line)   
			#~ t.close() #Close temporary file, will cause it to be deleted
			#~ o.close()
			
##############################################################################
########## compute the halo mass function for mass range M1
##############################################################################
		#~ print np.max(Hmass_a), np.min(Hmass_a), np.log10(np.max(Hmass_a))
		#~ print len(Hmass_a)
		#~ bins1 = np.logspace(np.log10(limM[0]),np.log10(5e15),50)

		#~ hist1, binedge1 = np.histogram(Hmass_a, bins1)
		#~ deltaM1=binedge1[1:]-binedge1[:-1] #size of the bin
		#~ M_middle1=10**(0.5*(np.log10(binedge1[1:])+np.log10(binedge1[:-1]))) #center of the bin
		
		#~ print hist1
		#~ dndm1 = hist1/1e9/deltaM1
		
		#~ Hist1, Binedge1, binnumber  = stats.binned_statistic(Hmass_a, Hmass_a, 'mean', bins=bins1)
		#~ print Hist1
		
		#~ #--------------------------------------------------
		#~ cname1 = '/home/dvalcin/plots/hmf1_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname1, 'w') as fid_file:
				#~ for index_k in xrange(len(dndm1)):
					#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( dndm1[index_k],M_middle1[index_k], Hist1[index_k], deltaM1[index_k]))
			#~ fid_file.close()
		#~ else:
			#~ #Create temporary file read/write
			#~ t1 = tempfile.NamedTemporaryFile(mode="r+")
			#~ #Open input file read-only
			#~ i = open(cname1, 'r')
			#~ #Copy input file to temporary file, modifying as we go
			#~ for line in i:
				#~ t1.write(line.rstrip()+"\n")
			#~ i.close() #Close input file
			#~ t1.seek(0) #Rewind temporary file to beginning
			#~ o = open(cname1, "w")  #Reopen input file writable
			#~ #Overwriting original file with temporary file contents          
			#~ for i,line in enumerate(t1):
				#~ o.write('%.8g' %(dndm1[i])+' '+line)    
			#~ t1.close() #Close temporary file, will cause it to be deleted
			#~ o.close()
			
#~ ##############################################################################
#~ ########## compute the halo mass function for mass range M2
#~ ##############################################################################
		#~ print np.max(Hmass_b), np.min(Hmass_b), np.log10(np.max(Hmass_b))
		#~ print len(Hmass_b)
		#~ bins2 = np.logspace(np.log10(limM[1]),np.log10(5e15),50)

		#~ hist2, binedge2 = np.histogram(Hmass_b, bins2)
		#~ deltaM2=binedge2[1:]-binedge2[:-1] #size of the bin
		#~ M_middle2=10**(0.5*(np.log10(binedge2[1:])+np.log10(binedge2[:-1]))) #center of the bin
		
		#~ print hist2
		#~ dndm2 = hist2/1e9/deltaM2
		
		#~ Hist2, Binedge2, binnumber  = stats.binned_statistic(Hmass_b, Hmass_b, 'mean', bins=bins2)
		#~ print Hist2
		
		#~ #--------------------------------------------------
		#~ cname2 = '/home/dvalcin/plots/hmf2_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname2, 'w') as fid_file:
				#~ for index_k in xrange(len(dndm2)):
					#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( dndm2[index_k],M_middle2[index_k], Hist2[index_k], deltaM2[index_k]))
			#~ fid_file.close()
		#~ else:
			#~ #Create temporary file read/write
			#~ t2 = tempfile.NamedTemporaryFile(mode="r+")
			#~ #Open input file read-only
			#~ i = open(cname2, 'r')
			#~ #Copy input file to temporary file, modifying as we go
			#~ for line in i:
				#~ t2.write(line.rstrip()+"\n")
			#~ i.close() #Close input file
			#~ t2.seek(0) #Rewind temporary file to beginning
			#~ o = open(cname2, "w")  #Reopen input file writable
			#~ #Overwriting original file with temporary file contents          
			#~ for i,line in enumerate(t2):
				#~ o.write('%.8g' %(dndm2[i])+' '+line)   
			#~ t2.close() #Close temporary file, will cause it to be deleted
			#~ o.close()
			
#~ ##############################################################################
#~ ########## compute the halo mass function for mass range M3
#~ ##############################################################################
		#~ print np.max(Hmass_c), np.min(Hmass_c), np.log10(np.max(Hmass_c))
		#~ print len(Hmass_c)
		#~ bins3 = np.logspace(np.log10(limM[2]),np.log10(5e15),50)

		#~ hist3, binedge3 = np.histogram(Hmass_c, bins3)
		#~ deltaM3=binedge3[1:]-binedge3[:-1] #size of the bin
		#~ M_middle3=10**(0.5*(np.log10(binedge3[1:])+np.log10(binedge3[:-1]))) #center of the bin
		
		#~ print hist3
		#~ dndm3 = hist3/1e9/deltaM3
		
		#~ Hist3, Binedge3, binnumber  = stats.binned_statistic(Hmass_c, Hmass_c, 'mean', bins=bins3)
		#~ print Hist3
		
		#~ #--------------------------------------------------
		#~ cname3 = '/home/dvalcin/plots/hmf3_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname3, 'w') as fid_file:
				#~ for index_k in xrange(len(dndm3)):
					#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( dndm3[index_k],M_middle3[index_k], Hist3[index_k], deltaM3[index_k]))
			#~ fid_file.close()
		#~ else:
			#~ #Create temporary file read/write
			#~ t3 = tempfile.NamedTemporaryFile(mode="r+")
			#~ #Open input file read-only
			#~ i = open(cname3, 'r')
			#~ #Copy input file to temporary file, modifying as we go
			#~ for line in i:
				#~ t3.write(line.rstrip()+"\n")
			#~ i.close() #Close input file
			#~ t3.seek(0) #Rewind temporary file to beginning
			#~ o = open(cname3, "w")  #Reopen input file writable
			#~ #Overwriting original file with temporary file contents          
			#~ for i,line in enumerate(t3):
				#~ o.write('%.8g' %(dndm3[i])+' '+line)   
			#~ t3.close() #Close temporary file, will cause it to be deleted
			#~ o.close()
			
#~ ##############################################################################
#~ ########## compute the halo mass function for mass range M4
#~ ##############################################################################
		#~ print np.max(Hmass_d), np.min(Hmass_d), np.log10(np.max(Hmass_d))
		#~ print len(Hmass_d)
		
		#~ bins4 = np.logspace(np.log10(limM[3]),np.log10(5e15),50)
		#~ hist4, binedge4 = np.histogram(Hmass_d, bins4)
		#~ deltaM4=binedge4[1:]-binedge4[:-1] #size of the bin
		#~ M_middle4=10**(0.5*(np.log10(binedge4[1:])+np.log10(binedge4[:-1]))) #center of the bin
		
		#~ print hist4
		#~ dndm4 = hist4/1e9/deltaM4
		
		#~ Hist4, Binedge4, binnumber  = stats.binned_statistic(Hmass_d, Hmass_d, 'mean', bins=bins4)
		#~ print Hist4
		
		#~ #--------------------------------------------------
		#~ cname4 = '/home/dvalcin/plots/hmf4_z='+str(z)+'.txt'
		#~ if i == 1:
			#~ with open(cname4, 'w') as fid_file:
				#~ for index_k in xrange(len(dndm4)):
					#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % ( dndm4[index_k],M_middle4[index_k], Hist4[index_k], deltaM4[index_k]))
			#~ fid_file.close()
		#~ else:
			#~ #Create temporary file read/write
			#~ t4 = tempfile.NamedTemporaryFile(mode="r+")
			#~ #Open input file read-only
			#~ i = open(cname4, 'r')
			#~ #Copy input file to temporary file, modifying as we go
			#~ for line in i:
				#~ t4.write(line.rstrip()+"\n")
			#~ i.close() #Close input file
			#~ t4.seek(0) #Rewind temporary file to beginning
			#~ o = open(cname4, "w")  #Reopen input file writable
			#~ #Overwriting original file with temporary file contents          
			#~ for i,line in enumerate(t4):
				#~ o.write('%.8g' %(dndm4[i])+' '+line)   
			#~ t4.close() #Close temporary file, will cause it to be deleted
			#~ o.close()
		#~ #--------------------------------------------------------------
		
		
		

############################################################################
############################################################################

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





