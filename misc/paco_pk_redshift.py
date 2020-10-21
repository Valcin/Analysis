
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

redshift = [0.0, 0.5, 1.0, 2.0]
#~ redshift = [2.0]
axis = [0,1,2] #in redshift-space distortion axis
for z in redshift:
	for axe in axis:

		dims = 1200 #the number of cells in the density field is dims^3
		#~ axis = 2 #in redshift-space distortion axis
		cores = 16 #number of openmp threads used to compute Pk

		#~ #######################

		# neutrino parameters
		hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
		Mnu       = 0.00  #eV
		Nnu       = 0  #number of massive neutrinos
		Neff      = 3.046

		# neutrino parameters
		hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
		Mnu       = 0.00  #eV
		Nnu       = 0  #number of massive neutrinos
		Neff      = 3.046

		# cosmological parameters
		h       = 0.6711
		Omega_c = 0.2685 - Mnu/(93.14*h**2)
		Omega_b = 0.049
		Omega_m = Omega_b + Omega_c
		Omega_k = 0.0
		Omega_l = 0.6825
		tau     = None

		#~ # read snapshot properties
		#~ head = readsnap.snapshot_header(snapshot_fname)
		BoxSize = 1000.0 #Mpc/h                                         
		#~ redshift = head.redshift
		Hubble = 100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l)#km/s/(Mpc/h)
		#~ h = head.hubble


########################################################################
############## 3D average ##############################################
########################################################################

#~ # 3D Pk
#~ k = (kx + ky + kz)/3 
#~ Pk0 = (Pk0x + Pk0y + Pk0z)/3  #monopole
#~ Pk2_av = (Pk2x_av + Pk2y_av + Pk2z_av)/3  #quadrupole
#~ Pk4 = (Pk4x + Pk4y + Pk4z)/3 #hexapole
#~ Nmodes = (Nmodesx + Nmodesy + Nmodesz)/3  #number of modes in each Pk bin
 


#~ f1.close()
#~ f2.close()

#~ del f1
#~ del f2


#~ # 2D Pk
#~ #kpar = Pk.kpar
#~ #kperp = Pk.kper
#~ #Pk2D = Pk.Pk2D
#~ #Nmodes2D = Pk.Nmodes2D

#~ # 1D Pk
#~ #k1D = Pk.k1D
#~ #Pk1D = Pk.Pk1D
#~ #Nmodes1D = Pk.Nmodes1D

#~ ####################################################################
#~ #### Addition for matter
#~ ####################################################################

#~ cname = '/home/dvalcin/plots/Pk_red_z='+str(z)+'.txt'
#~ with open(cname, 'w') as fid_file:
	#~ for index_k in xrange(len(k)):
		#~ fid_file.write('%.8g %.8g %.8g\n' % (k[index_k], Pk0[index_k], Pk2_av[index_k]))
#~ print '\n'

########################################################################
############# scale factor 
########################################################################
#~ red = ['0.0','0.5','1.0','2.0','3.0']
#~ ind = red.index(str(z))
#~ f = [0.518,0.754,0.872,0.956,0.98]
#~ mono = (1 + 2/3.*(f[ind]) + 1/5.*(f[ind])**2) 
#~ quadru = (4/3.*(f[ind]) + 4/7.*(f[ind])**2)



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



		for i in xrange(1,11):
			base = '/home/dvalcin/codes/Paco/data2/0.0eV/NCV'+str(i)
			
			if i == 1 or i == 2:
				fof = FoF_catalog(base, snapz,read_IDs=False)
			else:
				fof = FoF_catalog(base, snapz2,read_IDs=False)

			Masses=fof.GroupMass * 1e10 # for masses 10^10 Msun/h
			pos = fof.GroupPos/1e3 #Mpc/h

			vel = fof.GroupVel * (1+z) # for physical velocities in km/s
			
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
			##### move particles to redshift-space
			########################################################################

			RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, z, axe)

		#####################################################################
		########## compute the halo mass function
		#####################################################################
		#~ print np.max(Masses1), np.min(Masses1), np.log10(np.max(Masses1))
		#~ bins = np.logspace(11,16,32)
		#~ N = len(Masses1)
		#~ weights = np.ones(N) * 1/float(N)/1e9
		#~ print 1/float(N)
		#~ cmf = MFL.Crocce_mass_function(kcamb,pcamb,Omega_b + Omega_c,z,1e11,1e16,32,Masses=None)
		#~ print np.shape(cmf)

		# halo mass function
		#~ plt.figure()
		#~ hist, binedge = np.histogram(Masses1, bins)
		#~ plt.plot(bins,cmf[1], color='r', label='Crocce mass function')
		#~ plt.hist(Masses1,bins, color='g', label='NCV1')
		#~ plt.hist(Masses2,bins, color='b', label='NCV2')
		#~ plt.axvline(limM[0])
		#~ plt.axvline(limM[1])
		#~ plt.axvline(limM[2])
		#~ plt.axvline(limM[3])
		#~ plt.xlabel('Mass Msun/h')
		#~ plt.ylabel('number')
		#~ plt.legend(loc='upper right')
		#~ plt.yscale('log')
		#~ plt.xscale('log')
		#~ plt.show()
		#~ plt.savefig('/home/dvalcin/plots/halo_mass_funct_at_z= '+str(z)+'.png', dpi = 500)

		# cumulative halo mass function
		#~ plt.figure()
		#~ plt.hist(Masses1,bins, color='g', label='NCV1', cumulative=True, normed=True)
		#~ plt.hist(Masses2,bins, color='b', label='NCV2',cumulative=True, normed=True)
		#~ plt.axvline(limM[0])
		#~ plt.axvline(limM[1])
		#~ plt.axvline(limM[2])
		#~ plt.axvline(limM[3])
		#~ plt.xlabel('Mass Msun/h')
		#~ plt.ylabel('number')
		#~ plt.legend(loc='upper right')
		#~ plt.yscale('log')
		#~ plt.xscale('log')
		#~ plt.show()
		#~ plt.savefig('/home/dvalcin/plots/cumul_halo_mass_funct_at_z= '+str(z)+'.png', dpi = 500)







		##~ ##################################################################
			#~ ########## First mass range 
			#~ ##################################################################


			delta1a = np.zeros((dims,dims,dims), dtype=np.float32)
			MASL.MA(pos[Hmass_ind_a], delta1a, BoxSize, MAS='CIC', W=None)  
			delta1a = delta1a/np.mean(delta1a, dtype=np.float64) - 1.0

			# compute power spectra
			Pk1a = PKL.Pk(delta1a, BoxSize, axis=axe, MAS='CIC', threads=4) #Pk here is a class with all power spectra

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

			cname = '/home/dvalcin/plots/Phh1_realisation_red_axis_'+str(axe)+'_z='+str(z)+'.txt'
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
			Pk1b = PKL.Pk(delta1b, BoxSize, axis=axe, MAS='CIC', threads=4) #Pk here is a class with all power spectra

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
			cname = '/home/dvalcin/plots/Phh2_realisation_red_axis_'+str(axe)+'_z='+str(z)+'.txt'
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
			Pk1c = PKL.Pk(delta1c, BoxSize, axis=axe, MAS='CIC', threads=4) #Pk here is a class with all power spectra

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
			cname = '/home/dvalcin/plots/Phh3_realisation_red_axis_'+str(axe)+'_z='+str(z)+'.txt'
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
			Pk1d = PKL.Pk(delta1d, BoxSize, axis=axe, MAS='CIC', threads=4) #Pk here is a class with all power spectra

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
			cname = '/home/dvalcin/plots/Phh4_realisation_red_axis_'+str(axe)+'_z='+str(z)+'.txt'
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


#~ #########################################################################
#~ ################# Shoit noise
#~ #########################################################################

#~ Pshot_m1 = (1/(len(Hmass_1a)/BoxSize**3) + 1/(len(Hmass_2a)/BoxSize**3))/2
#~ Pshot_m2 = (1/(len(Hmass_1b)/BoxSize**3) + 1/(len(Hmass_2b)/BoxSize**3))/2
#~ Pshot_m3 = (1/(len(Hmass_1c)/BoxSize**3) + 1/(len(Hmass_2c)/BoxSize**3))/2
#~ Pshot_m4 = (1/(len(Hmass_1d)/BoxSize**3) + 1/(len(Hmass_2d)/BoxSize**3))/2

#~ Pk0_m1 = Pk0_m1 - Pshot_m1
#~ Pk0_m2 = Pk0_m2 - Pshot_m2
#~ Pk0_m3 = Pk0_m3 - Pshot_m3
#~ Pk0_m4 = Pk0_m4 - Pshot_m4

#~ ########################################################################
#~ ############ compute the halo bias
#~ ########################################################################
#~ bhh1 = np.sqrt((Pk0_m1 - Pshot_m1)/Pk0)
#~ bhh2 = np.sqrt((Pk0_m2 - Pshot_m2)/Pk0)
#~ bhh3 = np.sqrt((Pk0_m3 - Pshot_m3)/Pk0)
#~ bhh4 = np.sqrt((Pk0_m4 - Pshot_m4)/Pk0)


# TINKER
#~ Tb1 = halo_bias('Tinker', z, limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
#~ Tb2 = halo_bias('Tinker', z, limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
#~ Tb3 = halo_bias('Tinker', z, limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
#~ Tb4 = halo_bias('Tinker', z, limM[3],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )

# TORMEN
#~ Sb1 = halo_bias('SMT01', z, limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
#~ Sb2 = halo_bias('SMT01', z, limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
#~ Sb3 = halo_bias('SMT01', z, limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
#~ Sb4 = halo_bias('SMT01', z, limM[3],np.max(Masses1), cname,Omega_c, Omega_b, do_DM=True )



########################################################################
###### save the halo power spectrum
########################################################################

#~ cname = '/home/dvalcin/plots/Phh_red_axis'+str(axis)+'_z='+str(z)+'.txt'
#~ with open(cname, 'w') as fid_file:
	#~ fid_file.write('%s %s %s %s %s %s %s %s %s\n' % ('k', 'm1-monopole', 'm1-quadrupole', 'm2-monopole', 'm2-quadrupole', 'm3-monopole', 'm3-quadrupole', 'm4-monopole', 'm4-quadrupole'))
	#~ fid_file.write('\n')
	#~ for index_k in xrange(len(k_m1)):
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (k_m1[index_k], Pk0_m1[index_k], Pk2_m1[index_k], Pk0_m2[index_k], Pk2_m2[index_k], Pk0_m3[index_k], Pk2_m3[index_k],Pk0_m4[index_k], Pk2_m4[index_k]))
#~ print '\n'

########################################################################
########### Plots
#######################################################################

#~ plt.figure()
#~ plt.plot(k_m1,Pk0_m1 - Pshot_m1, label=str(limM[0]) + ' =< M =< ' + str(limM[1]), color='b')
#~ plt.plot(k_m2,Pk0_m2 - Pshot_m2, label=str(limM[1]) + ' =< M =< ' + str(limM[2]), color='r')
#~ plt.plot(k_m3,Pk0_m3 - Pshot_m3, label=str(limM[2]) + ' =< M =< ' + str(limM[3]), color='g')
#~ plt.plot(k_m4,Pk0_m4 - Pshot_m4, label=str(limM[3]) + ' =< M ', color='y')
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
#~ plt.ylim(0,Tb4+2)
#~ ##~ plt.yscale('log')
#~ plt.ylabel('bias')
#~ plt.savefig('/home/dvalcin/plots/halo_bias_at_z= '+str(z)+'.png', dpi=1200)  
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
