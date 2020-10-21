
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


#~ z = 0.0
dims = 32 #the number of cells in the density field is dims^3
axis = 0 #in redshift-space distortion axis
cores = 16 #number of openmp threads used to compute Pk

########################################################################

redshift = [0.0, 0.5, 1.0, 2.0]

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
BoxSize = 62.5 #Mpc/h                                         
#~ redshift = head.redshift
#~ Hubble = 100.0*np.sqrt(Omega_m*(1.0+z)**3+Omega_l)#km/s/(Mpc/h)
#~ h = head.hubble

z = 0.0

########################################################################
############# scale factor 
########################################################################
#~ red = ['0.0','0.5','1.0','2.0','3.0']
#~ ind = red.index(str(z))
#~ f = [0.518,0.754,0.872,0.956,0.98]
#~ mono = (1 + 2/3.*(f[ind]) + 1/5.*(f[ind])**2) 
#~ quadru = (4/3.*(f[ind]) + 4/7.*(f[ind])**2)



Mille = np.loadtxt('/home/dvalcin/codes/Paco/data2/0.0eV/millenium.txt', delimiter =',', skiprows = 50)

Masses = Mille[:,12]* 1e10 #10^10 Msun
x = Mille[:,16] #Mpc/h
y = Mille[:,17]
Z = Mille[:,18]

pos = np.sqrt(x**2 + y**2 + Z**2)


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

cname = '/home/dvalcin/plots/Phh1_realisation_'+str(Mnu)+'_z='+str(z)+'.txt'
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
cname = '/home/dvalcin/plots/Phh2_realisation_'+str(Mnu)+'_z='+str(z)+'.txt'
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
cname = '/home/dvalcin/plots/Phh3_realisation_'+str(Mnu)+'_z='+str(z)+'.txt'
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
