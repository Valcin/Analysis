
####################################################################
import numpy as np
import readsnap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pyximport
pyximport.install()
import redshift_space_library as RSL
from readfof import FoF_catalog
import MAS_library as MASL
import Pk_library as PKL
from classy import Class

######## INPUT ########

dims = 512 #the number of cells in the density field is dims^3
axis = 0 #in redshift-space distortion axis
cores = 16 #number of openmp threads used to compute Pk

#######################

Omega_m = 0.3175
Omega_l = 0.6825

# read snapshot properties
head = readsnap.snapshot_header(snapshot_fname)
BoxSize = head.boxsize/1e3 #Mpc/h                      
redshift = head.redshift
Hubble = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#km/s/(Mpc/h)
h = head.hubble



# read the positions and velocities of the particles
pos = readsnap.read_block(snapshot_fname,"POS ",parttype=1)/1e3 #Mpc/h
vel = readsnap.read_block(snapshot_fname,"VEL ",parttype=1) #km/s

# move particles to redshift-space
RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

# compute density field in redshift-space
delta = np.zeros((dims, dims, dims), dtype=np.float32) #this should be your density field
MASL.MA(pos, delta, BoxSize, MAS='CIC') # computes the density in each cell of the grid
delta = delta/np.mean(delta) - 1.0

# compute power spectra
Pk = PKL.Pk(delta, BoxSize, axis, MAS='CIC', threads=cores) #Pk here is a class with all power spectra

# 3D Pk
k = Pk.k3D
Pk0 = Pk.Pk[:,0] #monopole
Pk2 = Pk.Pk[:,1] #quadrupole
Pk4 = Pk.Pk[:,2] #hexadecapole
Nmodes = Pk.Nmodes3D #number of modes in each Pk bin


# 2D Pk
kpar = Pk.kpar
kperp = Pk.kper
Pk2D = Pk.Pk2D
Nmodes2D = Pk.Nmodes2D

# 1D Pk
k1D = Pk.k1D
Pk1D = Pk.Pk1D
Nmodes1D = Pk.Nmodes1D
#~ ####################################################################
#~ #### Addition for matter
#~ ####################################################################

d = np.loadtxt('/home/david/codes/class/output/test_pk.dat', skiprows = 4)
e = np.loadtxt('/home/david/codes/class/output/test_pk_nl.dat', skiprows = 4)
kclass = d[:,0]
kpk = d[:,1]
kpk_nl = e[:,1]
f = Class.scale_independent_growth_factor_f(self,z=2.)
f = 0.45
mono = (1 + 2/3.*(f) + 1/5.*(f)**2) 
quadru = (4/3.*(f) + 4/7.*(f)**2)


def jen(pk,var1, var2):
	if var1 == 'delta' and var2 == 'theta':
		alpha0 = -12288.7
		alpha1 = 1.43
		alpha2 = 1367.7
		alpha3 = 1.54
		
	elif var1 == 'theta' and var2 == 'theta':
		alpha0 = -12462.1
		alpha1 = 0.839
		alpha2 = 1446.6 
		alpha3 = 0.806
		
	pxy = (alpha0 * np.sqrt(kpk) + alpha1 * kpk**2) /(alpha2 + alpha3 * kpk)
	
	return pxy
	

pdt = jen(kpk_nl,'delta','theta')
ptt = jen(kpk_nl,'theta','theta')
qnk = kpk_nl + 2/3.*f*pdt + 1/5.*f**2 *ptt

plt.figure()
plt.plot(k,Pk0, label='monopole simu', color='b')
plt.plot(kclass,kpk*mono, label='monopole class + linear kaiser',linestyle ='--', color='b')
plt.plot(kclass,qnk, label='monopole class + quasi linear kaiser',linestyle =':', color='b')
plt.plot(k,Pk2, label='quadrupole simu', color='r')
plt.plot(kclass,kpk*quadru, label='quadrupole class + linear kaiser',linestyle ='--', color='r')
plt.title('3D power spectrum for snapshot 002' )
plt.legend(loc='upper right')
plt.xscale('log')
plt.xlabel('k')
plt.tick_params(labelleft=True, labelright=True)
plt.xlim(8e-3,0.5)
plt.ylim(1e1,1e5)
plt.yscale('log')
plt.ylabel('P(k)')
plt.show()  

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

#~ base1 = '/home/david/codes/Paco/data2/0.0eV/NCV1'
#~ base2 = '/home/david/codes/Paco/data2/0.0eV/NCV2'
#~ fof1 = FoF_catalog(base1, 25,read_IDs=False)
#~ fof2 = FoF_catalog(base2, 25,read_IDs=False)

#Vel = fof.GroupVel * (1+z) # for physical velocities in km/s
#~ Masses1=fof1.GroupMass * 1e10 # for masses 10^10 Msun/h
#~ Masses2=fof2.GroupMass * 1e10 # for masses 10^10 Msun/h
#~ pos1 = fof1.GroupPos/1e3 #Mpc/h
#~ pos2 = fof2.GroupPos/1e3 #Mpc/h
#~ limM = [3.2e12,3.2e13]
#~ Hmass_ind_1a = np.where(Masses1 >= limM[0] )[0]
#~ Hmass_ind_2a = np.where(Masses2 >= limM[0] )[0]
#~ Hmass_ind_1b = np.where(Masses1 >= limM[1] )[0]
#~ Hmass_ind_2b = np.where(Masses2 >= limM[1] )[0]

#~ Hmass_1a = Masses1[Hmass_ind_1a]
#~ Hmass_1b = Masses2[Hmass_ind_1b]
#~ Hmass_2a = Masses1[Hmass_ind_2a]
#~ Hmass_2b = Masses2[Hmass_ind_2b]

#~ minm = np.min(Masses1)

##################################################################
########## ALL haloes 
##################################################################


#~ delta1 = np.zeros((dims,dims,dims), dtype=np.float32)
#~ MASL.MA(pos1, delta1, BoxSize, MAS='CIC', W=None)  
#~ delta1 = delta1/np.mean(delta1, dtype=np.float64) - 1.0

#~ delta2 = np.zeros((dims,dims,dims), dtype=np.float32)
#~ MASL.MA(pos2, delta2, BoxSize, MAS='CIC', W=None)  
#~ delta2 = delta2/np.mean(delta2, dtype=np.float64) - 1.0


#~ # compute power spectra
#~ Pk1 = PKL.Pk(delta1, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra
#~ Pk2 = PKL.Pk(delta2, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra


#~ # 3D Pk
#~ k_m1= (Pk1.k3D + Pk2.k3D)/2
#~ Pk0_m1 = (Pk1.Pk[:,0] + Pk2.Pk[:,0])/2 #monopole
#~ Pk2_m1 = (Pk1.Pk[:,1] + Pk2.Pk[:,1])/2 #quadrupole
#~ Pk4_m1 = (Pk1.Pk[:,2] + Pk2.Pk[:,2])/2 #hexadecapole
#~ Nmodes_m1 = (Pk1.Nmodes3D + Pk2.Nmodes3D)/2 #number of modes in each Pk bin


#~ # 2D Pk
#~ kpar_m1 = (Pk1.kpar + Pk2.kpar)/2
#~ kperp_m1 = (Pk1.kper + Pk2.kper)/2
#~ Pk2D_m1 = (Pk1.Pk2D + Pk2.Pk2D)/2
#~ Nmodes2D_m1 = (Pk1.Nmodes2D + Pk2.Nmodes2D)/2



####################################################################
########## First mass limit
####################################################################


#~ delta1a = np.zeros((dims,dims,dims), dtype=np.float32)
#~ MASL.MA(pos1[Hmass_ind_1a], delta1a, BoxSize, MAS='CIC', W=None)  
#~ delta1a = delta1a/np.mean(delta1a, dtype=np.float64) - 1.0

#~ delta2a = np.zeros((dims,dims,dims), dtype=np.float32)
#~ MASL.MA(pos2[Hmass_ind_2a], delta2a, BoxSize, MAS='CIC', W=None)  
#~ delta2a = delta2a/np.mean(delta2a, dtype=np.float64) - 1.0


#~ # compute power spectra
#~ Pk1a = PKL.Pk(delta1a, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra
#~ Pk2a = PKL.Pk(delta2a, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra

#~ # 3D Pk
#~ k_m2= (Pk1a.k3D + Pk2a.k3D)/2
#~ Pk0_m2 = (Pk1a.Pk[:,0] + Pk2a.Pk[:,0])/2 #monopole
#~ Pk2_m2 = (Pk1a.Pk[:,1] + Pk2a.Pk[:,1])/2 #quadrupole
#~ Pk4_m2 = (Pk1a.Pk[:,2] + Pk2a.Pk[:,2])/2 #hexadecapole
#~ Nmodes_m2 = (Pk1a.Nmodes3D + Pk2a.Nmodes3D)/2 #number of modes in each Pk bin

#~ # 2D Pk
#~ kpar_m2 = (Pk1a.kpar + Pk2a.kpar)/2
#~ kperp_m2 = (Pk1a.kper + Pk2a.kper)/2
#~ Pk2D_m2 = (Pk1a.Pk2D + Pk2a.Pk2D)/2
#~ Nmodes2D_m2 = (Pk1a.Nmodes2D + Pk2a.Nmodes2D)/2


###############################################################
######## second mass limit
###############################################################


#~ delta1b = np.zeros((dims,dims,dims), dtype=np.float32)
#~ MASL.MA(pos1[Hmass_ind_1b], delta1b, BoxSize, MAS='CIC', W=None)  
#~ delta1b = delta1b/np.mean(delta1b, dtype=np.float64) - 1.0

#~ delta2b = np.zeros((dims,dims,dims), dtype=np.float32)
#~ MASL.MA(pos2[Hmass_ind_2b], delta2b, BoxSize, MAS='CIC', W=None)  
#~ delta2b = delta2b/np.mean(delta2b, dtype=np.float64) - 1.0


#~ # compute power spectra
#~ Pk1b = PKL.Pk(delta1b, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra
#~ Pk2b = PKL.Pk(delta2b, BoxSize, axis=0, MAS='CIC', threads=4) #Pk here is a class with all power spectra

#~ # 3D Pk
#~ k_m3= (Pk1b.k3D + Pk2b.k3D)/2
#~ Pk0_m3 = (Pk1b.Pk[:,0] + Pk2b.Pk[:,0])/2 #monopole
#~ Pk2_m3 = (Pk1b.Pk[:,1] + Pk2b.Pk[:,1])/2 #quadrupole
#~ Pk4_m3 = (Pk1b.Pk[:,2] + Pk2b.Pk[:,2])/2 #hexadecapole
#~ Nmodes_m3 = (Pk1b.Nmodes3D + Pk2b.Nmodes3D)/2 #number of modes in each Pk bin

#~ # 2D Pk
#~ kpar_m3 = (Pk1b.kpar + Pk2b.kpar)/2
#~ kperp_m3 = (Pk1b.kper + Pk2b.kper)/2
#~ Pk2D_m3 = (Pk1b.Pk2D + Pk2b.Pk2D)/2
#~ Nmodes2D_m3 = (Pk1b.Nmodes2D + Pk2b.Nmodes2D)/2

#########################################################################
################# Shoit noise
#########################################################################
#~ Pshot_m1 = (1/(len(Masses1)/BoxSize**3) + 1/(len(Masses2)/BoxSize**3))/2
#~ Pshot_m2 = (1/(len(Hmass_1a)/BoxSize**3) + 1/(len(Hmass_2a)/BoxSize**3))/2
#~ Pshot_m3 = (1/(len(Hmass_1b)/BoxSize**3) + 1/(len(Hmass_2b)/BoxSize**3))/2



########################################################################
########### Plots
#######################################################################

#~ plt.figure()
#~ plt.plot(k_m1,Pk0_m1 - Pshot_m1, label=' All haloes with min mass = '+str(minm), color='b')
#~ plt.plot(k_m2,Pk0_m2 - Pshot_m2, label='haloes with M >= '+str(limM[0]), color='g')
#~ plt.plot(k_m3,Pk0_m3 - Pshot_m3, label='haloes with M >= '+str(limM[1]), color='r')
#~ plt.title('3D halo spectrum at z=0 minus shot noise' )
#~ plt.legend(loc='upper right')
#~ plt.xscale('log')
#~ plt.xlabel('k')
#~ plt.tick_params(labelleft=True, labelright=True)
#~ plt.xlim(8e-3,1)
#~ plt.ylim(1e3)
#~ plt.yscale('log')
#~ plt.ylabel('P(k)')
#~ plt.show()  

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
