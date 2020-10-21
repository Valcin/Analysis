import myFASTPT as FPT
from time import time
import numpy as np
import h5py
import math
import readsnap
import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import scipy.interpolate as sp
import pyximport
pyximport.install()
import redshift_space_library as RSL
from readfof import FoF_catalog
import MAS_library as MASL
import Pk_library as PKL
import mass_function_library as MFL
import tempfile
import expected_CF
from time import time
from bias_library import halo_bias
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.special import gamma
from fit_emcee import coeffit_pl, coeffit_exp1, coeffit_exp2,coeffit_Kaiser, coeffit_Scocci, coeffit_TNS, coeffit_eTNS

H0 = 70
z = [0.0,0.5,1.0,2.0]
#~ z = [0.0,2.0]
#~ mu = 0.5
#~ kmax = 1
mass_range = ['m1','m2','m3','m4']
#~ mass_range = ['m1', 'm2']
#~ mass_range = ['m1']
#~ axis = 0 #in redshift-space distortion axis

# neutrino parameters
hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
Mnu       = 0.00  #eV
Nnu       = 0  #number of massive neutrinos
Neff      = 3.046

# cosmological parameters
h       = 0.6711
Omega_c = 0.2685 - Mnu/(93.14*h**2)
Omega_b = 0.049
Omega_l = 0.6825
Omega_k = 0.0
tau     = None

#~ # read snapshot properties
#~ head = readsnap.snapshot_header(snapshot_fname)
BoxSize = 1000.0 #Mpc/h                                         
#~ redshift = head.redshift
#~ Hubble = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#km/s/(Mpc/h)
#~ h = head.hubble

start = time()

for j in xrange(0,len(z)):
########################################################################
########################################################################
	####################################################################
	############# scale factor 
	####################################################################
	red = ['0.0','0.5','1.0','2.0','3.0']
	ind = red.index(str(z[j]))
	f = [0.518,0.754,0.872,0.956,0.98]
	print 'For redshift z = ' + str(z[j])

	########################################################################
	############# 	0.0 eV Masseless neutrino 
	########################################################################
	nv = 0.0
	
	#~ #------------------------------------------------
	#~ #-------- data from scoccimaro 2004 -------------
	#~ #------------------------------------------------
	scoccidd = np.loadtxt('//home/david/delta.txt')
	psdd = scoccidd[:,0]
	ksdd = scoccidd[:,1]
	scoccidt = np.loadtxt('//home/david/deltheta.txt')
	psdt = scoccidt[:,0]
	ksdt = scoccidt[:,1]
	scoccitt = np.loadtxt('//home/david/theta.txt')
	pstt = scoccitt[:,0]
	kstt = scoccitt[:,1]


	#~ #-------------------------------------------------
	#~ #---------------- Camb ---------------------------
	#~ #-------------------------------------------------
	camb = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/expected_Pk_z='+str(z[j])+'.txt')
	#~ #camb = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_mm_z='+str(z)+'00.txt')
	#cname = '/home/dvalcin/codes/Paco/data2/0.0eV/CAMB/expected_Pk_z='+str(z)+'.txt'
	kcamb = camb[:,0]
	Pcamb = camb[:,1]

	#~ k = np.logspace(np.min(np.log10(kcamb)), np.max(np.log10(kcamb)), 120)
	#~ Pcamb = np.interp(k,kcamb,Pcamb)
	#~ Plin = Pcamb

	#~ #-------------------------------------------------
	#~ #---------------- Class ---------------------------
	#~ #-------------------------------------------------
	Class = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class/test_z'+str(j+1)+'_pk.dat')
	kclass = Class[:,0]
	Pclass = Class[:,1]
	

	# check if array is evenly log space as required by FAST PT 
	#----------------------------------------------------------

	# fast PT needs an even scale size
	#~ if len(k) % 2 != 0: #odd
		#~ k = k[:-1]
		#~ P = P[:-1]
		
	#~ print len(k)

	k1 = np.logspace(np.min(np.log10(kclass)), np.max(np.log10(kclass)), len(kclass))
	Pclass = np.interp(k1,kclass,Pclass)
	Plin = Pclass



	#-------------------------------------------------
	#------------matter  Real space --------
	#-------------------------------------------------
	d = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Pcc_realisation_z='+str(z[j])+'.txt')
	kmat = d[:,10]
	Pmat = np.zeros((len(kmat),10))
	for i in xrange(0,10):
		Pmat[:,i]= d[:,i]
	
	
	#~ #---------------------------------------------------
	#~ #--------- halo real space -------------------------
	#~ #---------------------------------------------------
	#first mass range
	d1 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh1_realisation_z='+str(z[j])+'.txt')
	k = d1[:,19]
	Phh1 = np.zeros((len(k),10))
	Pshot1 = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Phh1[:,i]= d1[:,pnum1[i]]
		Pshot1[i]= d1[0,pnum2[i]]
	# second mass range
	d2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh2_realisation_z='+str(z[j])+'.txt')
	k = d2[:,19]
	Phh2 = np.zeros((len(k),10))
	Pshot2 = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Phh2[:,i]= d2[:,pnum1[i]]
		Pshot2[i]= d2[0,pnum2[i]]
	# third mass range
	d3 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh3_realisation_z='+str(z[j])+'.txt')
	k = d3[:,19]
	Phh3 = np.zeros((len(k),10))
	Pshot3 = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Phh3[:,i]= d3[:,pnum1[i]]
		Pshot3[i]= d3[0,pnum2[i]]
	# fourth mass range
	d4 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh4_realisation_z='+str(z[j])+'.txt')
	k = d4[:,19]
	Phh4 = np.zeros((len(k),10))
	Pshot4 = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Phh4[:,i]= d4[:,pnum1[i]]
		Pshot4[i]= d4[0,pnum2[i]]

	### compute where shot noise is 80% of the ps
	
	#~ kk1t = np.zeros(10)
	#~ kk2t = np.zeros(10)
	#~ kk3t = np.zeros(10)
	#~ kk4t = np.zeros(10)
	
	#~ for i in xrange(0,10):
		#~ dede1 = np.where(0.8*Phh1[:,i] < Pshot1[i])[0]
		#~ dede2 = np.where(0.8*Phh2[:,i] < Pshot2[i])[0]
		#~ dede3 = np.where(0.8*Phh3[:,i] < Pshot3[i])[0]
		#~ dede4 = np.where(0.8*Phh4[:,i] < Pshot4[i])[0]
		#~ dedemin1 = np.min(dede1)
		#~ dedemin2 = np.min(dede2)
		#~ dedemin3 = np.min(dede3)
		#~ dedemin4 = np.min(dede4)
		#~ kk1t[i] = k[dedemin1]
		#~ kk2t[i] = k[dedemin2]
		#~ kk3t[i] = k[dedemin3]
		#~ kk4t[i] = k[dedemin4]
		
	#~ kk1 = np.mean(kk1t)
	#~ kk2 = np.mean(kk2t)
	#~ kk3 = np.mean(kk3t)
	#~ kk4 = np.mean(kk4t)
	#~ kk4 = np.mean(kk4t)

	
	
	### do the mean over quantitites ###
	
	#~ Pmm = np.mean(Pmat[:,0:11], axis=1)
	#~ Pshot1 = np.mean(Pshot1)
	#~ Pshot2 = np.mean(Pshot2)
	#~ Pshot3 = np.mean(Pshot3)
	#~ Pshot4 = np.mean(Pshot4)
	#~ PH1 = np.mean(Phh1[:,0:11], axis=1)
	#~ PH2 = np.mean(Phh2[:,0:11], axis=1)
	#~ PH3 = np.mean(Phh3[:,0:11], axis=1)
	#~ PH4 = np.mean(Phh4[:,0:11], axis=1)
	
	#~ errPhh1 = np.std(Phh1[:,0:11], axis=1)
	#~ errPhh2 = np.std(Phh2[:,0:11], axis=1)
	#~ errPhh3 = np.std(Phh3[:,0:11], axis=1)
	#~ errPhh4 = np.std(Phh4[:,0:11], axis=1)
	
	
	
	#~ plt.figure()
	#~ M1, =plt.plot(k, PH1, label='halo Power spectrum')
	#~ M2, =plt.plot(k, PH2)
	#~ M3, =plt.plot(k, PH3)
	#~ M4, =plt.plot(k, PH4)
	#~ plt.axhline( Pshot1, color='C0', linestyle='--', label='shot noise')
	#~ plt.axhline( Pshot2, color='C1', linestyle='--')
	#~ plt.axhline( Pshot3, color='C2', linestyle='--')
	#~ plt.axhline( Pshot4, color='C3', linestyle='--')
	#~ plt.axvline( kk1, color='C0', linestyle='--', label='shot noise = 80% of P(k)')
	#~ plt.axvline( kk2, color='C1', linestyle='--')
	#~ plt.axvline( kk3, color='C2', linestyle='--')
	#~ plt.axvline( kk4, color='C3', linestyle='--')
	#~ plt.legend(loc = 'upper right', title='z = '+str(z[j]), fancybox=True)
	#~ plt.legend(loc = 'lower left', title='z = '+str(z[j]), fancybox=True)
	#~ plt.figlegend( (M1,M2,M3,M4), ('mass range M1','mass range M2','mass range M3','mass range M4'), \
	#~ loc = 'upper center', ncol=2, labelspacing=0. )
	#~ plt.xlabel('k')
	#~ plt.ylabel('P(k)')
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(8e-3,3)
	#~ plt.ylim(1e1,1e5)
	#~ plt.show()
	
	#~ kill
	
	#-------------------------------------------------------------------
	#----remove shot noise, compute bias and bias variance -------------
	#-------------------------------------------------------------------
	bhh1 = np.zeros((len(k),10))
	bhh2 = np.zeros((len(k),10))
	bhh3 = np.zeros((len(k),10))
	bhh4 = np.zeros((len(k),10))
	for i in xrange(0,10):
		Phh1[:,i] = Phh1[:,i]-Pshot1[i]
		Phh2[:,i] = Phh2[:,i]-Pshot2[i]
		Phh3[:,i] = Phh3[:,i]-Pshot3[i]
		Phh4[:,i] = Phh4[:,i]-Pshot4[i]
		nul1 = np.where(Phh1[:,i] < 0)[0]
		nul2 = np.where(Phh2[:,i] < 0)[0]
		nul3 = np.where(Phh3[:,i] < 0)[0]
		nul4 = np.where(Phh4[:,i] < 0)[0]
		Phh1[nul1,i] = 0
		Phh2[nul2,i] = 0
		Phh3[nul3,i] = 0
		Phh4[nul4,i] = 0
		bhh1[:,i] = np.sqrt(Phh1[:,i]/Pmat[:,i])
		bhh2[:,i] = np.sqrt(Phh2[:,i]/Pmat[:,i])
		bhh3[:,i] = np.sqrt(Phh3[:,i]/Pmat[:,i])
		bhh4[:,i] = np.sqrt(Phh4[:,i]/Pmat[:,i])
		
		
	### do the mean over quantitites ###
	
	Pmm = np.mean(Pmat[:,0:11], axis=1)
	PH1 = np.mean(Phh1[:,0:11], axis=1)
	PH2 = np.mean(Phh2[:,0:11], axis=1)
	PH3 = np.mean(Phh3[:,0:11], axis=1)
	PH4 = np.mean(Phh4[:,0:11], axis=1)

	
	bias1 = np.mean(bhh1[:,0:11], axis=1)
	bias2 = np.mean(bhh2[:,0:11], axis=1)
	bias3 = np.mean(bhh3[:,0:11], axis=1)
	bias4 = np.mean(bhh4[:,0:11], axis=1)
	
	errb1 = np.std(bhh1[:,0:11], axis=1)
	errb2 = np.std(bhh2[:,0:11], axis=1)
	errb3 = np.std(bhh3[:,0:11], axis=1)
	errb4 = np.std(bhh4[:,0:11], axis=1)
	
	errPhh1 = np.std(Phh1[:,0:11], axis=1)
	errPhh2 = np.std(Phh2[:,0:11], axis=1)
	errPhh3 = np.std(Phh3[:,0:11], axis=1)
	errPhh4 = np.std(Phh4[:,0:11], axis=1)
	

	
	
	
	#~ Phh_bis = np.zeros((len(k),len(mass_range)))
	#~ Phshot_bis = np.zeros((len(k),len(mass_range)))
	#~ bias_bis = np.zeros((len(k),len(mass_range)))
	#~ for i in xrange(0,len(mass_range)):
		#~ Phh_bis[:,i] = np.interp(k,kh,Phh[:,i])
		#~ Phshot_bis[:,i] = np.interp(k,kh,Phshot[:,i])
		#~ bias_bis[:,i] = np.interp(k,kh,bias[:,i])

	#---------------------------------------------------
	#------------ matter Redshift space -----
	#---------------------------------------------------
	d1 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Pcc_realisation_axis_0_z='+str(z[j])+'.txt')
	d2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Pcc_realisation_axis_1_z='+str(z[j])+'.txt')
	d3 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Pcc_realisation_axis_2_z='+str(z[j])+'.txt')
	kred = d[:,10]
	Pmat_r1 = np.zeros((len(kmat),10))
	Pmat_r2 = np.zeros((len(kmat),10))
	Pmat_r3 = np.zeros((len(kmat),10))
	for i in xrange(0,10):
		Pmat_r1[:,i]= d1[:,i]
		Pmat_r2[:,i]= d2[:,i]
		Pmat_r3[:,i]= d3[:,i]

	Pmat_r = (Pmat_r1 + Pmat_r2 + Pmat_r3)/3
	Pred = np.mean(Pmat_r[:,0:11], axis=1)


	#---------------------------------------------------
	#--------- halo redshift space -------------------------
	#---------------------------------------------------
	d1a = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh1_realisation_red_axis_0_z='+str(z[j])+'.txt')
	d1b = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh2_realisation_red_axis_0_z='+str(z[j])+'.txt')
	d1c = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh3_realisation_red_axis_0_z='+str(z[j])+'.txt')
	d1d = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh4_realisation_red_axis_0_z='+str(z[j])+'.txt')
	d2a = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh1_realisation_red_axis_1_z='+str(z[j])+'.txt')
	d2b = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh2_realisation_red_axis_1_z='+str(z[j])+'.txt')
	d2c = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh3_realisation_red_axis_1_z='+str(z[j])+'.txt')
	d2d = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh4_realisation_red_axis_1_z='+str(z[j])+'.txt')
	d3a = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh1_realisation_red_axis_2_z='+str(z[j])+'.txt')
	d3b = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh2_realisation_red_axis_2_z='+str(z[j])+'.txt')
	d3c = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh3_realisation_red_axis_2_z='+str(z[j])+'.txt')
	d3d = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh4_realisation_red_axis_2_z='+str(z[j])+'.txt')


	kx1a = d1a[:,19]
	Px1a = np.zeros((len(kx1a),10))
	Px1b = np.zeros((len(kx1a),10))
	Px1c = np.zeros((len(kx1a),10))
	Px1d = np.zeros((len(kx1a),10))
	Pxshot1a = np.zeros((10))
	Pxshot1b = np.zeros((10))
	Pxshot1c = np.zeros((10))
	Pxshot1d = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Px1a[:,i]= d1a[:,pnum1[i]]
		Px1b[:,i]= d1b[:,pnum1[i]]
		Px1c[:,i]= d1c[:,pnum1[i]]
		Px1d[:,i]= d1d[:,pnum1[i]]
		Pxshot1a[i]= d1a[0,pnum2[i]]
		Pxshot1b[i]= d1b[0,pnum2[i]]
		Pxshot1c[i]= d1c[0,pnum2[i]]
		Pxshot1d[i]= d1d[0,pnum2[i]]
	kx2a = d2a[:,19]
	Px2a = np.zeros((len(kx2a),10))
	Px2b = np.zeros((len(kx2a),10))
	Px2c = np.zeros((len(kx2a),10))
	Px2d = np.zeros((len(kx2a),10))
	Pxshot2a = np.zeros((10))
	Pxshot2b = np.zeros((10))
	Pxshot2c = np.zeros((10))
	Pxshot2d = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Px2a[:,i]= d2a[:,pnum1[i]]
		Px2b[:,i]= d2b[:,pnum1[i]]
		Px2c[:,i]= d2c[:,pnum1[i]]
		Px2d[:,i]= d2d[:,pnum1[i]]
		Pxshot2a[i]= d2a[0,pnum2[i]]
		Pxshot2b[i]= d2b[0,pnum2[i]]
		Pxshot2c[i]= d2c[0,pnum2[i]]
		Pxshot2d[i]= d2d[0,pnum2[i]]
	kx3a = d3a[:,19]
	Px3a = np.zeros((len(kx3a),10))
	Px3b = np.zeros((len(kx3a),10))
	Px3c = np.zeros((len(kx3a),10))
	Px3d = np.zeros((len(kx3a),10))
	Pxshot3a = np.zeros((10))
	Pxshot3b = np.zeros((10))
	Pxshot3c = np.zeros((10))
	Pxshot3d = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Px3a[:,i]= d3a[:,pnum1[i]]
		Px3b[:,i]= d3b[:,pnum1[i]]
		Px3c[:,i]= d3c[:,pnum1[i]]
		Px3d[:,i]= d3d[:,pnum1[i]]
		Pxshot3a[i]= d3a[0,pnum2[i]]
		Pxshot3b[i]= d3b[0,pnum2[i]]
		Pxshot3c[i]= d3c[0,pnum2[i]]
		Pxshot3d[i]= d3d[0,pnum2[i]]
		
	for i in xrange(0,10):
		Px1a[:,i] = Px1a[:,i]-Pxshot1a[i]
		Px1b[:,i] = Px1b[:,i]-Pxshot1b[i]
		Px1c[:,i] = Px1c[:,i]-Pxshot1c[i]
		Px1d[:,i] = Px1d[:,i]-Pxshot1d[i]
		Px2a[:,i] = Px2a[:,i]-Pxshot2a[i]
		Px2b[:,i] = Px2b[:,i]-Pxshot2b[i]
		Px2c[:,i] = Px2c[:,i]-Pxshot2c[i]
		Px2d[:,i] = Px2d[:,i]-Pxshot2d[i]
		Px3a[:,i] = Px3a[:,i]-Pxshot3a[i]
		Px3b[:,i] = Px3b[:,i]-Pxshot3b[i]
		Px3c[:,i] = Px3c[:,i]-Pxshot3c[i]
		Px3d[:,i] = Px3d[:,i]-Pxshot3d[i]
		
		nul1a = np.where(Px1a[:,i] < 0)[0]
		Px1a[nul1a,i] = 0
		nul1b = np.where(Px1b[:,i] < 0)[0]
		Px1b[nul1b,i] = 0
		nul1c = np.where(Px1c[:,i] < 0)[0]
		Px1c[nul1c,i] = 0
		nul1d = np.where(Px1d[:,i] < 0)[0]
		Px1d[nul1d,i] = 0
		nul2a = np.where(Px2a[:,i] < 0)[0]
		Px2a[nul2a,i] = 0
		nul2b = np.where(Px2b[:,i] < 0)[0]
		Px2b[nul2b,i] = 0
		nul2c = np.where(Px2c[:,i] < 0)[0]
		Px2c[nul2c,i] = 0
		nul2d = np.where(Px2d[:,i] < 0)[0]
		Px2d[nul2d,i] = 0
		nul3a = np.where(Px3a[:,i] < 0)[0]
		Px3a[nul3a,i] = 0
		nul3b = np.where(Px3b[:,i] < 0)[0]
		Px3b[nul3b,i] = 0
		nul3c = np.where(Px3c[:,i] < 0)[0]
		Px3c[nul3c,i] = 0
		nul3d = np.where(Px3d[:,i] < 0)[0]
		Px3d[nul3d,i] = 0

	Pmono1temp = (Px1a + Px2a + Px3a)/3
	Pmono2temp = (Px1b + Px2b + Px3b)/3
	Pmono3temp = (Px1c + Px2c + Px3c)/3
	Pmono4temp = (Px1d + Px2d + Px3d)/3


	### do the mean and std over quantitites ###
	
	Pmono1 = np.mean(Pmono1temp[:,0:11], axis=1)
	Pmono2 = np.mean(Pmono2temp[:,0:11], axis=1)
	Pmono3 = np.mean(Pmono3temp[:,0:11], axis=1)
	Pmono4 = np.mean(Pmono4temp[:,0:11], axis=1)
	
	
	errPr1 = np.std(Pmono1temp[:,0:11], axis=1)
	errPr2 = np.std(Pmono2temp[:,0:11], axis=1)
	errPr3 = np.std(Pmono3temp[:,0:11], axis=1)
	errPr4 = np.std(Pmono4temp[:,0:11], axis=1)

	#-------------------------------------------------------------------
	#--- compute bias and bias variance -------------
	#-------------------------------------------------------------------
	bredhh1 = np.zeros((len(k),10))
	bredhh2 = np.zeros((len(k),10))
	bredhh3 = np.zeros((len(k),10))
	bredhh4 = np.zeros((len(k),10))
	for i in xrange(0,10):
		bredhh1[:,i] = np.sqrt(Pmono1temp[:,i]/Pmat_r[:,i])
		bredhh2[:,i] = np.sqrt(Pmono2temp[:,i]/Pmat_r[:,i])
		bredhh3[:,i] = np.sqrt(Pmono3temp[:,i]/Pmat_r[:,i])
		bredhh4[:,i] = np.sqrt(Pmono4temp[:,i]/Pmat_r[:,i])
		
		
	### do the mean over quantitites ###


	
	biasred1 = np.mean(bredhh1[:,0:11], axis=1)
	biasred2 = np.mean(bredhh2[:,0:11], axis=1)
	biasred3 = np.mean(bredhh3[:,0:11], axis=1)
	biasred4 = np.mean(bredhh4[:,0:11], axis=1)
	
	errbred1 = np.std(bredhh1[:,0:11], axis=1)
	errbred2 = np.std(bredhh2[:,0:11], axis=1)
	errbred3 = np.std(bredhh3[:,0:11], axis=1)
	errbred4 = np.std(bredhh4[:,0:11], axis=1)



	#~ #----------------------------------------------------------------
	#~ #----------Tinker, Crocce param bias ----------------------------
	#~ #----------------------------------------------------------------

	for i in xrange(0,len(mass_range)):
		e = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/coeff_large-scale_z='+str(z[j])+'.txt', skiprows=1)
		Tb1 = e[0]
		Tb2 = e[1]
		Tb3 = e[2]
		Tb4 = e[3]
		Cb1 = e[4]
		Cb2 = e[5]
		Cb3 = e[6]
		Cb4 = e[7]

	#~ print Tb1, Cb1
	
	
	#~ plt.figure()
	#~ M1, = plt.plot(kh1, bias1, label='$b_{cc}$')
	#~ M2, = plt.plot(kh2, bias2)
	#~ M3, = plt.plot(kh3, bias3)
	#~ M4, = plt.plot(kh4, bias4)
	#~ plt.axhline(Tb1, color='C0', linestyle='--', label='Tinker effective bias')
	#~ plt.axhline(Tb2, color='C1', linestyle='--')
	#~ plt.axhline(Tb3, color='C2', linestyle='--')
	#~ plt.axhline(Tb4, color='C3', linestyle='--')
	#~ plt.fill_between(k1,bias1-errb1, bias1+errb1, alpha=0.6)
	#~ plt.fill_between(k2,bias2-errb2, bias2+errb2, alpha=0.6)
	#~ plt.fill_between(k3,bias3-errb3, bias3+errb3, alpha=0.6)
	#~ plt.fill_between(k4,bias4-errb4, bias4+errb4, alpha=0.6)
	#~ plt.legend(loc = 'upper right', title='z = '+str(z[j]), fancybox=True)
	#~ plt.figlegend( (M1,M2,M3,M4), ('mass range M1','mass range M2','mass range M3','mass range M4'), \
	#~ loc = 'upper center', ncol=2, labelspacing=0. )
	#~ plt.xlabel('k')
	#~ plt.ylabel('b(k)')
	#~ plt.xscale('log')
	#~ plt.xlim(8e-3,4)
	#~ plt.ylim(0,2)
	#~ plt.show()
	
	#~ kill

	########################################################################
	######### 0.15 eV Massive neutrino 
	########################################################################
	#~ nv = 0.15
	#~ #-------------------------------------------------
	#~ #---------------- Class --------------------------
	#~ #-------------------------------------------------
	#~ Class = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/expected/expected_Pk_z='+str(z[j])+'.txt')
	#~ #Class_nl = np.loadtxt('/home/david/codes/class/output/test_pk_nl.dat')
	#~ Class_trans = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/test_'+str(z[j])+'_tk.dat')
	#~ kclass = Class[:,0]
	#~ Pclass = Class[:,1]
	#~ #kclass_nl = Class_nl[:,0]
	#~ #Pclass_nl = Class_nl[:,1]
	#~ ktrans = Class_trans[:,0]
	#~ Tb = Class_trans[:,2]
	#~ Tcdm = Class_trans[:,3]
	#~ Tm = Class_trans[:,8]

	#~ k = np.logspace(np.min(np.log10(kclass)), np.max(np.log10(kclass)), 150)
	#~ Tb = np.interp(k,ktrans,Tb)
	#~ Tcdm =  np.interp(k,ktrans,Tcdm)
	#~ Tm =  np.interp(k,ktrans,Tm)
	#~ Pclass  = np.interp(k,kclass,Pclass)
	#~ #Pclass_nl = np.interp(k,kclass_nl,Pclass_nl)


	#~ #-----------------------------------------------------------------------
	#~ #-------- get the transfer function and Pcc ----------------------------
	#~ #-----------------------------------------------------------------------
	#~ Tcb = (Omega_c * Tcdm + Omega_b * Tb)/(Omega_c + Omega_b)
	#~ Pcc = Pclass * (Tcb/Tm)**2
	#~ #Pcc_nl = Pclass_nl * (Tcb/Tm)**2
	#~ Plin = Pcc

#-----------------------------------------------------------------------
	#---------------- matter neutrino Real space ---------------------------
	#-----------------------------------------------------------------------
	#~ d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/NCV1/analysis/Pk_c_z='+str(z[j])+'.txt')
	#~ e = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/NCV2/analysis/Pk_c_z='+str(z[j])+'.txt')
	#~ k1 = d[:,0]
	#~ p1 = d[:,1]
	#~ k2 = e[:,0]
	#~ p2 = e[:,1]


	#~ d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Pcc_realisation_'+str(nv)+'z='+str(z[j])+'_.txt')
	#~ kmat = d[:,8]
	#~ Pmat = np.zeros((len(kmat),10))
	#~ for i in xrange(0,8):
		#~ Pmat[:,i]= d[:,i]
	
	#~ Pmat[:,8] = p1
	#~ Pmat[:,9] = p2


	#~ #-----------------------------------------------------------------------
	#~ #---------------- halo neutrino Real space ---------------------------
	#~ #-----------------------------------------------------------------------
	#~ d1 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_0.15_z='+str(z[j])+'.txt')
	#~ k = d1[:,19]
	#~ Phh1 = np.zeros((len(k),10))
	#~ Pshot1 = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Phh1[:,i]= d1[:,pnum1[i]]
		#~ Pshot1[i]= d1[0,pnum2[i]]
	#~ # second mass range
	#~ d2 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_0.15_z='+str(z[j])+'.txt')
	#~ k = d2[:,19]
	#~ Phh2 = np.zeros((len(k),10))
	#~ Pshot2 = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Phh2[:,i]= d2[:,pnum1[i]]
		#~ Pshot2[i]= d2[0,pnum2[i]]
	#~ # third mass range
	#~ d3 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_0.15_z='+str(z[j])+'.txt')
	#~ k = d3[:,19]
	#~ Phh3 = np.zeros((len(k),10))
	#~ Pshot3 = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Phh3[:,i]= d3[:,pnum1[i]]
		#~ Pshot3[i]= d3[0,pnum2[i]]
	#~ # fourth mass range
	#~ d4 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_0.15_z='+str(z[j])+'.txt')
	#~ k = d4[:,19]
	#~ Phh4 = np.zeros((len(k),10))
	#~ Pshot4 = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Phh4[:,i]= d4[:,pnum1[i]]
		#~ Pshot4[i]= d4[0,pnum2[i]]


	
	#~ #-------------------------------------------------------------------
	#~ #----remove shot noise, compute bias and bias variance -------------
	#~ #-------------------------------------------------------------------
	#~ bhh1 = np.zeros((len(k),10))
	#~ bhh2 = np.zeros((len(k),10))
	#~ bhh3 = np.zeros((len(k),10))
	#~ bhh4 = np.zeros((len(k),10))
	#~ for i in xrange(0,10):
		#~ Phh1[:,i] = Phh1[:,i]-Pshot1[i]
		#~ Phh2[:,i] = Phh2[:,i]-Pshot2[i]
		#~ Phh3[:,i] = Phh3[:,i]-Pshot3[i]
		#~ Phh4[:,i] = Phh4[:,i]-Pshot4[i]
		#~ nul1 = np.where(Phh1[:,i] < 0)[0]
		#~ nul2 = np.where(Phh2[:,i] < 0)[0]
		#~ nul3 = np.where(Phh3[:,i] < 0)[0]
		#~ nul4 = np.where(Phh4[:,i] < 0)[0]
		#~ Phh1[nul1,i] = 0
		#~ Phh2[nul2,i] = 0
		#~ Phh3[nul3,i] = 0
		#~ Phh4[nul4,i] = 0
		#~ bhh1[:,i] = np.sqrt(Phh1[:,i]/Pmat[:,i])
		#~ bhh2[:,i] = np.sqrt(Phh2[:,i]/Pmat[:,i])
		#~ bhh3[:,i] = np.sqrt(Phh3[:,i]/Pmat[:,i])
		#~ bhh4[:,i] = np.sqrt(Phh4[:,i]/Pmat[:,i])
		
		
	#~ ### do the mean over quantitites ###
	
	#~ Pmm = np.mean(Pmat[:,0:11], axis=1)
	#~ PH1 = np.mean(Phh1[:,0:11], axis=1)
	#~ PH2 = np.mean(Phh2[:,0:11], axis=1)
	#~ PH3 = np.mean(Phh3[:,0:11], axis=1)
	#~ PH4 = np.mean(Phh4[:,0:11], axis=1)

	
	#~ bias1 = np.mean(bhh1[:,0:11], axis=1)
	#~ bias2 = np.mean(bhh2[:,0:11], axis=1)
	#~ bias3 = np.mean(bhh3[:,0:11], axis=1)
	#~ bias4 = np.mean(bhh4[:,0:11], axis=1)
	
	#~ errb1 = np.std(bhh1[:,0:11], axis=1)
	#~ errb2 = np.std(bhh2[:,0:11], axis=1)
	#~ errb3 = np.std(bhh3[:,0:11], axis=1)
	#~ errb4 = np.std(bhh4[:,0:11], axis=1)
	
	#~ errPhh1 = np.std(Phh1[:,0:11], axis=1)
	#~ errPhh2 = np.std(Phh2[:,0:11], axis=1)
	#~ errPhh3 = np.std(Phh3[:,0:11], axis=1)
	#~ errPhh4 = np.std(Phh4[:,0:11], axis=1)
	
	
	#~ #-----------------------------------------------------------------------
	#~ #---------------- halo neutrino Redshift space ---------------------------
	#~ #-----------------------------------------------------------------------
	#~ d1a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
	#~ d1b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
	#~ d1c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
	#~ d1d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
	#~ d2a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
	#~ d2b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
	#~ d2c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
	#~ d2d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
	#~ d3a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
	#~ d3b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
	#~ d3c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
	#~ d3d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')


	#~ kx1a = d1a[:,19]
	#~ Px1a = np.zeros((len(kx1a),10))
	#~ Px1b = np.zeros((len(kx1a),10))
	#~ Px1c = np.zeros((len(kx1a),10))
	#~ Px1d = np.zeros((len(kx1a),10))
	#~ Pxshot1a = np.zeros((10))
	#~ Pxshot1b = np.zeros((10))
	#~ Pxshot1c = np.zeros((10))
	#~ Pxshot1d = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Px1a[:,i]= d1a[:,pnum1[i]]
		#~ Px1b[:,i]= d1b[:,pnum1[i]]
		#~ Px1c[:,i]= d1c[:,pnum1[i]]
		#~ Px1d[:,i]= d1d[:,pnum1[i]]
		#~ Pxshot1a[i]= d1a[0,pnum2[i]]
		#~ Pxshot1b[i]= d1b[0,pnum2[i]]
		#~ Pxshot1c[i]= d1c[0,pnum2[i]]
		#~ Pxshot1d[i]= d1d[0,pnum2[i]]
	#~ kx2a = d2a[:,19]
	#~ Px2a = np.zeros((len(kx2a),10))
	#~ Px2b = np.zeros((len(kx2a),10))
	#~ Px2c = np.zeros((len(kx2a),10))
	#~ Px2d = np.zeros((len(kx2a),10))
	#~ Pxshot2a = np.zeros((10))
	#~ Pxshot2b = np.zeros((10))
	#~ Pxshot2c = np.zeros((10))
	#~ Pxshot2d = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Px2a[:,i]= d2a[:,pnum1[i]]
		#~ Px2b[:,i]= d2b[:,pnum1[i]]
		#~ Px2c[:,i]= d2c[:,pnum1[i]]
		#~ Px2d[:,i]= d2d[:,pnum1[i]]
		#~ Pxshot2a[i]= d2a[0,pnum2[i]]
		#~ Pxshot2b[i]= d2b[0,pnum2[i]]
		#~ Pxshot2c[i]= d2c[0,pnum2[i]]
		#~ Pxshot2d[i]= d2d[0,pnum2[i]]
	#~ kx3a = d3a[:,19]
	#~ Px3a = np.zeros((len(kx3a),10))
	#~ Px3b = np.zeros((len(kx3a),10))
	#~ Px3c = np.zeros((len(kx3a),10))
	#~ Px3d = np.zeros((len(kx3a),10))
	#~ Pxshot3a = np.zeros((10))
	#~ Pxshot3b = np.zeros((10))
	#~ Pxshot3c = np.zeros((10))
	#~ Pxshot3d = np.zeros((10))
	#~ pnum1 = [0,2,4,6,8,10,12,14,16,18]
	#~ pnum2 = [1,3,5,7,9,11,13,15,17,20]
	#~ for i in xrange(0,10):
		#~ Px3a[:,i]= d3a[:,pnum1[i]]
		#~ Px3b[:,i]= d3b[:,pnum1[i]]
		#~ Px3c[:,i]= d3c[:,pnum1[i]]
		#~ Px3d[:,i]= d3d[:,pnum1[i]]
		#~ Pxshot3a[i]= d3a[0,pnum2[i]]
		#~ Pxshot3b[i]= d3b[0,pnum2[i]]
		#~ Pxshot3c[i]= d3c[0,pnum2[i]]
		#~ Pxshot3d[i]= d3d[0,pnum2[i]]
		
	#~ for i in xrange(0,10):
		#~ Px1a[:,i] = Px1a[:,i]-Pxshot1a[i]
		#~ Px1b[:,i] = Px1b[:,i]-Pxshot1b[i]
		#~ Px1c[:,i] = Px1c[:,i]-Pxshot1c[i]
		#~ Px1d[:,i] = Px1d[:,i]-Pxshot1d[i]
		#~ Px2a[:,i] = Px2a[:,i]-Pxshot2a[i]
		#~ Px2b[:,i] = Px2b[:,i]-Pxshot2b[i]
		#~ Px2c[:,i] = Px2c[:,i]-Pxshot2c[i]
		#~ Px2d[:,i] = Px2d[:,i]-Pxshot2d[i]
		#~ Px3a[:,i] = Px3a[:,i]-Pxshot3a[i]
		#~ Px3b[:,i] = Px3b[:,i]-Pxshot3b[i]
		#~ Px3c[:,i] = Px3c[:,i]-Pxshot3c[i]
		#~ Px3d[:,i] = Px3d[:,i]-Pxshot3d[i]
		
		#~ nul1a = np.where(Px1a[:,i] < 0)[0]
		#~ Px1a[nul1a,i] = 0
		#~ nul1b = np.where(Px1b[:,i] < 0)[0]
		#~ Px1b[nul1b,i] = 0
		#~ nul1c = np.where(Px1c[:,i] < 0)[0]
		#~ Px1c[nul1c,i] = 0
		#~ nul1d = np.where(Px1d[:,i] < 0)[0]
		#~ Px1d[nul1d,i] = 0
		#~ nul2a = np.where(Px2a[:,i] < 0)[0]
		#~ Px2a[nul2a,i] = 0
		#~ nul2b = np.where(Px2b[:,i] < 0)[0]
		#~ Px2b[nul2b,i] = 0
		#~ nul2c = np.where(Px2c[:,i] < 0)[0]
		#~ Px2c[nul2c,i] = 0
		#~ nul2d = np.where(Px2d[:,i] < 0)[0]
		#~ Px2d[nul2d,i] = 0
		#~ nul3a = np.where(Px3a[:,i] < 0)[0]
		#~ Px3a[nul3a,i] = 0
		#~ nul3b = np.where(Px3b[:,i] < 0)[0]
		#~ Px3b[nul3b,i] = 0
		#~ nul3c = np.where(Px3c[:,i] < 0)[0]
		#~ Px3c[nul3c,i] = 0
		#~ nul3d = np.where(Px3d[:,i] < 0)[0]
		#~ Px3d[nul3d,i] = 0

	#~ Pmono1temp = (Px1a + Px2a + Px3a)/3
	#~ Pmono2temp = (Px1b + Px2b + Px3b)/3
	#~ Pmono3temp = (Px1c + Px2c + Px3c)/3
	#~ Pmono4temp = (Px1d + Px2d + Px3d)/3


	#~ ### do the mean and std over quantitites ###
	
	#~ Pmono1 = np.mean(Pmono1temp[:,0:11], axis=1)
	#~ Pmono2 = np.mean(Pmono2temp[:,0:11], axis=1)
	#~ Pmono3 = np.mean(Pmono3temp[:,0:11], axis=1)
	#~ Pmono4 = np.mean(Pmono4temp[:,0:11], axis=1)
	
	
	#~ errPr1 = np.std(Pmono1temp[:,0:11], axis=1)
	#~ errPr2 = np.std(Pmono2temp[:,0:11], axis=1)
	#~ errPr3 = np.std(Pmono3temp[:,0:11], axis=1)
	#~ errPr4 = np.std(Pmono4temp[:,0:11], axis=1)

	#~ #-------------------------------------------------------------------
	#~ #--- compute bias and bias variance -------------
	#~ #-------------------------------------------------------------------
	#~ bredhh1 = np.zeros((len(k),10))
	#~ bredhh2 = np.zeros((len(k),10))
	#~ bredhh3 = np.zeros((len(k),10))
	#~ bredhh4 = np.zeros((len(k),10))
	#~ for i in xrange(0,10):
		#~ bredhh1[:,i] = np.sqrt(Pmono1temp[:,i]/Pmat_r[:,i])
		#~ bredhh2[:,i] = np.sqrt(Pmono2temp[:,i]/Pmat_r[:,i])
		#~ bredhh3[:,i] = np.sqrt(Pmono3temp[:,i]/Pmat_r[:,i])
		#~ bredhh4[:,i] = np.sqrt(Pmono4temp[:,i]/Pmat_r[:,i])
		
		
	#~ ### do the mean over quantitites ###


	
	#~ biasred1 = np.mean(bredhh1[:,0:11], axis=1)
	#~ biasred2 = np.mean(bredhh2[:,0:11], axis=1)
	#~ biasred3 = np.mean(bredhh3[:,0:11], axis=1)
	#~ biasred4 = np.mean(bredhh4[:,0:11], axis=1)
	
	#~ errbred1 = np.std(bredhh1[:,0:11], axis=1)
	#~ errbred2 = np.std(bredhh2[:,0:11], axis=1)
	#~ errbred3 = np.std(bredhh3[:,0:11], axis=1)
	#~ errbred4 = np.std(bredhh4[:,0:11], axis=1)




	
	####################################################################
	###### compute the one loop correction 
	####################################################################

	# set the parameters for the power spectrum window and
	# Fourier coefficient window 
	#P_window=np.array([.2,.2])  
	C_window=.75

	# padding length 
	nu=-2; n_pad=len(k1)
	n_pad=int(0.5*len(k1))
	to_do=['all']
					
	# initialize the FASTPT class 
	# including extrapolation to higher and lower k  
	# time the operation
	t1=time()
	fastpt=FPT.FASTPT(k1,to_do=to_do,n_pad=n_pad) 
	t2=time()
		
	# calculate 1loop SPT (and time the operation) for density
	P_spt_dd=fastpt.one_loop_dd(Plin,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	# calculate 1loop SPT (and time the operation) for velocity
	P_spt_tt=fastpt.one_loop_tt(Plin,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	# calculate 1loop SPT (and time the operation) for velocity - density
	P_spt_dt=fastpt.one_loop_dt(Plin,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	#calculate tidal torque EE and BB P(k)
	#~ P_RSD=fastpt.RSD_components(P,1.0,C_window=C_window)	

	# update the power spectrum
	Pmod_dd=Plin+P_spt_dd[0]
	Pmod_dt=Plin+P_spt_dt[0]
	Pmod_tt=Plin+P_spt_tt[0]
	
	A = P_spt_dd[2]
	B = P_spt_dd[3]
	C = P_spt_dd[4]
	D = P_spt_dd[5]
	E = P_spt_dd[6]
	F = P_spt_dd[7]
	G = P_spt_dt[2]
	H = P_spt_dt[3]
	
	
	#~ plt.figure()
	#~ plt.plot(k1,k1**1.5*F, color='C3', label=r'$\sigma_{3}^{2}(k) P^{lin}$')
	#~ plt.plot(k1,k1**1.5*Pmod_dd, color='k', label=r'$P_{\delta\delta}$')
	#~ plt.plot(k1,k1**1.5*A, color='C0', linestyle=':' , label=r'$P_{b2,\delta}$')
	#~ plt.plot(k1,k1**1.5*G, color='C1', linestyle=':' , label=r'$P_{b2,\theta}$')
	#~ plt.plot(k1,k1**1.5*C, color='C2', linestyle='--', label=r'$P_{bs2,\delta}$')
	#~ plt.legend(loc='upper left', ncol=2, fancybox=True)
	#~ plt.xlim(0.01,0.2)
	#~ plt.xlabel('k')
	#~ plt.ylabel(r'$k^{1.5} \times P(k)$ [(Mpc/h)]')
	#~ plt.xscale('log')
	#~ plt.ylim(ymax=250)
	#~ plt.show()
	
	#~ kill
	
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file1.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], Pmod_dd[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file2.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], Pmod_dt[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file3.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], Pmod_tt[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file4.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], A[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file5.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], B[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file6.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], C[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file7.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], D[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file8.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], E[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file9.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], F[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file10.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], G[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file11.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], H[index_k]))
	fid_file.close()
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file12.txt', 'w+') as fid_file:
		for index_k in xrange(len(k1)):
			fid_file.write('%.8g %.8g\n' % ( k1[index_k], Plin[index_k]))
	fid_file.close()
	
	#~ expected_CF.expected(j)
	

	
	####################################################################
	########## compute Kaiser coefficient 
	####################################################################
	
	#~ mono = (1 + 2/3.*(f[ind]) + 1/5.*(f[ind])**2) 
	#~ quadru = (4/3.*(f[ind]) + 4/7.*(f[ind])**2)


	#~ monobias = np.zeros((len(k),len(mass_range)))
	#~ monobias_simu = np.zeros((len(k),len(mass_range)))
	#~ quadrubias = np.zeros((len(k),len(mass_range)))
	#~ for i in xrange(0,len(mass_range)):
		#~ monobias[:,i] = (1 + 2/3.*(f[ind]/biasF[:,i]) + 1/5.*(f[ind]/biasF[:,i])**2) 
		#~ monobias_simu[:,i] = (1 + 2/3.*(f[ind]/bias_bis[:,i]) + 1/5.*(f[ind]/bias_bis[:,i])**2) 
		#~ quadrubias[:,i] = (4/3.*(f[ind]/biasF[:,i]) + 4/7.*(f[ind]/biasF[:,i])**2)

	### compute mean linear bias for each simu mass bin
	#~ lin = np.where(k<=0.05)[0]
	#~ blin = np.mean(bias_bis[lin,:], axis=0)


	
	####################################################################
	###### Read expected PT terms
	####################################################################
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
	Pmod_dd = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
	Pmod_dt = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected3-'+str(z[j])+'.txt')
	Pmod_tt = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected4-'+str(z[j])+'.txt')
	A = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected5-'+str(z[j])+'.txt')
	B = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected6-'+str(z[j])+'.txt')
	C = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected7-'+str(z[j])+'.txt')
	D = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected8-'+str(z[j])+'.txt')
	E = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected9-'+str(z[j])+'.txt')
	F = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected10-'+str(z[j])+'.txt')
	G = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected11-'+str(z[j])+'.txt')
	H = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected12-'+str(z[j])+'.txt')
	Plin2 = pte[:,1]
	
	print len(F)
	
	#~ kill

	

	
	red = ['0.0','0.5','1.0','2.0','3.0']
	ind = red.index(str(z[j]))

	kstop1 = [0.16,0.2,0.25,0.35]
	kstop2 = [0.12,0.16,0.2,0.2]
	kstop3 = [0.15,0.15,0.15,0.15]

	kstop = kstop2[ind]
	
	####################################################################
	##### interpolate data to have more point on fitting scales
	####################################################################
	##### real space
	kbis = np.logspace(np.log10(np.min(k)), np.log10(kstop), 200)
	bias1bis = np.interp(kbis, k, bias1)
	bias2bis = np.interp(kbis, k, bias2)
	bias3bis = np.interp(kbis, k, bias3)
	bias4bis = np.interp(kbis, k, bias4)
	errb1bis = np.interp(kbis, k, errb1)
	errb2bis = np.interp(kbis, k, errb2)
	errb3bis = np.interp(kbis, k, errb3)
	errb4bis = np.interp(kbis, k, errb4)
	Plinbis = np.interp(kbis, k, Plin2)
	Pmmbis = np.interp(kbis, k, Pmm)
	Pmod_dtbis = np.interp(kbis, k, Pmod_dt)
	Pmod_ttbis = np.interp(kbis, k, Pmod_tt)
	Abis = np.interp(kbis, k, A)
	Bbis = np.interp(kbis, k, B)
	Cbis = np.interp(kbis, k, C)
	Dbis = np.interp(kbis, k, D)
	Ebis = np.interp(kbis, k, E)
	Fbis = np.interp(kbis, k, F)
	Gbis = np.interp(kbis, k, G)
	Hbis = np.interp(kbis, k, H)
	PH1bis = np.interp(kbis, k, PH1)
	PH2bis = np.interp(kbis, k, PH2)
	PH3bis = np.interp(kbis, k, PH3)
	PH4bis = np.interp(kbis, k, PH4)
	errPhh1bis = np.interp(kbis, k, errPhh1)
	errPhh2bis = np.interp(kbis, k, errPhh2)
	errPhh3bis = np.interp(kbis, k, errPhh3)
	errPhh4bis = np.interp(kbis, k, errPhh4)

	##### redshift space

	biasred1bis = np.interp(kbis, k, biasred1)
	biasred2bis = np.interp(kbis, k, biasred2)
	biasred3bis = np.interp(kbis, k, biasred3)
	biasred4bis = np.interp(kbis, k, biasred4)
	errb1redbis = np.interp(kbis, k, errbred1)
	errb2redbis = np.interp(kbis, k, errbred2)
	errb3redbis = np.interp(kbis, k, errbred3)
	errb4redbis = np.interp(kbis, k, errbred4)
	Pmono1bis = np.interp(kbis, k, Pmono1)
	Pmono2bis = np.interp(kbis, k, Pmono2)
	Pmono3bis = np.interp(kbis, k, Pmono3)
	Pmono4bis = np.interp(kbis, k, Pmono4)
	errPr1bis = np.interp(kbis, k, errPr1)
	errPr2bis = np.interp(kbis, k, errPr2)
	errPr3bis = np.interp(kbis, k, errPr3)
	errPr4bis = np.interp(kbis, k, errPr4)
	
	
	
	####################################################################
	###### define bias fitting formulae for real space and redshift space
	####################################################################
	def funcb(k, b1, b2, b3, b4):
		return b1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	def funcbis(k, b1, b2, b4):
		return b1 + b2 * k**2 + b4 * k**4 
	
	
	# here kh because the simu scale
	popF1, pcovF1 = curve_fit(funcb, kbis, bias1bis, sigma = errb1bis,  check_finite=True, maxfev=500000)
	popF2, pcovF2 = curve_fit(funcb, kbis, bias2bis, sigma = errb2bis,  check_finite=True, maxfev=500000)
	popF3, pcovF3 = curve_fit(funcb, kbis, bias3bis, sigma = errb3bis,  check_finite=True, maxfev=500000)
	popF4, pcovF4 = curve_fit(funcb, kbis, bias4bis, sigma = errb4bis,  check_finite=True, maxfev=500000)

	#~ popF1bis, pcovF1bis = curve_fit(funcbis, kbis, bias1bis, sigma = errb1bis,  check_finite=True, maxfev=500000)
	#~ popF2bis, pcovF2bis = curve_fit(funcbis, kbis, bias2bis, sigma = errb2bis,  check_finite=True, maxfev=500000)
	#~ popF3bis, pcovF3bis = curve_fit(funcbis, kbis, bias3bis, sigma = errb3bis,  check_finite=True, maxfev=500000)
	#~ popF4bis, pcovF4bis = curve_fit(funcbis, kbis, bias4bis, sigma = errb4bis,  check_finite=True, maxfev=500000)
	



	########################################################################
	##### get fitted coefficient of the non linear bias  from the tidal model 
	#### (E.g Baldauf et el. Arxiv:1201.4827) given by Fast-PT bias function
	########################################################################
	


	def funcbias1(Pdd, b1, b2, bs):
		klim = np.arange(len(Pdd))
		return np.sqrt((b1**2*Pdd + b1*b2*Abis + 1/4.*b2**2*Bbis + b1*bs*Cbis + 1/2.*b2*bs*Dbis + 1/4.*bs**2*Ebis)/Pdd)
	
	def funcbias2(Pdd, b1, b2, bs, b3nl):
		klim = np.arange(len(Pdd))
		return np.sqrt((b1**2*Pdd + b1*b2*Abis + 1/4.*b2**2*Bbis + b1*bs*Cbis + 1/2.*b2*bs*Dbis + 1/4.*bs**2*Ebis \
		+ 2*b1*b3nl*Fbis)/Pdd)
	
	def funcbias3(Pdd, b1, b2, bs):
		klim = np.arange(len(Pdd))
		b3nl = 32/315.*(b1-1)
		return np.sqrt((b1**2*Pdd + b1*b2*Abis + 1/4.*b2**2*Bbis + b1*bs*Cbis + 1/2.*b2*bs*Dbis + 1/4.*bs**2*Ebis \
		+ 2*b1*b3nl*Fbis)/Pdd)
			
	#~ for i in xrange(0,len(mass_range)):
	pop1, pcov1 = curve_fit(funcbias1, Pmmbis, bias1bis, sigma = errb1bis, check_finite=True, maxfev=500000)
	pop2, pcov2 = curve_fit(funcbias1, Pmmbis, bias2bis, sigma = errb2bis, check_finite=True, maxfev=500000)
	pop3, pcov3 = curve_fit(funcbias1, Pmmbis, bias3bis, sigma = errb3bis, check_finite=True, maxfev=500000)
	pop4, pcov4 = curve_fit(funcbias1, Pmmbis, bias4bis, sigma = errb4bis, check_finite=True, maxfev=500000)

	
	popbis1, pcovbis1 = curve_fit(funcbias2, Pmmbis, bias1bis, sigma = errb1bis,check_finite=True, maxfev=500000)
	popbis2, pcovbis2 = curve_fit(funcbias2, Pmmbis, bias2bis, sigma = errb2bis,check_finite=True, maxfev=500000)
	popbis3, pcovbis3 = curve_fit(funcbias2, Pmmbis, bias3bis, sigma = errb3bis,check_finite=True, maxfev=500000)
	popbis4, pcovbis4 = curve_fit(funcbias2, Pmmbis, bias4bis, sigma = errb4bis,check_finite=True, maxfev=500000)

	
	#~ popter1, pcovter1 = curve_fit(funcbias3, Pmmbis, bias1bis, sigma = errb1bis,check_finite=True, maxfev=500000)
	#~ popter2, pcovter2 = curve_fit(funcbias3, Pmmbis, bias2bis, sigma = errb2bis,check_finite=True, maxfev=500000)
	#~ popter3, pcovter3 = curve_fit(funcbias3, Pmmbis, bias3bis, sigma = errb3bis,check_finite=True, maxfev=500000)
	#~ popter4, pcovter4 = curve_fit(funcbias3, Pmmbis, bias4bis, sigma = errb4bis,check_finite=True, maxfev=500000)


	####################################################################
	###### compute linear bias and error
	####################################################################
	
	# on interpolated array
	toto = np.where(kbis < 0.05)[0]
	lb1 = np.mean(bias1bis[toto])
	lb2 = np.mean(bias2bis[toto])
	lb3 = np.mean(bias3bis[toto])
	lb4 = np.mean(bias4bis[toto])
	errlb1 = np.mean(errb1bis[toto])
	errlb2 = np.mean(errb2bis[toto])
	errlb3 = np.mean(errb3bis[toto])
	errlb4 = np.mean(errb4bis[toto])
	
	# on simulation array
	Toto = np.where(k < 0.05)[0]
	Lb1 = np.mean(bias1[Toto])
	Lb2 = np.mean(bias2[Toto])
	Lb3 = np.mean(bias3[Toto])
	Lb4 = np.mean(bias4[Toto])
	errLb1 = np.mean(errb1[Toto])
	errLb2 = np.mean(errb2[Toto])
	errLb3 = np.mean(errb3[Toto])
	errLb4 = np.mean(errb4[Toto])



	####################################################################
	##### compute coefficient with emcee
	####################################################################
	b1x1_mcmc, b2x1_mcmc, b3x1_mcmc, b4x1_mcmc = coeffit_pl(lb1, errlb1, popF1, kbis, bias1bis, errb1bis)
	b1x2_mcmc, b2x2_mcmc, b3x2_mcmc, b4x2_mcmc = coeffit_pl(lb2, errlb2, popF2, kbis, bias2bis, errb2bis)
	b1x3_mcmc, b2x3_mcmc, b3x3_mcmc, b4x3_mcmc = coeffit_pl(lb3, errlb3, popF3, kbis, bias3bis, errb3bis)
	b1x4_mcmc, b2x4_mcmc, b3x4_mcmc, b4x4_mcmc = coeffit_pl(lb4, errlb4, popF4, kbis, bias4bis, errb4bis)
	#~ #-----------------------------------------------------------------------------------------------------
	b1y1_mcmc, b2y1_mcmc, bsy1_mcmc = coeffit_exp1(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, lb1,errlb1, pop1, kbis ,bias1bis ,errb1bis)
	b1y2_mcmc, b2y2_mcmc, bsy2_mcmc = coeffit_exp1(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, lb2,errlb2, pop2, kbis ,bias2bis ,errb2bis)
	b1y3_mcmc, b2y3_mcmc, bsy3_mcmc = coeffit_exp1(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, lb3,errlb3, pop3, kbis ,bias3bis ,errb3bis)
	b1y4_mcmc, b2y4_mcmc, bsy4_mcmc = coeffit_exp1(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, lb4,errlb4, pop4, kbis ,bias4bis ,errb4bis)
	#~ #--------------------------------------------------------------------------------------------------------
	b1z1_mcmc, b2z1_mcmc, bsz1_mcmc, b3z1_mcmc = coeffit_exp2(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, Fbis, lb1, errlb1, popbis1,\
	kbis ,bias1bis ,errb1bis)
	b1z2_mcmc, b2z2_mcmc, bsz2_mcmc, b3z2_mcmc = coeffit_exp2(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, Fbis, lb2, errlb2, popbis2,\
	kbis ,bias2bis ,errb2bis)
	b1z3_mcmc, b2z3_mcmc, bsz3_mcmc, b3z3_mcmc = coeffit_exp2(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, Fbis, lb3, errlb3, popbis3,\
	kbis ,bias3bis ,errb3bis)
	b1z4_mcmc, b2z4_mcmc, bsz4_mcmc, b3z4_mcmc = coeffit_exp2(Pmmbis, Abis, Bbis, Cbis, Dbis, Ebis, Fbis, lb4, errlb4, popbis4,\
	kbis ,bias4bis ,errb4bis)
		
	
	#~ cname1 = 'coeff_pl_'+str(nv)+'_z='+str(z[j])+'.txt'
	#~ cname2 = 'coeff_2exp_'+str(nv)+'_z='+str(z[j])+'.txt'
	#~ cname3 = 'coeff_3exp_'+str(nv)+'_z='+str(z[j])+'.txt'
	#~ with open(cname1, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x1_mcmc[0], b2x1_mcmc[0], b3x1_mcmc[0], b4x1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x2_mcmc[0], b2x2_mcmc[0], b3x2_mcmc[0], b4x2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x3_mcmc[0], b2x3_mcmc[0], b3x3_mcmc[0], b4x3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x4_mcmc[0], b2x4_mcmc[0], b3x4_mcmc[0], b4x4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname2, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y1_mcmc[0], b2y1_mcmc[0], bsy1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y2_mcmc[0], b2y2_mcmc[0], bsy2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y3_mcmc[0], b2y3_mcmc[0], bsy3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y4_mcmc[0], b2y4_mcmc[0], bsy4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname3, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z1_mcmc[0], b2z1_mcmc[0], bsz1_mcmc[0], b3z1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z2_mcmc[0], b2z2_mcmc[0], bsz2_mcmc[0], b3z2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z3_mcmc[0], b2z3_mcmc[0], bsz3_mcmc[0], b3z3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z4_mcmc[0], b2z4_mcmc[0], bsz4_mcmc[0], b3z4_mcmc[0]))
	#~ fid_file.close()
	
	#~ np.savetxt('/home/david/coeff.txt' ,(b1x1_mcmc, b2x1_mcmc, b3x1_mcmc, b4x1_mcmc), delimiter=',', fmt = '%2f')
	#~ np.savetxt('/home/david/coeff.txt' ,(b1x2_mcmc, b2x2_mcmc, b3x2_mcmc, b4x2_mcmc), delimiter=',', fmt = '%2f')
	#~ np.savetxt('/home/david/coeff.txt' ,(b1y1_mcmc, b2y1_mcmc, bsy1_mcmc), delimiter=',', fmt = '%2f')
	#~ np.savetxt('/home/david/coeff.txt' ,(b1z1_mcmc, b2z1_mcmc, bsz1_mcmc, b3z1_mcmc), delimiter=',', fmt = '%2f')
	#~ np.savetxt('/home/david/coeff.txt' ,(b1z2_mcmc, b2z2_mcmc, bsz2_mcmc, b3z2_mcmc), delimiter=',', fmt = '%2f')
	#~ kill

	####################################################################
	#### get results from mcmc analysis to plots the different biases
	####################################################################
	biasF1 = b1x1_mcmc[0] + b2x1_mcmc[0] * k**2 + b3x1_mcmc[0] * k**3 + b4x1_mcmc[0] * k**4
	biasF2 = b1x2_mcmc[0] + b2x2_mcmc[0] * k**2 + b3x2_mcmc[0] * k**3 + b4x2_mcmc[0] * k**4
	biasF3 = b1x3_mcmc[0] + b2x3_mcmc[0] * k**2 + b3x3_mcmc[0] * k**3 + b4x3_mcmc[0] * k**4
	biasF4 = b1x4_mcmc[0] + b2x4_mcmc[0] * k**2 + b3x4_mcmc[0] * k**3 + b4x4_mcmc[0] * k**4
	#------------------------------------------------------------------- 
	bias2PT1 = np.sqrt((b1y1_mcmc[0]**2 * Pmm+ b1y1_mcmc[0]*b2y1_mcmc[0]*A + 1/4.*b2y1_mcmc[0]**2*B + b1y1_mcmc[0]*bsy1_mcmc[0]*C +\
	1/2.*b2y1_mcmc[0]*bsy1_mcmc[0]*D + 1/4.*bsy1_mcmc[0]**2*E )/Pmm)
	bias2PT2 = np.sqrt((b1y2_mcmc[0]**2 * Pmm+ b1y2_mcmc[0]*b2y2_mcmc[0]*A + 1/4.*b2y2_mcmc[0]**2*B + b1y2_mcmc[0]*bsy2_mcmc[0]*C +\
	1/2.*b2y2_mcmc[0]*bsy2_mcmc[0]*D + 1/4.*bsy2_mcmc[0]**2*E )/Pmm)
	bias2PT3 = np.sqrt((b1y3_mcmc[0]**2 * Pmm+ b1y3_mcmc[0]*b2y3_mcmc[0]*A + 1/4.*b2y3_mcmc[0]**2*B + b1y3_mcmc[0]*bsy3_mcmc[0]*C +\
	1/2.*b2y3_mcmc[0]*bsy3_mcmc[0]*D + 1/4.*bsy3_mcmc[0]**2*E )/Pmm)
	bias2PT4 = np.sqrt((b1y4_mcmc[0]**2 * Pmm+ b1y4_mcmc[0]*b2y4_mcmc[0]*A + 1/4.*b2y4_mcmc[0]**2*B + b1y4_mcmc[0]*bsy4_mcmc[0]*C +\
	1/2.*b2y4_mcmc[0]*bsy4_mcmc[0]*D + 1/4.*bsy4_mcmc[0]**2*E )/Pmm)
	#-------------------------------------------------------------------
	bias3PT1 = np.sqrt((b1z1_mcmc[0]**2 * Pmm+ b1z1_mcmc[0]*b2z1_mcmc[0]*A + 1/4.*b2z1_mcmc[0]**2*B + b1z1_mcmc[0]*bsz1_mcmc[0]*C +\
	1/2.*b2z1_mcmc[0]*bsz1_mcmc[0]*D + 1/4.*bsz1_mcmc[0]**2*E + 2*b1z1_mcmc[0]*b3z1_mcmc[0]*F )/Pmm)
	bias3PT2 = np.sqrt((b1z2_mcmc[0]**2 * Pmm+ b1z2_mcmc[0]*b2z2_mcmc[0]*A + 1/4.*b2z2_mcmc[0]**2*B + b1z2_mcmc[0]*bsz2_mcmc[0]*C +\
	1/2.*b2z2_mcmc[0]*bsz2_mcmc[0]*D + 1/4.*bsz2_mcmc[0]**2*E + 2*b1z2_mcmc[0]*b3z2_mcmc[0]*F )/Pmm)
	bias3PT3 = np.sqrt((b1z3_mcmc[0]**2 * Pmm+ b1z3_mcmc[0]*b2z3_mcmc[0]*A + 1/4.*b2z3_mcmc[0]**2*B + b1z3_mcmc[0]*bsz3_mcmc[0]*C +\
	1/2.*b2z3_mcmc[0]*bsz3_mcmc[0]*D + 1/4.*bsz3_mcmc[0]**2*E + 2*b1z3_mcmc[0]*b3z3_mcmc[0]*F )/Pmm)
	bias3PT4 = np.sqrt((b1z4_mcmc[0]**2 * Pmm+ b1z4_mcmc[0]*b2z4_mcmc[0]*A + 1/4.*b2z4_mcmc[0]**2*B + b1z4_mcmc[0]*bsz4_mcmc[0]*C +\
	1/2.*b2z4_mcmc[0]*bsz4_mcmc[0]*D + 1/4.*bsz4_mcmc[0]**2*E + 2*b1z4_mcmc[0]*b3z4_mcmc[0]*F )/Pmm)
	
	#~ ### 3rd order with fixed b3nl
	#~ b3nlTa = 32/315.*(b1tera-1)
	#~ b3nlTb = 32/315.*(b1terb-1)
	#~ b3nlTc = 32/315.*(b1terc-1)
	#~ b3nlTd = 32/315.*(b1terd-1)
	#~ PsptD3r1 = b1tera**2 * Pmm + b1tera*b2tera*A + 1/4.*b2tera**2*B + b1tera*bstera*C + 1/2.*b2tera*bstera*D + 1/4.*bstera**2*E + 2*b1tera*b3nlTa*F
	#~ PsptD3r2 = b1terb**2 * Pmm + b1terb*b2terb*A + 1/4.*b2terb**2*B + b1terb*bsterb*C + 1/2.*b2terb*bsterb*D + 1/4.*bsterb**2*E + 2*b1terb*b3nlTb*F
	#~ PsptD3r3 = b1terc**2 * Pmm + b1terc*b2terc*A + 1/4.*b2terc**2*B + b1terc*bsterc*C + 1/2.*b2terc*bsterc*D + 1/4.*bsterc**2*E + 2*b1terc*b3nlTc*F
	#~ PsptD3r4 = b1terd**2 * Pmm + b1terd*b2terd*A + 1/4.*b2terd**2*B + b1terd*bsterd*C + 1/2.*b2terd*bsterd*D + 1/4.*bsterd**2*E + 2*b1terd*b3nlTd*F


	
	
	####################################################################
	##### different fit
	####################################################################
	
	#~ ### mean ####
	#~ B2 = np.array([bias1/bias1, bias2/bias2, bias3/bias3, bias4/bias4])
	#~ B3 = np.array([biasF1/bias1, biasF2/bias2, biasF3/bias3, biasF4/bias4])
	#~ B3bis = np.array([biasF1bis/bias1, biasF2bis/bias2, biasF3bis/bias3, biasF4bis/bias4])
	#~ B4 = np.array([Tb1/bias1, Tb2/bias2, Tb3/bias3, Tb4/bias4])
	#~ b2 = np.mean(B2,axis=0)
	#~ b3 = np.mean(B3,axis=0)
	#~ b3bis = np.mean(B3bis,axis=0)
	#~ b4 = np.mean(B4,axis=0)
	
	#~ plt.figure()
	#~ B3, = plt.plot(k, b3, label='$b_{cc} = b_{1} + b_{2}*k^{2} + b_{3}*k^{3} + b_{4}*k^{4}$ ')
	#~ B3bis, = plt.plot(k, b3bis, label='$b_{cc} = b_{1} + b_{2}*k^{2} + b_{4}*k^{4}$ ')
	#~ B2, = plt.plot(k, b2, label='N-body', color='k')
	#~ plt.axhline(1, color='k', linestyle='--')
	#~ plt.axhline(1.01, color='k', linestyle=':')
	#~ plt.axhline(0.99, color='k', linestyle=':')
	#~ plt.legend(loc = 'upper left', title='z = '+str(z[j])+'$,  k_{max} = 0.1 h/Mpc$' , fancybox=True)
	#~ plt.figlegend( (B3,B3bis,B2), ('$b_{cc} = b_{1} + b_{2}*k^{2} + b_{3}*k^{3} + b_{4}*k^{4}$ ',\
	#~ '$b_{cc} = b_{1} + b_{2}*k^{2} + b_{4}*k^{4}$ ','N-body'), \
	#~ loc = 'upper center', ncol=2, labelspacing=0., title='z = '+str(z[j])+'$,  k_{max} = 0.1 h^{-1}Mpc$' )
	#~ plt.xlabel('k')
	#~ plt.ylabel('b(k) / $b_{ref}$ (k)')
	#~ plt.xscale('log')
	#~ plt.xlim(8e-3,0.3)
	#~ plt.ylim(0.9,1.1)
	#~ plt.show()



	#~ P1 = np.array([PH1, PH2, PH3, PH4])
	#~ B1 = np.array([np.sqrt(PsptD1r1/Pmm)/bias1, np.sqrt(PsptD1r2/Pmm)/bias2, np.sqrt(PsptD1r3/Pmm)/bias3, np.sqrt(PsptD1r4/Pmm)/bias4])
	#~ B1bis = np.array([np.sqrt(PsptD2r1/Pmm)/bias1, np.sqrt(PsptD2r2/Pmm)/bias2, np.sqrt(PsptD2r3/Pmm)/bias3, np.sqrt(PsptD2r4/Pmm)/bias4])
	#~ B1ter = np.array([np.sqrt(PsptD3r1/Pmm)/bias1, np.sqrt(PsptD3r2/Pmm)/bias2, np.sqrt(PsptD3r3/Pmm)/bias3, np.sqrt(PsptD3r4/Pmm)/bias4])
	#~ B2 = np.array([bias1/bias1, bias2/bias2, bias3/bias3, bias4/bias4])
	#~ B3 = np.array([biasF1/bias1, biasF2/bias2, biasF3/bias3, biasF4/bias4])
	#~ p1 = np.mean(P1, axis=0)
	#~ b1 = np.mean(B1,axis=0)
	#~ b1bis = np.mean(B1bis,axis=0)
	#~ b1ter = np.mean(B1ter,axis=0)
	#~ b2 = np.mean(B2,axis=0)
	#~ b3 = np.mean(B3,axis=0)

		
	#~ plt.figure()
	#~ B3, = plt.plot(k, b3, label='Power law', color='C0')
	#~ B1, = plt.plot(k, b1, label='2nd order expansion', color='C1')
	#~ B1bis, = plt.plot(k, b1bis, label='3rd order expansion with free $b_{3nl}$ ', color='C2')
	#~ B1ter, = plt.plot(k, b1ter, label=r'3rd order expansion with fixed $b_{3nl}$', color='C3')
	#~ B2, = plt.plot(k, b2, label='N-body', color='k')
	
	#~ plt.axhline(1, color='k', linestyle='--')
	#~ plt.axhline(1.01, color='k', linestyle=':')
	#~ plt.axhline(0.99, color='k', linestyle=':')
	#~ plt.legend(loc = 'upper left', fancybox=True)
	#~ plt.title('z = '+str(z[j])+'$,\:  k_{max} = 0.1 h/Mpc$' )
	#~ plt.xlabel('k')
	#~ plt.ylabel('b(k) / $b_{ref}$ (k)')
	#~ plt.xscale('log')
	#~ plt.xlim(8e-3,0.3)
	#~ plt.ylim(0.95,1.05)
	#~ plt.show()
	
		
	#~ kill
	
	
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------
	#~ # set the parameters for the power spectrum window and
	#~ # Fourier coefficient window 
	#~ #P_window=np.array([.2,.2])  
	C_window=.75
	#~ # padding length 
	nu=-2; n_pad=len(kbis)
	n_pad=int(0.5*len(kbis))
	to_do=['all']
	#~ # initialize the FASTPT class 
	#~ # including extrapolation to higher and lower k  
	#~ # time the operation
	fastpt2=FPT.FASTPT(kbis,to_do=to_do,n_pad=n_pad) 
	####################################################################
	#### compute tns coefficient then interpolate for more points
	
	AB2_1bis,AB4_1bis,AB6_1bis,AB8_1bis = fastpt2.RSD_ABsum_components(Plinbis,f[ind],Lb1 ,C_window=C_window)
	#~ AB2_2,AB4_2,AB6_2,AB8_2 = fastpt.RSD_ABsum_components(Plin,f[ind],Lb2 ,C_window=C_window)
	#~ AB2_3,AB4_3,AB6_3,AB8_3 = fastpt.RSD_ABsum_components(Plin,f[ind],Lb3,C_window=C_window)
	#~ AB2_4,AB4_4,AB6_4,AB8_4 = fastpt.RSD_ABsum_components(Plin,f[ind],Lb4,C_window=C_window)
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file13a.txt', 'w+') as fid_file:
		for index_k in xrange(len(kbis)):
			fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kbis[index_k], AB2_1bis[index_k],\
			AB4_1bis[index_k], AB6_1bis[index_k], AB8_1bis[index_k]))
	fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file13b.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k1)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k1[index_k], AB2_2[index_k],\
			#~ AB4_2[index_k], AB6_2[index_k], AB8_2[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file13c.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k1)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k1[index_k], AB2_3[index_k],\
			#~ AB4_3[index_k], AB6_3[index_k], AB8_3[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file13d.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k1)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k1[index_k], AB2_4[index_k],\
			#~ AB4_4[index_k], AB6_4[index_k], AB8_4[index_k]))
	#~ fid_file.close()
	
	######################################################################
	AB2bis_1bis,AB4bis_1bis,AB6bis_1bis,AB8bis_1bis = fastpt2.RSD_ABsum_components(Plinbis,f[ind],b1z1_mcmc[0] ,C_window=C_window)
	#~ AB2bis_2,AB4bis_2,AB6bis_2,AB8bis_2 = fastpt.RSD_ABsum_components(Plin,f[ind],b1z2_mcmc[0] ,C_window=C_window)
	#~ AB2bis_3,AB4bis_3,AB6bis_3,AB8bis_3 = fastpt.RSD_ABsum_components(Plin,f[ind],b1z3_mcmc[0] ,C_window=C_window)
	#~ AB2bis_4,AB4bis_4,AB6bis_4,AB8bis_4 = fastpt.RSD_ABsum_components(Plin,f[ind],b1z4_mcmc[0] ,C_window=C_window)
	with open('/home/david/codes/Paco/data2/0.0eV/exp/file14a.txt', 'w+') as fid_file:
		for index_k in xrange(len(kbis)):
			fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kbis[index_k], AB2bis_1bis[index_k],\
			AB4bis_1bis[index_k], AB6bis_1bis[index_k], AB8bis_1bis[index_k]))
	fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file14b.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k1)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k1[index_k], AB2bis_2[index_k],\
			#~ AB4bis_2[index_k], AB6bis_2[index_k], AB8bis_2[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file14c.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k1)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k1[index_k], AB2bis_3[index_k],\
			#~ AB4bis_3[index_k], AB6bis_3[index_k], AB8bis_3[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file14d.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k1)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k1[index_k], AB2bis_4[index_k],\
			#~ AB4bis_4[index_k], AB6bis_4[index_k], AB8bis_4[index_k]))
	#~ fid_file.close()
	
	#####################################################################
	#### compute tns coefficient on the same bins as the simulation
	expected_CF.expected(j)
	
	####################################################################
	#### read new coefficient
	pte1 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected13a-'+str(z[j])+'.txt')
	#~ pte2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected13b-'+str(z[j])+'.txt')
	#~ pte3 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected13c-'+str(z[j])+'.txt')
	#~ pte4 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected13d-'+str(z[j])+'.txt')
	AB2_1 = pte1[:,3];AB4_1 = pte1[:,2];AB6_1 = pte1[:,1];AB8_1 = pte1[:,0]
	#~ AB2_2 = pte2[:,3];AB4_2 = pte2[:,2];AB6_2 = pte2[:,1];AB8_2 = pte2[:,0]
	#~ AB2_3 = pte3[:,3];AB4_3 = pte3[:,2];AB6_3 = pte3[:,1];AB8_3 = pte3[:,0]
	#~ AB2_4 = pte4[:,3];AB4_4 = pte4[:,2];AB6_4 = pte4[:,1];AB8_4 = pte4[:,0]
	
	#-------------------------------------------------------------------
	pte1 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected14a-'+str(z[j])+'.txt')
	#~ pte2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected14b-'+str(z[j])+'.txt')
	#~ pte3 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected14c-'+str(z[j])+'.txt')
	#~ pte4 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected14d-'+str(z[j])+'.txt')
	AB2bis_1 = pte1[:,3];AB4bis_1 = pte1[:,2];AB6bis_1 = pte1[:,1];AB8bis_1 = pte1[:,0]
	#~ AB2bis_2 = pte2[:,3];AB4bis_2 = pte2[:,2];AB6bis_2 = pte2[:,1];AB8bis_2 = pte2[:,0]
	#~ AB2bis_3 = pte3[:,3];AB4bis_3 = pte3[:,2];AB6bis_3 = pte3[:,1];AB8bis_3 = pte3[:,0]
	#~ AB2bis_4 = pte4[:,3];AB4bis_4 = pte4[:,2];AB6bis_4 = pte4[:,1];AB8bis_4 = pte4[:,0]

	####################################################################
	###### fit the Finger of God effect
	####################################################################

	def KaiserFog(X,sigma):
		k,b = X
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		#~ return  Pmmbis*(bias1bis**2 +  2/3.*bias1bis*f[ind] + 1/5.*f[ind]**2)
		return  Pmmbis*(b**2 +  2/3.*b*f[ind] + 1/5.*f[ind]**2)

	def ScocciFog(X, sigma):
		k,b = X
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		#~ return bias1bis**2*Pmmbis*coeffA + 2/3.*bias1bis*f[ind]*Pmod_dtbis*coeffB + 1/5.*f[ind]**2*Pmod_ttbis*coeffC
		return b**2*Pmmbis*coeffA + 2/3.*b*f[ind]*Pmod_dtbis*coeffB + 1/5.*f[ind]**2*Pmod_ttbis*coeffC

		
	### define a second grid for fast pt
	fastpt2=FPT.FASTPT(kbis,to_do=to_do,n_pad=int(0.5*len(kbis))) 
	
	def TnsFog(X,sigma):
		k,b = X
		AB2,AB4,AB6,AB8 = fastpt2.RSD_ABsum_components(Plinbis,f[ind], b,C_window=C_window)
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		#~ return bias1bis**2*Pmmbis*coeffA + 2/3.*bias1bis*f[ind]*Pmod_dtbis*coeffB + 1/5.*f[ind]**2*Pmod_ttbis*coeffC \
		#~ + 1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE
		return b**2*Pmmbis*coeffA + 2/3.*b*f[ind]*Pmod_dtbis*coeffB + 1/5.*f[ind]**2*Pmod_ttbis*coeffC \
		+ 1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE



	def PTtnsFog(X, sigma):
		k, b1, b2, bs, b3nl = X
		AB2,AB4,AB6,AB8 = fastpt2.RSD_ABsum_components(Plinbis,f[ind], b1,C_window=C_window)
		PsptD1z = b1**2*Pmmbis + b1*b2*Abis + 1/4.*b2**2*Bbis + b1*bs*Cbis + 1/2.*b2*bs*Dbis + 1/4.*bs**2*Ebis + 2*b1*b3nl*Fbis
		PsptT = b1* Pmod_dtbis + b2*Gbis + bs*Hbis + b3nl * Fbis
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		return PsptD1z*coeffA + 2/3.*f[ind]*PsptT*coeffB + 1/5.*f[ind]**2*Pmod_ttbis*coeffC \
		+ 1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE
		
	#### compute least square fit of halo ps
	pop2a, pcov2a = curve_fit(KaiserFog,(kbis,lb1), Pmono1bis, sigma = errPr1bis, check_finite=True, maxfev=500000)
	#~ pop2b, pcov2b = curve_fit(KaiserFog,(kbis,lb2), Pmono2bis, sigma = errPr2bis, check_finite=True, maxfev=500000)
	#~ pop2c, pcov2c = curve_fit(KaiserFog,(kbis,lb3), Pmono3bis, sigma = errPr3bis, check_finite=True, maxfev=500000)
	#~ pop2d, pcov2d = curve_fit(KaiserFog,(kbis,lb4), Pmono4bis, sigma = errPr4bis, check_finite=True, maxfev=500000)
	#------------------------------------------------------------------------------------------------
	pop3a, pcov3a = curve_fit(ScocciFog,(kbis,lb1), Pmono1bis, sigma = errPr1bis, check_finite=True, maxfev=500000)
	#~ pop3b, pcov3b = curve_fit(ScocciFog,(kbis,lb2), Pmono2bis, sigma = errPr2bis, check_finite=True, maxfev=500000)
	#~ pop3c, pcov3c = curve_fit(ScocciFog,(kbis,lb3), Pmono3bis, sigma = errPr3bis, check_finite=True, maxfev=500000)
	#~ pop3d, pcov3d = curve_fit(ScocciFog,(kbis,lb4), Pmono4bis, sigma = errPr4bis, check_finite=True, maxfev=500000)
	#~ #------------------------------------------------------------------------------------------------
	pop4a, pcov4a = curve_fit(TnsFog,(kbis, lb1, AB2_1bis, AB4_1bis, AB6_1bis, AB8_1bis), Pmono1bis,\
	sigma = errPr1bis, check_finite=True, maxfev=500000)
	#~ pop4b, pcov4b = curve_fit(TnsFog,(kbis, lb2, AB2_2bis, AB4_2bis, AB6_2bis, AB8_2bis), Pmono2bis,\
	#~ sigma = errPr2bis, check_finite=True, maxfev=500000)
	#~ pop4c, pcov4c = curve_fit(TnsFog,(kbis, lb3, AB2_3bis, AB4_3bis, AB6_3bis, AB8_3bis), Pmono3bis,\
	#~ sigma = errPr3bis, check_finite=True, maxfev=500000)
	#~ pop4d, pcov4d = curve_fit(TnsFog,(kbis, lb4, AB2_4bis, AB4_4bis, AB6_4bis, AB8_4bis), Pmono4bis,\
	#~ sigma = errPr4bis, check_finite=True, maxfev=500000)
	
	#------------------------------------------------------------------------------------------------
	pop6a, pcov6a = curve_fit(PTtnsFog,(kbis,b1z1_mcmc[0], b2z1_mcmc[0], bsz1_mcmc[0], b3z1_mcmc[0], AB2bis_1bis, AB4bis_1bis,\
	AB6bis_1bis, AB8bis_1bis) , Pmono1bis, sigma = errPr1bis, check_finite=True, maxfev=500000)
	
	#~ pop6b, pcov6b = curve_fit(PTtnsFog,(kbis,b1z2_mcmc[0], b2z2_mcmc[0], bsz2_mcmc[0], b3z2_mcmc[0], AB2bis_2bis, AB4bis_2bis,\
	#~ AB6bis_2bis, AB8bis_2bis) , Pmono2bis, sigma = errPr2bis, check_finite=True, maxfev=500000)
	
	#~ pop6c, pcov6c = curve_fit(PTtnsFog,(kbis,b1z3_mcmc[0], b2z3_mcmc[0], bsz3_mcmc[0], b3z3_mcmc[0], AB2bis_3bis, AB4bis_3bis,\
	#~ AB6bis_3bis, AB8bis_3bis) , Pmono3bis, sigma = errPr3bis, check_finite=True, maxfev=500000)
	
	#~ pop6d, pcov6d = curve_fit(PTtnsFog,(kbis,b1z4_mcmc[0], b2z4_mcmc[0], bsz4_mcmc[0], b3z4_mcmc[0], AB2bis_4bis, AB4bis_4bis,\
	#~ AB6bis_4bis, AB8bis_4bis) , Pmono4bis, sigma = errPr4bis, check_finite=True, maxfev=500000)
	


	#### compute mcmc coefficient of halo ps fit
	print 'kaiser'
	bK1 = coeffit_Kaiser(j,Pmmbis, lb1, errb1bis, pop2a, kbis, Pmono1bis, errPr1bis)
	#~ bK2 = coeffit_Kaiser(j,Pmmbis, lb2, errb2bis, pop2b, kbis, Pmono2bis, errPr2bis)
	#~ bK3 = coeffit_Kaiser(j,Pmmbis, lb3, errb3bis, pop2c, kbis, Pmono3bis, errPr3bis)
	#~ bK4 = coeffit_Kaiser(j,Pmmbis, lb4, errb4bis, pop2d, kbis, Pmono4bis, errPr4bis)
	#~ #----------------------------------------------------------------------------------------
	print 'Scoccimaro'
	bsco1 = coeffit_Scocci(j,Pmmbis,Pmod_dtbis, Pmod_ttbis, lb1, errb1bis, pop3a, kbis, Pmono1bis, errPr1bis )
	#~ bsco2 = coeffit_Scocci(j,Pmmbis,Pmod_dtbis, Pmod_ttbis, lb2, errb2bis, pop3b, kbis, Pmono2bis, errPr2bis )
	#~ bsco3 = coeffit_Scocci(j,Pmmbis,Pmod_dtbis, Pmod_ttbis, lb3, errb3bis, pop3c, kbis, Pmono3bis, errPr3bis )
	#~ bsco4 = coeffit_Scocci(j,Pmmbis,Pmod_dtbis, Pmod_ttbis, lb4, errb4bis, pop3d, kbis, Pmono4bis, errPr4bis )
	#~ #----------------------------------------------------------------------------------------
	print 'Tns'
	btns1 = coeffit_TNS(j,Pmmbis,Pmod_dtbis, Pmod_ttbis,Plinbis, lb1, pop4a, kbis, Pmono1bis, errPr1bis  )
	#~ btns2 = coeffit_TNS(j,Pmmbis,Pmod_dtbis, Pmod_ttbis,Plinbis, lb2, pop4b, kbis, Pmono2bis, errPr2bis  )
	#~ btns3 = coeffit_TNS(j,Pmmbis,Pmod_dtbis, Pmod_ttbis,Plinbis, lb3, pop4c, kbis, Pmono3bis, errPr3bis  )
	#~ btns4 = coeffit_TNS(j,Pmmbis,Pmod_dtbis, Pmod_ttbis,Plinbis, lb4, pop4d, kbis, Pmono4bis, errPr4bis  )
	#----------------------------------------------------------------------------------------
	print 'eTns'
	betns1 = coeffit_eTNS(j, b1z1_mcmc[0], b2z1_mcmc[0], bsz1_mcmc[0], b3z1_mcmc[0], Pmmbis,Pmod_dtbis, Pmod_ttbis,\
	Abis, Bbis, Cbis, Dbis, Ebis, Fbis, Gbis, Hbis, Plinbis, pop6a, kbis, Pmono1bis, errPr1bis)
	
	#~ betns2 = coeffit_eTNS(j, b1z2_mcmc[0], b2z2_mcmc[0], bsz2_mcmc[0], b3z2_mcmc[0], Pmmbis,Pmod_dtbis, Pmod_ttbis,\
	#~ Abis, Bbis, Cbis, Dbis, Ebis, Fbis, Gbis, Hbis, Plinbis, pop6b, kbis, Pmono2bis, errPr2bis)

	#~ betns3 = coeffit_eTNS(j, b1z3_mcmc[0], b2z3_mcmc[0], bsz3_mcmc[0], b3z3_mcmc[0], Pmmbis,Pmod_dtbis, Pmod_ttbis, \
	#~ Abis, Bbis, Cbis, Dbis, Ebis, Fbis, Gbis, Hbis, Plinbis, pop6c, kbis, Pmono3bis, errPr3bis)

	#~ betns4 = coeffit_eTNS(j, b1z4_mcmc[0], b2z4_mcmc[0], bsz4_mcmc[0], b3z4_mcmc[0], Pmmbis,Pmod_dtbis, Pmod_ttbis, \
	#~ Abis, Bbis, Cbis, Dbis, Ebis, Fbis, Gbis, Hbis, Plinbis, pop6d, kbis, Pmono4bis, errPr4bis)

	
	

	#### compute the different power spectra given the mcmc results
	def kaips(b,sigma):
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		return Pmm*(b**2*coeffA +  2/3.*b*f[ind]*coeffB + 1/5.*f[ind]**2*coeffC)
	kai1 = kaips(Lb1, bK1[0][0])
	#~ kai2 = kaips(Lb2, bK2[0][0])
	#~ kai3 = kaips(Lb3, bK3[0][0])
	#~ kai4 = kaips(Lb4, bK4[0][0])
	#---------------------------------------------------------------------------------------
	def scops(b,sigma):
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		return b**2*Pmm*coeffA + 2/3.*b*f[ind]*Pmod_dt*coeffB + 1/5.*f[ind]**2*Pmod_tt*coeffC
	sco1 = scops(Lb1, bsco1[0][0])
	#~ sco2 = scops(Lb2, bsco2[0][0])
	#~ sco3 = scops(Lb3, bsco3[0][0])
	#~ sco4 = scops(Lb4, bsco4[0][0])
	#~ #---------------------------------------------------------------------------------------
	def tnsps(b,sigma, AB2, AB4, AB6, AB8):
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		return b**2*Pmm*coeffA + 2/3.*b*f[ind]*Pmod_dt*coeffB + 1/5.*f[ind]**2*Pmod_tt*coeffC \
		+ 1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE
	tns1 = tnsps(Lb1,btns1[0][0], AB2_1, AB4_1, AB6_1, AB8_1)
	#~ tns2 = tnsps(Lb2,btns2[0][0], AB2_2, AB4_2, AB6_2, AB8_2)
	#~ tns3 = tnsps(Lb3,btns3[0][0], AB2_3, AB4_3, AB6_3, AB8_3)
	#~ tns4 = tnsps(Lb4,btns4[0][0], AB2_4, AB4_4, AB6_4, AB8_4)
	#-------------------------------------------------------------------
	def etnsps(b1,b2,bs,b3nl,sigma, AB2bis, AB4bis, AB6bis, AB8bis):
		PsptD1z = b1**2*Pmm + b1*b2*A+ 1/4.*b2**2*B+ b1*bs*C+ 1/2.*b2*bs*D+ 1/4.*bs**2*E+ 2*b1*b3nl*F
		PsptT = b1* Pmod_dt+ b2*G+ bs*H + b3nl*F
		kappa = k*sigma
		coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		return  PsptD1z*coeffA + 2/3.*f[ind]*PsptT*coeffB + 1/5.*f[ind]**2*Pmod_tt*coeffC \
		+ 1/3.*AB2bis*coeffB+ 1/5.*AB4bis*coeffC+ 1/7.*AB6bis*coeffD+ 1/9.*AB8bis*coeffE
	etns1 = etnsps(b1z1_mcmc[0], b2z1_mcmc[0], bsz1_mcmc[0], b3z1_mcmc[0],betns1[0][0], AB2bis_1, AB4bis_1, AB6bis_1, AB8bis_1)  
	#~ etns2 = etnsps(b1z2_mcmc[0], b2z2_mcmc[0], bsz2_mcmc[0], b3z2_mcmc[0],betns2[0][0], AB2bis_2, AB4bis_2, AB6bis_2, AB8bis_2)  
	#~ etns3 = etnsps(b1z3_mcmc[0], b2z3_mcmc[0], bsz3_mcmc[0], b3z3_mcmc[0],betns3[0][0], AB2bis_3, AB4bis_3, AB6bis_3, AB8bis_3)  
	#~ etns4 = etnsps(b1z4_mcmc[0], b2z4_mcmc[0], bsz4_mcmc[0], b3z4_mcmc[0],betns4[0][0], AB2bis_4, AB4bis_4, AB6bis_4, AB8bis_4)  
 


	#~ plt.figure()
	#~ plt.plot(k, kai1/Pmono1, color='r')
	#~ plt.plot(k, sco1/Pmono1, color='b')
	#~ plt.plot(k, tns1/Pmono1, color='g')
	#~ plt.plot(k, etns1/Pmono1, color='c')
	#~ plt.axhline(1., color='k')
	#~ plt.axhline(1.01, color='k', linestyle='--')
	#~ plt.axhline(0.99, color='k', linestyle='--')
	#~ plt.xscale('log')
	#~ plt.xlim(0.008,1.)
	#~ plt.ylim(0.8,1.2)
	#~ plt.show()
	#~ kill
	

	
	
	####################################################################
	##### compute the mean and error of different quantities
	####################################################################


	
	### mean ####
	B1 = np.array([np.sqrt(PsptD1r1/Pmm)/bias1, np.sqrt(PsptD1r2/Pmm)/bias2, np.sqrt(PsptD1r3/Pmm)/bias3, np.sqrt(PsptD1r4/Pmm)/bias4])
	B1bis = np.array([np.sqrt(PsptD2r1/Pmm)/bias1, np.sqrt(PsptD2r2/Pmm)/bias2, np.sqrt(PsptD2r3/Pmm)/bias3, np.sqrt(PsptD2r4/Pmm)/bias4])
	B2 = np.array([bias1/bias1, bias2/bias2, bias3/bias3, bias4/bias4])
	B3 = np.array([biasF1/bias1, biasF2/bias2, biasF3/bias3, biasF4/bias4])
	B4 = np.array([Tb1/bias1, Tb2/bias2, Tb3/bias3, Tb4/bias4])
	b1 = np.mean(B1,axis=0)
	b1bis = np.mean(B1bis,axis=0)
	b2 = np.mean(B2,axis=0)
	b3 = np.mean(B3,axis=0)
	b4 = np.mean(B4,axis=0)
	
	E1 = np.array([errb1, errb2, errb3, errb4])
	e1 = np.mean(E1, axis =0)

	

	
	# p is number of free param
	# pf = 4
	#~ F1 = (biasF1-bias1bis)**2/errb1bis**2
	#~ F2 = (biasF2-bias2bis)**2/errb2bis**2
	#~ F3 = (biasF3-bias3bis)**2/errb3bis**2
	#~ F4 = (biasF4-bias4bis)**2/errb4bis**2
	#~ chi2F1 = np.sum(F1)/(len(kbis) - 5)
	#~ chi2F2 = np.sum(F2)/(len(kbis) - 5)
	#~ chi2F3 = np.sum(F3)/(len(kbis) - 5)
	#~ chi2F4 = np.sum(F4)/(len(kbis) - 5)
	#-------------------------------------------------
	# ppt = 3
	#~ PT1 = (np.sqrt(PsptD1r1/Pmmbis)- bias1bis)**2/errb1bis**2
	#~ PT2 = (np.sqrt(PsptD1r2/Pmmbis)- bias2bis)**2/errb2bis**2
	#~ PT3 = (np.sqrt(PsptD1r3/Pmmbis)- bias3bis)**2/errb3bis**2
	#~ PT4 = (np.sqrt(PsptD1r4/Pmmbis)- bias4bis)**2/errb4bis**2
	#~ chi2PT1 = np.sum(PT1)/(len(kbis) - 5)
	#~ chi2PT2 = np.sum(PT2)/(len(kbis) - 5)
	#~ chi2PT3 = np.sum(PT3)/(len(kbis) - 5)
	#~ chi2PT4 = np.sum(PT4)/(len(kbis) - 5)
	#-------------------------------------------------
	# pptbis = 4
	#~ PTbis1 = (np.sqrt(PsptD2r1/Pmmbis)- bias1bis)**2/errb1bis**2
	#~ PTbis2 = (np.sqrt(PsptD2r2/Pmmbis)- bias2bis)**2/errb2bis**2
	#~ PTbis3 = (np.sqrt(PsptD2r3/Pmmbis)- bias3bis)**2/errb3bis**2
	#~ PTbis4 = (np.sqrt(PsptD2r4/Pmmbis)- bias4bis)**2/errb4bis**2
	#~ chi2PTbis1 = np.sum(PTbis1)/(len(kbis) - 4)
	#~ chi2PTbis2 = np.sum(PTbis2)/(len(kbis) - 4)
	#~ chi2PTbis3 = np.sum(PTbis3)/(len(kbis) - 4)
	#~ chi2PTbis4 = np.sum(PTbis4)/(len(kbis) - 4)
	
	
	#~ print chi2F1, chi2F2, chi2F3, chi2F4
	#~ print chi2PTbis1, chi2PTbis2, chi2PTbis3, chi2PTbis4
	

	#~ cname = '/home/david/chi2_case_z='+str(z[j])+'.txt'
	#~ with open(cname, 'a+') as fid_file:

		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop,\
		#~ chi2F1, chi2F2, chi2F3, chi2F4, chi2PT1, chi2PT2, chi2PT3, chi2PT4, chi2PTbis1, chi2PTbis2, chi2PTbis3, chi2PTbis4))
	#~ print '\n'
			

	
	
	#~ ####################################################################
	#~ ### compute of the 4 mass bins
	#~ ####################################################################
	p1 = np.array([Pmono1/Pmono1, Pmono2/Pmono2, Pmono3/Pmono3, Pmono4/Pmono4])
	P1 = np.mean(p1, axis=0)
	p2 = np.array([kai1/Pmono1, kai2/Pmono2, kai3/Pmono3, kai4/Pmono4])
	P2 = np.mean(p2, axis=0)
	p3 = np.array([sco1/Pmono1, sco2/Pmono2, sco3/Pmono3, sco4/Pmono4])
	P3 = np.mean(p3, axis=0)
	p4 = np.array([tns1/Pmono1, tns2/Pmono2, tns3/Pmono3, tns4/Pmono4])
	P4 = np.mean(p4, axis=0)
	p6 = np.array([etns1/Pmono1, etns2/Pmono2, etns3/Pmono3, etns4/Pmono4])
	P6 = np.mean(p6, axis=0)
	
	
	
	
	


	########################################################################
	############## plot ####################################################
	########################################################################
	col = ['b','r','g','k']
	
	
	
		#--------- mean and std of bias and ps ratio ---------------------------
	if j == z[0]:
		fig2 = plt.figure()
	J = j + 1
	
	if len(z) == 1:
		ax2 = fig2.add_subplot(1, len(z), J)
	elif len(z) == 2:
		ax2 = fig2.add_subplot(1, 2, J)
	elif len(z) > 2:
		ax2 = fig2.add_subplot(2, 2, J)
	################################
	######### bias #################
	#~ B3, = ax2.plot(k,b3, color='C3', linewidth=2)
	#~ B2, = ax2.plot(k,b2, color='k',linewidth=2)
	#~ B1, = ax2.plot(k,b1, color='C0',linewidth=2)
	#~ B1bis, = ax2.plot(k,b1bis, color='C2',linewidth=2)
	###################################
	########### power spectrum ########
	P1, =ax2.plot(k,P1, color='k', label='z = '+str(z[j]))
	P2, =ax2.plot(k,P2, color='b')
	P3, =ax2.plot(k,P3, color='r')
	P4, =ax2.plot(k,P4, color='g')
	P6, =ax2.plot(k,P6, color='c')
	#~ plt.title('3D monopole power spectrum at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range)+', mu = '+str(mu) )
	#~ plt.title('3D halo power spectrum at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range))
	#~ plt.title('3D quadrupole power spectrum at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range) +', mu = '+str(mu))
	#~ plt.title('bias z = '+str(z[j])+', nu = '+str(nv) )
	######################################
	ax2.axhline(1, color='k', linestyle='--')
	ax2.axhline(1.01, color='k', linestyle=':')
	ax2.axhline(0.99, color='k', linestyle=':')
	ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, fontsize=9)
	#~ plt.figlegend( (P1,P2,P3), ('N-body','Power law fitted with free param','Scoccimaro'), \
	#~ plt.figlegend( (P1,P3,P5), ('N-body','Power law fitted with free param','Fast-PT'), \
	#~ plt.figlegend( (P1,P4,P6), ('N-body','Power law fitted with free param','Fast-PT'), \
	#~ plt.figlegend( (B1,B1bis,B2,B3), ('FAST-PT 2nd order','FAST-PT 3rd order','N-body','Power law '), \
	#~ plt.figlegend( (B4,B2,B3), ('bias fitted with Tinker','bias from simu','bias fitted with free parameters '), \
	#~ plt.figlegend( (B2, B4_2, B4_3, B4), ('bias from simu', 'bias fitted Tinker for bins $M_{1}, M_{2}$ ',\
	#~ 'bias fitted Tinker for bins $M_{1}, M_{2}, M_{3}$ ','bias fitted Tinker for all mass bins '), \
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(nv)+', case II ')
	plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	ax2.set_xscale('log')
	if j == 0 :
		ax2.tick_params(bottom='off', labelbottom='off')
		#~ ax2.set_xlabel('k')
		#~ ax2.set_ylabel(r'P(k) / $P_{ref}$')
		ax2.set_ylabel(r'b(k) / $b_{ref}$')
	if j == 1 :
		#~ ax2.set_xlabel('k')
		ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.tick_params(labelright=True, right= True,labelleft='off', left='off')
		#~ ax2.set_ylabel(r'P(k) / $P_{ref}$')
		ax2.set_ylabel(r'b(k) / $b_{ref}$')
		ax2.yaxis.set_label_position("right")
	if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		#~ ax2.set_ylabel(r'P(k) / $P_{ref}$')
		ax2.set_ylabel(r'b(k) / $b_{ref}$')
		ax2.set_xlabel('k')
	if j == 3 :
		ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		ax2.set_xlabel('k')
		#~ ax2.set_ylabel(r'P(k) / $P_{ref}$')
		ax2.set_ylabel(r'b(k) / $b_{ref}$')
		ax2.yaxis.set_label_position("right")
	ax2.set_xlim(8e-3,1)
	ax2.set_ylim(0.9,1.1)
	#~ #plt.ylim(0.7,1.3)
	if j == len(z) -1:
		plt.show()
	
	


		#~ #--------- plot the one loop contribution ------------------------------
	#~ plt.figure()
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.ylabel(r'P(k) ', size=10)
	#~ plt.ylabel(r'$2b_{1}*b_{3nl}*\sigma_{3}^{2}*P^{lin}$ ', size=10)
	#~ plt.xlabel(r'$k$', size=10)
	#~ plt.plot(k,PH1,label='N-body', color='k')
	#~ plt.plot(k,PsptD1r1, color='C1', label='2nd order expansion')
	#~ plt.plot(k,PsptD2r1,  color='C2',label=r'3rd order expansion with free $b_{3nl}$')
	#~ plt.plot(k,PsptD3r1, color='C3',label=r'3rd order expansion with fixed $b_{3nl}$')
	#~ plt.plot(k,Pmod_dd,label=r'$ \delta \delta $')
	#~ plt.plot(k,Pmod_dt,label=r'$ \delta \theta $')
	#~ plt.plot(k,Pmod_tt,label=r'$ \theta \theta $')
	#~ plt.plot(k,P_spt_dd[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#~ plt.plot(k,P_spt_dd[2], label='A' )
	#~ plt.plot(k,P_spt_dd[3], label=r'B' )
	#~ plt.plot(k,P_spt_dd[4], label=r'C' )
	#~ plt.plot(k,P_spt_dd[5], label=r'D' )
	#~ plt.plot(k,P_spt_dd[6], label=r'E' )
	#~ plt.plot(k,2.*b1bisa*b3nla*F, label=r'3rd order correction with free b3nl', color='C2', linestyle ='--' )
	#~ plt.plot(k,2.*b1tera*b3nlTa*F, label=r'3rd order correction with fixed b3nl', color='C3', linestyle='--' )
	#~ plt.plot(k,P_spt_dt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#~ plt.plot(k,P_spt_tt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#~ plt.title('z = '+str(z[j])+', $k_{max}$ = 0.1, mass range M1' )
	#~ plt.xlim(0.008,6)
	#~ plt.ylim(1e1,1e5)
	#~ plt.ylim(-150,0)
	#~ plt.legend(loc='lower left') 
	#~ plt.legend(loc='lower right') 
	#~ plt.show()
	
	#~ kill

	 


	#-------------plot the spectra -----------------------------------------
	#~ plt.figure()
	########### redshift space ###################
	#~ plt.plot(k,Phh, label='Paco', color='y')
	#~ plt.plot(k,Pmono_m1, label='Paco', color='r')
	#~ plt.plot(k,Ptns, label='TNS', color='y')
	#~ plt.plot(k,Pnlk, label='non linear kaiser', color='b')
	#~ plt.plot(k,Pcamb * mono, label='kaiser using $ P_{lin}$',linestyle ='--', color='k')
	#~ plt.plot(k,Pcamb * quadru, label='kaiser using $ P_{lin}$',linestyle ='--', color='k')
	#~ plt.plot(k,Pmod_dd*mono, label='kaiser using $P_{\delta \delta}$',linestyle ='-.', color='g')
	#~ plt.plot(k,Pmod_dd*quadru, label='kaiser using $P_{\delta \delta}$',linestyle ='-.', color='g')
	#~ plt.title('3D monopole cdm power spectrum without FoG at z = '+str(z[j])+', nu = '+str(nv))
	#~ plt.title('3D quadrupole cdm power spectrum without FoG at z = '+str(z[j])+', nu = '+str(nv))
	################ real space###################
	#~ for i in xrange(0, len(mass_range)):
		#~ plt.plot(k,Pmono_bis[:,i], label=r'simu ' + str(mass_range[i]), color=col[i])
		#~ plt.plot(k,Pmod_dd*monobias[:,i]*bias_bis[:,i]**2, label=r'simu ' + str(mass_range[i]), color=col[i], linestyle=':')
		#~ plt.plot(k,Pmod_dd*monobias1[:,i]*b1fit_bis[:,i]**2, label=r'simu ' + str(mass_range[i]), color=col[i], linestyle='--')
		#~ plt.plot(k, Pspt[:,i], label=r' simu / FAST-Pt bias' + str(mass_range[i]), color=col[i], linestyle=':')
	#~ plt.plot(k,Pmm, label='Paco', color='gold')
	#~ plt.plot(k,Pcamb, label=r'Pcamb',linestyle ='--', color='b')
	#~ plt.plot(k,Pnu, label='Paco', color='r')
	#~ plt.plot(k,Pmod_dd * bias**2, label=r'$ \delta \delta $ + bias',linestyle ='--', color='g')
	#~ plt.plot(k,Pmod_dd * Tbias**2, label=r'$ \delta \delta $ + tinker bias',linestyle ='-.', color='g')
	#~ plt.plot(k,Pmod_dd * Sbias**2, label=r'$ \delta \delta + smt bias$',linestyle =':', color='g')
	#~ plt.plot(k,Pcamb * bias**2, label=r'Pcamb + bias',linestyle ='--', color='b')
	#~ plt.plot(k,Pcamb * Tbias**2, label=r'Pcamb + tinker smt bias',linestyle ='-.', color='b')
	#~ plt.plot(k,Pcamb * Sbias**2, label=r'Pcamb + bias',linestyle =':', color='b')
	#~ plt.title('3D power spectrum at z = '+str(z[j])+', nu = '+str(nv))
	#~ plt.title('3D halo power spectrum at z = '+str([j])+', nu = '+str(nv))
	##############################################
	#~ plt.legend(loc='upper right')
	#~ plt.xscale('log')
	#~ plt.xlabel('k')
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.xlim(8e-3,1)
	#~ plt.ylim(1e1,1e5)
	#~ plt.yscale('log')
	#~ plt.ylabel('P(k)')
	#~ plt.show()  
	#~ kill

	#-------- test the accuracy of velocity divergence spectra -------------
	#~ plt.figure()
	#~ plt.suptitle('z = '+str(z[j])+' ,expansion at 11th order, class h = 0.7, omega_b =0.05, omega_cdm = 0.25')
	#~ ax1=plt.subplot(311)
	#~ ax1.plot(k,Pmod_dd/P,label=r'$ \delta \delta FAST PT $', color='r')
	#~ ax1.plot(ksdd,psdd, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ ax2=plt.subplot(312)
	#~ ax2.plot(k,Pmod_dt/P,label=r'$ \delta \theta FAST PT $',color='r')
	#~ ax2.plot(ksdt,psdt, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ ax3=plt.subplot(313)
	#~ ax3.plot(k,Pmod_tt/P,label=r'$ \theta \theta FAST PT$', color='r')
	#~ ax3.plot(kstt,pstt, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.show()


	#~ #---------- plot the bias ratio ----------------------------------------
	#~ if j == z[0]:
		#~ fig1 = plt.figure()
	#~ J = j + 1
	
	#~ if len(z) == 1:
		#~ ax = fig1.add_subplot(1, len(z), J)
	#~ elif len(z) == 2:
		#~ ax = fig1.add_subplot(1, 2, J)
	#~ elif len(z) > 2:
		#~ ax = fig1.add_subplot(2, 2, J)
	#~ ax.plot(k,bias_bis[:,0], label='z = '+str(z[j]), color='b', linestyle='--')
	#~ plt.plot(k,biasF[:,0], color='b', linestyle=':')
	#~ plt.plot(k,np.sqrt(PsptD1r[:,0]/Pmm), color='b')
	#~ plt.fill_between(k,bias_bis[:,0]-errb[:,0],bias_bis[:,0]+errb[:,0], color=col[i], alpha=0.4)
	#~ plt.fill_between(k,bias_bis[:,0]-errb2[:,0],bias_bis[:,0]+errb2[:,0], color=col[i], alpha=0.4)
	#~ plt.plot(k,Tinker_bis[:,0], color='b',linestyle='--')
	#~ plt.plot(k,SMT_bis[:,0], label=r' SMT formula ' , color='b')
	#~ plt.plot(k,Crocce_bis[:,0], color='b', linestyle=':')
	#~ plt.plot(k,Watson_bis[:,0], label=r' Watson MF ', color='b', linestyle='-.')
	#~ ax.plot(k,p1temp[:,0], color=col[0], label='z = '+str(z[j]), linestyle='-')
	#~ ax.plot(k,p2temp_simu[:,0], color=col[0], linestyle=':')
	#~ ax.plot(k,p3temp_simu[:,0], color=col[0], linestyle='--')
	#~ ax.plot(k,p4temp[:,0], color=col[0], linestyle='--')
	#~ for i in xrange(1, len(mass_range)):
		#~ ax.plot(k,b1fit_bis[:,i], color=col[i])
		#~ ax.plot(k,Pmod_dd * monobias[:,i]* biasF[:,i]**2, color=col[i], linestyle='--')
		#~ B, =ax.plot(k,p1temp[:,i], color=col[i], linestyle='-')
		#~ K, =ax.plot(k,p2temp_simu[:,i], color=col[i], linestyle=':')
		#~ SC, =ax.plot(k,p3temp_simu[:,i], color=col[i], linestyle='--')
		#~ TNS, =ax.plot(k,p4temp[:,i], color=col[i], linestyle='--')
		#~ plt.plot(k,Phh_bis[:,i]/(Pmm * biasT[:,i]**2), label=r' fit simu  Tinker ' + str(mass_range[i]), color=col[i], linestyle=':')
		#~ plt.plot(k,Phh_bis[:,i]/(Pmm * biasF[:,i]**2), label=r' fit simu Free ' + str(mass_range[i]), color=col[i], linestyle='-')
		#~ plt.plot(k,Phh_bis[:,i]/(Pmm * biasC[:,i]**2), label=r' fit simu Crocce ' + str(mass_range[i]), color=col[i], linestyle='-')
		#~ plt.plot(k,Phh_bis[:,i]/(Pmod_dd * biasS[:,i]**2), label=r' fir simu SMT ' + str(mass_range[i]), color=col[i], linestyle='--')
		#~ B, = ax.plot(k,bias_bis[:,i],  color=col[i],linestyle='--')
		#~ ax.fill_between(k,bias_bis[:,i]-errb[:,i],bias_bis[:,i]+errb[:,i], color=col[i], alpha=0.4)
		#~ ax.fill_between(k,bias_bis[:,i]-errb2[:,i],bias_bis[:,i]+errb2[:,i], color=col[i], alpha=0.4)
		#~ T, = ax.plot(k,Tinker_bis[:,i], color=col[i],linestyle='--')
		#~ plt.plot(k,SMT_bis[:,i],  color=col[i])
		#~ C, = ax.plot(k,Crocce_bis[:,i],  color=col[i], linestyle=':')
		#~ plt.plot(k,Watson_bis[:,i],  color=col[i], linestyle='-.')
		#~ plt.plot(k,SMT_bis[:,i], label=r' SMT MF ' + str(mass_range[i]), color=col[i], linestyle='-')
		#~ F, = ax.plot(k,biasF[:,i], color=col[i], linestyle=':')
		#~ plt.plot(k,biasF[:,i], label=r'bias fit with free param ', color='r', linestyle='--')
		#~ PT, = ax.plot(k,np.sqrt(PsptD1r[:,i]/Pmm), color=col[i])
		#~ plt.plot(k,biasT[:,i], label=r'bias fit with Tinker', color='b', linestyle='--')
		#~ plt.plot(k,biasS[:,i]/bias_bis[:,i], label=r'bias fit with SMT ', color='g', linestyle='--')
	#plt.title('bias z = '+str(z[j])+', nu = '+str(nv) )
	#plt.title('influence of bias at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range) )
	#~ ax.legend(loc = 'upper right', handlelength=0, handletextpad=0, fancybox=True)
	#~ ax.legend(loc = 'upper left', handlelength=0, handletextpad=0, fancybox=True)
	#~ plt.figlegend( (B,K,SC), ('N-body','Linear Kaiser','Scoccimarro '), loc = 'upper center',\
	#~ plt.figlegend( (B,TNS), ('N-body', 'TNS'), loc = 'upper center',\
	#~ ncol=5, labelspacing=0.,title =r'M$\nu$ = '+str(nv)  )
	#~ plt.figlegend( (B,T,C), ('bias from N-body','Tinker effective bias ', 'Crocce effective bias '), loc = 'upper center',\
	#~ ncol=5, labelspacing=0.,title =r'M$\nu$ = '+str(nv)  )
	#~ plt.figlegend( (B,F,PT), ('N-body','Power law fitted with free param ', 'FAST '), loc = 'upper center',\
	#~ ncol=5, labelspacing=0.,title =r'M$\nu$ = '+str(nv) )
	#~ plt.figlegend( (B,C), ('bias from N-body','Crocce effective bias '), loc = 'upper center', ncol=5, labelspacing=0. )
	#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#plt.tight_layout()
	#~ ax.set_xscale('log')
	#~ ax.set_yscale('log')
	#~ if j == 0 :
		#~ ax.tick_params(bottom='off', labelbottom='off')
		#ax.set_xlabel('k')
		#~ ax.set_ylabel('b(k)')
	#~ if j == 1 :
		#ax.set_xlabel('k')
		#~ ax.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax.tick_params(labelright=True, right= True,labelleft='off', left='off')
		#~ ax.set_ylabel('b(k)')
		#~ ax.yaxis.set_label_position("right")
	#~ if j == 2 :
		#~ ax.tick_params(labelleft=True)
		#~ ax.set_ylabel('b(k)')
		#~ ax.set_xlabel('k')
	#~ if j == 3 :
		#~ ax.tick_params(labelright=True, right= True, labelleft='off', left='off')
		#~ ax.set_xlabel('k')
		#~ ax.set_ylabel('b(k)')
		#~ ax.yaxis.set_label_position("right")
	#~ ax.set_xlim(8e-3,0.9)
	#plt.ylim(0,10)
	### real space ###
	#~ if j == 0:
		#~ ax.set_ylim(0.5,1.5)
	#~ elif j == 1:
		#~ ax.set_ylim(0.5,2)
	#~ elif j == 2:
		#~ ax.set_ylim(0.5,3)
	#~ elif j == 3:
		#~ ax.set_ylim(1.,7)
	### redshift space ###
	#~ if j == 0:
		#~ ax.set_ylim(1e3,1e5)
	#~ elif j == 1:
		#~ ax.set_ylim(1e3,1e5)
	#~ elif j == 2:
		#~ ax.set_ylim(1e3,1e5)
	#~ elif j == 3:
		#~ ax.set_ylim(1e3,1e5)
	#~ #plt.suptitle('bias z = '+str(z[j])+', nu = '+str(nv) )
	#~ if j == len(z) -1:
		#~ plt.show()


	#-------- plot velocity dispersion -------------------------------------
	#~ red2 = [0.0,0.5,1.0,2.0]
	#~ limM = [5e11,1e12,3e12,1e13]
	#~ znumber = len(red2)
	#~ from astropy.cosmology import Planck13 as cosmo
	#~ from hmf import MassFunction
	#~ hmf1 = MassFunction(Mmin = np.log10(limM[0]),Mmax = np.log10(limM[1]), lnk_min = -2.22, lnk_max = -0.30)
	#~ hmf2 = MassFunction(Mmin = np.log10(limM[1]),Mmax = np.log10(limM[2]), lnk_min = -2.22, lnk_max = -0.30)
	#~ hmf3 = MassFunction(Mmin = np.log10(limM[2]),Mmax = np.log10(limM[3]), lnk_min = -2.22, lnk_max = -0.30)
	#~ hmf4 = MassFunction(Mmin = np.log10(limM[3]), lnk_min = -2.22, lnk_max = -0.30)
	#~ Vs = 1000**3 # for a periodic box of 1000 h-1Mpc
	#~ h_z = np.zeros(znumber, 'float64')
	#~ v_disp = np.zeros(znumber, 'float64')
	#~ G = 4.302e-9 # in Mpc Msol-1 (km/s)**2


	#~ plt.figure()

	#~ for j in xrange(0,4):
		#~ z2 = red2[j]
		#~ for i in xrange(0, len(mass_range)):
			#~ e = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh_bias_'+str(mass_range[i])+'_z='+str(z2)+'.txt', skiprows=1)
			#~ vel_disp2 = e[:,17]
			#~ plt.scatter(z2,vel_disp2[0], label=r' v_disp from simu ', color=col[i])
		#~ hmf1.update(z=z2)
		#~ hmf2.update(z=z2)
		#~ hmf3.update(z=z2)
		#~ hmf4.update(z=z2)
		#~ h_z = cosmo.H(z2).value / 100
		#~ Omeg_m_z = Omega_c * (1 + z2)**3 / (Omega_c * (1 + z2)**3 + Omega_l) # with Omega_k = 0
		#~ mass_func = hmf1.dndm
		#~ mass = hmf1.m
		#~ n = mass * mass_func* Vs
		#~ b = np.average(mass, weights=n)
		#~ r_vir = (0.784 *(b/ 1e8)**(1/3.) * (Omega_c / Omeg_m_z * 200 / 18 / math.pi**2)**(-1/3.)* ((1+z2)/10)**(-1))/1000 #in mpc
		#~ M_200 = 100 * r_vir**3 * (h_z*100)**2 / G	#/ (1+redshift[i])**3
		#~ v_disp1 = 10**(math.log10(1082.9) + (0.3361)* math.log10(h_z* M_200 / 1e15))
		#~ plt.scatter(z2,v_disp1*5, label=r' v_disp from theory', color='y')
			
		
	#~ plt.title('velocity dispersion simu vs theory')
	#~ plt.legend(loc='upper right')
	#~ plt.xlabel('z')
	#~ plt.ylabel('velocity dispersion')
	#~ plt.xlim(-0.05,2.05)
	#~ plt.show()


	#--------- plot monopole and quadrupole ratio --------------------------

	#~ plt.figure()
	######### real space #################
	#~ plt.plot(k,Pnu/(Pmod_dd), label=r'$ Pk_{simu }/ P_{\delta \delta} $', color='g')
	#~ plt.plot(k,Pnu/Pcamb, label='$ Pk_{simu }/ P_{lin}$', color='r')
	#~ for i in xrange(0, len(mass_range)):
		#~ plt.plot(k,Phh_bis[:,i]/(Pmod_dd* bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{kaiser}$ using $P_{\delta \delta} $', linestyle=':', color=col[i])
		#~ plt.plot(k,Phh_bis[:,i]/(Pnlk[:,i]*bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{nlkaiser} $', linestyle='--', color=col[i])
		#~ plt.plot(k,Phh_bis[:,i]/(Pnlk_vel[:,i]*bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{nlkaiser} $', linestyle='--', color=col[i])
		#~ plt.plot(k,Phh_bis[:,i]/(Ptns[:,i] *bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{TNS} $ '+str(mass_range[i]) , color=col[i])

	#~ plt.title('3D power spectrum at z = '+str(z[j])+', nu = '+str(nv) )
	#~ plt.title('3D halo power spectrum at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range) )
	######### redshift space #############
	#~ plt.plot(k,Pmono/(Pmod_dd * mono), label=r'$ Pk_{simu }/ P_{kaiser}$ using $P_{\delta \delta} $', color='g')
	#~ plt.plot(k,Pmono/(Pnlk_m), label=r'$ Pk_{simu }/ P_{NLkaiser} $', color='r')
	#~ plt.plot(k,Pmono/(Ptns_m), label=r'$ Pk_{simu }/ P_{TNS} $', color='b')
	#~ plt.plot(k,Pquadru/(Pmod_dd * quadru), label=r'$ Pk_{simu }/ P_{kaiser}$ using $P_{\delta \delta} $', color='g')
	#~ plt.plot(k,Pquadru/(Pnlk_q), label=r'$ Pk_{simu }/ P_{NLkaiser} $', color='r')
	#~ plt.plot(k,Pquadru/(Ptns_q), label=r'$ Pk_{simu }/ P_{TNS} $', color='b')

	#~ for i in xrange(0, len(mass_range)):
		#~ plt.plot(k,Pquadru_bis[:,i]/(Pmod_dd * quadrubias[:,i]* bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{kaiser}$ using $P_{\delta \delta} $', linestyle=':', color=col[i])
		#~ plt.plot(k,Pquadru_bis[:,i]/(Pnlk_q[:,i]*bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{nlkaiser} $', linestyle='--', color=col[i])
		#~ plt.plot(k,Pquadru_bis[:,i]/(Ptns_q[:,i] *bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{TNS} $', color=col[i])
	#~ for i in xrange(0, len(mass_range)):
		#~ plt.plot(k,Pmono_bis[:,i]/(Pmod_dd * monobias[:,i]* bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{kaiser}$ using $P_{\delta \delta} $', linestyle=':', color=col[i])
		#~ plt.plot(k,Pmono_bis[:,i]/(Pnlk_m[:,i]*bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{nlkaiser} $', linestyle='--', color=col[i])
		#~ plt.plot(k,Pmono_bis[:,i]/(Ptns_m[:,i] *bias_bis[:,i]**2), label=r'$ Pk_{simu }/ P_{TNS} $ '+str(mass_range[i]) , color=col[i])
	#~ plt.title('3D monopole power spectrum at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range)+', mu = '+str(mu) )
	#~ plt.title('3D quadrupole power spectrum at z = '+str(z[j])+', nu = '+str(nv)+', mass range '+str(mass_range) +', mu = '+str(mu))
	#~ ######################################
	#~ plt.axhline(1, color='k', linestyle='--')
	#~ plt.axhline(1.01, color='k', linestyle=':')
	#~ plt.axhline(0.99, color='k', linestyle=':')
	#~ plt.legend(loc='upper right')
	#~ plt.xscale('log')
	#~ plt.xlabel('k')
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.xlim(8e-3,0.5)
	#~ plt.ylim(0.85,1.15)
	#~ plt.ylabel('P(k) ratio')
	#~ plt.show()
	 
end = time()
print 'total time is '+str((end - start))	 

