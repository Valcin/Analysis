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
sys.path.append('/home/david/codes/FAST-PT')
import myFASTPT as FPT
import scipy.interpolate as sp
import pyximport
pyximport.install()
import redshift_space_library as RSL
from readfof import FoF_catalog
import MAS_library as MASL
import Pk_library as PKL
import mass_function_library as MFL
import bias_library as BL
import tempfile
import expected_CF
import exp2
from time import time
from loop_pt import pt_terms
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma
from fit_emcee import coeffit_pl,coeffit_pl2,coeffit_exp1, coeffit_exp2, coeffit_exp3,coeffit_Kaiser, coeffit_Scocci, coeffit_TNS, coeffit_eTNS



def ld_data(mv, z, j):
	
	###########################
	Nnu       = 0  #number of massive neutrinos
	Neff      = 3.046

	# cosmological parameters
	h       = 0.6711
	Omega_c = 0.2685 - mv/(93.14*h**2)
	Omega_b = 0.049
	Omega_l = 0.6825
	Omega_k = 0.0
	Omega_m = Omega_c + Omega_b
	tau     = None
############################################################################################################################
############# 	0.0 eV Masseless neutrino ##################################################################################


	if mv == 0.0:

		#------------------------------------------------
		#-------- data from scoccimaro 2004 -------------
		#------------------------------------------------
		scoccidd = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class_scocci/delta.txt')
		psdd = scoccidd[:,0]
		ksdd = scoccidd[:,1]
		scoccidt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class_scocci/deltheta.txt')
		psdt = scoccidt[:,0]
		ksdt = scoccidt[:,1]
		scoccitt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class_scocci/theta.txt')
		pstt = scoccitt[:,0]
		kstt = scoccitt[:,1]

		classS = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class_scocci/test_pk.dat')
		ks = classS[:,0]
		pks = classS[:,1]
		
		ksbis = np.logspace(np.log10(np.min(ks)),np.log10(np.max(ks)), 100)
		pksbis = np.interp(ksbis, ks,pks)
		Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(ksbis, pksbis)
		
		fig = plt.figure()
		ax1 = fig.add_subplot(3, 1, 1)
		plt.gca().axes.get_xaxis().set_visible(False)
		B1, =ax1.plot(ksdd, psdd, c='b')
		B2, =ax1.plot(ksbis, Pmod_dd/pksbis, c='r')
		ax1.axhline(1, linestyle='--', c='k')
		ax1.set_xscale('log')
		ax1.set_xlim(2e-2, 0.2)
		ax1.set_ylim(0.5, 1.5)
		ax1.set_ylabel(r'$P_{\delta\delta}$ / $P_{lin}$', fontsize=16)
		#------------------
		ax2 = fig.add_subplot(3, 1, 2)
		plt.gca().axes.get_xaxis().set_visible(False)
		ax2.plot(ksdt, psdt, c='b')
		ax2.plot(ksbis, Pmod_dt/pksbis, c='r')
		ax2.axhline(1, linestyle='--', c='k')
		ax2.set_xscale('log')
		ax2.set_xlim(2e-2, 0.2)
		ax2.set_ylim(0.5, 1.5)
		ax2.set_ylabel(r'$P_{\delta\theta}$ / $P_{lin}$', fontsize=16)
		#-------------------
		ax3 = fig.add_subplot(3, 1, 3)
		ax3.plot(kstt, pstt, c='b')
		ax3.plot(ksbis, Pmod_tt/pksbis, c='r')
		ax3.axhline(1, linestyle='--', c='k')
		ax3.set_xscale('log')
		ax3.set_xlim(2e-2, 0.2)
		ax3.set_ylim(0.5, 1.5)
		ax3.set_xlabel('k [h/Mpc]', fontsize=14)
		ax3.set_ylabel(r'$P_{\theta\theta}$ / $P_{lin}$', fontsize=16)
		#-------------------
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.figlegend( (B1,B2), ('Scoccimarro','FAST-PT implementation'), \
		#~ ######################################
		loc = 'upper center', ncol=3, labelspacing=0., title = ' z = 0', fontsize=14)
		
		plt.show()
		kill
		
		#~ #-------------------------------------------------
		#~ #---------------- Camb ---------------------------
		#~ #-------------------------------------------------
		camb = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb = camb[:,0]
		Pcamb = camb[:,1]
		#~ Plin = Pcamb
		#~ klin = kcamb
		
	
		#~ #-------------------------------------------------
		#~ #---------------- Class ---------------------------
		#~ #-------------------------------------------------
		Class = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class/test_z'+str(j+1)+'_pk.dat')
		kclass = Class[:,0]
		Pclass = Class[:,1]
		Class_trans = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/class/test_z'+str(j+1)+'_tk.dat')
		Tb = Class_trans[:,2]
		Tcdm = Class_trans[:,3]
		Tm = Class_trans[:,5]
		Tcb = (Omega_c * Tcdm + Omega_b * Tb)/(Omega_c + Omega_b)
		

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

		#~ ### compute where shot noise is 80% of the ps
		
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
		
		#~ with open('test.txt', 'w') as fid_file:
			#~ for index_k in xrange(len(k)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % ( PH1[index_k],\
				#~ PH2[index_k],PH3[index_k],PH4[index_k],\
				#~ Pshot1, Pshot2, Pshot3, Pshot4, kk1, kk2, kk3, kk4 ))
		#~ fid_file.close()

		
		#~ plt.figure()
		#~ M1, =plt.plot(k, PH1, label='halo Power spectrum')
		#~ M2, =plt.plot(k, PH2)
		#~ M3, =plt.plot(k, PH3)
		#~ M4, =plt.plot(k, PH4, c='k')
		#~ for i in range(10):
			#~ plt.scatter(k, Phh4[:,i])
		#~ plt.axhline( Pshot1, color='C0', linestyle='--', label='shot noise')
		#~ plt.axhline( Pshot2, color='C1', linestyle='--')
		#~ plt.axhline( Pshot3, color='C2', linestyle='--')
		#~ plt.axhline( Pshot4, color='k', linestyle='--', label='shot noise')
		#~ plt.fill_between(k,PH1-errPhh1, PH1+errPhh1, alpha=0.6)
		#~ plt.fill_between(k,PH2-errPhh2, PH2+errPhh2, alpha=0.6)
		#~ plt.fill_between(k,PH3-errPhh3, PH3+errPhh3, alpha=0.6)
		#~ plt.fill_between(k,PH4-errPhh4, PH4+errPhh4, alpha=0.6)
		#~ plt.axvline( kk1, color='C0', linestyle=':', label='shot noise = 80% of P(k)')
		#~ plt.axvline( kk2, color='C1', linestyle=':')
		#~ plt.axvline( kk3, color='C2', linestyle=':')
		#~ plt.axvline( kk4, color='k', linestyle=':', label='shot noise = 80% of P(k)')
		#~ plt.legend(loc = 'upper right', title='z = '+str(z[j]), fancybox=True)
		#~ plt.legend(loc = 'lower left', title='z = '+str(z[j]), fancybox=True, fontsize = 14)
		#~ plt.figlegend( (M1,M2,M3,M4), ('mass range M1','mass range M2','mass range M3','mass range M4'), \
		#~ loc = 'upper center', ncol=2, labelspacing=0. , title ='Mv = 0.0eV ', fontsize = 14)
		#~ plt.xlabel('k [h/Mpc]', fontsize=14)
		#~ plt.ylabel('P(k)', fontsize=14)
		#~ plt.xscale('log')
		#~ plt.yscale('log')
		#~ plt.xlim(8e-3,2)
		#~ plt.ylim(1e4,2e5)
		#~ plt.show()
		
		#~ kill
		
		#~ #-------------------------------------------------------------------
		#~ #----remove shot noise, compute bias and bias variance -------------
		#~ #-------------------------------------------------------------------
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
			

		#~ ### do the mean over quantitites ###
		
		Pmm = np.mean(Pmat[:,0:11], axis=1)
		PH1 = np.mean(Phh1[:,0:11], axis=1)
		PH2 = np.mean(Phh2[:,0:11], axis=1)
		PH3 = np.mean(Phh3[:,0:11], axis=1)
		PH4 = np.mean(Phh4[:,0:11], axis=1)
		
		bias1 = np.mean(bhh1[:,0:11], axis=1)
		bias2 = np.mean(bhh2[:,0:11], axis=1)
		bias3 = np.mean(bhh3[:,0:11], axis=1)
		bias4 = np.mean(bhh4[:,0:11], axis=1)
		
		#~ hsize = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hsize_z='+str(z[j])+'.txt')
		#~ hsize_m1 = hsize[:,0]
		#~ hsize_m2 = hsize[:,1]
		#~ hsize_m3 = hsize[:,2]
		#~ hsize_m4 = hsize[:,3]
		#~ mostmass = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/mostmass_z='+str(z[j])+'.txt')
		#~ mostmass_m1 = mostmass[:,0]
		#~ mostmass_m2 = mostmass[:,1]
		#~ mostmass_m3 = mostmass[:,2]
		#~ mostmass_m4 = mostmass[:,3]
		
		
		#~ sn1 = Phh1[:,0:11]/Pshot1[0:11]
		#~ sn2 = Phh2[:,0:11]/Pshot2[0:11]
		#~ sn3 = Phh3[:,0:11]/Pshot3[0:11]
		#~ sn4 = Phh4[:,0:11]/Pshot4[0:11]
		
		#~ Sn1 = np.sum(sn1,axis=1)
		#~ wei1 = sn1/Sn1[:,None]
		
	
		

		errb1 = np.std(bhh1[:,0:11], axis=1)
		errb2 = np.std(bhh2[:,0:11], axis=1)
		errb3 = np.std(bhh3[:,0:11], axis=1)
		errb4 = np.std(bhh4[:,0:11], axis=1)
		
		
		errPhh1 = np.std(Phh1[:,0:11], axis=1)
		errPhh2 = np.std(Phh2[:,0:11], axis=1)
		errPhh3 = np.std(Phh3[:,0:11], axis=1)
		errPhh4 = np.std(Phh4[:,0:11], axis=1)
		
		noise1 = np.std(Pshot1[0:11])
		noise2 = np.std(Pshot2[0:11])
		noise3 = np.std(Pshot3[0:11])
		noise4 = np.std(Pshot4[0:11])
		
		### smooth the curve
		from scipy.signal import savgol_filter
		bias1s = savgol_filter(bias1, 51, 3) 
		bias2s = savgol_filter(bias2, 51, 3) 
		bias3s = savgol_filter(bias3, 51, 3) 
		bias4s = savgol_filter(bias4, 51, 3) 
		
		
		
		#~ plt.figure()
		#~ M1, =plt.plot(k, bias1, label=r'$b_{cc}$',c='r')
		#~ M1, =plt.plot(k, bias1bis,c='k')
		#~ M2, =plt.plot(k, bias2)
		#~ M2, =plt.plot(k, bias2bis)
		#~ M3, =plt.plot(k, bias3)
		#~ M3, =plt.plot(k, bias3bis)
		#~ M4, =plt.plot(k, bias4,c='C0', label='No smoothing')
		#~ plt.fill_between(k,bias1-errb1, bias1+errb1, alpha=0.6)
		#~ plt.fill_between(k,bias2-errb2, bias2+errb2, alpha=0.6)
		#~ plt.fill_between(k,bias3-errb3, bias3+errb3, alpha=0.6)
		#~ plt.fill_between(k,bias4-errb4, bias4+errb4, alpha=0.6)
		#~ plt.errorbar(k, bias4, yerr=errb4,fmt='o', capsize=2, )
		#~ plt.axvline( kk1, color='C0', linestyle=':', label='shot noise = 80% of P(k)')
		#~ plt.axvline( kk2, color='C1', linestyle=':')
		#~ plt.axvline( kk3, color='C2', linestyle=':')
		#~ plt.axvline( kk4, color='C3', linestyle=':')
		#~ plt.legend(loc = 'upper right', title='z = '+str(z[j]), fancybox=True)
		#~ plt.legend(loc = 'lower left', title='z = '+str(z[j]), fancybox=True, fontsize = 14)
		#~ plt.figlegend( (M1,M2,M3,M4), ('mass range M1','mass range M2','mass range M3','mass range M4'), \
		#~ loc = 'upper center', ncol=2, labelspacing=0. , fontsize = 14)
		#~ plt.ylim(0.0,bias4[0]*1.4)
		#~ #---------------------------------------------------------------
		#~ M1, =plt.plot(k, bias1s, label='4th order smoothing',linestyle='--', color='C0')
		#~ M1, =plt.plot(k, bias2s,linestyle='--', color='C1')
		#~ M1, =plt.plot(k, bias3s,linestyle='--', color='C2')
		#~ M1, =plt.plot(k, bias4s,linestyle='--', color='C3', label='smoothing 3rd order')
		#~ plt.plot(k, PH1, c='r')
		#~ plt.plot(k, PH1bis, c='r',linestyle='--')
		#~ plt.plot(k, PH2, c='b')
		#~ plt.plot(k, PH2bis,c='b',linestyle='--')
		#~ plt.plot(k, PH3, c='g')
		#~ plt.plot(k, PH3bis, c='g',linestyle='--')
		#~ plt.plot(k, PH4, c='c')
		#~ plt.plot(k, Pmm, color='k')
		
		#~ for i in range(10):
			#~ print np.min(hsize_m3[i])
			#~ plt.scatter(k, Phh3[:,i], c = np.full((len(k)),Phh3[:,i]/Pshot3[i]),\
			#~ norm=matplotlib.colors.Normalize(np.min(Phh3[:,i]/Pshot3[i]),np.max(Phh3[:,i]/Pshot3[i])))
			#~ plt.scatter(k, Phh3[:,i], c = np.full((len(k)),hsize_m3[i]),\
			#~ norm=matplotlib.colors.Normalize(np.min(hsize_m3),np.max(hsize_m3)))
			#~ plt.scatter(k, Phh3[:,i], c = np.full((len(k)),Pshot3[i]),\
			#~ norm=matplotlib.colors.Normalize(np.min(Pshot3),np.max(Pshot3)))
		
		#~ cbar= plt.colorbar()
		#~ plt.ylim(0.7,0.9)
		#~ plt.ylim(0.75,0.95)
		#~ plt.ylim(0.9,1.1)
		#~ plt.ylim(1.2,1.4)
		#~ plt.title('Mass bin M4')
		#~ plt.legend(loc = 'upper right', title='z = '+str(z[j]), fancybox=True, fontsize=14)
		#~ plt.ylim(0,0.05)
		#------------------
		#~ plt.xlabel('k [h/Mpc]', fontsize=14)
		#~ plt.ylabel('b(k)', fontsize=16)
		#~ plt.xscale('log')
		#~ plt.xlim(8e-3,0.2)
		#~ plt.show()
		
		#~ kill
		

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
		errPred = np.std(Pmat_r[:,0:11], axis=1)
		
		


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


		#~ ### do the mean and std over quantitites ###
		
		Pmono1 = np.mean(Pmono1temp[:,0:11], axis=1)
		Pmono2 = np.mean(Pmono2temp[:,0:11], axis=1)
		Pmono3 = np.mean(Pmono3temp[:,0:11], axis=1)
		Pmono4 = np.mean(Pmono4temp[:,0:11], axis=1)
				
		errPr1 = np.std(Pmono1temp[:,0:11], axis=1)
		errPr2 = np.std(Pmono2temp[:,0:11], axis=1)
		errPr3 = np.std(Pmono3temp[:,0:11], axis=1)
		errPr4 = np.std(Pmono4temp[:,0:11], axis=1)

		return kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
		bias1s, bias2s, bias3s, bias4s, errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3,\
		errPr4, kclass, Tm, Tcb, noise1, noise2, noise3, noise4
		#~ return kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1s, bias2s, bias3s, bias4s, \
		#~ errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4
		
#####################################################################################################################################
#####################################################################################################################################
######### 0.15 eV Massive neutrino 
#####################################################################################################################################	
#####################################################################################################################################
	if mv == 0.15:
		#-------------------------------------------------
		#---------------- Class ---------------------------
		#-------------------------------------------------
		Class = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/class/test_z'+str(j+1)+'_pk.dat')
		kclass = Class[:,0]
		Pclass = Class[:,1]
		Class_trans = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/class/test_z'+str(j+1)+'_tk.dat')

		ktrans = Class_trans[:,0]
		Tb = Class_trans[:,2]
		Tcdm = Class_trans[:,3]
		Tm = Class_trans[:,8]
		Tcb = (Omega_c * Tcdm + Omega_b * Tb)/(Omega_c + Omega_b)

		#-----------------------------------------------------------------------
		#-------- get the transfer function and Pcc ----------------------------
		#-----------------------------------------------------------------------
		Pcc = Pclass * (Tcb/Tm)**2
		Plin = Pcc
		klin = kclass
		with open('/home/david/codes/Paco/data2/0.15eV/Pcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			for index_k in xrange(len(klin)):
				fid_file.write('%.8g %.8g\n' % ( klin[index_k], Plin[index_k]))
		fid_file.close()
		
		#-------------------------------------------------
		#---------------- Camb ---------------------------
		#-------------------------------------------------
		camb = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb = camb[:,0]
		Pcamb = camb[:,1]
		Plin = Pcamb
		klin = kcamb

	#~ #-----------------------------------------------------------------------
		#~ #---------------- matter neutrino Real space ---------------------------
		#~ #-----------------------------------------------------------------------
		d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/NCV1/analysis/Pk_c_z='+str(z[j])+'.txt')
		e = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/NCV2/analysis/Pk_c_z='+str(z[j])+'.txt')
		k1 = d[:,0]
		p1 = d[:,1]
		k2 = e[:,0]
		p2 = e[:,1]


		d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Pcc_realisation_'+str(mv)+'_z='+str(z[j])+'.txt')
		kmat = d[:,8]
		Pmat = np.zeros((len(kmat),10))
		for i in xrange(0,8):
			Pmat[:,i]= d[:,i]
		
		Pmat[:,8] = p1
		Pmat[:,9] = p2


		#-----------------------------------------------------------------------
		#---------------- halo neutrino Real space ---------------------------
		#-----------------------------------------------------------------------
		d1 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_0.15_z='+str(z[j])+'.txt')
		k = d1[:,19]
		Phh1 = np.zeros((len(k),10))
		Pshot1 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh1[:,i]= d1[:,pnum1[i]]
			Pshot1[i]= d1[0,pnum2[i]]
		# second mass range
		d2 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_0.15_z='+str(z[j])+'.txt')
		k = d2[:,19]
		Phh2 = np.zeros((len(k),10))
		Pshot2 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh2[:,i]= d2[:,pnum1[i]]
			Pshot2[i]= d2[0,pnum2[i]]
		# third mass range
		d3 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_0.15_z='+str(z[j])+'.txt')
		k = d3[:,19]
		Phh3 = np.zeros((len(k),10))
		Pshot3 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh3[:,i]= d3[:,pnum1[i]]
			Pshot3[i]= d3[0,pnum2[i]]
		# fourth mass range
		d4 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_0.15_z='+str(z[j])+'.txt')
		k = d4[:,19]
		Phh4 = np.zeros((len(k),10))
		Pshot4 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh4[:,i]= d4[:,pnum1[i]]
			Pshot4[i]= d4[0,pnum2[i]]
	
	
		#~ ### compute where shot noise is 80% of the ps
		
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
		
		#~ test = np.loadtxt('test.txt')
		#~ PH1bis = test[:,0]
		#~ PH2bis = test[:,1]
		#~ PH3bis = test[:,2]
		#~ PH4bis = test[:,3]
		#~ Pshot1bis = test[0,4]
		#~ Pshot2bis = test[0,5]
		#~ Pshot3bis = test[0,6]
		#~ Pshot4bis = test[0,7]
		#~ kk1bis = test[0,8]
		#~ kk2bis = test[0,9]
		#~ kk3bis = test[0,10]
		#~ kk4bis = test[0,11]
		
		#~ plt.figure()
		#~ M1, =plt.plot(k, PH1, c ='C0', label='halo Power spectrum')
		#~ M2, =plt.plot(k, PH2, c ='C1',)
		#~ M3, =plt.plot(k, PH3, c ='C2',)
		#~ M4, =plt.plot(k, PH4, c ='C3',)
		#~ M1, =plt.plot(k, PH1bis, c ='C0',)
		#~ M2, =plt.plot(k, PH2bis, c ='C1',)
		#~ M3, =plt.plot(k, PH3bis, c ='C2',)
		#~ M4, =plt.plot(k, PH4bis, c ='C3',)
		#~ plt.axhline( Pshot1, color='C0', linestyle='--', label='shot noise')
		#~ plt.axhline( Pshot2, color='C1', linestyle='--')
		#~ plt.axhline( Pshot3, color='C2', linestyle='--')
		#~ plt.axhline( Pshot4, color='C3', linestyle='--')
		#~ plt.axhline( Pshot1bis, color='C0', linestyle='--')
		#~ plt.axhline( Pshot2bis, color='C1', linestyle='--')
		#~ plt.axhline( Pshot3bis, color='C2', linestyle='--')
		#~ plt.axhline( Pshot4bis, color='C3', linestyle='--')
		#~ plt.fill_between(k,PH1-errPhh1, PH1+errPhh1, alpha=0.6)
		#~ plt.fill_between(k,PH2-errPhh2, PH2+errPhh2, alpha=0.6)
		#~ plt.fill_between(k,PH3-errPhh3, PH3+errPhh3, alpha=0.6)
		#~ plt.fill_between(k,PH4-errPhh4, PH4+errPhh4, alpha=0.6)
		#~ plt.axvline( kk1, color='C0', linestyle=':', label='shot noise = 80% of P(k)')
		#~ plt.axvline( kk2, color='C1', linestyle=':')
		#~ plt.axvline( kk3, color='C2', linestyle=':')
		#~ plt.axvline( kk4, color='C3', linestyle=':')
		#~ plt.axvline( kk1bis, color='C0', linestyle=':')
		#~ plt.axvline( kk2bis, color='C1', linestyle=':')
		#~ plt.axvline( kk3bis, color='C2', linestyle=':')
		#~ plt.axvline( kk4bis, color='C3', linestyle=':')
		#~ plt.legend(loc = 'upper right', title='z = '+str(z[j]), fancybox=True)
		#~ plt.legend(loc = 'lower left', title='z = '+str(z[j]), fancybox=True, fontsize = 14)
		#~ plt.figlegend( (M1,M2,M3,M4), ('mass range M1','mass range M2','mass range M3','mass range M4'), \
		#~ loc = 'upper center', ncol=2, labelspacing=0. , title ='Mv = 0.15eV ', fontsize = 14)
		#~ plt.xlabel('k [h/Mpc]', fontsize=14)
		#~ plt.ylabel('P(k)', fontsize=14)
		#~ plt.xscale('log')
		#~ plt.yscale('log')
		#~ plt.xlim(8e-3,2)
		#~ plt.ylim(1e2,2e5)
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
			
			
		#~ ### do the mean over quantitites ###
		
		Pmm = np.mean(Pmat[:,0:11], axis=1)
		PH1 = np.mean(Phh1[:,0:11], axis=1)
		PH2 = np.mean(Phh2[:,0:11], axis=1)
		PH3 = np.mean(Phh3[:,0:11], axis=1)
		PH4 = np.mean(Phh4[:,0:11], axis=1)

		
		bias1 = np.mean(bhh1[:,0:11], axis=1)
		bias2 = np.mean(bhh2[:,0:11], axis=1)
		bias3 = np.mean(bhh3[:,0:11], axis=1)
		bias4 = np.mean(bhh4[:,0:11], axis=1)
		
		### smooth the curve
		from scipy.signal import savgol_filter
		bias1s = savgol_filter(bias1, 51, 3) 
		bias2s = savgol_filter(bias2, 51, 3) 
		bias3s = savgol_filter(bias3, 51, 3) 
		bias4s = savgol_filter(bias4, 51, 3) 
		
		errb1 = np.std(bhh1[:,0:11], axis=1)
		errb2 = np.std(bhh2[:,0:11], axis=1)
		errb3 = np.std(bhh3[:,0:11], axis=1)
		errb4 = np.std(bhh4[:,0:11], axis=1)
		
		errPhh1 = np.std(Phh1[:,0:11], axis=1)
		errPhh2 = np.std(Phh2[:,0:11], axis=1)
		errPhh3 = np.std(Phh3[:,0:11], axis=1)
		errPhh4 = np.std(Phh4[:,0:11], axis=1)
		
		noise1 = np.std(Pshot1[0:11])
		noise2 = np.std(Pshot2[0:11])
		noise3 = np.std(Pshot3[0:11])
		noise4 = np.std(Pshot4[0:11])
		
		
		
		#-----------------------------------------------------------------------
		#---------------- halo neutrino Redshift space ---------------------------
		#-----------------------------------------------------------------------
		d1a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d1b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d1c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d1d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d2a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d2b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d2c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d2d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d3a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
		d3b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
		d3c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
		d3d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')


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
		
		return kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
		bias1s, bias2s, bias3s, bias4s, errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3,\
		errPr4, kclass, Tm, Tcb, noise1, noise2, noise3, noise4
		#~ return kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1s, bias2s, bias3s, bias4s, \
		#~ errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4
