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
from load_data import ld_data
from rescaling import rescal
from loop_pt import pt_terms
from polynomial import poly
from perturbation import perturb
#~ from hmf_test import htest
from time import time
from rsd import RSD
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma
from fit_emcee import coeffit_pl,coeffit_pl2,coeffit_exp1, coeffit_exp2, coeffit_exp3,coeffit_Kaiser, coeffit_Scocci, coeffit_TNS, coeffit_eTNS


z = [0.0,0.5,1.0,2.0]
#~ z = [0.0,2.0]
#~ mu = 0.5
#~ kmax = 1
#~ mass_range = ['m1','m2','m3','m4']
#~ mass_range = ['m1', 'm2']
#~ mass_range = ['m1']
#~ axis = 0 #in redshift-space distortion axis

# neutrino parameters
hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
###########################
Mnu       = 0.0 #eV
###########################
Nnu       = 0  #number of massive neutrinos
Neff      = 3.046

# cosmological parameters
h       = 0.6711
Omega_c = 0.2685 - Mnu/(93.14*h**2)
Omega_b = 0.049
Omega_l = 0.6825
Omega_k = 0.0
Omega_m = Omega_c + Omega_b
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
	##### scale factor 
	red = ['0.0','0.5','1.0','2.0']
	ind = red.index(str(z[j]))
	#~ fz = [0.524,0.759,0.875,0.958]
	Dz = [ 1.,0.77,0.61,0.42]
	print 'For redshift z = ' + str(z[j])
	
	Omeg_m_z = Omega_m * (1 + z[j])**3 / (Omega_m * (1 + z[j])**3 + Omega_l)
	fz = Omeg_m_z**0.55
	
#########################################################################
#### load data from simualtion 

	#### load data from simualtion 

	kcamb, Pcamb, k, Pmm, _, _, _ , _, _, _, _, _, _, _,_, _, \
	_, _, _, _, _, _, _, _, _, _, _, _, _, _,\
	_, _ = ld_data(Mnu, z, j)
	
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
	# fourth mass range
	#~ d4 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/Phh4_realisation_z='+str(z[j])+'.txt')
	d4 = np.loadtxt('/home/david/codes/Analysis/cosmo_test/Phh4_realisation_0.0_z=0.0.txt')
	k = d4[:,19]
	Phh4 = np.zeros((len(k),10))
	Pshot4 = np.zeros((10))
	pnum1 = [0,2,4,6,8,10,12,14,16,18]
	pnum2 = [1,3,5,7,9,11,13,15,17,20]
	for i in xrange(0,10):
		Phh4[:,i]= d4[:,pnum1[i]]
		Pshot4[i]= d4[0,pnum2[i]]
	
	#~ #-------------------------------------------------------------------
	#~ #----remove shot noise, compute bias and bias variance -------------
	#~ #-------------------------------------------------------------------
	bhh4 = np.zeros((len(k),10))
	
	
	for i in xrange(0,10):
		Phh4[:,i] = Phh4[:,i]-Pshot4[i]
		nul4 = np.where(Phh4[:,i] < 0)[0]
		Phh4[nul4,i] = 0
		bhh4[:,i] = np.sqrt(Phh4[:,i]/Pmat[:,i])
		

	#~ ### do the mean over quantitites ###
	
	Pmm = np.mean(Pmat[:,0:11], axis=1)
	ePmm = np.std(Pmat[:,0:11], axis=1)
	PH4 = np.mean(Phh4[:,0:11], axis=1)
	
	bias4 = np.mean(bhh4[:,0:11], axis=1)
	errb4 = np.std(bhh4[:,0:11], axis=1)
	
	
	kfid = k
	bfid = bias4
	efid = errb4
	
	
	####################################################################
	####################################################################
	# load other cosmology ps
	
	fid = np.loadtxt('/home/david/codes/Analysis/cosmo_test/b_z=0.txt')
	kfid = fid[:,0]
	bfid_square = fid[:,1]
	bfid = np.sqrt(bfid_square)
	efid_square = fid[:,2]
	efid = efid_square/2./bfid
	
	fid2 = np.loadtxt('/home/david/codes/Analysis/cosmo_test/Pk_z=0.txt')
	Pmmfid = fid2[:,1]
	
	#~ plt.plot(k,PH4)
	#~ plt.plot(k,np.sqrt(PH4/Pmm))
	#~ plt.plot(kfid, bfid)
	#~ plt.fill_between(kfid, bfid -efid, bfid + efid, alpha = 0.6)
	#~ plt.plot(k, bias4)
	#~ plt.fill_between(k, bias4 -errb4, bias4 + errb4, alpha = 0.6)
	#~ plt.yscale('log')
	#~ plt.xscale('log')
	#~ plt.xlim(1e-2, 0.4)
	#~ plt.ylim(0.7, 0.9)
	#~ plt.show()
	#~ kill
	

	test1 = np.loadtxt('/home/david/codes/Analysis/cosmo_test/s8_p/b_z=0.txt')
	ktest1 = test1[:,0]
	btest1_square = test1[:,1]
	btest1 = np.sqrt(btest1_square)
	test2 = np.loadtxt('/home/david/codes/Analysis/cosmo_test/s8_m/b_z=0.txt')
	ktest2 = test2[:,0]
	btest2_square = test2[:,1]
	btest2 = np.sqrt(btest2_square)
	test3 = np.loadtxt('/home/david/codes/Analysis/cosmo_test/Om_m/b_z=0.txt')
	ktest3 = test3[:,0]
	btest3_square = test3[:,1]
	btest3 = np.sqrt(btest3_square)
	test4 = np.loadtxt('/home/david/codes/Analysis/cosmo_test/Om_p/b_z=0.txt')
	ktest4 = test4[:,0]
	btest4_square = test4[:,1]
	btest4 = np.sqrt(btest4_square)
	
	#####################################################################
	#####################################################################
	##### define the maximum scale for the fit 
	kstop1 = [0.16,0.2,0.25,0.35]
	kstop2 = [0.12,0.16,0.2,0.2]
	kstop3 = [0.15,0.15,0.15,0.15]
	
	#### the case 
	case = 2
	
	if case == 1:
		kstop = kstop1[ind]
	elif case == 2:
		kstop = kstop2[ind]
	elif case == 3:
		kstop = kstop3[ind]
		
		
	#### other kstop
	#~ kstoplim = [0.5,0.5,0.5,0.4]
	#~ kstop = kstoplim[ind]
	
	###################################################################
	#Plin = Pclass
	#~ #klin = kclass
	#Plin = pks
	#klin = ks
	Plin = Pcamb
	klin = kcamb

#######################################################################

	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
	Plin = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
	Tm = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected3-'+str(z[j])+'.txt')
	Tcb = pte[:,1]
	
	# interpolate to have more points and create an evenly logged array
	#~ kfid2= np.logspace(np.log10(np.min(kfid)), np.log10(np.max(kfid)), 1000)
	#~ kbis = np.logspace(np.log10(np.min(kcamb)), np.log10(np.max(kcamb)), 200)
	#~ Plinbis = np.interp(kbis, k, Plin)
	
	lim = np.where((k < kstop)&(k > 1e-2))[0]

	
	# on interpolated array
	toto = np.where(kfid < 0.05)[0]
	lb4 = np.mean(bfid[toto])
	errlb4 = np.mean(efid[toto])
	#~ toto = np.where(k < 0.05)[0]
	#~ lb4 = np.mean(bias4[toto])
	#~ errlb4 = np.mean(errb4[toto])
	
	print lb4, errlb4
	
	
####################################################################
#### compute pt terms
	

	Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(kcamb, Pcamb)
	
	### interpolate to create more point and for comparison
	k2= np.logspace(np.log10(np.min(kfid)), np.log10(np.max(kfid)), 500)
	#~ bfid = np.interp(k2, kfid, bfid)
	#~ efid = np.interp(k2, kfid, efid)
	
	Pmod_dd = np.interp(kfid, kcamb, Pmod_dd)
	Pmod_dt = np.interp(kfid, kcamb, Pmod_dt)
	Pmod_tt = np.interp(kfid, kcamb, Pmod_tt)
	A = np.interp(kfid, kcamb, A)
	B = np.interp(kfid, kcamb, B)
	C = np.interp(kfid, kcamb, C)
	D = np.interp(kfid, kcamb, D)
	E = np.interp(kfid, kcamb, E)
	F = np.interp(kfid, kcamb, F)
	G = np.interp(kfid, kcamb, G)
	H = np.interp(kfid, kcamb, H)
	#~ Pmod_dd = np.interp(k2, kcamb, Pmod_dd)
	#~ Pmod_dt = np.interp(k2, kcamb, Pmod_dt)
	#~ Pmod_tt = np.interp(k2, kcamb, Pmod_tt)
	#~ A = np.interp(k2, kcamb, A)
	#~ B = np.interp(k2, kcamb, B)
	#~ C = np.interp(k2, kcamb, C)
	#~ D = np.interp(k2, kcamb, D)
	#~ E = np.interp(k2, kcamb, E)
	#~ F = np.interp(k2, kcamb, F)
	#~ G = np.interp(k2, kcamb, G)
	#~ H = np.interp(k2, kcamb, H)
	
	#~ plt.plot(k, bias4)
	#~ plt.plot(k, bias4s)
	#~ plt.ylim(1.2, 1.4)
	#~ plt.plot(k, errb4)
	#~ plt.plot(k, errb4s)
	#~ plt.ylim(0,0.2)
	#~ plt.plot(kfid, Pmmfid, c='k')
	#~ plt.plot(kfid, Pmod_dd, c='k')
	#~ plt.plot(kfid, A,label='A')
	#~ plt.plot(kfid, B,label='B')
	#~ plt.plot(kfid, C,label='C')
	#~ plt.plot(kfid, D,label='D')
	#~ plt.plot(kfid, E,label='E')
	#~ plt.plot(kfid, F,label='F')
	#~ plt.plot(k2, A,label='A')
	#~ plt.plot(k2, B,label='B')
	#~ plt.plot(k2, C,label='C')
	#~ plt.plot(k2, D,label='D')
	#~ plt.plot(k2, E,label='E')
	#~ plt.plot(k2, F,label='F')
	#~ plt.plot(k, PH4, c='r')
	#~ plt.ylim(1e2,1e5)
	#~ plt.legend(loc='upper right')
	#~ plt.yscale('log')
	#~ plt.xscale('log')
	#~ plt.xlim(1e-2, 1)
	#~ plt.title('A')
	#~ plt.show()
	#~ kill
	
	
####################################################################
#### get fitted coefficients

	print 'polynomial'
	lim = np.where((k < kstop)&(k > 1e-2))[0]

	

	def funcb(k, b1, b2, b3, b4):
		return b1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
		
	def funcbis(k, b1, b2, b4):
		return b1 + b2 * k**2 + b4 * k**4 
	
	
	# here kh because the simu scale
	popF4, pcovF4 = curve_fit(funcb, kfid[lim], bfid[lim], sigma = efid[lim],  check_finite=True, maxfev=500000)
	b1x4_ml, b2x4_ml, b3x4_ml, b4x4_ml = coeffit_pl(kstop, lb4, errlb4, popF4, kfid, bfid, efid)
	biasF4 = b1x4_ml + b2x4_ml * kfid**2 + b3x4_ml * kfid**3 + b4x4_ml * kfid**4
	

#-------------------------------------------------------------------

	print 'perturbation'
	popbis4 = [lb4, -25.,-4/7.*(lb4-1),32/315.*(lb4-1)]
	
	
	#~ popbis4 = [lb4,1]
	
	#~ print efid
	#~ print bfid
	#~ kill
	#3rd order free -----------------------------------------------------------------------------------------------
	
	
	
	popbis4 = [lb4, 1.,-4/7.*(lb4-1),32/315.*(lb4-1),100]
	b1z4_ml, b2z4_ml, bsz4_ml, b3z4_ml, N_ml  = coeffit_exp2(kstop, Pmod_dd, A, B, C, D, E, F, lb4, errlb4, popbis4,\
	kfid ,bfid ,efid)
	bias3PT4 = np.sqrt((b1z4_ml**2 * Pmod_dd+ b1z4_ml*b2z4_ml*A + 1/4.*b2z4_ml**2*B + b1z4_ml*bsz4_ml*C +\
	1/2.*b2z4_ml*bsz4_ml*D + 1/4.*bsz4_ml**2*E + 2*b1z4_ml*b3z4_ml*F + N_ml)/Pmod_dd)
	
	#~ popbis4 = [lb4, 1.,-4/7.*(lb4-1),100]
	#~ b1z4_mcmc, b2z4_mcmc, bsz4_mcmc, N, b1z4_ml, b2z4_ml, bsz4_ml, N_ml = coeffit_exp1(kstop, Pmod_dd, A, B, C, D, E, lb4, errlb4, popbis4,\
	#~ kfid ,bfid ,efid)
	#~ bias3PT4 = np.sqrt((b1z4_mcmc[0]**2 * Pmod_dd+ b1z4_mcmc[0]*b2z4_mcmc[0]*A + 1/4.*b2z4_mcmc[0]**2*B + b1z4_mcmc[0]*bsz4_mcmc[0]*C +\
	#~ 1/2.*b2z4_mcmc[0]*bsz4_mcmc[0]*D + 1/4.*bsz4_mcmc[0]**2*E + N[0])/Pmod_dd)
	#~ bias3PT4bis = np.sqrt((b1z4_ml**2 * Pmod_dd+ b1z4_ml*b2z4_ml*A + 1/4.*b2z4_ml**2*B + b1z4_ml*bsz4_ml*C +\
	#~ 1/2.*b2z4_ml*bsz4_ml*D + 1/4.*bsz4_ml**2*E + N_ml)/Pmod_dd)
	
		
	
	####################################################################
	#~ plt.figure()
	#~ plt.plot(k, bias4)
	#~ plt.plot(kfid, bfid, c='k')
	#~ plt.plot(ktest1, btest1)
	#~ plt.plot(ktest2, btest2)
	#~ plt.plot(ktest3, btest3)
	#~ plt.plot(ktest4, btest4)

	#~ plt.plot(kbis, biasF4, c='C0')
	#~ plt.plot(kbis, bias3PT4, c='C1')
	#~ plt.xscale('log')
	#~ plt.xlim(1e-2, 1)
	#~ plt.ylim(1.6,2.2)
	#~ plt.show()
	
	#~ plt.figure()
	plt.axhline(1, c='k')
	r1, =plt.plot(kfid, biasF4/bfid, c='k', label = 'Polynomial')
	r2, =plt.plot(kfid, biasF4/btest1, c='C0')
	r3, =plt.plot(kfid, biasF4/btest2, c='C1')
	r4, =plt.plot(kfid, biasF4/btest3, c='C2')
	r5, =plt.plot(kfid, biasF4/btest4, c='C3')
	#--------------------------------------------
	r1, =plt.plot(kfid, bias3PT4/bfid, c='k', label = 'Perturbation theory', linestyle='--')
	r2, =plt.plot(kfid, bias3PT4/btest1, c='C0', linestyle='--')
	r3, =plt.plot(kfid, bias3PT4/btest2, c='C1', linestyle='--')
	r4, =plt.plot(kfid, bias3PT4/btest3, c='C2', linestyle='--')
	r5, =plt.plot(kfid, bias3PT4/btest4, c='C3', linestyle='--')
	plt.axhline(1.01, linestyle = ':', c='k')
	plt.axhline(0.99, linestyle = ':', c='k')
	plt.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ plt.figlegend( (r1, r2, r3, r4, r5), ('fiducial', r'$\Omega_m$ = 0.3175, $\sigma_8$ = 0.849',\
	#~ r'$\Omega_m$ = 0.3175, $\sigma_8$ = 0.819', r'$\Omega_m$ = 0.3075, $\sigma_8$ = 0.834',\
	#~ r'$\Omega_m$ = 0.3275, $\sigma_8$ = 0.834'), loc = 'upper center', ncol=5, labelspacing=0.,\
	#~ title =r' M$\nu$ = '+str(Mnu)+', case '+str(case), fontsize=12)
	plt.legend(loc = 'upper left', title = 'z = '+str(z[j]), fancybox=True, fontsize=14)
	plt.xscale('log')
	plt.xlabel('k [h/Mpc]', fontsize=16)
	plt.ylabel(r'$b_{fit}$ / $b_{cosmo}$', fontsize=18)
	plt.xlim(1e-2, 1)
	plt.ylim(0.85,1.15)
	plt.show()
	
	kill
	
