import numpy as np
import h5py
import math
import scipy
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
from loop_pt import pt_terms
from polynomial import poly
from perturbation import perturb
from interp import interp_simu
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
Mnu       = 0.0  #eV
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

for j in xrange(0,len(z)): #chmahe this after test
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

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	bias1s, bias2s, bias3s, bias4s, errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2,\
	errPr3, errPr4, kclass, Tm, Tcb = ld_data(Mnu, z, j)

####################################################################
##### define the maximum scale for the fit 
	kstop1 = [0.16,0.2,0.25,0.35]
	kstop2 = [0.12,0.16,0.2,0.2]
	kstop3 = [0.15,0.15,0.15,0.15]
	
#### the case 
	case = 3
	
	if case == 1:
		kstop = kstop1[ind]
	elif case == 2:
		kstop = kstop2[ind]
	elif case == 3:
		kstop = kstop3[ind]
		
		
#~ #*********************************************************************************************
#~ #*********************************************************************************************
	#~ ktest = np.logspace(np.log10(0.05),np.log10(0.55),15)
	ktest = np.logspace(np.log10(0.04),np.log10(0.5),15)
	for kstop in ktest:
		print kstop
	
		
		# interpolate to have more points and create an evenly logged array
		#~ kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 200)
		#~ Plinbis = np.interp(kbis, k, Plin)
		lim = np.where((k < kstop)&(k > 1e-2))[0]

		
	####################################################################
	##### compute linear bias and error
		
		# on interpolated array
		if kstop < 0.05:
			toto = np.where(k < kstop)[0]
		else:
			toto = np.where(k < 0.05)[0]
		lb1 = np.mean(bias1[toto])
		lb2 = np.mean(bias2[toto])
		lb3 = np.mean(bias3[toto])
		lb4 = np.mean(bias4[toto])
		errlb1 = np.mean(errb1[toto])
		errlb2 = np.mean(errb2[toto])
		errlb3 = np.mean(errb3[toto])
		errlb4 = np.mean(errb4[toto])
		
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
	#### compute pt terms

		Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(kcamb, Pcamb)
		
		### interpolate on simulation k
		Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H  = interp_simu(z,j ,k, kcamb, Pcamb, Pmod_dd, Pmod_dt, Pmod_tt,\
		A, B, C, D, E, F, G, H, 1)
		
	####################################################################
	#### get fitted coefficients


		print 'polynomial'
		biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, lb1, lb2, lb3, lb4,\
		errlb1, errlb2, errlb3, errlb4, k, bias1, bias2, bias3, bias4, errb1, errb2, errb3, errb4,Mnu, z, j, case)
		
		#~ biasF1, biasF2, biasF3, biasF4 = poly(kstop, k, lb1, lb2, lb3, lb4,\
		#~ errlb1, errlb2, errlb3, errlb4, kbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis,Mnu, z, j, case)


	#-------------------------------------------------------------------
		print 'perturbation'
		#~ bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
		#~ bias3PTbis2, bias3PTbis3, bias3PTbis4, PsptD1r1, PsptD2r1, PsptD3r1 = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
		#~ bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j, case)
		
		bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
		bias3PTbis2, bias3PTbis3, bias3PTbis4 = perturb(kstop,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmod_dd, k, bias1,\
		bias2, bias3, bias4, errb1, errb2, errb3, errb4, A, B, C, D, E, F,Mnu, z, j, case)
	
		
		#~ bins = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 50)
		#~ stats, binedges, binnumber = scipy.stats.binned_statistic(kbis,bias4bis,'mean',bins )
		#~ binmean=10**(0.5*(np.log10(binedges[1:])+np.log10(binedges[:-1])))
		#~ bin_width = (binedges[1] - binedges[0])
		#~ bin_centers = binedges[1:] - bin_width/2
		#~ print binmean
		
		#~ plt.figure()
		#~ plt.plot(kbis,bias1bis)
		#~ plt.plot(kbis,bias3PT1)
		#~ plt.ylim(0.7, 0.9)
		#-------------------
		#~ plt.plot(kbis,bias2bis)
		#~ plt.plot(kbis,bias3PT2)
		#~ plt.ylim(0.75, 0.95)
		#~ #-------------------
		#~ plt.plot(kbis,bias3bis)
		#~ plt.plot(kbis,bias3PT3)
		#~ plt.ylim(0.9, 1.1)
		#-------------------
		#~ plt.plot(kbis,bias4bis)
		#~ plt.plot(kbis,bias3PT4)
		#~ plt.ylim(1.2, 1.4)
		#--------------------
		#~ plt.xlim(1e-2, kstop)
		#~ plt.xscale('log')
		#~ plt.axvline(0.15)
		#~ plt.axvspan(kstop, 7, alpha=0.2, color='grey')
		#~ plt.show()

		#~ kill
	####################################################################
	##### compute the chi2 of different quantities
	####################################################################

		# p is number of free param
		F1 = (biasF1[lim]-bias1[lim])**2/errb1[lim]**2
		F2 = (biasF2[lim]-bias2[lim])**2/errb2[lim]**2
		F3 = (biasF3[lim]-bias3[lim])**2/errb3[lim]**2
		F4 = (biasF4[lim]-bias4[lim])**2/errb4[lim]**2
		chi2F1 = np.sum(F1)
		chi2F2 = np.sum(F2)
		chi2F3 = np.sum(F3)
		chi2F4 = np.sum(F4)
		#-------------------------------------------------

		PT1 = (bias2PT1[lim]- bias1[lim])**2/errb1[lim]**2
		PT2 = (bias2PT2[lim]- bias2[lim])**2/errb2[lim]**2
		PT3 = (bias2PT3[lim]- bias3[lim])**2/errb3[lim]**2
		PT4 = (bias2PT4[lim]- bias4[lim])**2/errb4[lim]**2
		chi2PT1 = np.sum(PT1)
		chi2PT2 = np.sum(PT2)
		chi2PT3 = np.sum(PT3)
		chi2PT4 = np.sum(PT4)
		#-------------------------------------------------
		PTbis1 = (bias3PT1[lim]- bias1[lim])**2/errb1[lim]**2
		PTbis2 = (bias3PT2[lim]- bias2[lim])**2/errb2[lim]**2
		PTbis3 = (bias3PT3[lim]- bias3[lim])**2/errb3[lim]**2
		PTbis4 = (bias3PT4[lim]- bias4[lim])**2/errb4[lim]**2
		chi2PTbis1 = np.sum(PTbis1)
		chi2PTbis2 = np.sum(PTbis2)
		chi2PTbis3 = np.sum(PTbis3)
		chi2PTbis4 = np.sum(PTbis4)
		#-------------------------------------------------
		PTter1 = (bias3PTbis1[lim]- bias1[lim])**2/errb1[lim]**2
		PTter2 = (bias3PTbis2[lim]- bias2[lim])**2/errb2[lim]**2
		PTter3 = (bias3PTbis3[lim]- bias3[lim])**2/errb3[lim]**2
		PTter4 = (bias3PTbis4[lim]- bias4[lim])**2/errb4[lim]**2
		chi2PTter1 = np.sum(PTter1)
		chi2PTter2 = np.sum(PTter2)
		chi2PTter3 = np.sum(PTter3)
		chi2PTter4 = np.sum(PTter4)
		
		cname = 'chi2a_z='+str(z[j])+'.txt'
		with open(cname, 'a+') as fid_file:

			fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop,\
			chi2F1, chi2F2, chi2F3, chi2F4, chi2PT1, chi2PT2, chi2PT3, chi2PT4, chi2PTbis1, chi2PTbis2, chi2PTbis3, chi2PTbis4,\
			chi2PTter1, chi2PTter2, chi2PTter3, chi2PTter4))
		print '\n'

		#~ del  pte, Plin, Tm, Tcb, kbis,\
		#~ Plinbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis,errb3bis,errb4bis, Pmmbis,PH1bis,PH2bis,PH3bis,PH4bis,\
		#~ errPhh1bis,errPhh2bis,errPhh3bis,errPhh4bis,Pmono1bis,Pmono2bis,Pmono3bis,Pmono4bis,errPr1bis,errPr2bis,errPr3bis,errPr4bis,\
		#~ biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis, bias2PT1, bias2PT2, bias2PT3, bias2PT4,\
		#~ bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1, bias3PTbis2, bias3PTbis3,bias3PTbis4,F1 ,F2,F3,F4,chi2F1,chi2F2,chi2F3,chi2F4,PT1,PT2,PT3,PT4,chi2PT1,chi2PT2,chi2PT3,chi2PT4,PTbis1,PTbis2,PTbis3\
		#~ ,PTbis4,chi2PTbis1,chi2PTbis2,chi2PTbis3,chi2PTbis4
		#~ del  pte, Tm, Tcb, bias1, bias2, bias3, bias4, errb1, errb2,errb3,errb4, Pmm,PH1,PH2,PH3,PH4,\
		#~ errPhh1,errPhh2,errPhh3,errPhh4,Pmono1,Pmono2,Pmono3,Pmono4,errPr1,errPr2,errPr3,errPr4,\
		#~ biasF1, biasF2, biasF3, biasF4, bias2PT1, bias2PT2, bias2PT3, bias2PT4,\
		#~ bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1, bias3PTbis2, bias3PTbis3,bias3PTbis4,F1 ,F2,F3,F4,chi2F1,chi2F2,chi2F3,chi2F4,PT1,PT2,PT3,PT4,chi2PT1,chi2PT2,chi2PT3,chi2PT4,PTbis1,PTbis2,PTbis3\
		#~ ,PTbis4,chi2PTbis1,chi2PTbis2,chi2PTbis3,chi2PTbis4
	kill
end = time()
print 'total time is '+str((end - start))	 

