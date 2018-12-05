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
from loop_pt import pt_terms
from polynomial import poly
from perturbation import perturb
#~ from hmf_test import htest
from time import time
from rsd import RSD1
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma
from fit_emcee import coeffit_pl,coeffit_pl2,coeffit_exp1, coeffit_exp2, coeffit_exp3,coeffit_Kaiser, coeffit_Scocci, coeffit_TNS, coeffit_eTNS


def rescal(j, case):

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
	
	start = time()

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

############################################################################################################
############################################################################################################
############################################################################################################
### load rescaling coefficients for analytic model
	sca1, sca2, sca3, sca4 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.15eV/large_scale/rescaling_z='+str(z[j])+'_.txt')

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	errb1, errb2, errb3, errb4, pmono1, pmono2, pmono3, pmono4, errPr1, errPr2, errPr3, errPr4 = ld_data(0.0, z, j)
	
	kstop1 = [0.16,0.2,0.25,0.35]
	kstop2 = [0.12,0.16,0.2,0.2]
	kstop3 = [0.15,0.15,0.15,0.15]
	
#### the case 
	print 'this is the case '+str(case)
	
	if case == 1:
		kstop = kstop1[ind]
	elif case == 2:
		kstop = kstop2[ind]
	elif case == 3:
		kstop = kstop3[ind]

	Plin = Pcamb
	klin = kcamb

	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
	Plin = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
	Tm = pte[:,1]
	pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected3-'+str(z[j])+'.txt')
	Tcb = pte[:,1]
	
	# interpolate to have more points and create an evenly logged array
	kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 250)
	Plinbis = np.interp(kbis, k, Plin)
	lim = np.where((k < kstop)&(k > 1e-2))[0]

##### real space
	bias1bis = np.interp(kbis, k, bias1)
	bias2bis = np.interp(kbis, k, bias2)
	bias3bis = np.interp(kbis, k, bias3)
	bias4bis = np.interp(kbis, k, bias4)
	errb1bis = np.interp(kbis, k, errb1)
	errb2bis = np.interp(kbis, k, errb2)
	errb3bis = np.interp(kbis, k, errb3)
	errb4bis = np.interp(kbis, k, errb4)
	Pmmbis = np.interp(kbis, k, Pmm)
	PH1bis = np.interp(kbis, k, PH1)
	PH2bis = np.interp(kbis, k, PH2)
	PH3bis = np.interp(kbis, k, PH3)
	PH4bis = np.interp(kbis, k, PH4)
	errPhh1bis = np.interp(kbis, k, errPhh1)
	errPhh2bis = np.interp(kbis, k, errPhh2)
	errPhh3bis = np.interp(kbis, k, errPhh3)
	errPhh4bis = np.interp(kbis, k, errPhh4)

	##### redshift space
	Pmono1bis = np.interp(kbis, k, pmono1)
	Pmono2bis = np.interp(kbis, k, pmono2)
	Pmono3bis = np.interp(kbis, k, pmono3)
	Pmono4bis = np.interp(kbis, k, pmono4)
	errPr1bis = np.interp(kbis, k, errPr1)
	errPr2bis = np.interp(kbis, k, errPr2)
	errPr3bis = np.interp(kbis, k, errPr3)
	errPr4bis = np.interp(kbis, k, errPr4)
	Tm =  np.interp(kbis,k,Tm)
	Tcb =  np.interp(kbis,k,Tcb)
	
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


	Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(kbis, Plinbis)
	
	print 'polynomial'
	biasF1, biasF2, biasF3, biasF4,biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, k, lb1, lb2, lb3, lb4,\
	errlb1, errlb2, errlb3, errlb4, kbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis,0.0, z, j, case)

	print 'perturbation'
	bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	bias3PTbis2, bias3PTbis3, bias3PTbis4 = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
	bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,0.0, z, j, case)

### mean ####
	#~ b1 = np.array([bias2PT1*sca1/bias1bis, bias2PT2*sca2/bias2bis, bias2PT3*sca3/bias3bis, bias2PT4*sca4/bias4bis])
	#~ b1bis = np.array([bias3PT1*sca1/bias1bis, bias3PT2*sca2/bias2bis, bias3PT3*sca3/bias3bis, bias3PT4*sca4/bias4bis])
	#~ b1ter = np.array([bias3PTbis1*sca1/bias1bis, bias3PTbis2*sca2/bias2bis, bias3PTbis3*sca3/bias3bis, bias3PTbis4*sca4/bias4bis])
	#~ b3 = np.array([biasF1*sca1/bias1bis, biasF2*sca2/bias2bis, biasF3*sca3/bias3bis, biasF4*sca4/bias4bis])
	#~ b3bis = np.array([biasF1bis*sca1/bias1bis, biasF2bis*sca2/bias2bis, biasF3bis*sca3/bias3bis, biasF4bis*sca4/bias4bis])
	#~ b1 = np.mean(b1,axis=0)
	#~ b1bis = np.mean(b1bis,axis=0)
	#~ b1ter = np.mean(b1ter,axis=0)
	#~ b3 = np.mean(b3,axis=0)
	#~ b3bis = np.mean(b3bis,axis=0)
	
	


	b2PT1 = bias2PT1*sca1
	b2PT2 = bias2PT2*sca2
	b2PT3 = bias2PT3*sca3
	b2PT4 = bias2PT4*sca4
	b3PT1 = bias3PT1*sca1
	b3PT2 = bias3PT2*sca2
	b3PT3 = bias3PT3*sca3
	b3PT4 =	bias3PT4*sca4
	bF1 = biasF1*sca1
	bF2 = biasF2*sca2
	bF3 = biasF3*sca3
	bF4 = biasF4*sca4
	b3PTbis1 = bias3PTbis1*sca1
	b3PTbis2 = bias3PTbis2*sca2
	b3PTbis3 = bias3PTbis3*sca3
	b3PTbis4 = bias3PTbis4*sca4
	
	#~ return b2PT1, b2PT2, b2PT3, b2PT4,b3PT1, b3PT2, b3PT3, b3PT4,bF1, bF2, bF3, bF4,\
	#~ b3PTbis1, b3PTbis2, b3PTbis3, b3PTbis4
#~ ####################################################################
#~ ### compute of the 4 mass bins
#~ ####################################################################
	fcc = fz * (Tm/ Tcb)
	
	#~ kai1, kai2, kai3, kai4, sco1, sco2, sco3, sco4, tns1, tns2, tns3, tns4, etns1, etns2, etns3, etns4 = RSD(fz,fcc, Dz[ind]\
	#~ , j, kstop, Pmmbis, biasF1, biasF2, biasF3, biasF4, kbis, Plinbis, Pmono1bis, Pmono2bis, Pmono3bis, \
	#~ Pmono4bis, errPr1bis, errPr2bis, errPr3bis, errPr4bis, Pmod_dt, Pmod_tt, case,z,0.0, A, B, C, D, E, F, G, H )
	
	
	#~ return kai1*sca1, kai2*sca2, kai3*sca3, kai4*sca4, sco1*sca1, sco2*sca2, sco3*sca3, sco4*sca4, tns1*sca1, tns2*sca2,\
	#~ tns3*sca3, tns4*sca4, etns1*sca1, etns2*sca2, etns3*sca3, etns4*sca4
	#---------------------------------------------------
	kai1, kai2, kai3, kai4, sco1, sco2, sco3, sco4, tns1, tns2, tns3, tns4, etns1, etns2, etns3, etns4 = RSD1(fz,fcc, Dz[ind]\
	, j, kstop, Pmmbis, bF1, bF2, bF3, bF4, kbis, Plinbis, Pmono1bis, Pmono2bis, Pmono3bis, Pmono4bis, errPr1bis, errPr2bis,\
	errPr3bis, errPr4bis, Pmod_dt, Pmod_tt, case,z,0.0, A, B, C, D, E, F, G, H, sca1, sca2, sca3, sca4 )
	
	
	return kai1, kai2, kai3, kai4, sco1, sco2, sco3, sco4*sca4, tns1, tns2,\
	tns3, tns4, etns1, etns2, etns3, etns4
	
