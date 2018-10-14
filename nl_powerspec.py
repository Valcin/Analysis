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
from time import time
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

for j in xrange(0,len(z)):
########################################################################
########################################################################
	####################################################################
	##### scale factor 

	red = ['0.0','0.5','1.0','2.0']
	ind = red.index(str(z[j]))
	f = [0.524,0.759,0.875,0.958]
	Dz = [ 1.,0.77,0.61,0.42]
	print 'For redshift z = ' + str(z[j])
	
	Omeg_m_z = Omega_m * (1 + z[j])**3 / (Omega_m * (1 + z[j])**3 + Omega_l)
	
#########################################################################
	#### load data from simualtion 


	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4 = ld_data(Mnu, z, j)


####################################################################
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
		
	kstoplim = [0.5,0.5,0.5,0.4]
	kstop = kstoplim[ind]
	print kstop
	
	
	# put identation to the rest to loop over kstop
	#~ #kstop_arr = np.logspace(np.log10(0.05),np.log10(0.6),20)
	#~ #for kstop in kstop_arr:
	#	print kstop
####################################################################
	#Plin = Pclass
	#~ #klin = kclass
	#Plin = pks
	#klin = ks
	Plin = Pcamb
	klin = kcamb

#######################################################################
	if Mnu == 0.0:
		# compute the linear ps on the simulation bins
		#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file1.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(klin)):
				#~ fid_file.write('%.8g %.8g\n' % ( klin[index_k], Plin[index_k]))
		#~ fid_file.close()
		#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file2.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(kclass)):
				#~ fid_file.write('%.8g %.8g\n' % ( kclass[index_k], Tm[index_k]))
		#~ fid_file.close()
		#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file3.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(kclass)):
				#~ fid_file.write('%.8g %.8g\n' % ( kclass[index_k], Tcb[index_k]))
		#~ fid_file.close()
		#~ ###
		#~ #exp2.expected(j)
		#~ ###
			
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
		Plin = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
		Tm = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected3-'+str(z[j])+'.txt')
		Tcb = pte[:,1]
		
	elif Mnu == 0.15:
		#~ # compute the linear ps on the simulation bins
		#~ with open('/home/david/codes/Paco/data2/0.15eV/exp/file1.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(klin)):
				#~ fid_file.write('%.8g %.8g\n' % ( klin[index_k], Plin[index_k]))
		#~ fid_file.close()
		#~ with open('/home/david/codes/Paco/data2/0.15eV/exp/file2.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(kclass)):
				#~ fid_file.write('%.8g %.8g\n' % ( kclass[index_k], Tm[index_k]))
		#~ fid_file.close()
		#~ with open('/home/david/codes/Paco/data2/0.15eV/exp/file3.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(kclass)):
				#~ fid_file.write('%.8g %.8g\n' % ( kclass[index_k], Tcb[index_k]))
		#~ fid_file.close()
		#~ ###
		#exp2.expected(j)
		###
		
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/exp/expected1-'+str(z[j])+'.txt')
		Plin = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/exp/expected2-'+str(z[j])+'.txt')
		Tm = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/exp/expected3-'+str(z[j])+'.txt')
		Tcb = pte[:,1]
	
	# interpolate to have more points and create an evenly logged array
	kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 250)
	#~ kbis = np.logspace(np.log10(np.min(kcamb)), np.log10(np.max(kcamb)), 200)
	Plinbis = np.interp(kbis, k, Plin)
	lim = np.where((kbis < kstop))[0]

	#~ plt.figure()
	#~ plt.plot(kcamb,Pcamb)
	#~ plt.plot(kbis,Plinbis)
	#~ plt.plot(k,Plin)
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(1e-3,10)
	#~ plt.ylim(1e-1,4e4)
	#~ plt.show()
	#~ kill

########################################################################################################################################
#######################################################################################################################################
##### interpolate data to have more point on fitting scales
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
	Pmono1bis = np.interp(kbis, k, Pmono1)
	Pmono2bis = np.interp(kbis, k, Pmono2)
	Pmono3bis = np.interp(kbis, k, Pmono3)
	Pmono4bis = np.interp(kbis, k, Pmono4)
	errPr1bis = np.interp(kbis, k, errPr1)
	errPr2bis = np.interp(kbis, k, errPr2)
	errPr3bis = np.interp(kbis, k, errPr3)
	errPr4bis = np.interp(kbis, k, errPr4)
	#~ Predbis = np.interp(kbis,k, Pred)
	#~ errPredbis = np.interp(kbis,k, errPred)
	Tm =  np.interp(kbis,k,Tm)
	Tcb =  np.interp(kbis,k,Tcb)


	
####################################################################
##### compute linear bias and error
	
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
#### compute pt terms

	A, B, C, D, E, F, G, H  = pt_terms(kbis, Plinbis)
	
####################################################################
#### get fitted coefficients


	biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, k, lb1, lb2, lb3, lb4,\
	errlb1, errlb2, errlb3, errlb4, kbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis,Mnu, z, j)
	#~ biasF1, biasF2, biasF3, biasF4 = poly(kstop, k, lb1, lb2, lb3, lb4,\
	#~ errlb1, errlb2, errlb3, errlb4, kbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis,Mnu, z, j)
	
	#~ bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1, bias3PTbis2, bias3PTbis3,\
	#~ bias3PTbis4 = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
	#~ bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j)
		
######################################################################
### mean of mass bins

	#~ B1 = np.array([bias2PT1/bias1bis, bias2PT2/bias2bis, bias2PT3/bias3bis, bias2PT4/bias4bis])
	#~ B1bis = np.array([bias3PT1/bias1bis, bias3PT2/bias2bis, bias3PT3/bias3bis, bias3PT4/bias4bis])
	#~ B1ter = np.array([bias3PTbis1/bias1bis, bias3PTbis2/bias2bis, bias3PTbis3/bias3bis, bias3PTbis4/bias4bis])
	B2 = np.array([bias1bis/bias1bis, bias2bis/bias2bis, bias3bis/bias3bis, bias4bis/bias4bis])
	B3 = np.array([biasF1/bias1bis, biasF2/bias2bis, biasF3/bias3bis, biasF4/bias4bis])
	B3bis = np.array([biasF1bis/bias1bis, biasF2bis/bias2bis, biasF3bis/bias3bis, biasF4bis/bias4bis])
	#~ b1 = np.mean(B1,axis=0)
	#~ b1bis = np.mean(B1bis,axis=0)
	#~ b1ter = np.mean(B1ter,axis=0)
	b2 = np.mean(B2,axis=0)
	b3 = np.mean(B3,axis=0)
	b3bis = np.mean(B3bis,axis=0)
	

	#####################################################################
	#####################################################################
	### read 0.0ev coeff to use rule of 3 for massive neutrinos
	nv0 = 0.0
	#~ m1pl, m2pl, m3pl, m4pl = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(nv0)+'eV/case'+str(case)+'/coeff_pl_'+str(nv0)+'_z='+str(z[j])+'.txt')
	#~ m1plbis, m2plbis, m3plbis, m4plbis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(nv0)+'eV/case'+str(case)+'/coeff_ple_'+str(nv0)+'_z='+str(z[j])+'.txt')
	#~ m1pt2, m2pt2, m3pt2, m4pt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(nv0)+'eV/case'+str(case)+'/coeff_2exp_'+str(nv0)+'_z='+str(z[j])+'.txt')
	#~ m1pt3, m2pt3, m3pt3, m4pt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(nv0)+'eV/case'+str(case)+'/coeff_3exp_'+str(nv0)+'_z='+str(z[j])+'.txt')
	#~ m1pt3bis, m2pt3bis, m3pt3bis, m4pt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(nv0)+'eV/case'+str(case)+'/coeff_3exp_fixed_'+str(nv0)+'_z='+str(z[j])+'.txt')

	####################################################################
	#### get results from mcmc analysis to plots the different biases
	#~ if Mnu == 0.15:
		# power law odd ----------------------------------------------------------------------------
		#~ bF1 = (m1pl[0] + m1pl[1] * kbis**2 + m1pl[2] * kbis**3 + m1pl[3] * kbis**4)*Bias_eff_t1/Bias_eff0_t1
		#~ bF2 = (m2pl[0] + m2pl[1] * kbis**2 + m2pl[2] * kbis**3 + m2pl[3] * kbis**4)*Bias_eff_t2/Bias_eff0_t2
		#~ bF3 = (m3pl[0] + m3pl[1] * kbis**2 + m3pl[2] * kbis**3 + m3pl[3] * kbis**4)*Bias_eff_t3/Bias_eff0_t3
		#~ bF4 = (m4pl[0] + m4pl[1] * kbis**2 + m4pl[2] * kbis**3 + m4pl[3] * kbis**4)*Bias_eff_t4/Bias_eff0_t4
		#~ bF1 = (m1pl[0] + m1pl[1] * kbis**2 + m1pl[2] * kbis**3 + m1pl[3] * kbis**4)*Bias_eff_t1/Bias_eff0_t1
		#~ bF2 = (m2pl[0] + m2pl[1] * kbis**2 + m2pl[2] * kbis**3 + m2pl[3] * kbis**4)*Bias_eff_t2/Bias_eff0_t2
		#~ bF3 = (m3pl[0] + m3pl[1] * kbis**2 + m3pl[2] * kbis**3 + m3pl[3] * kbis**4)*Bias_eff_t3/Bias_eff0_t3
		#~ bF4 = (m4pl[0] + m4pl[1] * kbis**2 + m4pl[2] * kbis**3 + m4pl[3] * kbis**4)*Bias_eff_t4/Bias_eff0_t4
		#~ # power law odd ----------------------------------------------------------------------------
		#~ biasF1 = b1x1_mcmc[0] + b2x1_mcmc[0] * kbis**2 + b3x1_mcmc[0] * kbis**3 + b4x1_mcmc[0] * kbis**4
		#~ biasF2 = b1x2_mcmc[0] + b2x2_mcmc[1] * kbis**2 + b3x2_mcmc[0] * kbis**3 + b4x2_mcmc[0] * kbis**4
		#~ biasF3 = b1x3_mcmc[0] + b2x3_mcmc[1] * kbis**2 + b3x3_mcmc[0] * kbis**3 + b4x3_mcmc[0] * kbis**4
		#~ biasF4 = b1x4_mcmc[0] + b2x4_mcmc[1] * kbis**2 + b3x4_mcmc[0] * kbis**3 + b4x4_mcmc[0] * kbis**4
		
		# power law even -------------------------------------------------------------------------------------------
		#~ biasF1bis = b1w1_mcmc[0] + b2w1_mcmc[0] * kbis**2 + b4w1_mcmc[0] * kbis**4
		#~ biasF2bis = b1w2_mcmc[0] + b2w2_mcmc[0] * kbis**2 + b4w2_mcmc[0] * kbis**4
		#~ biasF3bis = b1w3_mcmc[0] + b2w3_mcmc[0] * kbis**2 + b4w3_mcmc[0] * kbis**4
		#~ biasF4bis = b1w4_mcmc[0] + b2w4_mcmc[0] * kbis**2 + b4w4_mcmc[0] * kbis**4
		
		# 2nd order ------------------------------------------------------------------ 
		#~ bias2PT1 = np.sqrt((b1y1_mcmc[0]**2 * Pmmbis+ b1y1_mcmc[0]*b2y1_mcmc[0]*A + 1/4.*b2y1_mcmc[0]**2*B + b1y1_mcmc[0]*bsy1_mcmc[0]*C +\
		#~ 1/2.*b2y1_mcmc[0]*bsy1_mcmc[0]*D + 1/4.*bsy1_mcmc[0]**2*E )/Pmmbis)
		#~ bias2PT2 = np.sqrt((b1y2_mcmc[0]**2 * Pmmbis+ b1y2_mcmc[0]*b2y2_mcmc[0]*A + 1/4.*b2y2_mcmc[0]**2*B + b1y2_mcmc[0]*bsy2_mcmc[0]*C +\
		#~ 1/2.*b2y2_mcmc[0]*bsy2_mcmc[0]*D + 1/4.*bsy2_mcmc[0]**2*E )/Pmmbis)
		#~ bias2PT3 = np.sqrt((b1y3_mcmc[0]**2 * Pmmbis+ b1y3_mcmc[0]*b2y3_mcmc[0]*A + 1/4.*b2y3_mcmc[0]**2*B + b1y3_mcmc[0]*bsy3_mcmc[0]*C +\
		#~ 1/2.*b2y3_mcmc[0]*bsy3_mcmc[0]*D + 1/4.*bsy3_mcmc[0]**2*E )/Pmmbis)
		#~ bias2PT4 = np.sqrt((b1y4_mcmc[0]**2 * Pmmbis+ b1y4_mcmc[0]*b2y4_mcmc[0]*A + 1/4.*b2y4_mcmc[0]**2*B + b1y4_mcmc[0]*bsy4_mcmc[0]*C +\
		#~ 1/2.*b2y4_mcmc[0]*bsy4_mcmc[0]*D + 1/4.*bsy4_mcmc[0]**2*E )/Pmmbis)
		
		#~ # 3rd order free -------------------------------------------------------------------
		#~ bias3PT1 = np.sqrt((b1z1_mcmc[0]**2 * Pmmbis+ b1z1_mcmc[0]*b2z1_mcmc[0]*A + 1/4.*b2z1_mcmc[0]**2*B + b1z1_mcmc[0]*bsz1_mcmc[0]*C +\
		#~ 1/2.*b2z1_mcmc[0]*bsz1_mcmc[0]*D + 1/4.*bsz1_mcmc[0]**2*E + 2*b1z1_mcmc[0]*b3z1_mcmc[0]*F)/Pmmbis)
		#~ bias3PT2 = np.sqrt((b1z2_mcmc[0]**2 * Pmmbis+ b1z2_mcmc[0]*b2z2_mcmc[0]*A + 1/4.*b2z2_mcmc[0]**2*B + b1z2_mcmc[0]*bsz2_mcmc[0]*C +\
		#~ 1/2.*b2z2_mcmc[0]*bsz2_mcmc[0]*D + 1/4.*bsz2_mcmc[0]**2*E + 2*b1z2_mcmc[0]*b3z2_mcmc[0]*F)/Pmmbis)
		#~ bias3PT3 = np.sqrt((b1z3_mcmc[0]**2 * Pmmbis+ b1z3_mcmc[0]*b2z3_mcmc[0]*A + 1/4.*b2z3_mcmc[0]**2*B + b1z3_mcmc[0]*bsz3_mcmc[0]*C +\
		#~ 1/2.*b2z3_mcmc[0]*bsz3_mcmc[0]*D + 1/4.*bsz3_mcmc[0]**2*E + 2*b1z3_mcmc[0]*b3z3_mcmc[0]*F)/Pmmbis)
		#~ bias3PT4 = np.sqrt((b1z4_mcmc[0]**2 * Pmmbis+ b1z4_mcmc[0]*b2z4_mcmc[0]*A + 1/4.*b2z4_mcmc[0]**2*B + b1z4_mcmc[0]*bsz4_mcmc[0]*C +\
		#~ 1/2.*b2z4_mcmc[0]*bsz4_mcmc[0]*D + 1/4.*bsz4_mcmc[0]**2*E + 2*b1z4_mcmc[0]*b3z4_mcmc[0]*F)/Pmmbis)
		#~ # 3rd order fixed --------------------------------------------------------------------------------
		#~ B3nlTa = 32/315.*(b1u1_mcmc[0]-1)
		#~ B3nlTb = 32/315.*(b1u2_mcmc[0]-1)
		#~ B3nlTc = 32/315.*(b1u3_mcmc[0]-1)
		#~ B3nlTd = 32/315.*(b1u4_mcmc[0]-1)
		#~ bias3PTbis1 = np.sqrt((b1u1_mcmc[0]**2 * Pmmbis+ b1u1_mcmc[0]*b2u1_mcmc[0]*A + 1/4.*b2u1_mcmc[0]**2*B + b1u1_mcmc[0]*bsu1_mcmc[0]*C +\
		#~ 1/2.*b2u1_mcmc[0]*bsu1_mcmc[0]*D + 1/4.*bsu1_mcmc[0]**2*E + 2*b1u1_mcmc[0]*b3u1_mcmc[0]*F)/Pmmbis)
		#~ bias3PTbis2 = np.sqrt((b1u2_mcmc[0]**2 * Pmmbis+ b1u2_mcmc[0]*b2u2_mcmc[0]*A + 1/4.*b2u2_mcmc[0]**2*B + b1u2_mcmc[0]*bsu2_mcmc[0]*C +\
		#~ 1/2.*b2u2_mcmc[0]*bsu2_mcmc[0]*D + 1/4.*bsu2_mcmc[0]**2*E + 2*b1u2_mcmc[0]*b3u2_mcmc[0]*F)/Pmmbis)
		#~ bias3PTbis3 = np.sqrt((b1u3_mcmc[0]**2 * Pmmbis+ b1u3_mcmc[0]*b2u3_mcmc[0]*A + 1/4.*b2u3_mcmc[0]**2*B + b1u3_mcmc[0]*bsu3_mcmc[0]*C +\
		#~ 1/2.*b2u3_mcmc[0]*bsu3_mcmc[0]*D + 1/4.*bsu3_mcmc[0]**2*E + 2*b1u3_mcmc[0]*b3u3_mcmc[0]*F)/Pmmbis)
		#~ bias3PTbis4 = np.sqrt((b1u4_mcmc[0]**2 * Pmmbis+ b1u4_mcmc[0]*b2u4_mcmc[0]*A + 1/4.*b2u4_mcmc[0]**2*B + b1u4_mcmc[0]*bsu4_mcmc[0]*C +\
		#~ 1/2.*b2u4_mcmc[0]*bsu4_mcmc[0]*D + 1/4.*bsu4_mcmc[0]**2*E + 2*b1u4_mcmc[0]*b3u4_mcmc[0]*F)/Pmmbis)
		
		
		### mean ####
		#~ Bb1 = np.array([b2PT1/bias1bis, b2PT2/bias2bis, b2PT3/bias3bis, b2PT4/bias4bis])
		#~ Bb1bis = np.array([b3PT1/bias1bis, b3PT2/bias2bis, b3PT3/bias3bis, b3PT4/bias4bis])
		#~ Bb1ter = np.array([b3PTbis1/bias1bis, b3PTbis2/bias2bis, b3PTbis3/bias3bis, b3PTbis4/bias4bis])
		#~ Bb3 = np.array([bF1/bias1bis, bF2/bias2bis, bF3/bias3bis, bF4/bias4bis])
		#~ Bb3bis = np.array([bF1bis/bias1bis, bF2bis/bias2bis, bF3bis/bias3bis, bF4bis/bias4bis])
		#~ bb1 = np.mean(Bb1,axis=0)
		#~ bb1bis = np.mean(Bb1bis,axis=0)
		#~ bb1ter = np.mean(Bb1ter,axis=0)
		#~ bb3 = np.mean(Bb3,axis=0)
		#~ bb3bis = np.mean(Bb3bis,axis=0)
	######################################################################################
	######################################################################################

	#~ plt.figure()
	#~ plt.plot(kbis,kbis**1.5*F, color='C3', label=r'$\sigma_{3}^{2}(k) P^{lin}$')
	#~ plt.plot(kbis,kbis**1.5*Pmod_dd, color='k', label=r'$P_{\delta\delta}$')
	#~ plt.plot(kbis,kbis**1.5*A, color='C0', linestyle=':' , label=r'$P_{b2,\delta}$')
	#~ plt.plot(kbis,kbis**1.5*G, color='C1', linestyle=':' , label=r'$P_{b2,\theta}$')
	#~ plt.plot(kbis,kbis**1.5*C, color='C2', linestyle='--', label=r'$P_{bs2,\delta}$')
	#~ plt.legend(loc='upper left', ncol=2, fancybox=True)
	#~ plt.xlim(0.01,0.2)
	#~ plt.xlabel('k [h/Mpc]')
	#~ plt.ylabel(r'$k^{1.5} \times P(k)$ [(Mpc/h)]')
	#~ plt.xscale('log')
	#~ plt.ylim(-50,250)
	#~ plt.show()
	
	
	#~ kill
	####################################################################
	#~ PsptD1r1 = b1y1_mcmc[0]**2 * Pmmbis+ b1y1_mcmc[0]*b2y1_mcmc[0]*A + 1/4.*b2y1_mcmc[0]**2*B + b1y1_mcmc[0]*bsy1_mcmc[0]*C +\
	#~ 1/2.*b2y1_mcmc[0]*bsy1_mcmc[0]*D + 1/4.*bsy1_mcmc[0]**2*E
	#~ #------------------------------------------------------
	#~ PsptD2r1 = b1z1_mcmc[0]**2 * Pmmbis+ b1z1_mcmc[0]*b2z1_mcmc[0]*A + 1/4.*b2z1_mcmc[0]**2*B + b1z1_mcmc[0]*bsz1_mcmc[0]*C +\
	#~ 1/2.*b2z1_mcmc[0]*bsz1_mcmc[0]*D + 1/4.*bsz1_mcmc[0]**2*E + 2*b1z1_mcmc[0]*b3z1_mcmc[0]*F
	#~ #------------------------------------------------------
	#~ PsptD3r1 = b1u4_mcmc[0]**2 * Pmmbis+ b1u4_mcmc[0]*b2u4_mcmc[0]*A + 1/4.*b2u4_mcmc[0]**2*B + b1u4_mcmc[0]*bsu4_mcmc[0]*C +\
	#~ 1/2.*b2u4_mcmc[0]*bsu4_mcmc[0]*D + 1/4.*bsu4_mcmc[0]**2*E + 2*b1u4_mcmc[0]*B3nlTd*F
	
	####################################################################
	##### different fit
	####################################################################
	
	
	#~ plt.figure()
	#~ plt.plot(kbis, bias1bis)
	#~ plt.plot(kbis, bias2bis)
	#~ plt.plot(kbis, bias3bis)
	#~ plt.plot(kbis, bias4bis)
	#~ plt.plot(kbis, biasF1, color='C0', linestyle='--')
	#~ plt.plot(kbis, biasF2, color='C1', linestyle='--')
	#~ plt.plot(kbis, biasF3, color='C2', linestyle='--')
	#~ plt.plot(kbis, biasF4, color='C3', linestyle='--')
	#~ plt.plot(kbis, bF1, color='C0', linestyle=':')
	#~ plt.plot(kbis, bF2, color='C1', linestyle=':')
	#~ plt.plot(kbis, bF3, color='C2', linestyle=':')
	#~ plt.plot(kbis, bF4, color='C3', linestyle=':')
	#~ plt.ylim(bias1bis[0]*0.8,bias4bis[0]*1.2)
	#~ plt.xscale('log')
	#~ plt.xlim(0.008,1)
	#~ plt.show()
	
	
	
	#~ test1 = np.loadtxt('/home/david/b3nl1.txt') 
	#~ test2 = np.loadtxt('/home/david/b3nl2.txt') 
	#~ test3 = np.loadtxt('/home/david/b3nl3.txt') 
	#~ test4 = np.loadtxt('/home/david/b3nl4.txt') 
	#~ b1test1 = test1[:,1]
	#~ b3test1 = test1[:,4]
	#~ b1test2 = test2[:,1]
	#~ b3test2 = test2[:,4]
	#~ b1test3 = test3[:,1]
	#~ b3test3 = test3[:,4]
	#~ b1test4 = test4[:,1]
	#~ b3test4 = test4[:,4]
	#~ ktest = test1[:,0]
	#~ ktest2 = test3[:,0]
	#~ bins = np.logspace(np.log10(0.03),np.log10(0.2),20)
	#~ inds = np.digitize(ktest, bins)
	
	#~ print inds
	#~ mb1a = np.zeros(len(bins))
	#~ mb1b = np.zeros(len(bins))
	#~ mb1c = np.zeros(len(bins))
	#~ mb1d = np.zeros(len(bins))
	#~ mb3a = np.zeros(len(bins))
	#~ mb3b = np.zeros(len(bins))
	#~ mb3c = np.zeros(len(bins))
	#~ mb3d = np.zeros(len(bins))
	
	#~ for ind in xrange(1,len(bins)):
		#~ mb1a[ind-1] = np.mean(b1test1[np.where(inds == ind)[0]])
		#~ mb1b[ind-1] = np.mean(b1test2[np.where(inds == ind)[0]])
		#~ mb1a[ind-1] = np.mean(b1test3[np.where(inds == ind)[0]])
		#~ mb1a[ind-1] = np.mean(b1test4[np.where(inds == ind)[0]])
		#~ mb3a[ind-1] = np.mean(b3test1[np.where(inds == ind)[0]])
		#~ mb3b[ind-1] = np.mean(b3test2[np.where(inds == ind)[0]])
		#~ mb3c[ind-1] = np.mean(b3test3[np.where(inds == ind)[0]])
		#~ mb3d[ind-1] = np.mean(b3test4[np.where(inds == ind)[0]])

	#~ dM=binedge[1:]-binedge[:-1] #size of the bin
	#~ M_middle=10**(0.5*(np.log10(binedge[1:])+np.log10(binedge[:-1]))) #center of the bin
	#~ np.loadtxt('/home/david/errl.txt', 'a') 
	#~ np.loadtxt('/home/david/errh.txt', 'a') 
		

	#### compare the third order influence
	#~ plt.figure()
	#-----------------------------
	#~ plt.ylabel(r'$2b_{1}*b_{3nl}*\sigma_{3}^{2}*P^{lin}$ ', size=10)
	#~ plt.ylabel(r'$b_{3nl}$ / $(b_{1} - 1)$ ', size=10)
	#~ plt.xlabel(r'$k$ [h/Mpc] ', size=10)
	#~ plt.plot(kbis,2.*M1pt3[0]*M1pt3[3]*F, label=r'3rd order correction with free b3nl', color='C2', linestyle ='--' )
	#~ plt.plot(kbis,2.*M1pt3bis[0]*B3nlTa*F, label=r'3rd order correction with fixed b3nl', color='C3', linestyle='--' )
	#~ plt.scatter(bins,mb3a/(mb1a-1), label=r'3rd order correction with fixed b3nl', color='C0', marker='.' )
	#~ plt.scatter(bins,mb3b/(mb1b-1), label=r'3rd order correction with fixed b3nl', color='C1', marker='.' )
	#~ plt.scatter(bins,mb3c/(mb1c-1), label=r'3rd order correction with fixed b3nl', color='C2', marker='.' )
	#~ plt.scatter(bins,mb3d/(mb1d-1), label=r'3rd order correction with fixed b3nl', color='C3', marker='.' )
	#~ plt.scatter(ktest,b3test1/(b1test1-1), label=r'$M_1$', color='C0', marker='.' )
	#~ plt.scatter(ktest,b3test2/(b1test2-1), label=r'$M_2$', color='C1', marker='.' )
	#~ plt.scatter(ktest,b3test3/(b1test3-1), label=r'$M_3$', color='C2', marker='.' )
	#~ plt.scatter(ktest,b3test4/(b1test4-1), label=r'$M_4$', color='C3', marker='.' )
	#~ plt.axhline(32/315., label=r'local Lagrangian bias', color='k', linestyle='--' )
	#~ plt.ylim(-400,0)
	#~ plt.ylim(0,2)
	#~ plt.xlim(0.03,0.2)
	#~ plt.legend(loc='upper right') 
	#----------------------------
	#~ plt.yscale('log')
	#~ plt.xlabel(r'$k$ [h/Mpc] ', size=10)
	#~ plt.ylabel(r'P(k) ', size=10)
	#~ plt.plot(kbis,PH1bis,label='N-body', color='k')
	#~ plt.fill_between(kbis,PH1bis-errPhh1bis, PH1bis+errPhh1bis, alpha=0.6,color='k')
	#~ plt.plot(kbis,PsptD1r1, color='C1', label='2nd order expansion')
	#~ plt.plot(kbis,PsptD2r1,  color='C2',label=r'3rd order expansion with free $b_{3nl}$')
	#~ plt.plot(kbis,PsptD3r1, color='C3',label=r'3rd order expansion with fixed $b_{3nl}$')
	#~ plt.ylim(1e1,1e5)
	#~ plt.xlim(0.008,1)
	#----------------------------
	#~ plt.plot(k,Pmod_dd,label=r'$ \delta \delta $')
	#~ plt.plot(k,Pmod_dt,label=r'$ \delta \theta $')
	#~ plt.plot(k,Pmod_tt,label=r'$ \theta \theta $')
	#~ plt.plot(k,P_spt_dd[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#~ plt.plot(k,P_spt_dd[2], label='A' )
	#~ plt.plot(k,P_spt_dd[3], label=r'B' )
	#~ plt.plot(k,P_spt_dd[4], label=r'C' )
	#~ plt.plot(k,P_spt_dd[5], label=r'D' )
	#~ plt.plot(k,P_spt_dd[6], label=r'E' )
	#~ plt.plot(k,P_spt_dt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#~ plt.plot(k,P_spt_tt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
	#--------------------------------------------
	#~ plt.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ plt.xscale('log')
	#~ plt.title('z = '+str(z[j])+', $k_{max}$ = 0.12, mass range M1' )
	#~ plt.legend(loc='lower left') 
	#~ plt.show()

	#~ kill

	
	####################################################################
	#######--------- mean and std of bias and ps ratio ------------#####
	if j == z[0]:
		fig2 = plt.figure()
	J = j + 1
	
	if len(z) == 1:
		ax2 = fig2.add_subplot(1, len(z), J)
	elif len(z) == 2:
		ax2 = fig2.add_subplot(1, 2, J)
	elif len(z) > 2:
		ax2 = fig2.add_subplot(2, 2, J)
	#~ ######### pl residuals comparison #################
	ax2.set_ylim(0.9,1.1)
	ax2.axhline(1, color='k', linestyle='--')
	ax2.axhline(1.01, color='k', linestyle=':')
	ax2.axhline(0.99, color='k', linestyle=':')
	B3, = ax2.plot(kbis, b3)
	B3bis, = ax2.plot(kbis, b3bis)
	B2, = ax2.plot(kbis, b2, label='z = '+str(z[j]), color='k')
	#~ ax2.plot(kbis, biasF1bis/bias1bis)
	#~ ax2.plot(kbis, biasF2bis/bias2bis)
	#~ ax2.plot(kbis, biasF3bis/bias3bis)
	#~ ax2.plot(kbis, biasF4bis/bias4bis)
	#~ plt.figlegend( (B3bis,B3), (r'$b_{cc} = b_{1} + b_{2}*k^{2} + b_{4}*k^{4}$ ',\
	#~ r'$b_{cc} = b_{1} + b_{2}*k^{2} + b_{3}*k^{3} + b_{4}*k^{4}$ '), \
	####### comparison bias and != models #############################
	#~ M1 = ax2.errorbar(kbis, bias1bis, yerr= errb1bis,fmt='.', label='z = '+str(z[j]))
	#~ M2 = ax2.errorbar(kbis, bias2bis, yerr= errb2bis,fmt='.')
	#~ M3 = ax2.errorbar(kbis, bias3bis, yerr= errb3bis,fmt='.')
	#~ M4 = ax2.errorbar(kbis, bias4bis, yerr= errb4bis,fmt='.')
	#~ ax2.set_ylim(bias1bis[0]*0.8,bias4bis[0]*1.2)
	#~ Plo, =ax2.plot(kbis, biasF1, color='k')
	#~ Ple, =ax2.plot(kbis, biasF1bis, color='k', linestyle='--')
	#~ pt2, =ax2.plot(kbis, bias2PT1, color='k', linestyle='--')
	#~ pt3, =ax2.plot(kbis, bias3PT1, color='k', linestyle=':')
	#~ pt3bis, =ax2.plot(kbis, bias3PTbis1, color='k', linestyle='-.')
	#~ #--------
	#~ ax2.plot(kbis, biasF2, color='k')
	#~ ax2.plot(kbis, biasF2bis, color='k', linestyle='--')
	#~ ax2.plot(kbis, bias2PT2, color='k', linestyle='--' )
	#~ ax2.plot(kbis, bias3PT2, color='k', linestyle=':')
	#~ ax2.plot(kbis, bias3PTbis2, color='k', linestyle='-.')
	#~ #--------
	#~ ax2.plot(kbis, biasF3, color='k')
	#~ ax2.plot(kbis, biasF3bis, color='k', linestyle='--')
	#~ ax2.plot(kbis, bias2PT3, color='k', linestyle='--' )
	#~ ax2.plot(kbis, bias3PT3, color='k', linestyle=':')
	#~ ax2.plot(kbis, bias3PTbis3, color='k', linestyle='-.')
	#~ #--------
	#~ ax2.plot(kbis, biasF4, color='k')
	#~ ax2.plot(kbis, biasF4bis, color='k', linestyle='--')
	#~ ax2.plot(kbis, bias2PT4, color='k', linestyle='--')
	#~ ax2.plot(kbis, bias3PT4, color='k', linestyle=':')
	#~ ax2.plot(kbis, bias3PTbis4, color='k', linestyle='-.')
	#~ #--------
	#~ plt.figlegend( (M1,M2,M3,M4,Plo, Ple), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'PL with odd k','PL without odd k'), \
	#~ plt.figlegend( (M1,M2,M3,M4,Plo, pt2, pt3, pt3bis), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'PL with odd k'\
	#~ ,'2nd order bias expansion', r'3rd order with free $b_{3nl}$', r'3rd order with fixed $b_{3nl}$'), \
	###### compare all power model residuals ##########################
	#~ ax2.set_ylim(0.9,1.1)
	#~ ax2.set_yticks(np.linspace(0.9,1.1,5))
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ B3, = ax2.plot(kbis, b3,label=r'w/ $b_{sim}$', color='C0')
	#~ B1, = ax2.plot(kbis, b1, color='C1')	
	#~ B1bis, = ax2.plot(kbis, b1bis, color='C2')
	#~ B1ter, = ax2.plot(kbis, b1ter,  color='C3')
	#~ B2, = ax2.plot(kbis, b2, color='k')
	
	#~ B3anal, = ax2.plot(kbis, bb3,label=r'w/ $b_{fiducial}$', color='C0',linestyle='--')
	#~ B1anal, = ax2.plot(kbis, bb1, color='C1',linestyle='--')
	#~ B1bisanal, = ax2.plot(kbis, bb1bis, color='C2',linestyle='--')
	#~ B1teranal, = ax2.plot(kbis, bb1ter,  color='C3',linestyle='--')
	
	#~ plt.figlegend( (B1,B1bis,B1ter,B2,B3), ('2nd order expansion',r'3rd order expansion with free $b_{3nl}$',\
	#~ r'3rd order expansion with fixed $b_{3nl}$', 'N-body','Power law '), \
	######################################
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case II ')
	ax2.axvspan(kstop, 7, alpha=0.2, color='grey')
	ax2.legend(loc = 'upper left', fancybox=True, fontsize=9)
	plt.rcParams.update({'font.size': 14})
	plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	ax2.set_xscale('log')
	if j == 0 :
		#~ ax2.tick_params(bottom='off', labelbottom='off',labelleft=True)
		ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$', fontsize = 14)
		ax2.set_ylabel(r'$b_{cc}$')
	if j == 1 :
		ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$')
		ax2.set_ylabel(r'$b_{cc}$')
		ax2.yaxis.set_label_position("right")
	if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$')
		ax2.set_ylabel(r'$b_{cc}$')
		ax2.set_xlabel('k [h/Mpc]')
	if j == 3 :
		ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		ax2.set_xlabel('k [h/Mpc]')
		ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$')
		ax2.set_ylabel(r'$b_{cc}$')
		ax2.yaxis.set_label_position("right")
	ax2.set_xlim(8e-3,1)
	if j == len(z) -1:
		plt.show()
	
		
	#~ kill
	
	
	####################################################################
	#### compute fcc with transfer function
	fcc = f[ind] * (Tm/ Tcb)

	####################################################################
	###### fit the Finger of God effect
	####################################################################
	
	####################################################################
	#### compute tns coefficeints given mcmc results
	#~ AB2_1,AB4_1,AB6_1,AB8_1 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M1pl[0] ,C_window=C_window)
	#~ AB2_2,AB4_2,AB6_2,AB8_2 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M2pl[0] ,C_window=C_window)
	#~ AB2_3,AB4_3,AB6_3,AB8_3 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M3pl[0] ,C_window=C_window)
	#~ AB2_4,AB4_4,AB6_4,AB8_4 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M4pl[0] ,C_window=C_window)
	
	#~ ab2_1,ab4_1,ab6_1,ab8_1 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m1pl[0]*(Bias_eff_t1/Bias_eff0_t1) ,C_window=C_window)
	#~ ab2_2,ab4_2,ab6_2,ab8_2 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m2pl[0]*(Bias_eff_t2/Bias_eff0_t2) ,C_window=C_window)
	#~ ab2_3,ab4_3,ab6_3,ab8_3 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m3pl[0]*(Bias_eff_t3/Bias_eff0_t3) ,C_window=C_window)
	#~ ab2_4,ab4_4,ab6_4,ab8_4 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m4pl[0]*(Bias_eff_t4/Bias_eff0_t4)  ,C_window=C_window)
	#~ #--------------------------------------------------------------------------------------
	#~ AB2bis_1,AB4bis_1,AB6bis_1,AB8bis_1 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M1pt3[0] ,C_window=C_window)
	#~ AB2bis_2,AB4bis_2,AB6bis_2,AB8bis_2 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M2pt3[0] ,C_window=C_window)
	#~ AB2bis_3,AB4bis_3,AB6bis_3,AB8bis_3 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M3pt3[0] ,C_window=C_window)
	#~ AB2bis_4,AB4bis_4,AB6bis_4,AB8bis_4 = fastpt.RSD_ABsum_components(Plinbis,f[ind],M4pt3[0] ,C_window=C_window)
	
	#~ ab2bis_1,ab4bis_1,ab6bis_1,ab8bis_1 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m1pt3[0]*(Bias_eff_t1/Bias_eff0_t1) ,C_window=C_window)
	#~ ab2bis_2,ab4bis_2,ab6bis_2,ab8bis_2 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m2pt3[0]*(Bias_eff_t2/Bias_eff0_t2) ,C_window=C_window)
	#~ ab2bis_3,ab4bis_3,ab6bis_3,ab8bis_3 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m3pt3[0]*(Bias_eff_t3/Bias_eff0_t3) ,C_window=C_window)
	#~ ab2bis_4,ab4bis_4,ab6bis_4,ab8bis_4 = fastpt.RSD_ABsum_components(Plinbis,f[ind],m4pt3[0]*(Bias_eff_t4/Bias_eff0_t4) ,C_window=C_window)
	
	#~ print M1pt3[0], m1pt3[0], M1pt3[0]/m1pt3[0]
	#~ print Bias_eff_t1/Bias_eff0_t1
	#~ kill
	
	#~ dat_file_path = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/large_scale/'\
		#~ 'LS_z='+str(z[j])+'_.txt'
	#~ f = np.loadtxt(dat_file_path)
	#~ bls1 = f[0]
	#~ bls2 = f[1]
	#~ bls3 = f[2]
	#~ bls4 = f[3]
	#~ AB2ter_1,AB4ter_1,AB6ter_1,AB8ter_1 = fastpt.RSD_ABsum_components(Plinbis,f[ind],bls1 ,C_window=C_window)
	#~ AB2ter_2,AB4ter_2,AB6ter_2,AB8ter_2 = fastpt.RSD_ABsum_components(Plinbis,f[ind],bls2 ,C_window=C_window)
	#~ AB2ter_3,AB4ter_3,AB6ter_3,AB8ter_3 = fastpt.RSD_ABsum_components(Plinbis,f[ind],bls3 ,C_window=C_window)
	#~ AB2ter_4,AB4ter_4,AB6ter_4,AB8ter_4 = fastpt.RSD_ABsum_components(Plinbis,f[ind],bls4 ,C_window=C_window)
	
	#~ #-------------------------------------------------------
	#~ cname1m1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pl1_z='+str(z[j])+'.txt'
	#~ cname1m2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pl2_z='+str(z[j])+'.txt'
	#~ cname1m3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pl3_z='+str(z[j])+'.txt'
	#~ cname1m4 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pl4_z='+str(z[j])+'.txt'
	#~ cname2m1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pt1_z='+str(z[j])+'.txt'
	#~ cname2m2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pt2_z='+str(z[j])+'.txt'
	#~ cname2m3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pt3_z='+str(z[j])+'.txt'
	#~ cname2m4 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_pt4_z='+str(z[j])+'.txt'
	#~ cname3m1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_lin1_z='+str(z[j])+'.txt'
	#~ cname3m2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_lin2_z='+str(z[j])+'.txt'
	#~ cname3m3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_lin3_z='+str(z[j])+'.txt'
	#~ cname3m4 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/TNS_coeff/'\
	#~ 'tns_lin4_z='+str(z[j])+'.txt'


	#~ with open(cname1m1, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2_1[index_k],AB4_1[index_k],AB6_1[index_k],AB8_1[index_k]))
	#~ with open(cname1m2, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2_2[index_k],AB4_2[index_k],AB6_2[index_k],AB8_2[index_k]))
	#~ with open(cname1m3, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2_3[index_k],AB4_3[index_k],AB6_3[index_k],AB8_3[index_k]))
	#~ with open(cname1m4, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2_4[index_k],AB4_4[index_k],AB6_4[index_k],AB8_4[index_k]))
	#~ with open(cname2m1, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2bis_1[index_k],AB4bis_1[index_k],AB6bis_1[index_k],AB8bis_1[index_k]))
	#~ with open(cname2m2, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2bis_2[index_k],AB4bis_2[index_k],AB6bis_2[index_k],AB8bis_2[index_k]))
	#~ with open(cname2m3, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2bis_3[index_k],AB4bis_3[index_k],AB6bis_3[index_k],AB8bis_3[index_k]))
	#~ with open(cname2m4, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2bis_4[index_k],AB4bis_4[index_k],AB6bis_4[index_k],AB8bis_4[index_k]))
	#~ with open(cname3m1, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2ter_1[index_k],AB4ter_1[index_k],AB6ter_1[index_k],AB8ter_1[index_k]))
	#~ with open(cname3m2, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2ter_2[index_k],AB4ter_2[index_k],AB6ter_2[index_k],AB8ter_2[index_k]))
	#~ with open(cname3m3, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2ter_3[index_k],AB4ter_3[index_k],AB6ter_3[index_k],AB8ter_3[index_k]))
	#~ with open(cname3m4, 'w') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (AB2ter_4[index_k],AB4ter_4[index_k],AB6ter_4[index_k],AB8ter_4[index_k]))



	#----------------------------------------------------------------
	#### compute mcmc coefficient of halo ps fit
	print 'kaiser'
	#~ biasK1 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, bF1, kbis, Pmono1bis, errPr1bis)
	#~ biasK2 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, bF2, kbis, Pmono2bis, errPr2bis)
	#~ biasK3 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, bF3, kbis, Pmono3bis, errPr3bis)
	#~ biasK4 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, bF4, kbis, Pmono4bis, errPr4bis)
	#~ cn1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdispkaibis_z='+str(z[j])+'.txt'
	#~ with open(cn1, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (biasK1[0][0],biasK2[0][0],biasK3[0][0],biasK4[0][0]))
	#~ fid_file.close()
	
	#~ bK1 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, biasF1, kbis, Pmono1bis, errPr1bis)
	#~ bK2 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, biasF2, kbis, Pmono2bis, errPr2bis)
	#~ bK3 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, biasF3, kbis, Pmono3bis, errPr3bis)
	#~ bK4 = coeffit_Kaiser(j, fcc, kstop,Pmmbis, biasF4, kbis, Pmono4bis, errPr4bis)
	#~ cn1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdispkai_z='+str(z[j])+'.txt'
	#~ with open(cn1, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (bK1[0][0],bK2[0][0],bK3[0][0],bK4[0][0]))
	#~ fid_file.close()
	#----------------------------------------------------------------------------------------
	print 'Scoccimaro'
	#~ bs1 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF1, kbis, Pmono1bis, errPr1bis)
	#~ bs2 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF2, kbis, Pmono2bis, errPr2bis)
	#~ bs3 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF3, kbis, Pmono3bis, errPr3bis)
	#~ bs4 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF4, kbis, Pmono4bis, errPr4bis)
	#~ cn2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdispscobis_z='+str(z[j])+'.txt'
	#~ with open(cn2, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (bs1[0][0],bs2[0][0],bs3[0][0],bs4[0][0]))
	#~ fid_file.close()
	
	#~ bsco1 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF1, kbis, Pmono1bis, errPr1bis)
	#~ bsco2 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF2, kbis, Pmono2bis, errPr2bis)
	#~ bsco3 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF3, kbis, Pmono3bis, errPr3bis)
	#~ bsco4 = coeffit_Scocci(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF4, kbis, Pmono4bis, errPr4bis)
	#~ cn2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdispsco_z='+str(z[j])+'.txt'
	#~ with open(cn2, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (bsco1[0][0],bsco2[0][0],bsco3[0][0],bsco4[0][0]))
	#~ fid_file.close()
	#----------------------------------------------------------------------------------------
	print 'Tns'
	#~ btns1 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF1, kbis, Pmono1bis, errPr1bis, AB2_1, AB4_1, AB6_1, AB8_1)
	#~ btns2 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF2, kbis, Pmono2bis, errPr2bis, AB2_2, AB4_2, AB6_2, AB8_2)
	#~ btns3 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF3, kbis, Pmono3bis, errPr3bis, AB2_3, AB4_3, AB6_3, AB8_3)
	#~ btns4 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, biasF4, kbis, Pmono4bis, errPr4bis, AB2_4, AB4_4, AB6_4, AB8_4)
	#~ cn3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdisptns_z='+str(z[j])+'.txt'
	#~ with open(cn3, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (btns1[0][0],btns2[0][0],btns3[0][0],btns4[0][0]))
	#~ fid_file.close()
	
	#~ bt1 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF1, kbis, Pmono1bis, errPr1bis, ab2_1, ab4_1, ab6_1, ab8_1)
	#~ bt2 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF2, kbis, Pmono2bis, errPr2bis, ab2_2, ab4_2, ab6_2, ab8_2)
	#~ bt3 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF3, kbis, Pmono3bis, errPr3bis, ab2_3, ab4_3, ab6_3, ab8_3)
	#~ bt4 = coeffit_TNS(j, fcc, kstop,Pmmbis,Pmod_dt, Pmod_tt, bF4, kbis, Pmono4bis, errPr4bis, ab2_4, ab4_4, ab6_4, ab8_4)
	#~ cn3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdisptnsbis_z='+str(z[j])+'.txt'
	#~ with open(cn3, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (bt1[0][0],bt2[0][0],bt3[0][0],bt4[0][0]))
	#~ fid_file.close()
	#~ #----------------------------------------------------------------------------------------
	print 'eTns'
	#~ betns1 = coeffit_eTNS(j, fcc, kstop, M1pt3[0], M1pt3[1], M1pt3[2], M1pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono1bis, errPr1bis, AB2bis_1, AB4bis_1,\
	#~ AB6bis_1, AB8bis_1)
	#~ betns2 = coeffit_eTNS(j, fcc, kstop, M2pt3[0], M2pt3[1], M2pt3[2], M2pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono2bis, errPr2bis, AB2bis_2, AB4bis_2,\
	#~ AB6bis_2, AB8bis_2)
	#~ betns3 = coeffit_eTNS(j, fcc, kstop, M3pt3[0], M3pt3[1], M3pt3[2], M3pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono3bis, errPr3bis, AB2bis_3, AB4bis_3,\
	#~ AB6bis_3, AB8bis_3)
	#~ betns4 = coeffit_eTNS(j, fcc, kstop, M4pt3[0], M4pt3[1], M4pt3[2], M4pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono4bis, errPr4bis, AB2bis_4, AB4bis_4,\
	#~ AB6bis_4, AB8bis_4)
	#~ cn4 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdispetns_z='+str(z[j])+'.txt'
	#~ with open(cn4, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (betns1[0][0],betns2[0][0],betns3[0][0],betns4[0][0]))
	#~ fid_file.close()
	
	#~ be1 = coeffit_eTNS(j, fcc, kstop, m1pt3[0], m1pt3[1], m1pt3[2], m1pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono1bis, errPr1bis, ab2bis_1, ab4bis_1,\
	#~ ab6bis_1, ab8bis_1, (Bias_eff_t1/Bias_eff0_t1) )
	#~ be2 = coeffit_eTNS(j, fcc, kstop, m2pt3[0], m2pt3[1], m2pt3[2], m2pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono2bis, errPr2bis, ab2bis_2, ab4bis_2,\
	#~ ab6bis_2, ab8bis_2, (Bias_eff_t2/Bias_eff0_t2) )
	#~ be3 = coeffit_eTNS(j, fcc, kstop, m3pt3[0], m3pt3[1], m3pt3[2], m3pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono3bis, errPr3bis, ab2bis_3, ab4bis_3,\
	#~ ab6bis_3, ab8bis_3, (Bias_eff_t3/Bias_eff0_t3) )
	#~ be4 = coeffit_eTNS(j, fcc, kstop, m4pt3[0], m4pt3[1], m4pt3[2], m4pt3[3], Pmmbis, Pmod_dt, Pmod_tt,\
	#~ A, B, C, D, E, F, G, H, kbis, Pmono4bis, errPr4bis, ab2bis_4, ab4bis_4,\
	#~ ab6bis_4, ab8bis_4, (Bias_eff_t4/Bias_eff0_t4) )
	#~ cn4 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/vdispetnsbis_z='+str(z[j])+'.txt'
	#~ with open(cn4, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (be1[0][0],be2[0][0],be3[0][0],be4[0][0]))
	#~ fid_file.close()


	#### compute the different power spectra given the mcmc results
	#~ bK = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdispkai_z='+str(z[j])+'.txt')
	#~ BK = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdispkaibis_z='+str(z[j])+'.txt')
	#~ biasK = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/v_disp/vdispkai_z='+str(z[j])+'.txt')
	def kaips(b,sigma):
		kappa = kbis*sigma*fcc[ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		return Pmmbis*(b**2*coeffA +  2/3.*b*f[ind]*coeffB + 1/5.*f[ind]**2*coeffC)
		#~ return Pmmbis*(b**2 +  2/3.*b*f[ind] + 1/5.*f[ind]**2)
		

	#~ kai1 = kaips(biasF1, bK[0])
	#~ kai2 = kaips(biasF2, bK[1])
	#~ kai3 = kaips(biasF3, bK[2])
	#~ kai4 = kaips(biasF4, bK[3])
	
	#~ k1 = kaips(bF1, BK[0])
	#~ k2 = kaips(bF2, BK[1])
	#~ k3 = kaips(bF3, BK[2])
	#~ k4 = kaips(bF4, BK[3])

	#~ k1ter = kaips(bF1, biasK[0])
	#~ k2ter = kaips(bF2, biasK[1])
	#~ k3ter = kaips(bF3, biasK[2])
	#~ k4ter = kaips(bF4, biasK[3])

	#---------------------------------------------------------------------------------------
	#~ bsco = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdispsco_z='+str(z[j])+'.txt')
	#~ Bsco = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdispscobis_z='+str(z[j])+'.txt')
	#~ bs = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/v_disp/vdispsco_z='+str(z[j])+'.txt')
	def scops(b,sigma):
		kappa = kbis*sigma*fcc[ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		return b**2*Pmmbis*coeffA + 2/3.*b*f[ind]*Pmod_dt*coeffB + 1/5.*f[ind]**2*Pmod_tt*coeffC
		#~ return b**2*Pmmbis + 2/3.*b*f[ind]*Pmod_dt + 1/5.*f[ind]**2*Pmod_tt

	#~ sco1 = scops(biasF1, bsco[0])
	#~ sco2 = scops(biasF2, bsco[1])
	#~ sco3 = scops(biasF3, bsco[2])
	#~ sco4 = scops(biasF4, bsco[3])
	
	#~ s1 = scops(bF1, Bsco[0])
	#~ s2 = scops(bF2, Bsco[1])
	#~ s3 = scops(bF3, Bsco[2])
	#~ s4 = scops(bF4, Bsco[3])
	
	#~ s1ter = scops(bF1, bs[0])
	#~ s2ter = scops(bF2, bs[1])
	#~ s3ter = scops(bF3, bs[2])
	#~ s4ter = scops(bF4, bs[3])
	#~ #---------------------------------------------------------------------------------------
	#~ btns = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdisptns_z='+str(z[j])+'.txt')
	#~ Btns = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdisptnsbis_z='+str(z[j])+'.txt')
	#~ bt = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/v_disp/vdisptns_z='+str(z[j])+'.txt')
	def tnsps(b,sigma, AB2, AB4, AB6, AB8):
		kappa = kbis*sigma*fcc[ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		coeffD = -2/3./kappa**2*(coeffC - 2/(2+kappa**2))
		coeffE = -4/10./kappa**2*(7.*coeffD - 2/(2+kappa**2))
		return b**2*Pmmbis*coeffA + 2/3.*b*f[ind]*Pmod_dt*coeffB + 1/5.*f[ind]**2*Pmod_tt*coeffC \
		+ (1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE)
		#~ return b**2*Pmmbis + 2/3.*b*f[ind]*Pmod_dt + 1/5.*f[ind]**2*Pmod_tt \
		#~ + (1/3.*AB2+ 1/5.*AB4+ 1/7.*AB6+ 1/9.*AB8)

	#~ tns1 = tnsps(biasF1,btns[0], AB2_1, AB4_1, AB6_1, AB8_1)
	#~ tns2 = tnsps(biasF2,btns[1], AB2_2, AB4_2, AB6_2, AB8_2)
	#~ tns3 = tnsps(biasF3,btns[2], AB2_3, AB4_3, AB6_3, AB8_3)
	#~ tns4 = tnsps(biasF4,btns[3], AB2_4, AB4_4, AB6_4, AB8_4)
	
	#~ t1 = tnsps(bF1,Btns[0], ab2_1, ab4_1, ab6_1, ab8_1)
	#~ t2 = tnsps(bF2,Btns[1], ab2_2, ab4_2, ab6_2, ab8_2)
	#~ t3 = tnsps(bF3,Btns[2], ab2_3, ab4_3, ab6_3, ab8_3)
	#~ t4 = tnsps(bF4,Btns[3], ab2_4, ab4_4, ab6_4, ab8_4)
	
	#~ t1ter = tnsps(bF1,bt[0], ab2_1, ab4_1, ab6_1, ab8_1)
	#~ t2ter = tnsps(bF2,bt[1], ab2_2, ab4_2, ab6_2, ab8_2)
	#~ t3ter = tnsps(bF3,bt[2], ab2_3, ab4_3, ab6_3, ab8_3)
	#~ t4ter = tnsps(bF4,bt[3], ab2_4, ab4_4, ab6_4, ab8_4)
	#-------------------------------------------------------------------
	#~ betns = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdispetns_z='+str(z[j])+'.txt')
	#~ Betns = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/v_disp/vdispetnsbis_z='+str(z[j])+'.txt')
	#~ be = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/v_disp/vdispetns_z='+str(z[j])+'.txt')
	def etnsps(b1,b2,bs,b3nl,sigma, AB2, AB4, AB6, AB8, sca=None):
		PsptD1z = b1**2*Pmmbis + b1*b2*A+ 1/4.*b2**2*B+ b1*bs*C+ 1/2.*b2*bs*D+ 1/4.*bs**2*E+ 2*b1*b3nl*F
		PsptT = b1* Pmod_dt+ b2*G+ bs*H + b3nl*F
		#~ kappa = kbis*sigma
		#~ coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		#~ coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		#~ coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		#~ coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		#~ coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		kappa = kbis*sigma*fcc[ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		coeffD = -2/3./kappa**2*(coeffC - 2/(2+kappa**2))
		coeffE = -4/10./kappa**2*(7.*coeffD - 2/(2+kappa**2))
		if sca:
			return  PsptD1z*coeffA*sca**2 + 2/3.*f[ind]*PsptT*coeffB*sca + 1/5.*f[ind]**2*Pmod_tt*coeffC \
			+ (1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE)
		else:
			return  PsptD1z*coeffA + 2/3.*f[ind]*PsptT*coeffB + 1/5.*f[ind]**2*Pmod_tt*coeffC \
			+ (1/3.*AB2*coeffB+ 1/5.*AB4*coeffC+ 1/7.*AB6*coeffD+ 1/9.*AB8*coeffE)
		#~ return  PsptD1z + 2/3.*f[ind]*PsptT + 1/5.*f[ind]**2*Pmod_tt \
		#~ + (1/3.*AB2+ 1/5.*AB4+ 1/7.*AB6+ 1/9.*AB8) 
		
	#~ etns1 = etnsps(M1pt3[0], M1pt3[1], M1pt3[2], M1pt3[3], betns[0], AB2bis_1, AB4bis_1, AB6bis_1, AB8bis_1)  
	#~ etns2 = etnsps(M2pt3[0], M2pt3[1], M2pt3[2], M2pt3[3], betns[1], AB2bis_2, AB4bis_2, AB6bis_2, AB8bis_2)  
	#~ etns3 = etnsps(M3pt3[0], M3pt3[1], M3pt3[2], M3pt3[3], betns[2], AB2bis_3, AB4bis_3, AB6bis_3, AB8bis_3)  
	#~ etns4 = etnsps(M4pt3[0], M4pt3[1], M4pt3[2], M4pt3[3], betns[3], AB2bis_4, AB4bis_4, AB6bis_4, AB8bis_4) 
	 
	#~ e1 = etnsps(m1pt3[0], m1pt3[1], m1pt3[2], m1pt3[3], Betns[0], ab2bis_1, ab4bis_1, ab6bis_1, ab8bis_1,(Bias_eff_t1/Bias_eff0_t1))  
	#~ e2 = etnsps(m2pt3[0], m2pt3[1], m2pt3[2], m2pt3[3], Betns[1], ab2bis_2, ab4bis_2, ab6bis_2, ab8bis_2,(Bias_eff_t2/Bias_eff0_t2))  
	#~ e3 = etnsps(m3pt3[0], m3pt3[1], m3pt3[2], m3pt3[3], Betns[2], ab2bis_3, ab4bis_3, ab6bis_3, ab8bis_3,(Bias_eff_t3/Bias_eff0_t3))  
	#~ e4 = etnsps(m4pt3[0], m4pt3[1], m4pt3[2], m4pt3[3], Betns[3], ab2bis_4, ab4bis_4, ab6bis_4, ab8bis_4,(Bias_eff_t4/Bias_eff0_t4))
	  
	#~ e1ter = etnsps(m1pt3[0], m1pt3[1], m1pt3[2], m1pt3[3], be[0], ab2bis_1, ab4bis_1, ab6bis_1, ab8bis_1)  
	#~ e2ter = etnsps(m2pt3[0], m2pt3[1], m2pt3[2], m2pt3[3], be[1], ab2bis_2, ab4bis_2, ab6bis_2, ab8bis_2)  
	#~ e3ter = etnsps(m3pt3[0], m3pt3[1], m3pt3[2], m3pt3[3], be[2], ab2bis_3, ab4bis_3, ab6bis_3, ab8bis_3)  
	#~ e4ter = etnsps(m4pt3[0], m4pt3[1], m4pt3[2], m4pt3[3], be[3], ab2bis_4, ab4bis_4, ab6bis_4, ab8bis_4)  

 
	

	
	#~ plt.figure()
	#~ plt.plot(kbis, kai1/Pmono1bis, color='r')
	#~ plt.plot(kbis, sco1/Pmono1bis, color='b')
	#~ plt.plot(kbis, tns1/Pmono1bis, color='g')
	#~ plt.plot(kbis, etns1/Pmono1bis, color='c')
	#~ plt.plot(kbis, Pmono1bis, color='k')
	#~ plt.plot(kbis, Pmmbis, color='c')
	#~ plt.plot(kbis, kai1, color='r')
	#~ plt.plot(kbis, sco1, color='b')
	#~ plt.plot(kbis, tns1, color='g')
	#~ plt.plot(kbis, etns1, color='c')
	#~ plt.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ plt.axhline(1., color='k')
	#~ plt.axhline(1.01, color='k', linestyle='--')
	#~ plt.axhline(0.99, color='k', linestyle='--')
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(0.008,1.0)
	#~ plt.ylim(0.9,1.1)
	#~ plt.ylim(1e2,2e5)
	#~ plt.show()
	#~ kill
	

	#--------
	#~ plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1), title='z = '+str(z[j]), fancybox=True)
	#~ plt.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ plt.xlabel('k')
	#~ plt.ylabel('P(k)')
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(8e-3,0.4)
	#~ plt.ylim(1e3,2e5)
	#~ plt.show()
	#~ kill
	
	
	#~ ####################################################################
	#~ ### compute of the 4 mass bins
	#~ ####################################################################
	#~ p1 = np.array([Pmono1bis/Pmono1bis, Pmono2bis/Pmono2bis, Pmono3bis/Pmono3bis, Pmono4bis/Pmono4bis])
	#~ P1 = np.mean(p1, axis=0)
	#~ p2 = np.array([kai1/Pmono1bis, kai2/Pmono2bis, kai3/Pmono3bis, kai4/Pmono4bis])
	#~ P2 = np.mean(p2, axis=0)
	#~ p3 = np.array([sco1/Pmono1bis, sco2/Pmono2bis, sco3/Pmono3bis, sco4/Pmono4bis])
	#~ P3 = np.mean(p3, axis=0)
	#~ p4 = np.array([tns1/Pmono1bis, tns2/Pmono2bis, tns3/Pmono3bis, tns4/Pmono4bis])
	#~ P4 = np.mean(p4, axis=0)
	#~ p6 = np.array([etns1/Pmono1bis, etns2/Pmono2bis, etns3/Pmono3bis, etns4/Pmono4bis])
	#~ P6 = np.mean(p6, axis=0)
	
	#~ p2bis = np.array([k1/Pmono1bis, k2/Pmono2bis, k3/Pmono3bis, k4/Pmono4bis])
	#~ P2bis = np.mean(p2bis, axis=0)
	#~ p3bis = np.array([s1/Pmono1bis, s2/Pmono2bis, s3/Pmono3bis, s4/Pmono4bis])
	#~ P3bis = np.mean(p3bis, axis=0)
	#~ p4bis = np.array([t1/Pmono1bis, t2/Pmono2bis, t3/Pmono3bis, t4/Pmono4bis])
	#~ P4bis = np.mean(p4bis, axis=0)
	#~ p6bis = np.array([e1/Pmono1bis, e2/Pmono2bis, e3/Pmono3bis, e4/Pmono4bis])
	#~ P6bis = np.mean(p6bis, axis=0)
	
	#~ p2ter = np.array([k1ter/Pmono1bis, k2ter/Pmono2bis,	k3ter/Pmono3bis, k4ter/Pmono4bis])
	#~ P2ter = np.mean(p2ter, axis=0)
	#~ p3ter = np.array([s1ter/Pmono1bis, s2ter/Pmono2bis, s3ter/Pmono3bis, s4ter/Pmono4bis])
	#~ P3ter = np.mean(p3ter, axis=0)
	#~ p4ter = np.array([t1ter/Pmono1bis, t2ter/Pmono2bis, t3ter/Pmono3bis, t4ter/Pmono4bis])
	#~ P4ter = np.mean(p4ter, axis=0)
	#~ p6ter = np.array([e1ter*(Bias_eff_t1/Bias_eff0_t1)**2/Pmono1bis, e2ter*(Bias_eff_t2/Bias_eff0_t2)**2/Pmono2bis,\
	#~ e3ter*(Bias_eff_t3/Bias_eff0_t3)**2/Pmono3bis, e4ter*(Bias_eff_t4/Bias_eff0_t4)**2/Pmono4bis])
	#~ P6ter = np.mean(p6ter, axis=0)
	
	
	
	#######--------- mean and std of bias and ps ratio ------------#####
	#~ if j == z[0]:
		#~ fig2 = plt.figure()
	#~ J = j + 1
	
	#~ if len(z) == 1:
		#~ ax2 = fig2.add_subplot(1, len(z), J)
	#~ elif len(z) == 2:
		#~ ax2 = fig2.add_subplot(1, 2, J)
	#~ elif len(z) > 2:
		#~ ax2 = fig2.add_subplot(2, 2, J)
	########### power spectrum ########
	#~ ax2.set_ylim(0.9,1.1)
	#~ ax2.set_yticks(np.linspace(0.9,1.1,5))
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ P1, =ax2.plot(kbis,P1, color='k')
	#~ P2, =ax2.plot(kbis,P2, color='C3',label=r'w/ $b_{sim}$')
	#~ P3, =ax2.plot(kbis,P3, color='C0')
	#~ P4, =ax2.plot(kbis,P4, color='C1')
	#~ P6, =ax2.plot(kbis,P6, color='c')
	#~ #-------------------------------
	#~ ax2.plot(kbis,P2bis, color='C3', linestyle='--',label=r'w/ $b_{fiducial}$ and $\sigma_v$ free')
	#~ ax2.plot(kbis,P3bis, color='C0', linestyle='--')
	#~ ax2.plot(kbis,P4bis, color='C1', linestyle='--')
	#~ ax2.plot(kbis,P6bis, color='c', linestyle='--')
	#-------------------------------
	#~ ax2.plot(kbis,P2ter, color='C3', linestyle='--',label=r'w/ $b_{fiducial}$ and $\sigma_v$ fixed')
	#~ ax2.plot(kbis,P3ter, color='C0', linestyle='--')
	#~ ax2.plot(kbis,P4ter, color='C1', linestyle='--')
	#~ ax2.plot(kbis,P6ter, color='c', linestyle='--')
	
	#~ plt.figlegend( (P1,P2, P3, P4,P6), ('N-body','Power law + Kaiser','Power law + Scoccimarro','Power law + TNS','eTNS'), \
	####### comparison bias and != models #############################
	#~ ax2.set_yscale('log')
	#~ plt.ylim(2e2,3e5)
	#~ M1 = ax2.errorbar(kbis, Pmono1bis, yerr= errPr1bis,fmt='.', label='z = '+str(z[j]))
	#~ M2 = ax2.errorbar(kbis, Pmono2bis, yerr= errPr2bis,fmt='.')
	#~ M3 = ax2.errorbar(kbis, Pmono3bis, yerr= errPr3bis,fmt='.')
	#~ M4 = ax2.errorbar(kbis, Pmono4bis, yerr= errPr4bis,fmt='.')
	#~ nlk, = ax2.plot(kbis, kai1, color='k')
	#~ sco, = ax2.plot(kbis, sco1, color='k', linestyle='--')
	#~ tns, = ax2.plot(kbis, tns1, color='k', linestyle=':')
	#~ etns, = ax2.plot(kbis, etns1, color='k', linestyle='-.')
	#--------
	#~ ax2.plot(kbis, kai2, color='k')
	#~ ax2.plot(kbis, sco2, color='k', linestyle='--' )
	#~ ax2.plot(kbis, tns2, color='k', linestyle=':')
	#~ ax2.plot(kbis, etns2, color='k', linestyle='-.')
	#--------
	#~ ax2.plot(kbis, kai3, color='k')
	#~ ax2.plot(kbis, sco3, color='k', linestyle='--' )
	#~ ax2.plot(kbis, tns3, color='k', linestyle=':')
	#~ ax2.plot(kbis, etns3, color='k', linestyle='-.')
	#--------
	#~ ax2.plot(kbis, kai4, color='k')
	#~ ax2.plot(kbis, sco4, color='k', linestyle='--')
	#~ ax2.plot(kbis, tns4, color='k', linestyle=':')
	#~ ax2.plot(kbis, etns4, color='k', linestyle='-.')
	#~ #--------
	#~ plt.figlegend( (M1,M2,M3,M4,nlk,sco, tns, etns), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'non linear kaiser + PL'\
	#~ ,'Scoccimarro + PL', r'TNS + PL', r'eTNS'), \
	######################################
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case II ')
	#~ ax2.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, fontsize=9)
	#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#~ ax2.set_xscale('log')
	#~ if j == 0 :
		#~ ax2.tick_params(bottom='off', labelbottom='off')
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$')
		#~ ax2.set_ylabel(r'$P_{cc}$')
	#~ if j == 1 :
		#~ ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$')
		#~ ax2.set_ylabel(r'$P_{cc}$')
		#~ ax2.yaxis.set_label_position("right")
	#~ if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$')
		#~ ax2.set_ylabel(r'$P_{cc}$')
		#~ ax2.set_xlabel('k [h/Mpc]')
	#~ if j == 3 :
		#~ ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_xlabel('k [h/Mpc]')
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$')
		#~ ax2.set_ylabel(r'$P_{cc}$')
		#~ ax2.yaxis.set_label_position("right")
	#~ ax2.set_xlim(8e-3,1)
	#~ #plt.ylim(0.7,1.3)
	#~ if j == len(z) -1:
		#~ plt.show()
	
		
	#~ kill
	
	
	####################################################################
	##### compute the chi2 of different quantities
	####################################################################

	#~ # p is number of free param
	#~ F1 = (biasF1[lim]-bias1bis[lim])**2/errb1bis[lim]**2
	#~ F2 = (biasF2[lim]-bias2bis[lim])**2/errb2bis[lim]**2
	#~ F3 = (biasF3[lim]-bias3bis[lim])**2/errb3bis[lim]**2
	#~ F4 = (biasF4[lim]-bias4bis[lim])**2/errb4bis[lim]**2
	#~ chi2F1 = np.sum(F1)
	#~ chi2F2 = np.sum(F2)
	#~ chi2F3 = np.sum(F3)
	#~ chi2F4 = np.sum(F4)
	#~ #-------------------------------------------------

	#~ PT1 = (np.sqrt(PsptD1r1[lim]/Pmmbis[lim])- bias1bis[lim])**2/errb1bis[lim]**2
	#~ PT2 = (np.sqrt(PsptD1r2[lim]/Pmmbis[lim])- bias2bis[lim])**2/errb2bis[lim]**2
	#~ PT3 = (np.sqrt(PsptD1r3[lim]/Pmmbis[lim])- bias3bis[lim])**2/errb3bis[lim]**2
	#~ PT4 = (np.sqrt(PsptD1r4[lim]/Pmmbis[lim])- bias4bis[lim])**2/errb4bis[lim]**2
	#~ chi2PT1 = np.sum(PT1)
	#~ chi2PT2 = np.sum(PT2)
	#~ chi2PT3 = np.sum(PT3)
	#~ chi2PT4 = np.sum(PT4)
	#~ #-------------------------------------------------
	#~ PTbis1 = (np.sqrt(PsptD2r1[lim]/Pmmbis[lim])- bias1bis[lim])**2/errb1bis[lim]**2
	#~ PTbis2 = (np.sqrt(PsptD2r2[lim]/Pmmbis[lim])- bias2bis[lim])**2/errb2bis[lim]**2
	#~ PTbis3 = (np.sqrt(PsptD2r3[lim]/Pmmbis[lim])- bias3bis[lim])**2/errb3bis[lim]**2
	#~ PTbis4 = (np.sqrt(PsptD2r4[lim]/Pmmbis[lim])- bias4bis[lim])**2/errb4bis[lim]**2
	#~ chi2PTbis1 = np.sum(PTbis1)
	#~ chi2PTbis2 = np.sum(PTbis2)
	#~ chi2PTbis3 = np.sum(PTbis3)
	#~ chi2PTbis4 = np.sum(PTbis4)
	
	#~ cname = '/home/david/chi2_z='+str(z[j])+'.txt'
	#~ with open(cname, 'a+') as fid_file:

		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop,\
		#~ chi2F1, chi2F2, chi2F3, chi2F4, chi2PT1, chi2PT2, chi2PT3, chi2PT4, chi2PTbis1, chi2PTbis2, chi2PTbis3, chi2PTbis4))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (kstop, chi2F1, chi2PT1, chi2PTbis1))
	#~ print '\n'
	
	####################################################
	#### linear chi2
	#~ lim2 = np.where(kbis <= 0.1)[0]
	#~ # p is number of free param
	#~ F1bis = (biasF1[lim2]-bias1bis[lim2])**2/errb1bis[lim2]**2
	#~ F2bis = (biasF2[lim2]-bias2bis[lim2])**2/errb2bis[lim2]**2
	#~ F3bis = (biasF3[lim2]-bias3bis[lim2])**2/errb3bis[lim2]**2
	#~ F4bis = (biasF4[lim2]-bias4bis[lim2])**2/errb4bis[lim2]**2
	#~ chi2F1bis = np.sum(F1bis)
	#~ chi2F2bis = np.sum(F2bis)
	#~ chi2F3bis = np.sum(F3bis)
	#~ chi2F4bis = np.sum(F4bis)
	#~ #-------------------------------------------------

	#~ PT1bis = (np.sqrt(PsptD1r1[lim2]/Pmmbis[lim2])- bias1bis[lim2])**2/errb1bis[lim2]**2
	#~ PT2bis = (np.sqrt(PsptD1r2[lim2]/Pmmbis[lim2])- bias2bis[lim2])**2/errb2bis[lim2]**2
	#~ PT3bis = (np.sqrt(PsptD1r3[lim2]/Pmmbis[lim2])- bias3bis[lim2])**2/errb3bis[lim2]**2
	#~ PT4bis = (np.sqrt(PsptD1r4[lim2]/Pmmbis[lim2])- bias4bis[lim2])**2/errb4bis[lim2]**2
	#~ chi2PT1bis = np.sum(PT1bis)
	#~ chi2PT2bis = np.sum(PT2bis)
	#~ chi2PT3bis = np.sum(PT3bis)
	#~ chi2PT4bis = np.sum(PT4bis)
	#~ #-------------------------------------------------
	#~ PTbis1bis = (np.sqrt(PsptD2r1[lim2]/Pmmbis[lim2])- bias1bis[lim2])**2/errb1bis[lim2]**2
	#~ PTbis2bis = (np.sqrt(PsptD2r2[lim2]/Pmmbis[lim2])- bias2bis[lim2])**2/errb2bis[lim2]**2
	#~ PTbis3bis = (np.sqrt(PsptD2r3[lim2]/Pmmbis[lim2])- bias3bis[lim2])**2/errb3bis[lim2]**2
	#~ PTbis4bis = (np.sqrt(PsptD2r4[lim2]/Pmmbis[lim2])- bias4bis[lim2])**2/errb4bis[lim2]**2
	#~ chi2PTbis1bis = np.sum(PTbis1bis)
	#~ chi2PTbis2bis = np.sum(PTbis2bis)
	#~ chi2PTbis3bis = np.sum(PTbis3bis)
	#~ chi2PTbis4bis = np.sum(PTbis4bis)
	
	
	#~ cname = '/home/david/chi2bis_z='+str(z[j])+'.txt'
	#~ with open(cname, 'a+') as fid_file:

		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop,\
		#~ chi2F1bis, chi2F2bis, chi2F3bis, chi2F4bis, chi2PT1bis, chi2PT2bis, chi2PT3bis, chi2PT4bis,\
		#~ chi2PTbis1bis, chi2PTbis2bis, chi2PTbis3bis, chi2PTbis4bis))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (kstop, chi2F1bis, chi2PT1bis, chi2PTbis1bis))
	#~ print '\n'



	
	
	
	


	########################################################################
	############## plot ####################################################
	########################################################################
	col = ['b','r','g','k']
	
	
	
	
	
	


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
	#~ plt.title('3D monopole cdm power spectrum without FoG at z = '+str(z[j])+', nu = '+str(Mnu))
	#~ plt.title('3D quadrupole cdm power spectrum without FoG at z = '+str(z[j])+', nu = '+str(Mnu))
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
	#~ plt.title('3D power spectrum at z = '+str(z[j])+', nu = '+str(Mnu))
	#~ plt.title('3D halo power spectrum at z = '+str([j])+', nu = '+str(Mnu))
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
	#plt.title('bias z = '+str(z[j])+', nu = '+str(Mnu) )
	#plt.title('influence of bias at z = '+str(z[j])+', nu = '+str(Mnu)+', mass range '+str(mass_range) )
	#~ ax.legend(loc = 'upper right', handlelength=0, handletextpad=0, fancybox=True)
	#~ ax.legend(loc = 'upper left', handlelength=0, handletextpad=0, fancybox=True)
	#~ plt.figlegend( (B,K,SC), ('N-body','Linear Kaiser','Scoccimarro '), loc = 'upper center',\
	#~ plt.figlegend( (B,TNS), ('N-body', 'TNS'), loc = 'upper center',\
	#~ ncol=5, labelspacing=0.,title =r'M$\nu$ = '+str(Mnu)  )
	#~ plt.figlegend( (B,T,C), ('bias from N-body','Tinker effective bias ', 'Crocce effective bias '), loc = 'upper center',\
	#~ ncol=5, labelspacing=0.,title =r'M$\nu$ = '+str(Mnu)  )
	#~ plt.figlegend( (B,F,PT), ('N-body','Power law fitted with free param ', 'FAST '), loc = 'upper center',\
	#~ ncol=5, labelspacing=0.,title =r'M$\nu$ = '+str(Mnu) )
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
	#~ #plt.suptitle('bias z = '+str(z[j])+', nu = '+str(Mnu) )
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

	#~ plt.title('3D power spectrum at z = '+str(z[j])+', nu = '+str(Mnu) )
	#~ plt.title('3D halo power spectrum at z = '+str(z[j])+', nu = '+str(Mnu)+', mass range '+str(mass_range) )
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
	#~ plt.title('3D monopole power spectrum at z = '+str(z[j])+', nu = '+str(Mnu)+', mass range '+str(mass_range)+', mu = '+str(mu) )
	#~ plt.title('3D quadrupole power spectrum at z = '+str(z[j])+', nu = '+str(Mnu)+', mass range '+str(mass_range) +', mu = '+str(mu))
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

