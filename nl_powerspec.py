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

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4 = ld_data(Mnu, z, j)

####################################################################
##### define the maximum scale for the fit 
	kstop1 = [0.16,0.2,0.25,0.35]
	kstop2 = [0.12,0.16,0.2,0.2]
	kstop3 = [0.15,0.15,0.15,0.15]
	
#### the case 
	case = 1
	
	if case == 1:
		kstop = kstop1[ind]
	elif case == 2:
		kstop = kstop2[ind]
	elif case == 3:
		kstop = kstop3[ind]
		
		
#### other kstop
	#~ kstoplim = [0.5,0.5,0.5,0.4]
	#~ kstop = kstoplim[ind]
	
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
	lim = np.where((k < kstop)&(k > 1e-2))[0]

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

	Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(kbis, Plinbis)
	
####################################################################
#### get fitted coefficients

	print 'polynomial'
	biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, k, lb1, lb2, lb3, lb4,\
	errlb1, errlb2, errlb3, errlb4, kbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis,Mnu, z, j, case)


#-------------------------------------------------------------------

	print 'perturbation'
	#~ bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	#~ bias3PTbis2, bias3PTbis3, bias3PTbis4, PsptD1r1, PsptD2r1, PsptD3r1 = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
	#~ bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j, case)
	bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	bias3PTbis2, bias3PTbis3, bias3PTbis4 = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
	bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j, case)
	
	
	
######################################################################
### mean of mass bins

	B1 = np.array([bias2PT1/bias1bis, bias2PT2/bias2bis, bias2PT3/bias3bis, bias2PT4/bias4bis])
	B1bis = np.array([bias3PT1/bias1bis, bias3PT2/bias2bis, bias3PT3/bias3bis, bias3PT4/bias4bis])
	B1ter = np.array([bias3PTbis1/bias1bis, bias3PTbis2/bias2bis, bias3PTbis3/bias3bis, bias3PTbis4/bias4bis])
	B2 = np.array([bias1bis/bias1bis, bias2bis/bias2bis, bias3bis/bias3bis, bias4bis/bias4bis])
	B3 = np.array([biasF1/bias1bis, biasF2/bias2bis, biasF3/bias3bis, biasF4/bias4bis])
	B3bis = np.array([biasF1bis/bias1bis, biasF2bis/bias2bis, biasF3bis/bias3bis, biasF4bis/bias4bis])
	b1 = np.mean(B1,axis=0)
	b1bis = np.mean(B1bis,axis=0)
	b1ter = np.mean(B1ter,axis=0)
	b2 = np.mean(B2,axis=0)
	b3 = np.mean(B3,axis=0)
	b3bis = np.mean(B3bis,axis=0)
	

#####################################################################
#####################################################################
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
	#~ plt.legend(loc='upper left', ncol=2, fancybox=True,fontsize=14)
	#~ plt.xlim(0.01,0.2)
	#~ plt.xlabel('k [h/Mpc]',fontsize=14)
	#~ plt.ylabel(r'$k^{1.5} \times P(k)$ [(Mpc/h)]',fontsize=14)
	#~ plt.xscale('log')
	#~ plt.ylim(-50,250)
	#~ plt.show()
	
	
	#~ kill
	
####################################################################
##### different fit
####################################################################
	
	#### compare the third order influence
	#~ plt.figure()
	#-----------------------------
	#~ plt.ylabel(r'$2 \times b_{1} \times b_{3nl}\times \sigma_{3}^{2} \times P^{lin}$ ', fontsize = 14)
	#~ plt.ylabel(r'$b_{3nl}$ / $(b_{1} - 1)$ ', fontsize = 14)
	#~ plt.xlabel(r'$k$ [h/Mpc] ', fontsize = 14)
	#~ plt.plot(kbis,PsptD2r1, label=r'3rd order correction with free b3nl', color='C2', linestyle ='--' )
	#~ plt.plot(kbis,PsptD3r1, label=r'3rd order correction with fixed b3nl', color='C3', linestyle='--' )
	#~ plt.ylim(-400,0)
	#~ plt.ylim(0,2)
	#~ plt.xlim(0.03,0.2)
	#~ plt.legend(loc=6, fontsize = 12) 
	#----------------------------
	#~ plt.yscale('log')
	#~ plt.xlabel(r'$k$ [h/Mpc] ', fontsize = 14)
	#~ plt.ylabel(r'P(k) ', fontsize = 14)
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
	#~ plt.title('z = '+str(z[j])+', $k_{max}$ = 0.12, mass range M1', fontsize = 14 )
	#~ plt.legend(loc='lower left', fontsize = 14) 
	#~ plt.show()

	#~ kill

	
	####################################################################
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
	#~ ######### pl residuals comparison #################
	#~ ax2.set_ylim(0.9,1.1)
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ B3, = ax2.plot(kbis, b3, linewidth = 2)
	#~ B3bis, = ax2.plot(kbis, b3bis, linewidth = 2)
	#~ B2, = ax2.plot(kbis, b2, label='z = '+str(z[j]), color='k')
	#~ plt.figlegend( (B3bis,B3), (r'$b_{cc} = b_{1} + b_{2}*k^{2} + b_{4}*k^{4}$ ',\
	#~ r'$b_{cc} = b_{1} + b_{2}*k^{2} + b_{3}*k^{3} + b_{4}*k^{4}$ '), \
	####### comparison bias and != models #############################
	#~ M1 = ax2.errorbar(kbis, bias1bis, yerr= errb1bis,fmt='.', label='z = '+str(z[j]))
	#~ M2 = ax2.errorbar(kbis, bias2bis, yerr= errb2bis,fmt='.')
	#~ M3 = ax2.errorbar(kbis, bias3bis, yerr= errb3bis,fmt='.')
	#~ M4 = ax2.errorbar(kbis, bias4bis, yerr= errb4bis,fmt='.')
	#~ ax2.set_ylim(bias1bis[0]*0.8,bias4bis[0]*1.4)
	#~ Plo, =ax2.plot(kbis, biasF1, color='k')
	#~ Ple, =ax2.plot(kbis, biasF1bis, color='k', linestyle='--')
	#~ pt2, =ax2.plot(kbis, bias2PT1, color='k', linestyle='--')
	#~ pt3, =ax2.plot(kbis, bias3PT1, color='k', linestyle=':')
	#~ pt3bis, =ax2.plot(kbis, bias3PTbis1, color='k', linestyle='-.')
	#--------
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
	#--------
	#~ ax2.plot(kbis, biasF4, color='k')
	#~ ax2.plot(kbis, biasF4bis, color='k', linestyle='--')
	#~ ax2.plot(kbis, bias2PT4, color='k', linestyle='--')
	#~ ax2.plot(kbis, bias3PT4, color='k', linestyle=':')
	#~ ax2.plot(kbis, bias3PTbis4, color='k', linestyle='-.')
	#--------
	#~ plt.figlegend( (M1,M2,M3,M4,Plo, Ple), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'PL with odd k','PL without odd k'), \
	#~ plt.figlegend( (M1,M2,M3,M4,Plo, pt2, pt3, pt3bis), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'PL with odd k'\
	#~ ,'2nd order bias expansion', r'3rd order with free $b_{3nl}$', r'3rd order with fixed $b_{s}$, $b_{3nl}$'), \
	###### compare all power model residuals ##########################
	#~ ax2.set_ylim(0.9,1.1)
	#~ ax2.set_yticks(np.linspace(0.9,1.1,5))
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ B3, = ax2.plot(kbis, b3,label=r'w/ $b_{sim}$', color='C0')
	#~ B3, = ax2.plot(kbis, b3, label='z = '+str(z[j]), color='C0')
	#~ B1, = ax2.plot(kbis, b1, color='C1')	
	#~ B1bis, = ax2.plot(kbis, b1bis, color='C2')
	#~ B1ter, = ax2.plot(kbis, b1ter,  color='C3')
	#~ B2, = ax2.plot(kbis, b2, color='k')
	#~ ax2.plot(kbis, bias3PT1/bias1bis)
	#~ ax2.plot(kbis, bias3PT2/bias2bis)
	#~ ax2.plot(kbis, bias3PT3/bias3bis)
	#~ ax2.plot(kbis, bias3PT4/bias4bis)
	
	#~ B3anal, = ax2.plot(kbis, bb3,label=r'w/ $b_{fiducial}$', color='C0',linestyle='--')
	#~ B1anal, = ax2.plot(kbis, bb1, color='C1',linestyle='--')
	#~ B1bisanal, = ax2.plot(kbis, bb1bis, color='C2',linestyle='--')
	#~ B1teranal, = ax2.plot(kbis, bb1ter,  color='C3',linestyle='--')
	
	#~ plt.figlegend( (B1,B1bis,B1ter,B2,B3), ('2nd order expansion',r'3rd order expansion with free $b_{3nl}$',\
	#~ r'3rd order expansion with fixed $b_{3nl}$', 'N-body','Power law '), \
	######################################
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case '+str(case), fontsize=12)
	#~ ax2.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ ax2.legend(loc = 'upper left', fancybox=True, fontsize=14)
	#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#~ ax2.set_xscale('log')
	#~ if j == 0 :
		#~ ax2.tick_params(bottom='off', labelbottom='off',labelleft=True)
		#~ ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize=16)
	#~ if j == 1 :
		#~ ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize=16)
		#~ ax2.yaxis.set_label_position("right")
	#~ if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		#~ ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize=14)
		#~ ax2.set_xlabel('k [h/Mpc]', fontsize=16)
	#~ if j == 3 :
		#~ ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_xlabel('k [h/Mpc]', fontsize=14)
		#~ ax2.set_ylabel(r'$b_{cc}$ / $b_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize=16)
		#~ ax2.yaxis.set_label_position("right")
	#~ ax2.set_xlim(8e-3,1)
	#~ if j == len(z) -1:
		#~ plt.show()
	
		
	#~ kill
	

###############################################################################
###############################################################################
	

	
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
	
########################################################################
### load rescaling coefficients
	if Mnu == 0.15:
		sca1, sca2, sca3, sca4 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+\
		'eV/large_scale/rescaling_z='+str(z[j])+'_.txt')
	
####################################################################
### compute of the 4 mass bins
####################################################################
	
	#### compute fcc with transfer function
	fcc = fz * (Tm/ Tcb)
	
	
	kai1, kai2, kai3, kai4, sco1, sco2, sco3, sco4, tns1, tns2, tns3, tns4, etns1, etns2, etns3, etns4 = RSD(fz,fcc, Dz[ind]\
	, j, kstop, Pmmbis, biasF1, biasF2, biasF3, biasF4, kbis, Plinbis, Pmono1bis, Pmono2bis, Pmono3bis, \
	Pmono4bis, errPr1bis, errPr2bis, errPr3bis, errPr4bis, Pmod_dt, Pmod_tt, case,z,Mnu, A, B, C, D, E, F, G, H )


	p1 = np.array([Pmono1bis/Pmono1bis, Pmono2bis/Pmono2bis, Pmono3bis/Pmono3bis, Pmono4bis/Pmono4bis])
	P1 = np.mean(p1, axis=0)
	p2 = np.array([kai1/Pmono1bis, kai2/Pmono2bis, kai3/Pmono3bis, kai4/Pmono4bis])
	P2 = np.mean(p2, axis=0)
	p3 = np.array([sco1/Pmono1bis, sco2/Pmono2bis, sco3/Pmono3bis, sco4/Pmono4bis])
	P3 = np.mean(p3, axis=0)
	p4 = np.array([tns1/Pmono1bis, tns2/Pmono2bis, tns3/Pmono3bis, tns4/Pmono4bis])
	P4 = np.mean(p4, axis=0)
	p6 = np.array([etns1/Pmono1bis, etns2/Pmono2bis, etns3/Pmono3bis, etns4/Pmono4bis])
	P6 = np.mean(p6, axis=0)
	
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
	if j == z[0]:
		fig2 = plt.figure()
	J = j + 1
	
	if len(z) == 1:
		ax2 = fig2.add_subplot(1, len(z), J)
	elif len(z) == 2:
		ax2 = fig2.add_subplot(1, 2, J)
	elif len(z) > 2:
		ax2 = fig2.add_subplot(2, 2, J)
	########### power spectrum ########
	ax2.set_ylim(0.9,1.1)
	ax2.set_yticks(np.linspace(0.9,1.1,5))
	ax2.axhline(1, color='k', linestyle='--')
	ax2.axhline(1.01, color='k', linestyle=':')
	ax2.axhline(0.99, color='k', linestyle=':')
	P1, =ax2.plot(kbis,P1, color='k')
	P2, =ax2.plot(kbis,P2, color='C3',label=r'w/ $b_{sim}$')
	P2, =ax2.plot(kbis,P2, color='C3', label='z = '+str(z[j]))
	P3, =ax2.plot(kbis,P3, color='C0')
	P4, =ax2.plot(kbis,P4, color='C1')
	P6, =ax2.plot(kbis,P6, color='c')
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
	
	plt.figlegend( (P1,P2, P3, P4,P6), ('N-body','Power law + Kaiser','Power law + Scoccimarro','Power law + TNS','eTNS'), \
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
	loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case '+str(case), fontsize=14)
	ax2.axvspan(kstop, 7, alpha=0.2, color='grey')
	ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, fontsize=14)
	ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, fontsize=14)
	plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	ax2.set_xscale('log')
	if j == 0 :
		ax2.tick_params(bottom='off', labelbottom='off')
		ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cc}$', fontsize=16)
	if j == 1 :
		ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cc}$', fontsize=16)
		ax2.yaxis.set_label_position("right")
	if j == 2 :
		#ax.tick_params(labelleft=True)
		ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cc}$', fontsize=16)
		ax2.set_xlabel('k [h/Mpc]', fontsize=14)
	if j == 3 :
		ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		ax2.set_xlabel('k [h/Mpc]', fontsize=14)
		ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cc}$', fontsize=16)
		ax2.yaxis.set_label_position("right")
	ax2.set_xlim(8e-3,1)
	#plt.ylim(0.7,1.3)
	if j == len(z) -1:
		plt.show()
	
		
	#~ kill
	
	del kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4, pte, Plin, Tm, Tcb, kbis,\
	Plinbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis,errb3bis,errb4bis, Pmmbis,PH1bis,PH2bis,PH3bis,PH4bis,\
	errPhh1bis,errPhh2bis,errPhh3bis,errPhh4bis,Pmono1bis,Pmono2bis,Pmono3bis,Pmono4bis,errPr1bis,errPr2bis,errPr3bis,errPr4bis,\
	biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis, bias2PT1, bias2PT2, bias2PT3, bias2PT4,\
	bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1, bias3PTbis2, bias3PTbis3,bias3PTbis4,B1,B1bis,B1ter,B2,B3,B3bis,b1,b1bis,\
	b1ter,b2,b3,b3bis
	
########################################################################
############## plot ####################################################
########################################################################
	col = ['b','r','g','k']
	
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
#*********************************************************************************************
#*********************************************************************************************
	#~ ktest = np.logspace(np.log10(0.03),np.log10(0.55),15)
	#~ for kstop in ktest:
		#~ print kstop
	
	#~ ####################################################################
		#~ Plin = Pcamb
		#~ klin = kcamb

	#~ #######################################################################
		#~ if Mnu == 0.0:
				
			#~ pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
			#~ Plin = pte[:,1]
			#~ pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
			#~ Tm = pte[:,1]
			#~ pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected3-'+str(z[j])+'.txt')
			#~ Tcb = pte[:,1]
		
		#~ # interpolate to have more points and create an evenly logged array
		#~ kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 250)
		#~ Plinbis = np.interp(kbis, k, Plin)
		#~ lim = np.where((kbis < kstop))[0]

	#~ ########################################################################################################################################
	#~ #######################################################################################################################################
	#~ ##### interpolate data to have more point on fitting scales
	#~ ##### real space
		#~ bias1bis = np.interp(kbis, k, bias1)
		#~ bias2bis = np.interp(kbis, k, bias2)
		#~ bias3bis = np.interp(kbis, k, bias3)
		#~ bias4bis = np.interp(kbis, k, bias4)
		#~ errb1bis = np.interp(kbis, k, errb1)
		#~ errb2bis = np.interp(kbis, k, errb2)
		#~ errb3bis = np.interp(kbis, k, errb3)
		#~ errb4bis = np.interp(kbis, k, errb4)
		#~ Pmmbis = np.interp(kbis, k, Pmm)
		#~ PH1bis = np.interp(kbis, k, PH1)
		#~ PH2bis = np.interp(kbis, k, PH2)
		#~ PH3bis = np.interp(kbis, k, PH3)
		#~ PH4bis = np.interp(kbis, k, PH4)
		#~ errPhh1bis = np.interp(kbis, k, errPhh1)
		#~ errPhh2bis = np.interp(kbis, k, errPhh2)
		#~ errPhh3bis = np.interp(kbis, k, errPhh3)
		#~ errPhh4bis = np.interp(kbis, k, errPhh4)

		#~ ##### redshift space
		#~ Pmono1bis = np.interp(kbis, k, Pmono1)
		#~ Pmono2bis = np.interp(kbis, k, Pmono2)
		#~ Pmono3bis = np.interp(kbis, k, Pmono3)
		#~ Pmono4bis = np.interp(kbis, k, Pmono4)
		#~ errPr1bis = np.interp(kbis, k, errPr1)
		#~ errPr2bis = np.interp(kbis, k, errPr2)
		#~ errPr3bis = np.interp(kbis, k, errPr3)
		#~ errPr4bis = np.interp(kbis, k, errPr4)
		#~ Tm =  np.interp(kbis,k,Tm)
		#~ Tcb =  np.interp(kbis,k,Tcb)


		
	#~ ####################################################################
	#~ ##### compute linear bias and error
		
		#~ # on interpolated array
		#~ toto = np.where(kbis < 0.05)[0]
		#~ lb1 = np.mean(bias1bis[toto])
		#~ lb2 = np.mean(bias2bis[toto])
		#~ lb3 = np.mean(bias3bis[toto])
		#~ lb4 = np.mean(bias4bis[toto])
		#~ errlb1 = np.mean(errb1bis[toto])
		#~ errlb2 = np.mean(errb2bis[toto])
		#~ errlb3 = np.mean(errb3bis[toto])
		#~ errlb4 = np.mean(errb4bis[toto])
		
		#~ # on simulation array
		#~ Toto = np.where(k < 0.05)[0]
		#~ Lb1 = np.mean(bias1[Toto])
		#~ Lb2 = np.mean(bias2[Toto])
		#~ Lb3 = np.mean(bias3[Toto])
		#~ Lb4 = np.mean(bias4[Toto])
		#~ errLb1 = np.mean(errb1[Toto])
		#~ errLb2 = np.mean(errb2[Toto])
		#~ errLb3 = np.mean(errb3[Toto])
		#~ errLb4 = np.mean(errb4[Toto])
		
	#~ ####################################################################
	#~ #### compute pt terms

		#~ Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(kbis, Plinbis)
		
	#~ ####################################################################
	#~ #### get fitted coefficients


		#~ biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, k, lb1, lb2, lb3, lb4,\
		#~ errlb1, errlb2, errlb3, errlb4, kbis, bias1bis, bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis,Mnu, z, j, case)


	#~ #-------------------------------------------------------------------

		#~ bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1, bias3PTbis2, bias3PTbis3,\
		#~ bias3PTbis4 = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
		#~ bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j, case)
		
	
		
		
	#~ ####################################################################
	#~ ##### compute the chi2 of different quantities
	#~ ####################################################################

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

		#~ PT1 = (bias2PT1[lim]- bias1bis[lim])**2/errb1bis[lim]**2
		#~ PT2 = (bias2PT2[lim]- bias2bis[lim])**2/errb2bis[lim]**2
		#~ PT3 = (bias2PT3[lim]- bias3bis[lim])**2/errb3bis[lim]**2
		#~ PT4 = (bias2PT4[lim]- bias4bis[lim])**2/errb4bis[lim]**2
		#~ chi2PT1 = np.sum(PT1)
		#~ chi2PT2 = np.sum(PT2)
		#~ chi2PT3 = np.sum(PT3)
		#~ chi2PT4 = np.sum(PT4)
		#~ #-------------------------------------------------
		#~ PTbis1 = (bias3PT1[lim]- bias1bis[lim])**2/errb1bis[lim]**2
		#~ PTbis2 = (bias3PT2[lim]- bias2bis[lim])**2/errb2bis[lim]**2
		#~ PTbis3 = (bias3PT3[lim]- bias3bis[lim])**2/errb3bis[lim]**2
		#~ PTbis4 = (bias3PT4[lim]- bias4bis[lim])**2/errb4bis[lim]**2
		#~ chi2PTbis1 = np.sum(PTbis1)
		#~ chi2PTbis2 = np.sum(PTbis2)
		#~ chi2PTbis3 = np.sum(PTbis3)
		#~ chi2PTbis4 = np.sum(PTbis4)
		
		#~ cname = 'chi2_z='+str(z[j])+'.txt'
		#~ with open(cname, 'a+') as fid_file:

			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop,\
			#~ chi2F1, chi2F2, chi2F3, chi2F4, chi2PT1, chi2PT2, chi2PT3, chi2PT4, chi2PTbis1, chi2PTbis2, chi2PTbis3, chi2PTbis4))
		#~ print '\n'

	 
end = time()
print 'total time is '+str((end - start))	 

