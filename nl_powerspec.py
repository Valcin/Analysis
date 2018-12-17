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
from interp import interp_simu1, interp_simu3
#~ from hmf_test import htest
from time import time
from rsd import RSD1, RSD2
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma



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

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	bias1s, bias2s, bias3s, bias4s, errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3,\
	errPr4, kclass, Tm, Tcb, noise1, noise2, noise3, noise4 = ld_data(Mnu, z, j)
	
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
		
		
#### other kstop
	#~ kstoplim = [0.5,0.5,0.5,0.4]
	#~ kstoplim = [0.25,0.25,0.25,0.25]
	#~ kstop = kstoplim[ind]
	
	print kstop
	
	
	# put identation to the rest to loop over kstop
	#~ #kstop_arr = np.logspace(np.log10(0.05),np.log10(0.6),20)
	#~ #for kstop in kstop_arr:

#######################################################################

	# interpolate to have more points and create an evenly logged array
	#~ kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 250)
	#~ kbis = np.logspace(np.log10(np.min(kcamb)), np.log10(np.max(kcamb)), 200)
	#~ Plinbis = np.interp(kbis, k, Plin)
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


####################################################################
##### compute linear bias and error
	
	# on interpolated array
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
	Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H  = interp_simu1(z,j ,k, kcamb, Pcamb, Pmod_dd, Pmod_dt, Pmod_tt,\
	A, B, C, D, E, F, G, H, 2)
	
	
	#~ plt.plot(k,Pmm/Pmm)
	#~ plt.plot(k,Pmod_dd/Pmm)
	#~ plt.plot(k, Plin/Pmm)
	#~ plt.plot(k, Pcamb/Pmm)
	#~ plt.xscale('log')
	#~ plt.show()
	#~ kill
	
####################################################################
#### get fitted coefficients

	print 'polynomial'
	#~ biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, lb1, lb2, lb3, lb4,\
	#~ errlb1, errlb2, errlb3, errlb4, k, bias1, bias2, bias3, bias4, errb1, errb2, errb3, errb4,Mnu, z, j, case)

	#~ biasF1s, biasF2s, biasF3s, biasF4s, biasF1biss, biasF2biss, biasF3biss, biasF4biss = poly(kstop, k, lb1, lb2, lb3, lb4,\
	#~ errlb1, errlb2, errlb3, errlb4, kbis, bias1biss, bias2biss, bias3biss, bias4biss, errb1bis, errb2bis, errb3bis, errb4bis,Mnu, z, j, case)


#-------------------------------------------------------------------

	print 'perturbation'
	
	bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	bias3PTbis2, bias3PTbis3, bias3PTbis4 = perturb(kstop,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmod_dd, k, bias1,\
	bias2, bias3, bias4, errb1, errb2, errb3, errb4, A, B, C, D, E, F,Mnu, z, j, case, PH1, noise1, noise2, noise3, noise4)
	
	#~ bias2PT1s, bias2PT2s, bias2PT3s, bias2PT4s, bias3PT1s, bias3PT2s, bias3PT3s, bias3PT4s, bias3PTbis1s,\
	#~ bias3PTbis2s, bias3PTbis3s, bias3PTbis4s = perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis,\
	#~ kbis, bias1biss, bias2biss, bias3biss, bias4biss, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j, case)
	
	#~ kill
######################################################################
### mean of mass bins

	#~ B1 = np.array([bias2PT1/bias1, bias2PT2/bias2, bias2PT3/bias3, bias2PT4/bias4])
	#~ B1bis = np.array([bias3PT1/bias1, bias3PT2/bias2, bias3PT3/bias3, bias3PT4/bias4])
	#~ B1ter = np.array([bias3PTbis1/bias1, bias3PTbis2/bias2, bias3PTbis3/bias3, bias3PTbis4/bias4])
	#~ B2 = np.array([bias1/bias1, bias2/bias2, bias3/bias3, bias4/bias4])
	#~ B3 = np.array([biasF1/bias1, biasF2/bias2, biasF3/bias3, biasF4/bias4])
	#~ B3bis = np.array([biasF1bis/bias1, biasF2bis/bias2, biasF3bis/bias3, biasF4bis/bias4])
	#~ b1 = np.mean(B1,axis=0)
	#~ b1bis = np.mean(B1bis,axis=0)
	#~ b1ter = np.mean(B1ter,axis=0)
	#~ b2 = np.mean(B2,axis=0)
	#~ b3 = np.mean(B3,axis=0)
	#~ b3bis = np.mean(B3bis,axis=0)


######################################################################

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
#### get rescaled coefficients
	#~ bbias2PT1, bbias2PT2, bbias2PT3, bbias2PT4,bbias3PT1, bbias3PT2, bbias3PT3,\
	#~ bbias3PT4,bbiasF1, bbiasF2, bbiasF3, bbiasF4, bbias3PTbis1, bbias3PTbis2, bbias3PTbis3, bbias3PTbis4 = rescal(j, case)

	#~ Bb1 = np.array([bbias2PT1/bias1bis, bbias2PT2/bias2bis, bbias2PT3/bias3bis, bbias2PT4/bias4bis])
	#~ Bb1bis = np.array([bbias3PT1/bias1bis, bbias3PT2/bias2bis, bbias3PT3/bias3bis, bbias3PT4/bias4bis])
	#~ Bb1ter = np.array([bbias3PTbis1/bias1bis, bbias3PTbis2/bias2bis, bbias3PTbis3/bias3bis, bbias3PTbis4/bias4bis])
	#~ Bb3 = np.array([bbiasF1/bias1bis, bbiasF2/bias2bis, bbiasF3/bias3bis, bbiasF4/bias4bis])
	#~ bb1 = np.mean(Bb1,axis=0)
	#~ bb1bis = np.mean(Bb1bis,axis=0)
	#~ bb1ter = np.mean(Bb1ter,axis=0)
	#~ bb3 = np.mean(Bb3,axis=0)
	
	
####################################################################
##### different fit
####################################################################
	
	
	
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
	#~ ax2.set_yticks(np.linspace(0.9,1.1,5))
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ B3, = ax2.plot(k, b3, linewidth = 2)
	#~ B3bis, = ax2.plot(k, b3bis, linewidth = 2)
	#~ B2, = ax2.plot(k, b2, label='z = '+str(z[j]), color='k')
	#~ plt.figlegend( (B3bis,B3), (r'$b_{cb} = b_{1} + b_{2}k^{2} + b_{4}k^{4}$ ',\
	#~ r'$b_{cb} = b_{1} + b_{2}k^{2} + b_{3}k^{3} + b_{4}k^{4}$ '), \
	####### comparison bias and != models #############################
	#~ M1 = ax2.errorbar(k, bias1, yerr= errb1,fmt='.')
	#~ M2 = ax2.errorbar(k, bias2, yerr= errb2,fmt='.')
	#~ M3 = ax2.errorbar(k, bias3, yerr= errb3,fmt='.')
	#~ M4 = ax2.errorbar(k, bias4, yerr= errb4,fmt='.')
	#~ ax2.set_ylim(bias1[0]*0.8,bias4[0]*1.4)
	#~ Plo, =ax2.plot(k, biasF1, color='k', label='z = '+str(z[j]))
	#~ Ple, =ax2.plot(k, biasF1bis, color='k', linestyle='--')
	#~ pt2, =ax2.plot(k, bias2PT1, color='k', linestyle='--')
	#~ pt3, =ax2.plot(k, bias3PT1, color='k', linestyle=':')
	#~ pt3bis, =ax2.plot(k, bias3PTbis1, color='k', linestyle='-.')
	#~ #--------
	#~ ax2.plot(k, biasF2, color='k')
	#~ ax2.plot(k, biasF2bis, color='k', linestyle='--')
	#~ ax2.plot(k, bias2PT2, color='k', linestyle='--' )
	#~ ax2.plot(k, bias3PT2, color='k', linestyle=':')
	#~ ax2.plot(k, bias3PTbis2, color='k', linestyle='-.')
	#~ #--------
	#~ ax2.plot(k, biasF3, color='k')
	#~ ax2.plot(k, biasF3bis, color='k', linestyle='--')
	#~ ax2.plot(k, bias2PT3, color='k', linestyle='--' )
	#~ ax2.plot(k, bias3PT3, color='k', linestyle=':')
	#~ ax2.plot(k, bias3PTbis3, color='k', linestyle='-.')
	#~ #--------
	#~ ax2.plot(k, biasF4, color='k')
	#~ ax2.plot(k, biasF4bis, color='k', linestyle='--')
	#~ ax2.plot(k, bias2PT4, color='k', linestyle='--')
	#~ ax2.plot(k, bias3PT4, color='k', linestyle=':')
	#~ ax2.plot(k, bias3PTbis4, color='k', linestyle='-.')
	#--------
	#~ plt.figlegend( (M1,M2,M3,M4,Plo, Ple), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'PL with odd k','PL without odd k'), \
	#~ plt.figlegend( (M1,M2,M3,M4,Plo, pt2, pt3, pt3bis), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'PL with odd k'\
	#~ ,'2nd order bias expansion', r'3rd order with free $b_{3nl}$', r'3rd order with fixed $b_{s}$, $b_{3nl}$'), \
	#~ ###### compare all power model residuals ##########################
	#~ ax2.set_ylim(0.,2.1)
	#~ ax2.set_ylim(0.9,1.1)
	#~ ax2.set_yticks(np.linspace(0.9,1.1,5))
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ B3, = ax2.plot(k, b3, label='z = '+str(z[j]), color='C0')
	#~ B1, = ax2.plot(k, b1, color='C1')
	#~ B1bis, = ax2.plot(k, b1bis, color='C2')
	#~ B1ter, = ax2.plot(k, b1ter,  color='C3')
	#~ B2, = ax2.plot(k, b2, color='k')
	
	#~ B3, = ax2.plot(k, b3,label=r'w/ $b_{sim}$', color='C0')
	#~ B3anal, = ax2.plot(k, bb3,label=r'w/ $b_{model}$', color='C0',linestyle='--')
	#~ B1anal, = ax2.plot(k, bb1, color='C1',linestyle='--')
	#~ B1bisanal, = ax2.plot(k, bb1bis, color='C2',linestyle='--')
	#~ B1teranal, = ax2.plot(k, bb1ter,  color='C3',linestyle='--')
	
	#~ plt.figlegend( (B1,B1bis,B1ter,B2,B3), ('2nd order expansion with free $b_{s}$',r'3rd order expansion with free $b_s$, $b_{3nl}$',\
	#~ r'3rd order expansion with fixed $b_s$, $b_{3nl}$', 'N-body','Power law '), \
	######################################
	#~ loc = 'upper center', ncol=3, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case '+str(case), fontsize=12)
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu), fontsize=12)
	#~ ax2.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ ax2.legend(loc = 'upper left', fancybox=True, fontsize=14, handlelength=0, handletextpad=0)
	#~ ax2.legend(loc = 'upper left', title = 'z = '+str(z[j]), fancybox=True, fontsize=14)
	#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#~ ax2.set_xscale('log')
	#~ if j == 0 :
		#~ ax2.tick_params(bottom='off', labelbottom='off',labelleft=True)
		#~ ax2.set_ylabel(r'$b_{cb}$ / $b_{sim}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$b_{cb}$', fontsize=16)
	#~ if j == 1 :
		#~ ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_ylabel(r'$b_{cb}$ / $b_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$b_{cb}$', fontsize=16)
		#~ ax2.yaxis.set_label_position("right")
	#~ if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		#~ ax2.set_ylabel(r'$b_{cb}$ / $b_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$b_{cb}$', fontsize=14)
		#~ ax2.set_xlabel('k [h/Mpc]', fontsize=16)
	#~ if j == 3 :
		#~ ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_xlabel('k [h/Mpc]', fontsize=14)
		#~ ax2.set_ylabel(r'$b_{cb}$ / $b_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$b_{cb}$', fontsize=16)
		#~ ax2.yaxis.set_label_position("right")
	#~ ax2.set_xlim(8e-3,1)
	#~ if j == len(z) -1:
		#~ plt.show()
	
#####################################################################
#### compute fcc with transfer function
	Tm, Tcb = interp_simu3(z,j ,k, kclass, Tm, Tcb, 2)
	fcc = fz * (Tm/ Tcb)
	
#####################################################################
### only sigma is a free parameter
	#~ kai1, kai2, kai3, kai4, sco1, sco2, sco3, sco4, tns1, tns2, tns3, tns4, etns1, etns2, etns3, etns4 = RSD1(fz,fcc, Dz[ind]\
	#~ , j, kstop, kcamb, Pcamb, Pmod_dd, biasF1, biasF2, biasF3, biasF4, k, Pmono1, Pmono2, Pmono3, \
	#~ Pmono4, errPr1, errPr2, errPr3, errPr4, Pmod_dt, Pmod_tt, case,z,Mnu, A, B, C, D, E, F, G, H )

	#~ p1 = np.array([Pmono1/Pmono1, Pmono2/Pmono2, Pmono3/Pmono3, Pmono4/Pmono4])
	#~ P1 = np.mean(p1, axis=0)
	#~ p2 = np.array([kai1/Pmono1, kai2/Pmono2, kai3/Pmono3, kai4/Pmono4])
	#~ P2 = np.mean(p2, axis=0)
	#~ p3 = np.array([sco1/Pmono1, sco2/Pmono2, sco3/Pmono3, sco4/Pmono4])
	#~ P3 = np.mean(p3, axis=0)
	#~ p4 = np.array([tns1/Pmono1, tns2/Pmono2, tns3/Pmono3, tns4/Pmono4])
	#~ P4 = np.mean(p4, axis=0)
	#~ p6 = np.array([etns1/Pmono1, etns2/Pmono2, etns3/Pmono3, etns4/Pmono4])
	#~ P6 = np.mean(p6, axis=0)
	
###########################################################################
### bias is also a free parameter
	#~ kai1f, kai2f, kai3f, kai4f, sco1f, sco2f, sco3f, sco4f, tns1f, tns2f, tns3f, tns4f, etns1f, etns2f,\
	#~ etns3f, etns4f = RSD2(fz,fcc, Dz[ind], j, kstop, kcamb, Pcamb, Pmod_dd, biasF1, biasF2, biasF3, biasF4,\
	#~ k, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4, Pmod_dt, Pmod_tt, case,z,Mnu, A, B, C, D, E, F, G, H )


	#~ p2f = np.array([kai1f/Pmono1, kai2f/Pmono2, kai3f/Pmono3, kai4f/Pmono4])
	#~ P2f = np.mean(p2f, axis=0)
	#~ p3f = np.array([sco1f/Pmono1, sco2f/Pmono2, sco3f/Pmono3, sco4f/Pmono4])
	#~ P3f = np.mean(p3f, axis=0)
	#~ p4f = np.array([tns1f/Pmono1, tns2f/Pmono2, tns3f/Pmono3, tns4f/Pmono4])
	#~ P4f = np.mean(p4f, axis=0)
	#~ p6f = np.array([etns1f/Pmono1, etns2f/Pmono2, etns3f/Pmono3, etns4f/Pmono4])
	#~ P6f = np.mean(p6f, axis=0)
	


####################################################################
#### get rescaled coefficients
	#~ k1, k2, k3, k4, s1, s2, s3, s4, t1, t2, t3, t4, e1, e2, e3, e4 = rescal(j, case)
	
	
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
	
############################################################################################################
############################################################################################################
############################################################################################################
	
	#~ #######--------- mean and std of bias and ps ratio ------------#####
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
	#~ Ps1, =ax2.plot(k,P1, color='k')
	#~ Ps2, =ax2.plot(k,P2, color='C3')
	#~ Ps3, =ax2.plot(k,P3, color='C0')
	#~ Ps4, =ax2.plot(k,P4, color='C1')
	#~ Ps6, =ax2.plot(k,P6, color='c')
	#~ #--------------------------------
	#~ ax2.plot(k,P2f, color='C3', label='z = '+str(z[j]), linestyle='--')
	#~ ax2.plot(k,P3f, color='C0', linestyle='--')
	#~ ax2.plot(k,P4f, color='C1', linestyle='--')
	#~ ax2.plot(k,P6f, color='c', linestyle='--')
	#-------------------------------
	#~ ax2.plot(k,P2bis, color='C3', linestyle='--',label=r'w/ $b_{model}$ and $\sigma_v$ free')
	#~ ax2.plot(k,P3bis, color='C0', linestyle='--')
	#~ ax2.plot(k,P4bis, color='C1', linestyle='--')
	#~ ax2.plot(k,P6bis, color='c', linestyle='--')
	#-------------------------------
	#~ ax2.plot(k,P2ter, color='C3', linestyle='--',label=r'w/ $b_{model}$ and $\sigma_v$ fixed')
	#~ ax2.plot(k,P3ter, color='C0', linestyle='--')
	#~ ax2.plot(k,P4ter, color='C1', linestyle='--')
	#~ ax2.plot(k,P6ter, color='c', linestyle='--')
	
	#~ plt.figlegend( (Ps1,Ps2, Ps3, Ps4,Ps6), ('N-body','Polyn. + Kaiser','Polyn. + Scoccimarro','Polyn. + TNS','eTNS'), \
	####### comparison bias and != models #############################
	#~ ax2.set_yscale('log')
	#~ plt.ylim(2e2,3e5)
	#~ M1 = ax2.errorbar(k, Pmono1, yerr= errPr1,fmt='.')
	#~ M2 = ax2.errorbar(k, Pmono2, yerr= errPr2,fmt='.')
	#~ M3 = ax2.errorbar(k, Pmono3, yerr= errPr3,fmt='.')
	#~ M4 = ax2.errorbar(k, Pmono4, yerr= errPr4,fmt='.')
	#~ nlk, = ax2.plot(k, kai1, color='k', label='z = '+str(z[j]))
	#~ sco, = ax2.plot(k, sco1, color='k', linestyle='--')
	#~ tns, = ax2.plot(k, tns1, color='k', linestyle=':')
	#~ etns, = ax2.plot(k, etns1, color='k', linestyle='-.')
	#--------
	#~ ax2.plot(k, kai2, color='k')
	#~ ax2.plot(k, sco2, color='k', linestyle='--' )
	#~ ax2.plot(k, tns2, color='k', linestyle=':')
	#~ ax2.plot(k, etns2, color='k', linestyle='-.')
	#--------
	#~ ax2.plot(k, kai3, color='k')
	#~ ax2.plot(k, sco3, color='k', linestyle='--' )
	#~ ax2.plot(k, tns3, color='k', linestyle=':')
	#~ ax2.plot(k, etns3, color='k', linestyle='-.')
	#--------
	#~ ax2.plot(k, kai4, color='k')
	#~ ax2.plot(k, sco4, color='k', linestyle='--')
	#~ ax2.plot(k, tns4, color='k', linestyle=':')
	#~ ax2.plot(k, etns4, color='k', linestyle='-.')
	#~ #--------
	#~ plt.figlegend( (M1,M2,M3,M4,nlk,sco, tns, etns), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'non linear kaiser + PL'\
	#~ ,'Scoccimarro + PL', r'TNS + PL', r'eTNS'), \
	######################################
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case '+str(case), fontsize=14)
	#~ ax2.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, fontsize=14)
	#~ ax2.legend(loc = 'lower left', fancybox=True, fontsize=14, handlelength=0, handletextpad=0)
	#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#~ ax2.set_xscale('log')
	#~ if j == 0 :
		#~ ax2.tick_params(bottom='off', labelbottom='off')
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cb}$', fontsize=16)
	#~ if j == 1 :
		#~ ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cb}$', fontsize=16)
		#~ ax2.yaxis.set_label_position("right")
	#~ if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cb}$', fontsize=16)
		#~ ax2.set_xlabel('k [h/Mpc]', fontsize=14)
	#~ if j == 3 :
		#~ ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_xlabel('k [h/Mpc]', fontsize=14)
		#~ ax2.set_ylabel(r'P(k) / $P_{sim}$', fontsize=16)
		#~ ax2.set_ylabel(r'$P_{cb}$', fontsize=16)
		#~ ax2.yaxis.set_label_position("right")
	#~ ax2.set_xlim(8e-3,1.)
	#plt.ylim(0.7,1.3)
	#~ if j == len(z) -1:
		#~ plt.show()
	
		
	#~ kill
	
	#~ del kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	#~ errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4, Tm, Tcb, \
	#~ biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis, bias2PT1, bias2PT2, bias2PT3, bias2PT4,\
	#~ bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1, bias3PTbis2, bias3PTbis3,bias3PTbis4,B1,B1bis,B1ter,B2,B3,B3bis,b1,b1bis,\
	#~ b1ter,b2,b3,b3bis

	

	
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
		#~ k = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 250)
		#~ Plinbis = np.interp(k, k, Plin)
		#~ lim = np.where((k < kstop))[0]

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

