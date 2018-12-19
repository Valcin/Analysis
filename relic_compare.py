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
from interp import interp_simu1, interp_simu3
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma
from fit_emcee import coeffit_pl,coeffit_pl2,coeffit_exp1, coeffit_exp2, coeffit_exp3,coeffit_Kaiser, coeffit_Scocci, coeffit_TNS, coeffit_eTNS


#~ z = [0.0,0.5,1.0,2.0]
z = [0.0]
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
	
	## define the maximum scale for the fit 
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
		

	
#########################################################################
#### load data from simualtion 

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	bias1s, bias2s, bias3s, bias4s, errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3,\
	errPr4, kclass, Tm, Tcb, noise1, noise2, noise3, noise4 = ld_data(Mnu, z, j)

#########################################################################


	print 'perturbation'
	
	
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
	
	#### compute pt terms

	Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H   = pt_terms(kcamb, Pcamb)

	### interpolate on simulation k
	Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H  = interp_simu1(z,j ,k, kcamb, Pcamb, Pmod_dd, Pmod_dt, Pmod_tt,\
	A, B, C, D, E, F, G, H, 2)
	
	
	print 'polynomial'
	#~ biasF1, biasF2, biasF3, biasF4, biasF1bis, biasF2bis, biasF3bis, biasF4bis = poly(kstop, lb1, lb2, lb3, lb4,\
	#~ errlb1, errlb2, errlb3, errlb4, k, bias1, bias2, bias3, bias4, errb1, errb2, errb3, errb4,Mnu, z, j, case)
	

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




#~ #----------------------------------------------------------------
#~ #----------Tinker, Crocce param bias ----------------------------
#~ #----------------------------------------------------------------
	
	if Mnu == 0:
		#~ #compute tinker stuff
		limM = [5e11,1e12,3e12,1e13, 3.5e15]
		camb = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb = camb[:,0]
		Pcamb = camb[:,1]

		
##########################
		#~ #-------------------------------------------------------------
		dndMbis2=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]

		dndMbis2 = np.nan_to_num(dndMbis2)

###########################################
		
		#~ bt1=np.empty(len(m_middle1),dtype=np.float64)
		#~ bt2=np.empty(len(m_middle2),dtype=np.float64)
		#~ bt3=np.empty(len(m_middle3),dtype=np.float64)
		#~ bt4=np.empty(len(m_middle4),dtype=np.float64)		
		
		### Crocce/Tinker hmf ############################################

		#------------------------------
		bias_eff0_st2=np.sum(dndMbis2*dm2*bst2)/np.sum(dm2*dndMbis2)


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################
	#### plot all different bias test
	if j == z[0]:
		fig2 = plt.figure()
	J = j + 1
	
	if len(z) == 1:
		ax2 = fig2.add_subplot(1, len(z), J)
	elif len(z) == 2:
		ax2 = fig2.add_subplot(1, 2, J)
	elif len(z) > 2:
		ax2 = fig2.add_subplot(2, 2, J)

	###################################################################
	####### comparison bias and != models #############################
	#~ M1, = ax2.plot(k, bias1, label='$simulation$')
	#~ M2, = ax2.plot(k, bias2)
	#~ M3, = ax2.plot(k, bias3)
	#~ M4, = ax2.plot(k, bias4)
	#-----------------------------------------------
	#~ M1, = ax2.plot(k, bias1_0ev, linestyle = '--', color='C0', label=r'$b_{cc , M_{\nu} = 0.0eV} $')
	#~ M2, = ax2.plot(k, bias2_0ev, linestyle = '--', color='C1')
	#~ M3, = ax2.plot(k, bias3_0ev, linestyle = '--', color='C2')
	#~ M4, = ax2.plot(k, bias4_0ev, linestyle = '--', color='C3')
	#---------------------------------------------------
	#~ st2 =ax2.axhline(bias_eff0_t1, color='C0', linestyle='--', label='$Tinker 20008$')
	#~ ax2.axhline(bias_eff0_t2, color='C1', linestyle='--')
	#~ ax2.axhline(bias_eff0_t3, color='C2', linestyle='--')
	#~ ax2.axhline(bias_eff0_t4, color='C3', linestyle='--')
	#---------------------------------------------------
	#~ st2 =ax2.axhline(Bias_eff0_t1, color='C0', linestyle=':', label='$Tinker 2010$')
	#~ ax2.axhline(Bias_eff0_t2, color='C1', linestyle=':')
	#~ ax2.axhline(Bias_eff0_t3, color='C2', linestyle=':')
	#~ ax2.axhline(Bias_eff0_t4, color='C3', linestyle=':')
	#~ #---------------------------------------------------
	#~ st3 =ax2.axhline(bias_eff0_st1, color='C0', linestyle='--')
	#~ ax2.axhline(bias_eff0_st2, color='C1', linestyle='--')
	#~ ax2.axhline(bias_eff0_st3, color='C2', linestyle='--')
	#~ ax2.axhline(bias_eff0_st4, color='C3', linestyle='--')
	#~ #---------------------------------------------------
	#~ st4 =ax2.axhline(bias_eff0_smt1, color='C0', linestyle='-.')
	#~ ax2.axhline(bias_eff0_smt2, color='C1', linestyle='-.')
	#~ ax2.axhline(bias_eff0_smt3, color='C2', linestyle='-.')
	#~ ax2.axhline(bias_eff0_smt4, color='C3', linestyle='-.')
	#~ #---------------------------------------------------
	#~ st4 =ax2.axhline(bias_eff0_m1, color='C0', linestyle='-.', label='$Crocce$')
	#~ ax2.axhline(bias_eff0_m2, color='C1', linestyle='-.')
	#~ ax2.axhline(bias_eff0_m3, color='C2', linestyle='-.')
	#~ ax2.axhline(bias_eff0_m4, color='C3', linestyle='-.')
	#-----------------------------------------------
	#~ ax2.axvline( kk1, color='C0', linestyle=':', label='shot noise = 80% of P(k)')
	#~ ax2.axvline( kk2, color='C1', linestyle=':')
	#~ ax2.axvline( kk3, color='C2', linestyle=':')
	#~ ax2.axvline( kk4, color='C3', linestyle=':')
	#~ ax2.fill_between(k,bias1-errb1, bias1+errb1, alpha=0.6)
	#~ ax2.fill_between(k,bias2-errb2, bias2+errb2, alpha=0.6)
	#~ ax2.fill_between(k,bias3-errb3, bias3+errb3, alpha=0.6)
	#~ ax2.fill_between(k,bias4-errb4, bias4+errb4, alpha=0.6)
	#~ ax2.set_ylim(bias1[0]*0.8,bias4[0]*1.4)
	#~ ax2.set_xlim(8e-3,1)
	#~ plt.figlegend( (M1,M2,M3,M4, st2,st3), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'Sim hmf + Tinker bias', 'rescaled effective bias'), \
	#~ plt.figlegend( (M1,M2,M3,M4, st2,st3, st4), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'Tinker', 'ST', 'SMT'), \
	#~ plt.figlegend( (M1,M2,M3,M4), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$'), \
	####################################################################
	#######################################
	loc = 'upper center', ncol=3, labelspacing=0., title =r' M$\nu$ = '+str(Mnu), fontsize=12)
	ax2.legend(loc = 'upper left', fancybox=True, fontsize=14, handlelength=0, handletextpad=0)
	plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	ax2.set_xscale('log')
	#----------------------------
	if j == 0 :
		ax2.tick_params(bottom='off', labelbottom='off')
		ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$', fontsize = 16)
		ax2.set_ylabel(r'$b_{\rm model}$ / $b_{\rm sim}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$M^2 n(M)$', fontsize = 16)
		#ax2.grid()
	if j == 1 :
		ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$', fontsize = 16)
		ax2.set_ylabel(r'$b_{\rm model}$ / $b_{\rm sim}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$M^2 n(M)$', fontsize = 16)
		ax2.yaxis.set_label_position("right")
		#ax2.grid()
	if j == 2 :
		#~ #ax.tick_params(labelleft=True)
		ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$', fontsize = 16)
		ax2.set_ylabel(r'$b_{\rm model}$ / $b_{\rm sim}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$M^2 n(M)$', fontsize = 16)
		ax2.set_xlabel('k [h/Mpc]', fontsize = 14)
		#~ ax2.set_xlabel(r'M [$h^{-1} M_{\odot}$]', fontsize = 14)
		#ax2.grid()
	if j == 3 :
		ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		ax2.set_xlabel('k [h/Mpc]', fontsize = 16)
		#~ ax2.set_xlabel(r'M [$h^{-1} M_{\odot}$]', fontsize = 16)
		ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$', fontsize = 16)
		ax2.set_ylabel(r'$b_{\rm model}$ / $b_{\rm sim}$', fontsize = 16)
		#~ ax2.set_ylabel(r'$M^2 n(M)$', fontsize = 14)
		ax2.yaxis.set_label_position("right")
		#ax2.grid()
	#ax2.set_xlim(8e-3,0.05)
	if j == len(z) -1:
		plt.show()


end = time()
print 'total time is '+str((end - start))
