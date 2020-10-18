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
from rsd import RSD1, RSD2, RSD3
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import emcee
import sys
sys.path.append('/home/david/codes/FAST-PT')
import myFASTPT as FPT
from scipy.optimize import curve_fit
from scipy.special import erf


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
Mnu       = 0.15 #eV
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
	fz = [0.524,0.759,0.875,0.958]
	Dz = [ 1.,0.77,0.61,0.42]
	print 'For redshift z = ' + str(z[j])
	
	Omeg_m_z = Omega_m * (1 + z[j])**3 / (Omega_m * (1 + z[j])**3 + Omega_l)
	#~ fz = Omeg_m_z**0.55
	j = 1
#########################################################################
#### load data from simualtion 

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	bias1s, bias2s, bias3s, bias4s, errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3,\
	errPr4, kclass, Tm, Tcb, noise1, noise2, noise3, noise4 = ld_data(Mnu, z, j)
	
	kcamb55, Pcamb55, k55, Pmm55, PH155, PH255, PH553 , PH554, err55Phh1, errP55hh2, errPh55h3, e55rrPhh4, bia55s1, bi55as2, bia55s3, b55ias4, \
	bias551s, b55ias2s, b55ias3s, bi55as4s, errb551, e55rrb2, er55rb3, er55rb4, Pmon55o1, P55mono2, Pmo55no3, Pm55ono4, er55rPr1, err55Pr2, err55Pr3,\
	err55Pr4, kc55lass, Tm55, T55cb, noise51, noise552, noise553, noise554 = ld_data(0.0, z, j)
	
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

#-------------------------------------------------------------------
	Tm, Tcb = interp_simu3(z,j ,k, kclass, Tm, Tcb, 2)
	fcc = fz[j] * (Tm/ Tcb)

	#### compute tns coefficeints given mcmc results
	# set the parameters for the power spectrum window and
	# Fourier coefficient window 
	#P_window=np.array([.2,.2])  
	C_window=0.95

	# padding length 
	nu=-2; n_pad=len(kcamb)
	n_pad=int(0.5*len(kcamb))
	to_do=['all']
					
	# initialize the FASTPT class 
	# including extrapolation to higher and lower k  
	# time the operation
	t1=time()
	fastpt=FPT.FASTPT(kcamb,to_do=to_do,n_pad=n_pad, verbose=True) 
	t2=time()

	def coeffit_eTNS2(j, fcc, kstop, Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H, k ,b ,\
		errb,N, noise,Pcamb,fz):
		
		
		#~ lim = np.where(k < kstop)[0]
		lim = np.where((k < kstop)&(k > 1e-2))[0]
		
		def lnlike(theta, x, y, yerr,Pcamb,fz):
			b1, b2, N, sigma = theta
			#~ sigma = theta
			bs = -4/7.*(b1-1)
			b3nl = 32/315.*(b1-1)
			PsptD1z = b1**2*Pmod_dd[lim] + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim] \
			+ 2*b1*b3nl*F[lim] 
			PsptT = b1* Pmod_dt[lim] + b2*G[lim] + bs*H[lim] + b3nl*F[lim]
			kappa = x[lim]*sigma
			AB2,AB4,AB6,AB8 = fastpt.RSD_ABsum_components(Pcamb,fz, b1 ,C_window=C_window)
			#~ coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
			#~ coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
			#~ coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
			#~ coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
			#~ coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
			kappa = x[lim]*sigma*fcc[lim][ind]*Dz[ind]
			coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
			coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
			coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
			coeffD = -2/3./kappa**2*(coeffC - 2/(2+kappa**2))
			coeffE = -4/10./kappa**2*(7.*coeffD - 2/(2+kappa**2))
			
			model = PsptD1z*coeffA + 2/3.*fcc[lim]*PsptT*coeffB + 1/5.*fcc[lim]**2*Pmod_tt[lim]*coeffC \
			+ (1/3.*AB2[lim]*coeffB+ 1/5.*AB4[lim]*coeffC+ 1/7.*AB6[lim]*coeffD+ 1/9.*AB8[lim]*coeffE + N)
			inv_sigma2 = 1.0/(yerr[lim]**2)
			return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
			
		def lnprior(theta):
			b1, b2,N, sigma = theta
			#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1 :
			if 0 < sigma < 100 and  -3*noise < N < 3*noise:
				return 0.0
			return -np.inf
		
		def lnprob(theta, x, y, yerr,Pcamb,fz):
			lp = lnprior(theta)
			if not np.isfinite(lp):
				return -np.inf
			return lp + lnlike(theta, x, y, yerr,Pcamb,fz)
			
		z = [0.0,0.5,1.0,2.0]
		red = ['0.0','0.5','1.0','2.0','3.0']
		ind = red.index(str(z[j]))
		f = [0.518,0.754,0.872,0.956,0.98]
		Dz = [ 1.,0.77,0.61,0.42]

		pop = [1,1,1,6]
		nll = lambda *args: -lnlike(*args)
		result = op.minimize(nll, [pop],  method='Nelder-Mead', args=(k, b ,errb ,Pcamb,fz),  options={'maxfev': 5000} )
		b1_ml, b2_ml,N_ml, sigma_ml  = result["x"]
		print(result)

		return b1_ml, b2_ml,N_ml, sigma_ml 

####################################################################
##### compute coefficient with emcee
####################################################################
		
	bpt3 = np.loadtxt('/home/david/codes/montepython_public/montepython/likelihoods/BE_HaPPy/coefficients/'+\
	str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	b1pt3 = bpt3[:,0]
	b2pt3 = bpt3[:,1]
	bspt3 = bpt3[:,2]
	b3pt3 = bpt3[:,3]
	Npt3 = bpt3[:,4]
	AB2bis_1,AB4bis_1,AB6bis_1,AB8bis_1 = fastpt.RSD_ABsum_components(Pcamb,fz[j],b1pt3[0] ,C_window=C_window)
	
	be1a, be2a, besa, be3a, Nea, betns1 = coeffit_eTNS2(j, fcc, kstop, Pmm, Pmod_dt, Pmod_tt,\
	A, B, C, D, E, F, G, H, k, Pmono1, errPr1, Npt3[0], noise51,Pcamb,fz[j])

end = time()
print 'total time is '+str((end - start))	


 

