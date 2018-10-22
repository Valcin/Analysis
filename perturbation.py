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
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma
from fit_emcee import coeffit_pl,coeffit_pl2,coeffit_exp1, coeffit_exp2, coeffit_exp3,coeffit_Kaiser, coeffit_Scocci, coeffit_TNS, coeffit_eTNS


def perturb(kstop, k,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmmbis, kbis, bias1bis,\
	bias2bis, bias3bis, bias4bis, errb1bis, errb2bis, errb3bis, errb4bis, A, B, C, D, E, F,Mnu, z, j, case):
	lim = np.where(k < kstop)[0]

	print len(A)
	print len(A[lim])
	
	def funcbias1(Pdd, b1, b2, bs):
		return np.sqrt((b1**2*Pdd + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim])/Pdd)

	def funcbias2(Pdd, b1, b2, bs, b3nl):
		return np.sqrt((b1**2*Pdd + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim] \
		+ 2*b1*b3nl*F[lim])/Pdd)

	def funcbias3(Pdd, b1, b2, bs):
		b3nl = 32/315.*(b1-1)
		return np.sqrt((b1**2*Pdd + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim] \
		+ 2*b1*b3nl*F[lim])/Pdd)
			
	#~ pop1 = [lb1,1,-4/7.*(lb1-1)]
	#~ pop2 = [lb2,1,-4/7.*(lb2-1)]
	#~ pop3 = [lb3,1,-4/7.*(lb3-1)]
	#~ pop4 = [lb4,1,-4/7.*(lb4-1)]

	#~ popbis1 = [lb1,1,-4/7.*(lb1-1),32/315.*(lb1-1)]
	#~ popbis2 = [lb2,1,-4/7.*(lb2-1),32/315.*(lb2-1)]
	#~ popbis3 = [lb3,1,-4/7.*(lb3-1),32/315.*(lb3-1)]
	#~ popbis4 = [lb4,1,-4/7.*(lb4-1),32/315.*(lb4-1)]

	#~ popter1 = [lb1,1,-4/7.*(lb1-1)]
	#~ popter2 = [lb2,1,-4/7.*(lb2-1)]
	#~ popter3 = [lb3,1,-4/7.*(lb3-1)]
	#~ popter4 = [lb4,1,-4/7.*(lb4-1)]

	pop1, pcov1 = curve_fit(funcbias1, Pmmbis[lim], bias1bis[lim], sigma = errb1bis[lim], check_finite=True, maxfev=500000)
	pop2, pcov2 = curve_fit(funcbias1, Pmmbis[lim], bias2bis[lim], sigma = errb2bis[lim], check_finite=True, maxfev=500000)
	pop3, pcov3 = curve_fit(funcbias1, Pmmbis[lim], bias3bis[lim], sigma = errb3bis[lim], check_finite=True, maxfev=500000)
	pop4, pcov4 = curve_fit(funcbias1, Pmmbis[lim], bias4bis[lim], sigma = errb4bis[lim], check_finite=True, maxfev=500000)

	popbis1, pcovbis1 = curve_fit(funcbias2, Pmmbis[lim], bias1bis[lim], sigma = errb1bis[lim],check_finite=True, maxfev=500000)
	popbis2, pcovbis2 = curve_fit(funcbias2, Pmmbis[lim], bias2bis[lim], sigma = errb2bis[lim],check_finite=True, maxfev=500000)
	popbis3, pcovbis3 = curve_fit(funcbias2, Pmmbis[lim], bias3bis[lim], sigma = errb3bis[lim],check_finite=True, maxfev=500000)
	popbis4, pcovbis4 = curve_fit(funcbias2, Pmmbis[lim], bias4bis[lim], sigma = errb4bis[lim],check_finite=True, maxfev=500000)

	popter1, pcovter1 = curve_fit(funcbias3, Pmmbis[lim], bias1bis[lim], sigma = errb1bis[lim],check_finite=True, maxfev=500000)
	popter2, pcovter2 = curve_fit(funcbias3, Pmmbis[lim], bias2bis[lim], sigma = errb2bis[lim],check_finite=True, maxfev=500000)
	popter3, pcovter3 = curve_fit(funcbias3, Pmmbis[lim], bias3bis[lim], sigma = errb3bis[lim],check_finite=True, maxfev=500000)
	popter4, pcovter4 = curve_fit(funcbias3, Pmmbis[lim], bias4bis[lim], sigma = errb4bis[lim],check_finite=True, maxfev=500000)




####################################################################
##### compute coefficient with emcee
####################################################################
	# 2nd order bias ----------------------------------------------------------------------------------------------
	#~ b1y1_mcmc, b2y1_mcmc, bsy1_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb1,errlb1, pop1, kbis ,bias1bis ,errb1bis)
	#~ b1y2_mcmc, b2y2_mcmc, bsy2_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb2,errlb2, pop2, kbis ,bias2bis ,errb2bis)
	#~ b1y3_mcmc, b2y3_mcmc, bsy3_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb3,errlb3, pop3, kbis ,bias3bis ,errb3bis)
	#~ b1y4_mcmc, b2y4_mcmc, bsy4_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb4,errlb4, pop4, kbis ,bias4bis ,errb4bis)
	#~ #3rd order free -----------------------------------------------------------------------------------------------
	#~ b1z1_mcmc, b2z1_mcmc, bsz1_mcmc, b3z1_mcmc = coeffit_exp2(kstop, Pmmbis, A, B, C, D, E, F, lb1, errlb1, popbis1,\
	#~ kbis ,bias1bis ,errb1bis)
	#~ b1z2_mcmc, b2z2_mcmc, bsz2_mcmc, b3z2_mcmc = coeffit_exp2(kstop, Pmmbis, A, B, C, D, E, F, lb2, errlb2, popbis2,\
	#~ kbis ,bias2bis ,errb2bis)
	#~ b1z3_mcmc, b2z3_mcmc, bsz3_mcmc, b3z3_mcmc = coeffit_exp2(kstop, Pmmbis, A, B, C, D, E, F, lb3, errlb3, popbis3,\
	#~ kbis ,bias3bis ,errb3bis)
	#~ b1z4_mcmc, b2z4_mcmc, bsz4_mcmc, b3z4_mcmc = coeffit_exp2(kstop, Pmmbis, A, B, C, D, E, F, lb4, errlb4, popbis4,\
	#~ kbis ,bias4bis ,errb4bis)
	#~ #-3rd order fixed -------------------------------------------------------------------------------------------------
	#~ b1u1_mcmc, b2u1_mcmc, bsu1_mcmc = coeffit_exp3(kstop, Pmmbis, A, B, C, D, E, F, lb1, errlb1, popter1,\
	#~ kbis ,bias1bis ,errb1bis)
	#~ b1u2_mcmc, b2u2_mcmc, bsu2_mcmc = coeffit_exp3(kstop, Pmmbis, A, B, C, D, E, F, lb2, errlb2, popter2,\
	#~ kbis ,bias2bis ,errb2bis)
	#~ b1u3_mcmc, b2u3_mcmc, bsu3_mcmc = coeffit_exp3(kstop, Pmmbis, A, B, C, D, E, F, lb3, errlb3, popter3,\
	#~ kbis ,bias3bis ,errb3bis)
	#~ b1u4_mcmc, b2u4_mcmc, bsu4_mcmc = coeffit_exp3(kstop, Pmmbis, A, B, C, D, E, F, lb4, errlb4, popter4,\
	#~ kbis ,bias4bis ,errb4bis)
		
########################################################################
########################################################################
	# 2nd order ------------------------------------------------------------------ 
	#~ bias2PT1 = np.sqrt((b1y1_mcmc[0]**2 * Pmmbis+ b1y1_mcmc[0]*b2y1_mcmc[0]*A + 1/4.*b2y1_mcmc[0]**2*B + b1y1_mcmc[0]*bsy1_mcmc[0]*C +\
	#~ 1/2.*b2y1_mcmc[0]*bsy1_mcmc[0]*D + 1/4.*bsy1_mcmc[0]**2*E )/Pmmbis)
	#~ bias2PT2 = np.sqrt((b1y2_mcmc[0]**2 * Pmmbis+ b1y2_mcmc[0]*b2y2_mcmc[0]*A + 1/4.*b2y2_mcmc[0]**2*B + b1y2_mcmc[0]*bsy2_mcmc[0]*C +\
	#~ 1/2.*b2y2_mcmc[0]*bsy2_mcmc[0]*D + 1/4.*bsy2_mcmc[0]**2*E )/Pmmbis)
	#~ bias2PT3 = np.sqrt((b1y3_mcmc[0]**2 * Pmmbis+ b1y3_mcmc[0]*b2y3_mcmc[0]*A + 1/4.*b2y3_mcmc[0]**2*B + b1y3_mcmc[0]*bsy3_mcmc[0]*C +\
	#~ 1/2.*b2y3_mcmc[0]*bsy3_mcmc[0]*D + 1/4.*bsy3_mcmc[0]**2*E )/Pmmbis)
	#~ bias2PT4 = np.sqrt((b1y4_mcmc[0]**2 * Pmmbis+ b1y4_mcmc[0]*b2y4_mcmc[0]*A + 1/4.*b2y4_mcmc[0]**2*B + b1y4_mcmc[0]*bsy4_mcmc[0]*C +\
	#~ 1/2.*b2y4_mcmc[0]*bsy4_mcmc[0]*D + 1/4.*bsy4_mcmc[0]**2*E )/Pmmbis)

	# 3rd order free -------------------------------------------------------------------
	#~ bias3PT1 = np.sqrt((b1z1_mcmc[0]**2 * Pmmbis+ b1z1_mcmc[0]*b2z1_mcmc[0]*A + 1/4.*b2z1_mcmc[0]**2*B + b1z1_mcmc[0]*bsz1_mcmc[0]*C +\
	#~ 1/2.*b2z1_mcmc[0]*bsz1_mcmc[0]*D + 1/4.*bsz1_mcmc[0]**2*E + 2*b1z1_mcmc[0]*b3z1_mcmc[0]*F)/Pmmbis)
	#~ bias3PT2 = np.sqrt((b1z2_mcmc[0]**2 * Pmmbis+ b1z2_mcmc[0]*b2z2_mcmc[0]*A + 1/4.*b2z2_mcmc[0]**2*B + b1z2_mcmc[0]*bsz2_mcmc[0]*C +\
	#~ 1/2.*b2z2_mcmc[0]*bsz2_mcmc[0]*D + 1/4.*bsz2_mcmc[0]**2*E + 2*b1z2_mcmc[0]*b3z2_mcmc[0]*F)/Pmmbis)
	#~ bias3PT3 = np.sqrt((b1z3_mcmc[0]**2 * Pmmbis+ b1z3_mcmc[0]*b2z3_mcmc[0]*A + 1/4.*b2z3_mcmc[0]**2*B + b1z3_mcmc[0]*bsz3_mcmc[0]*C +\
	#~ 1/2.*b2z3_mcmc[0]*bsz3_mcmc[0]*D + 1/4.*bsz3_mcmc[0]**2*E + 2*b1z3_mcmc[0]*b3z3_mcmc[0]*F)/Pmmbis)
	#~ bias3PT4 = np.sqrt((b1z4_mcmc[0]**2 * Pmmbis+ b1z4_mcmc[0]*b2z4_mcmc[0]*A + 1/4.*b2z4_mcmc[0]**2*B + b1z4_mcmc[0]*bsz4_mcmc[0]*C +\
	#~ 1/2.*b2z4_mcmc[0]*bsz4_mcmc[0]*D + 1/4.*bsz4_mcmc[0]**2*E + 2*b1z4_mcmc[0]*b3z4_mcmc[0]*F)/Pmmbis)
	# 3rd order fixed --------------------------------------------------------------------------------
	#~ B3nlTa = 32/315.*(b1u1_mcmc[0]-1)
	#~ B3nlTb = 32/315.*(b1u2_mcmc[0]-1)
	#~ B3nlTc = 32/315.*(b1u3_mcmc[0]-1)
	#~ B3nlTd = 32/315.*(b1u4_mcmc[0]-1)
	#~ bias3PTbis1 = np.sqrt((b1u1_mcmc[0]**2 * Pmmbis+ b1u1_mcmc[0]*b2u1_mcmc[0]*A + 1/4.*b2u1_mcmc[0]**2*B + b1u1_mcmc[0]*bsu1_mcmc[0]*C +\
	#~ 1/2.*b2u1_mcmc[0]*bsu1_mcmc[0]*D + 1/4.*bsu1_mcmc[0]**2*E + 2*b1u1_mcmc[0]*B3nlTa*F)/Pmmbis)
	#~ bias3PTbis2 = np.sqrt((b1u2_mcmc[0]**2 * Pmmbis+ b1u2_mcmc[0]*b2u2_mcmc[0]*A + 1/4.*b2u2_mcmc[0]**2*B + b1u2_mcmc[0]*bsu2_mcmc[0]*C +\
	#~ 1/2.*b2u2_mcmc[0]*bsu2_mcmc[0]*D + 1/4.*bsu2_mcmc[0]**2*E + 2*b1u2_mcmc[0]*B3nlTb*F)/Pmmbis)
	#~ bias3PTbis3 = np.sqrt((b1u3_mcmc[0]**2 * Pmmbis+ b1u3_mcmc[0]*b2u3_mcmc[0]*A + 1/4.*b2u3_mcmc[0]**2*B + b1u3_mcmc[0]*bsu3_mcmc[0]*C +\
	#~ 1/2.*b2u3_mcmc[0]*bsu3_mcmc[0]*D + 1/4.*bsu3_mcmc[0]**2*E + 2*b1u3_mcmc[0]*B3nlTc*F)/Pmmbis)
	#~ bias3PTbis4 = np.sqrt((b1u4_mcmc[0]**2 * Pmmbis+ b1u4_mcmc[0]*b2u4_mcmc[0]*A + 1/4.*b2u4_mcmc[0]**2*B + b1u4_mcmc[0]*bsu4_mcmc[0]*C +\
	#~ 1/2.*b2u4_mcmc[0]*bsu4_mcmc[0]*D + 1/4.*bsu4_mcmc[0]**2*E + 2*b1u4_mcmc[0]*B3nlTd*F)/Pmmbis)
	


##########################################################################
##########################################################################

	#~ cname2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname2err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3bis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3errbis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#------------------------------------------------------------------------------------------------

	#~ cname2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname2err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3bis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3errbis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'


	#~ with open(cname2, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y1_mcmc[0], b2y1_mcmc[0], bsy1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y2_mcmc[0], b2y2_mcmc[0], bsy2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y3_mcmc[0], b2y3_mcmc[0], bsy3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1y4_mcmc[0], b2y4_mcmc[0], bsy4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname2err, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1y1_mcmc[1], b2y1_mcmc[1], bsy1_mcmc[1]\
		#~ ,b1y1_mcmc[2], b2y1_mcmc[2], bsy1_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1y2_mcmc[1], b2y2_mcmc[1], bsy2_mcmc[1]\
		#~ ,b1y2_mcmc[2], b2y2_mcmc[2], bsy2_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1y3_mcmc[1], b2y3_mcmc[1], bsy3_mcmc[1]\
		#~ ,b1y3_mcmc[2], b2y3_mcmc[2], bsy3_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1y4_mcmc[1], b2y4_mcmc[1], bsy4_mcmc[1]\
		#~ ,b1y4_mcmc[2], b2y4_mcmc[2], bsy4_mcmc[2]))
	#~ fid_file.close()
	#~ with open(cname3, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z1_mcmc[0], b2z1_mcmc[0], bsz1_mcmc[0], b3z1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z2_mcmc[0], b2z2_mcmc[0], bsz2_mcmc[0], b3z2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z3_mcmc[0], b2z3_mcmc[0], bsz3_mcmc[0], b3z3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1z4_mcmc[0], b2z4_mcmc[0], bsz4_mcmc[0], b3z4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname3err, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1z1_mcmc[1], b2z1_mcmc[1], bsz1_mcmc[1], b3z1_mcmc[1]\
		#~ ,b1z1_mcmc[2], b2z1_mcmc[2], bsz1_mcmc[2], b3z1_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1z2_mcmc[1], b2z2_mcmc[1], bsz2_mcmc[1], b3z2_mcmc[1]\
		#~ ,b1z2_mcmc[2], b2z2_mcmc[2], bsz2_mcmc[2], b3z2_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1z3_mcmc[1], b2z3_mcmc[1], bsz3_mcmc[1], b3z3_mcmc[1]\
		#~ ,b1z3_mcmc[2], b2z3_mcmc[2], bsz3_mcmc[2], b3z3_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1z4_mcmc[1], b2z4_mcmc[1], bsz4_mcmc[1], b3z4_mcmc[1]\
		#~ ,b1z4_mcmc[2], b2z4_mcmc[2], bsz4_mcmc[2], b3z4_mcmc[2]))
	#~ fid_file.close()
	#~ with open(cname3bis, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u1_mcmc[0], b2u1_mcmc[0], bsu1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u2_mcmc[0], b2u2_mcmc[0], bsu2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u3_mcmc[0], b2u3_mcmc[0], bsu3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u4_mcmc[0], b2u4_mcmc[0], bsu4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname3errbis, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1u1_mcmc[1], b2u1_mcmc[1], bsu1_mcmc[1]\
		#~ ,b1u1_mcmc[2], b2u1_mcmc[2], bsu1_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1u2_mcmc[1], b2u2_mcmc[1], bsu2_mcmc[1]\
		#~ ,b1u2_mcmc[2], b2u2_mcmc[2], bsu2_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1u3_mcmc[1], b2u3_mcmc[1], bsu3_mcmc[1]\
		#~ ,b1u3_mcmc[2], b2u3_mcmc[2], bsu3_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1u4_mcmc[1], b2u4_mcmc[1], bsu4_mcmc[1]\
		#~ ,b1u4_mcmc[2], b2u4_mcmc[2], bsu4_mcmc[2]))
	#~ fid_file.close()




#####################################################################
#####################################################################
	
	bpt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	str(Mnu)+'eV/case'+str(case)+'/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	bpt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	bpt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#---------------------------------------------------------------------
	
	#~ Mpt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ Mpt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ Mpt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	
	b1pt2 = bpt2[:,0]; b1pt3 = bpt3[:,0]; b1pt3bis = bpt3bis[:,0]
	b2pt2 = bpt2[:,1]; b2pt3 = bpt3[:,1]; b2pt3bis = bpt3bis[:,1]
	bspt2 = bpt2[:,2]; bspt3 = bpt3[:,2]; bspt3bis = bpt3bis[:,2]
	b3pt3 = bpt3[:,3]
	
	
	
	bias2PT1 = np.sqrt((b1pt2[0]**2 * Pmmbis+ b1pt2[0]*b2pt2[0]*A + 1/4.*b2pt2[0]**2*B + b1pt2[0]*bspt2[0]*C +\
	1/2.*b2pt2[0]*bspt2[0]*D + 1/4.*bspt2[0]**2*E )/Pmmbis)
	bias2PT2 = np.sqrt((b1pt2[1]**2 * Pmmbis+ b1pt2[1]*b2pt2[1]*A + 1/4.*b2pt2[1]**2*B + b1pt2[1]*bspt2[1]*C +\
	1/2.*b2pt2[1]*bspt2[1]*D + 1/4.*bspt2[1]**2*E )/Pmmbis)
	bias2PT3 = np.sqrt((b1pt2[2]**2 * Pmmbis+ b1pt2[2]*b2pt2[2]*A + 1/4.*b2pt2[2]**2*B + b1pt2[2]*bspt2[2]*C +\
	1/2.*b2pt2[2]*bspt2[2]*D + 1/4.*bspt2[2]**2*E )/Pmmbis)
	bias2PT4 = np.sqrt((b1pt2[3]**2 * Pmmbis+ b1pt2[3]*b2pt2[3]*A + 1/4.*b2pt2[3]**2*B + b1pt2[3]*bspt2[3]*C +\
	1/2.*b2pt2[3]*bspt2[3]*D + 1/4.*bspt2[3]**2*E )/Pmmbis)
	


	# 3rd order free -------------------------------------------------------------------
	bias3PT1 = np.sqrt((b1pt3[0]**2 * Pmmbis+ b1pt3[0]*b2pt3[0]*A + 1/4.*b2pt3[0]**2*B + b1pt3[0]*bspt3[0]*C +\
	1/2.*b2pt3[0]*bspt3[0]*D + 1/4.*bspt3[0]**2*E + 2*b1pt3[0]*b3pt3[0]*F)/Pmmbis)
	bias3PT2 = np.sqrt((b1pt3[1]**2 * Pmmbis+ b1pt3[1]*b2pt3[1]*A + 1/4.*b2pt3[1]**2*B + b1pt3[1]*bspt3[1]*C +\
	1/2.*b2pt3[1]*bspt3[1]*D + 1/4.*bspt3[1]**2*E + 2*b1pt3[1]*b3pt3[1]*F)/Pmmbis)
	bias3PT3 = np.sqrt((b1pt3[2]**2 * Pmmbis+ b1pt3[2]*b2pt3[2]*A + 1/4.*b2pt3[2]**2*B + b1pt3[2]*bspt3[2]*C +\
	1/2.*b2pt3[2]*bspt3[2]*D + 1/4.*bspt3[2]**2*E + 2*b1pt3[2]*b3pt3[2]*F)/Pmmbis)
	bias3PT4 = np.sqrt((b1pt3[3]**2 * Pmmbis+ b1pt3[3]*b2pt3[3]*A + 1/4.*b2pt3[3]**2*B + b1pt3[3]*bspt3[3]*C +\
	1/2.*b2pt3[3]*bspt3[3]*D + 1/4.*bspt3[3]**2*E + 2*b1pt3[3]*b3pt3[3]*F)/Pmmbis)
	
	#~ # 3rd order fixed --------------------------------------------------------------------------------
	B3nlTa = 32/315.*(b1pt3bis[0]-1)
	B3nlTb = 32/315.*(b1pt3bis[1]-1)
	B3nlTc = 32/315.*(b1pt3bis[2]-1)
	B3nlTd = 32/315.*(b1pt3bis[3]-1)
	
	bias3PTbis1 = np.sqrt((b1pt3bis[0]**2 * Pmmbis+ b1pt3bis[0]*b2pt3bis[0]*A + 1/4.*b2pt3bis[0]**2*B + b1pt3bis[0]*bspt3bis[0]*C +\
	1/2.*b2pt3bis[0]*bspt3bis[0]*D + 1/4.*bspt3bis[0]**2*E + 2*b1pt3bis[0]*B3nlTa*F)/Pmmbis)
	bias3PTbis2 = np.sqrt((b1pt3bis[1]**2 * Pmmbis+ b1pt3bis[1]*b2pt3bis[1]*A + 1/4.*b2pt3bis[1]**2*B + b1pt3bis[1]*bspt3bis[1]*C +\
	1/2.*b2pt3bis[1]*bspt3bis[1]*D + 1/4.*bspt3bis[1]**2*E + 2*b1pt3bis[1]*B3nlTb*F)/Pmmbis)
	bias3PTbis3 = np.sqrt((b1pt3bis[2]**2 * Pmmbis+ b1pt3bis[2]*b2pt3bis[2]*A + 1/4.*b2pt3bis[2]**2*B + b1pt3bis[2]*bspt3bis[2]*C +\
	1/2.*b2pt3bis[2]*bspt3bis[2]*D + 1/4.*bspt3bis[2]**2*E + 2*b1pt3bis[2]*B3nlTc*F)/Pmmbis)
	bias3PTbis4 = np.sqrt((b1pt3bis[3]**2 * Pmmbis+ b1pt3bis[3]*b2pt3bis[3]*A + 1/4.*b2pt3bis[3]**2*B + b1pt3bis[3]*bspt3bis[3]*C +\
	1/2.*b2pt3bis[3]*bspt3bis[3]*D + 1/4.*bspt3bis[3]**2*E + 2*b1pt3bis[3]*B3nlTd*F)/Pmmbis)
	
####################################################################
	PsptD1r1 = b1pt2[0]**2 * Pmmbis+ b1pt2[0]*b2pt2[0]*A + 1/4.*b2pt2[0]**2*B + b1pt2[0]*bspt2[0]*C +\
	1/2.*b2pt2[0]*bspt2[0]*D + 1/4.*bspt2[0]**2*E 
	#------------------------------------------------------
	PsptD2r1 = 2*b1pt3[0]*b3pt3[0]*F
	#~ #------------------------------------------------------
	PsptD3r1 =  2*b1pt3bis[0]*B3nlTa*F
	

####################################################################
####################################################################

	return bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	bias3PTbis2, bias3PTbis3, bias3PTbis4, PsptD1r1, PsptD2r1, PsptD3r1
	#~ return bias3PTbis1,	bias3PTbis2, bias3PTbis3, bias3PTbis4  
	#~ return bias2PT1, bias2PT2, bias2PT3, bias2PT4
	#~ return  bias3PT1, bias3PT2, bias3PT3, bias3PT4

