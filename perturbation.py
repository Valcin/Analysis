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


def perturb(kstop,  lb1, lb2, lb3, lb4, errlb1, errlb2, errlb3, errlb4, Pmm, k, bias1,\
	bias2, bias3, bias4, errb1, errb2, errb3, errb4, A, B, C, D, E, F,Mnu, z, j, case):
	lim = np.where((k < kstop)&(k > 1e-2))[0]

	
	#~ def funcbias1(Pdd, b1, b2, bs):
		#~ return np.sqrt((b1**2*Pdd + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim])/Pdd)

	#~ def funcbias2(Pdd, b1, b2, bs, b3nl):
		#~ return np.sqrt((b1**2*Pdd + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim] \
		#~ + 2*b1*b3nl*F[lim])/Pdd)

	#~ def funcbias3(Pdd, b1, b2, bs):
		#~ b3nl = 32/315.*(b1-1)
		#~ return np.sqrt((b1**2*Pdd + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim] \
		#~ + 2*b1*b3nl*F[lim])/Pdd)
			
	pop1 = [lb1,1,-4/7.*(lb1-1), 1]
	pop2 = [lb2,1,-4/7.*(lb2-1), 1]
	pop3 = [lb3,1,-4/7.*(lb3-1), 1]
	pop4 = [lb4,1,-4/7.*(lb4-1), 1]

	popbis1 = [lb1,1,-4/7.*(lb1-1),32/315.*(lb1-1), 1]
	popbis2 = [lb2,1,-4/7.*(lb2-1),32/315.*(lb2-1), 1]
	popbis3 = [lb3,1,-4/7.*(lb3-1),32/315.*(lb3-1), 1]
	popbis4 = [lb4,1,-4/7.*(lb4-1),32/315.*(lb4-1), 1]

	popter1 = [lb1,1, 1]
	popter2 = [lb2,1, 1]
	popter3 = [lb3,1, 1]
	popter4 = [lb4,1, 1]





####################################################################
##### compute coefficient with emcee
####################################################################
	# 2nd order bias ----------------------------------------------------------------------------------------------
	b1y1_mcmc, b2y1_mcmc, bsy1_mcmc, Ny1= coeffit_exp1(kstop, Pmm, A, B, C, D, E, lb1,errlb1, pop1, k ,bias1 ,errb1)
	b1y2_mcmc, b2y2_mcmc, bsy2_mcmc, Ny2 = coeffit_exp1(kstop, Pmm, A, B, C, D, E, lb2,errlb2, pop2, k ,bias2 ,errb2)
	b1y3_mcmc, b2y3_mcmc, bsy3_mcmc, Ny3 = coeffit_exp1(kstop, Pmm, A, B, C, D, E, lb3,errlb3, pop3, k ,bias3 ,errb3)
	b1y4_mcmc, b2y4_mcmc, bsy4_mcmc, Ny4 = coeffit_exp1(kstop, Pmm, A, B, C, D, E, lb4,errlb4, pop4, k ,bias4 ,errb4)
	#~ #3rd order free -----------------------------------------------------------------------------------------------
	b1z1_mcmc, b2z1_mcmc, bsz1_mcmc, b3z1_mcmc, Nz1 = coeffit_exp2(kstop, Pmm, A, B, C, D, E, F, lb1, errlb1, popbis1,\
	k ,bias1 ,errb1)
	b1z2_mcmc, b2z2_mcmc, bsz2_mcmc, b3z2_mcmc, Nz2 = coeffit_exp2(kstop, Pmm, A, B, C, D, E, F, lb2, errlb2, popbis2,\
	k ,bias2 ,errb2)
	b1z3_mcmc, b2z3_mcmc, bsz3_mcmc, b3z3_mcmc, Nz3 = coeffit_exp2(kstop, Pmm, A, B, C, D, E, F, lb3, errlb3, popbis3,\
	k ,bias3 ,errb3)
	b1z4_mcmc, b2z4_mcmc, bsz4_mcmc, b3z4_mcmc, Nz4 = coeffit_exp2(kstop, Pmm, A, B, C, D, E, F, lb4, errlb4, popbis4,\
	k ,bias4 ,errb4)
	#~ #-3rd order fixed -------------------------------------------------------------------------------------------------
	b1u1_mcmc, b2u1_mcmc, Nu1= coeffit_exp3(kstop, Pmm, A, B, C, D, E, F, lb1, errlb1, popter1,\
	k ,bias1 ,errb1)
	b1u2_mcmc, b2u2_mcmc, Nu2 = coeffit_exp3(kstop, Pmm, A, B, C, D, E, F, lb2, errlb2, popter2,\
	k ,bias2 ,errb2)
	b1u3_mcmc, b2u3_mcmc, Nu3 = coeffit_exp3(kstop, Pmm, A, B, C, D, E, F, lb3, errlb3, popter3,\
	k ,bias3 ,errb3)
	b1u4_mcmc, b2u4_mcmc, Nu4 = coeffit_exp3(kstop, Pmm, A, B, C, D, E, F, lb4, errlb4, popter4,\
	k ,bias4 ,errb4)
		
#~ ########################################################################
#~ ########################################################################
	# 2nd order ------------------------------------------------------------------ 
	#~ bias2PT1 = np.sqrt((b1y1_mcmc[0]**2 * Pmm+ b1y1_mcmc[0]*b2y1_mcmc[0]*A + 1/4.*b2y1_mcmc[0]**2*B + b1y1_mcmc[0]*bsy1_mcmc[0]*C +\
	#~ 1/2.*b2y1_mcmc[0]*bsy1_mcmc[0]*D + 1/4.*bsy1_mcmc[0]**2*E )/Pmm)
	#~ bias2PT2 = np.sqrt((b1y2_mcmc[0]**2 * Pmm+ b1y2_mcmc[0]*b2y2_mcmc[0]*A + 1/4.*b2y2_mcmc[0]**2*B + b1y2_mcmc[0]*bsy2_mcmc[0]*C +\
	#~ 1/2.*b2y2_mcmc[0]*bsy2_mcmc[0]*D + 1/4.*bsy2_mcmc[0]**2*E )/Pmm)
	#~ bias2PT3 = np.sqrt((b1y3_mcmc[0]**2 * Pmm+ b1y3_mcmc[0]*b2y3_mcmc[0]*A + 1/4.*b2y3_mcmc[0]**2*B + b1y3_mcmc[0]*bsy3_mcmc[0]*C +\
	#~ 1/2.*b2y3_mcmc[0]*bsy3_mcmc[0]*D + 1/4.*bsy3_mcmc[0]**2*E )/Pmm)
	#~ bias2PT4 = np.sqrt((b1y4_mcmc[0]**2 * Pmm+ b1y4_mcmc[0]*b2y4_mcmc[0]*A + 1/4.*b2y4_mcmc[0]**2*B + b1y4_mcmc[0]*bsy4_mcmc[0]*C +\
	#~ 1/2.*b2y4_mcmc[0]*bsy4_mcmc[0]*D + 1/4.*bsy4_mcmc[0]**2*E )/Pmm)
	bias2PT1 = np.sqrt((b1y1_mcmc**2 * Pmm+ b1y1_mcmc*b2y1_mcmc*A + 1/4.*b2y1_mcmc**2*B + b1y1_mcmc*bsy1_mcmc*C +\
	1/2.*b2y1_mcmc*bsy1_mcmc*D + 1/4.*bsy1_mcmc**2*E + Ny1)/Pmm)
	bias2PT2 = np.sqrt((b1y2_mcmc**2 * Pmm+ b1y2_mcmc*b2y2_mcmc*A + 1/4.*b2y2_mcmc**2*B + b1y2_mcmc*bsy2_mcmc*C +\
	1/2.*b2y2_mcmc*bsy2_mcmc*D + 1/4.*bsy2_mcmc**2*E + Ny2)/Pmm)
	bias2PT3 = np.sqrt((b1y3_mcmc**2 * Pmm+ b1y3_mcmc*b2y3_mcmc*A + 1/4.*b2y3_mcmc**2*B + b1y3_mcmc*bsy3_mcmc*C +\
	1/2.*b2y3_mcmc*bsy3_mcmc*D + 1/4.*bsy3_mcmc**2*E + Ny3)/Pmm)
	bias2PT4 = np.sqrt((b1y4_mcmc**2 * Pmm+ b1y4_mcmc*b2y4_mcmc*A + 1/4.*b2y4_mcmc**2*B + b1y4_mcmc*bsy4_mcmc*C +\
	1/2.*b2y4_mcmc*bsy4_mcmc*D + 1/4.*bsy4_mcmc**2*E + Ny4)/Pmm)

	#~ # 3rd order free -------------------------------------------------------------------
	#~ bias3PT1 = np.sqrt((b1z1_mcmc[0]**2 * Pmm+ b1z1_mcmc[0]*b2z1_mcmc[0]*A + 1/4.*b2z1_mcmc[0]**2*B + b1z1_mcmc[0]*bsz1_mcmc[0]*C +\
	#~ 1/2.*b2z1_mcmc[0]*bsz1_mcmc[0]*D + 1/4.*bsz1_mcmc[0]**2*E + 2*b1z1_mcmc[0]*b3z1_mcmc[0]*F)/Pmm)
	#~ bias3PT2 = np.sqrt((b1z2_mcmc[0]**2 * Pmm+ b1z2_mcmc[0]*b2z2_mcmc[0]*A + 1/4.*b2z2_mcmc[0]**2*B + b1z2_mcmc[0]*bsz2_mcmc[0]*C +\
	#~ 1/2.*b2z2_mcmc[0]*bsz2_mcmc[0]*D + 1/4.*bsz2_mcmc[0]**2*E + 2*b1z2_mcmc[0]*b3z2_mcmc[0]*F)/Pmm)
	#~ bias3PT3 = np.sqrt((b1z3_mcmc[0]**2 * Pmm+ b1z3_mcmc[0]*b2z3_mcmc[0]*A + 1/4.*b2z3_mcmc[0]**2*B + b1z3_mcmc[0]*bsz3_mcmc[0]*C +\
	#~ 1/2.*b2z3_mcmc[0]*bsz3_mcmc[0]*D + 1/4.*bsz3_mcmc[0]**2*E + 2*b1z3_mcmc[0]*b3z3_mcmc[0]*F)/Pmm)
	#~ bias3PT4 = np.sqrt((b1z4_mcmc[0]**2 * Pmm+ b1z4_mcmc[0]*b2z4_mcmc[0]*A + 1/4.*b2z4_mcmc[0]**2*B + b1z4_mcmc[0]*bsz4_mcmc[0]*C +\
	#~ 1/2.*b2z4_mcmc[0]*bsz4_mcmc[0]*D + 1/4.*bsz4_mcmc[0]**2*E + 2*b1z4_mcmc[0]*b3z4_mcmc[0]*F)/Pmm)
	bias3PT1 = np.sqrt((b1z1_mcmc**2 * Pmm+ b1z1_mcmc*b2z1_mcmc*A + 1/4.*b2z1_mcmc**2*B + b1z1_mcmc*bsz1_mcmc*C +\
	1/2.*b2z1_mcmc*bsz1_mcmc*D + 1/4.*bsz1_mcmc**2*E + 2*b1z1_mcmc*b3z1_mcmc*F + Nz1)/Pmm)
	bias3PT2 = np.sqrt((b1z2_mcmc**2 * Pmm+ b1z2_mcmc*b2z2_mcmc*A + 1/4.*b2z2_mcmc**2*B + b1z2_mcmc*bsz2_mcmc*C +\
	1/2.*b2z2_mcmc*bsz2_mcmc*D + 1/4.*bsz2_mcmc**2*E + 2*b1z2_mcmc*b3z2_mcmc*F + Nz2)/Pmm)
	bias3PT3 = np.sqrt((b1z3_mcmc**2 * Pmm+ b1z3_mcmc*b2z3_mcmc*A + 1/4.*b2z3_mcmc**2*B + b1z3_mcmc*bsz3_mcmc*C +\
	1/2.*b2z3_mcmc*bsz3_mcmc*D + 1/4.*bsz3_mcmc**2*E + 2*b1z3_mcmc*b3z3_mcmc*F + Nz3)/Pmm)
	bias3PT4 = np.sqrt((b1z4_mcmc**2 * Pmm+ b1z4_mcmc*b2z4_mcmc*A + 1/4.*b2z4_mcmc**2*B + b1z4_mcmc*bsz4_mcmc*C +\
	1/2.*b2z4_mcmc*bsz4_mcmc*D + 1/4.*bsz4_mcmc**2*E + 2*b1z4_mcmc*b3z4_mcmc*F + Nz4)/Pmm)
	#~ # 3rd order fixed --------------------------------------------------------------------------------
	#~ BsTa = -4/7.*(b1u1_mcmc[0]-1)
	#~ BsTb = -4/7.*(b1u2_mcmc[0]-1)
	#~ BsTc = -4/7.*(b1u3_mcmc[0]-1)
	#~ BsTd = -4/7.*(b1u4_mcmc[0]-1)
	#~ B3nlTa = 32/315.*(b1u1_mcmc[0]-1)
	#~ B3nlTb = 32/315.*(b1u2_mcmc[0]-1)
	#~ B3nlTc = 32/315.*(b1u3_mcmc[0]-1)
	#~ B3nlTd = 32/315.*(b1u4_mcmc[0]-1)
	#~ bias3PTbis1 = np.sqrt((b1u1_mcmc[0]**2 * Pmm+ b1u1_mcmc[0]*b2u1_mcmc[0]*A + 1/4.*b2u1_mcmc[0]**2*B + b1u1_mcmc[0]*BsTa*C +\
	#~ 1/2.*b2u1_mcmc[0]*BsTa*D + 1/4.*BsTa**2*E + 2*b1u1_mcmc[0]*B3nlTa*F)/Pmm)
	#~ bias3PTbis2 = np.sqrt((b1u2_mcmc[0]**2 * Pmm+ b1u2_mcmc[0]*b2u2_mcmc[0]*A + 1/4.*b2u2_mcmc[0]**2*B + b1u2_mcmc[0]*BsTb*C +\
	#~ 1/2.*b2u2_mcmc[0]*BsTb*D + 1/4.*BsTb**2*E + 2*b1u2_mcmc[0]*B3nlTb*F)/Pmm)
	#~ bias3PTbis3 = np.sqrt((b1u3_mcmc[0]**2 * Pmm+ b1u3_mcmc[0]*b2u3_mcmc[0]*A + 1/4.*b2u3_mcmc[0]**2*B + b1u3_mcmc[0]*BsTc*C +\
	#~ 1/2.*b2u3_mcmc[0]*BsTc*D + 1/4.*BsTc**2*E + 2*b1u3_mcmc[0]*B3nlTc*F)/Pmm)
	#~ bias3PTbis4 = np.sqrt((b1u4_mcmc[0]**2 * Pmm+ b1u4_mcmc[0]*b2u4_mcmc[0]*A + 1/4.*b2u4_mcmc[0]**2*B + b1u4_mcmc[0]*BsTd*C +\
	#~ 1/2.*b2u4_mcmc[0]*BsTd*D + 1/4.*BsTd**2*E + 2*b1u4_mcmc[0]*B3nlTd*F)/Pmm)
	BsTa = -4/7.*(b1u1_mcmc-1)
	BsTb = -4/7.*(b1u2_mcmc-1)
	BsTc = -4/7.*(b1u3_mcmc-1)
	BsTd = -4/7.*(b1u4_mcmc-1)
	B3nlTa = 32/315.*(b1u1_mcmc-1)
	B3nlTb = 32/315.*(b1u2_mcmc-1)
	B3nlTc = 32/315.*(b1u3_mcmc-1)
	B3nlTd = 32/315.*(b1u4_mcmc-1)
	bias3PTbis1 = np.sqrt((b1u1_mcmc**2 * Pmm+ b1u1_mcmc*b2u1_mcmc*A + 1/4.*b2u1_mcmc**2*B + b1u1_mcmc*BsTa*C +\
	1/2.*b2u1_mcmc*BsTa*D + 1/4.*BsTa**2*E + 2*b1u1_mcmc*B3nlTa*F + Nu1)/Pmm)
	bias3PTbis2 = np.sqrt((b1u2_mcmc**2 * Pmm+ b1u2_mcmc*b2u2_mcmc*A + 1/4.*b2u2_mcmc**2*B + b1u2_mcmc*BsTb*C +\
	1/2.*b2u2_mcmc*BsTb*D + 1/4.*BsTb**2*E + 2*b1u2_mcmc*B3nlTb*F + Nu2)/Pmm)
	bias3PTbis3 = np.sqrt((b1u3_mcmc**2 * Pmm+ b1u3_mcmc*b2u3_mcmc*A + 1/4.*b2u3_mcmc**2*B + b1u3_mcmc*BsTc*C +\
	1/2.*b2u3_mcmc*BsTc*D + 1/4.*BsTc**2*E + 2*b1u3_mcmc*B3nlTc*F + Nu3)/Pmm)
	bias3PTbis4 = np.sqrt((b1u4_mcmc**2 * Pmm+ b1u4_mcmc*b2u4_mcmc*A + 1/4.*b2u4_mcmc**2*B + b1u4_mcmc*BsTd*C +\
	1/2.*b2u4_mcmc*BsTd*D + 1/4.*BsTd**2*E + 2*b1u4_mcmc*B3nlTd*F + Nu4)/Pmm)
	
	#~ with open('3rdorder_'+str(z[j])+'.txt', 'a') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop, b1z1_mcmc[0], b1z2_mcmc[0], b1z3_mcmc[0],\
		#~ b1z4_mcmc[0], b3z1_mcmc[0], b3z2_mcmc[0], b3z3_mcmc[0], b3z4_mcmc[0]))
	#~ fid_file.close()
	#~ with open('3rdorder_'+str(z[j])+'.txt', 'a') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (kstop, b1z1_mcmc, b1z2_mcmc, b1z3_mcmc,\
		#~ b1z4_mcmc, b3z1_mcmc, b3z2_mcmc, b3z3_mcmc, b3z4_mcmc))
	#~ fid_file.close()

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
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u1_mcmc[0], b2u1_mcmc[0], BsTa))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u2_mcmc[0], b2u2_mcmc[0], BsTb))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u3_mcmc[0], b2u3_mcmc[0], BsTc))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1u4_mcmc[0], b2u4_mcmc[0], BsTd))
	#~ fid_file.close()
	#~ with open(cname3errbis, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1u1_mcmc[1], b2u1_mcmc[1]\
		#~ ,b1u1_mcmc[2], b2u1_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1u2_mcmc[1], b2u2_mcmc[1]\
		#~ ,b1u2_mcmc[2], b2u2_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1u3_mcmc[1], b2u3_mcmc[1]\
		#~ ,b1u3_mcmc[2], b2u3_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1u4_mcmc[1], b2u4_mcmc[1]\
		#~ ,b1u4_mcmc[2], b2u4_mcmc[2]))
	#~ fid_file.close()




#####################################################################
#####################################################################
	
	
	#~ Mpt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ Mpt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ Mpt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	
	
	#~ bpt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ bpt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ bpt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	
	
	#~ b1pt2 = bpt2[:,0]; b1pt3 = bpt3[:,0]; b1pt3bis = bpt3bis[:,0]
	#~ b2pt2 = bpt2[:,1]; b2pt3 = bpt3[:,1]; b2pt3bis = bpt3bis[:,1]
	#~ bspt2 = bpt2[:,2]; bspt3 = bpt3[:,2]; bspt3bis = bpt3bis[:,2]
	#~ b3pt3 = bpt3[:,3]
	
	
	#~ bias2PT1 = np.sqrt((b1pt2[0]**2 * Pmmbis+ b1pt2[0]*b2pt2[0]*A + 1/4.*b2pt2[0]**2*B + b1pt2[0]*bspt2[0]*C +\
	#~ 1/2.*b2pt2[0]*bspt2[0]*D + 1/4.*bspt2[0]**2*E )/Pmmbis)
	#~ bias2PT2 = np.sqrt((b1pt2[1]**2 * Pmmbis+ b1pt2[1]*b2pt2[1]*A + 1/4.*b2pt2[1]**2*B + b1pt2[1]*bspt2[1]*C +\
	#~ 1/2.*b2pt2[1]*bspt2[1]*D + 1/4.*bspt2[1]**2*E )/Pmmbis)
	#~ bias2PT3 = np.sqrt((b1pt2[2]**2 * Pmmbis+ b1pt2[2]*b2pt2[2]*A + 1/4.*b2pt2[2]**2*B + b1pt2[2]*bspt2[2]*C +\
	#~ 1/2.*b2pt2[2]*bspt2[2]*D + 1/4.*bspt2[2]**2*E )/Pmmbis)
	#~ bias2PT4 = np.sqrt((b1pt2[3]**2 * Pmmbis+ b1pt2[3]*b2pt2[3]*A + 1/4.*b2pt2[3]**2*B + b1pt2[3]*bspt2[3]*C +\
	#~ 1/2.*b2pt2[3]*bspt2[3]*D + 1/4.*bspt2[3]**2*E )/Pmmbis)
	


	# 3rd order free -------------------------------------------------------------------
	#~ bias3PT1 = np.sqrt((b1pt3[0]**2 * Pmmbis+ b1pt3[0]*b2pt3[0]*A + 1/4.*b2pt3[0]**2*B + b1pt3[0]*bspt3[0]*C +\
	#~ 1/2.*b2pt3[0]*bspt3[0]*D + 1/4.*bspt3[0]**2*E + 2*b1pt3[0]*b3pt3[0]*F)/Pmmbis)
	#~ bias3PT2 = np.sqrt((b1pt3[1]**2 * Pmmbis+ b1pt3[1]*b2pt3[1]*A + 1/4.*b2pt3[1]**2*B + b1pt3[1]*bspt3[1]*C +\
	#~ 1/2.*b2pt3[1]*bspt3[1]*D + 1/4.*bspt3[1]**2*E + 2*b1pt3[1]*b3pt3[1]*F)/Pmmbis)
	#~ bias3PT3 = np.sqrt((b1pt3[2]**2 * Pmmbis+ b1pt3[2]*b2pt3[2]*A + 1/4.*b2pt3[2]**2*B + b1pt3[2]*bspt3[2]*C +\
	#~ 1/2.*b2pt3[2]*bspt3[2]*D + 1/4.*bspt3[2]**2*E + 2*b1pt3[2]*b3pt3[2]*F)/Pmmbis)
	#~ bias3PT4 = np.sqrt((b1pt3[3]**2 * Pmmbis+ b1pt3[3]*b2pt3[3]*A + 1/4.*b2pt3[3]**2*B + b1pt3[3]*bspt3[3]*C +\
	#~ 1/2.*b2pt3[3]*bspt3[3]*D + 1/4.*bspt3[3]**2*E + 2*b1pt3[3]*b3pt3[3]*F)/Pmmbis)
	
	# 3rd order fixed --------------------------------------------------------------------------------
	#~ B3nlTa = 32/315.*(b1pt3bis[0]-1)
	#~ B3nlTb = 32/315.*(b1pt3bis[1]-1)
	#~ B3nlTc = 32/315.*(b1pt3bis[2]-1)
	#~ B3nlTd = 32/315.*(b1pt3bis[3]-1)
	
	
	#~ bias3PTbis1 = np.sqrt((b1pt3bis[0]**2 * Pmmbis+ b1pt3bis[0]*b2pt3bis[0]*A + 1/4.*b2pt3bis[0]**2*B + b1pt3bis[0]*bspt3bis[0]*C +\
	#~ 1/2.*b2pt3bis[0]*bspt3bis[0]*D + 1/4.*bspt3bis[0]**2*E + 2*b1pt3bis[0]*B3nlTa*F)/Pmmbis)
	#~ bias3PTbis2 = np.sqrt((b1pt3bis[1]**2 * Pmmbis+ b1pt3bis[1]*b2pt3bis[1]*A + 1/4.*b2pt3bis[1]**2*B + b1pt3bis[1]*bspt3bis[1]*C +\
	#~ 1/2.*b2pt3bis[1]*bspt3bis[1]*D + 1/4.*bspt3bis[1]**2*E + 2*b1pt3bis[1]*B3nlTb*F)/Pmmbis)
	#~ bias3PTbis3 = np.sqrt((b1pt3bis[2]**2 * Pmmbis+ b1pt3bis[2]*b2pt3bis[2]*A + 1/4.*b2pt3bis[2]**2*B + b1pt3bis[2]*bspt3bis[2]*C +\
	#~ 1/2.*b2pt3bis[2]*bspt3bis[2]*D + 1/4.*bspt3bis[2]**2*E + 2*b1pt3bis[2]*B3nlTc*F)/Pmmbis)
	#~ bias3PTbis4 = np.sqrt((b1pt3bis[3]**2 * Pmmbis+ b1pt3bis[3]*b2pt3bis[3]*A + 1/4.*b2pt3bis[3]**2*B + b1pt3bis[3]*bspt3bis[3]*C +\
	#~ 1/2.*b2pt3bis[3]*bspt3bis[3]*D + 1/4.*bspt3bis[3]**2*E + 2*b1pt3bis[3]*B3nlTd*F)/Pmmbis)
	
#~ ####################################################################
	#~ PsptD1r1 = b1pt2[0]**2 * Pmmbis+ b1pt2[0]*b2pt2[0]*A + 1/4.*b2pt2[0]**2*B + b1pt2[0]*bspt2[0]*C +\
	#~ 1/2.*b2pt2[0]*bspt2[0]*D + 1/4.*bspt2[0]**2*E 
	#~ #------------------------------------------------------
	#~ PsptD2r1 = 2*b1pt3[0]*b3pt3[0]*F
	#------------------------------------------------------
	#~ PsptD3r1 =  2*b1pt3bis[0]*B3nlTa*F
	

####################################################################
####################################################################

	#~ return bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	#~ bias3PTbis2, bias3PTbis3, bias3PTbis4, PsptD1r1, PsptD2r1, PsptD3r1
	
	return bias2PT1, bias2PT2, bias2PT3, bias2PT4, bias3PT1, bias3PT2, bias3PT3, bias3PT4, bias3PTbis1,\
	bias3PTbis2, bias3PTbis3, bias3PTbis4
	
	#~ return  bias3PT1, bias3PT2, bias3PT3, bias3PT4
	
	

