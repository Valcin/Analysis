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
Mnu       = 0.15  #eV
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
	
#########################################################################
#### load data from simualtion 

	kcamb, Pcamb, k, Pmm, PH1, PH2, PH3 , PH4, errPhh1, errPhh2, errPhh3, errPhh4, bias1, bias2, bias3, bias4, \
	errb1, errb2, errb3, errb4, Pmono1, Pmono2, Pmono3, Pmono4, errPr1, errPr2, errPr3, errPr4 = ld_data(Mnu, z, j)


#########################################################################

#~ #----------------------------------------------------------------
#~ #----------Tinker, Crocce param bias ----------------------------
#~ #----------------------------------------------------------------
	
	if Mnu == 0:
		#~ #compute tinker stuff
		limM = [5e11,1e12,3e12,1e13, 5e15]
		#~ limM = [4.2e11,1e12,3e12,1e13, 5e15]
		#~ loglim = [ 11.623, 12., 12.477, 13., 15.698]
		loglim = [ 11.698, 12., 12.477, 13., 15.54]
		camb = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb = camb[:,0]
		Pcamb = camb[:,1]

##########################################################################
#### get the mass function from simulation
		massf = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf_z='+str(z[j])+'.txt')
		m_middle = massf[:,10]
		dm = massf[:,11]
		hmf_temp = np.zeros((len(m_middle),10))
		for i in xrange(0,10):
			hmf_temp[:,i]= massf[:,i]
		hmf = np.mean(hmf_temp[:,0:11], axis=1)
		
######## for M1 
		massf1 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf1_z='+str(z[j])+'.txt')
		m_middle1 = massf1[:,10]
		dm1 = massf1[:,12]
		hmf_temp1 = np.zeros((len(m_middle1),10))
		for i in xrange(0,10):
			hmf_temp1[:,i]= massf1[:,i]
		hmf1 = np.mean(hmf_temp1[:,0:11], axis=1)
		#----------------
		bocc1 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/bocc1_z='+str(z[j])+'.txt')
		#~ bocc1 = np.nan_to_num(bocc1)
		M_middle1 = np.mean(bocc1, axis=1)
		#~ x = x[~numpy.isnan(x)]

		#~ print M_middle1
########## for M2
		massf2 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf2_z='+str(z[j])+'.txt')
		m_middle2 = massf2[:,10]
		dm2 = massf2[:,12]
		hmf_temp2 = np.zeros((len(m_middle2),10))
		for i in xrange(0,10):
			hmf_temp2[:,i]= massf2[:,i]
		hmf2 = np.mean(hmf_temp2[:,0:11], axis=1)
		#----------------
		bocc2 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/bocc2_z='+str(z[j])+'.txt')
		#~ bocc2 = np.nan_to_num(bocc2)
		M_middle2 = np.mean(bocc2, axis=1)
		
		#~ print M_middle2
		
########## for M3
		massf3 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf3_z='+str(z[j])+'.txt')
		m_middle3 = massf3[:,10]
		dm3 = massf3[:,12]
		hmf_temp3 = np.zeros((len(m_middle3),10))
		for i in xrange(0,10):
			hmf_temp3[:,i]= massf3[:,i]
		hmf3 = np.mean(hmf_temp3[:,0:11], axis=1)
		#----------------
		bocc3 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/bocc3_z='+str(z[j])+'.txt')
		#~ bocc3 = np.nan_to_num(bocc3)
		M_middle3 = np.mean(bocc3, axis=1)
		
########## for M4
		massf4 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf4_z='+str(z[j])+'.txt')
		m_middle4 = massf4[:,10]
		dm4 = massf4[:,12]
		hmf_temp4 = np.zeros((len(m_middle4),10))
		for i in xrange(0,10):
			hmf_temp4[:,i]= massf4[:,i]
		hmf4 = np.mean(hmf_temp4[:,0:11], axis=1)
		#----------------
		bocc4 = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/bocc4_z='+str(z[j])+'.txt')
		#~ bocc4 = np.nan_to_num(bocc4)
		M_middle4 = np.mean(bocc4, axis=1)
		
##########################
		#-------------------------------------------------------
		#~ dndM1=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[0],limM[4],len(m_middle1), Masses = m_middle1)[1]
		#~ dndM2=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]
		#~ dndM3=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[2],limM[4],len(m_middle3), Masses = m_middle3)[1]
		#~ dndM4=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[3],limM[4],len(m_middle4), Masses = m_middle4)[1]
		#-------------------------------------------------------
		DndM1=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[0],limM[4],len(m_middle1), Masses = m_middle1)[1]
		DndM2=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]
		DndM3=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[2],limM[4],len(m_middle3), Masses = m_middle3)[1]
		DndM4=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[3],limM[4],len(m_middle4), Masses = m_middle4)[1]
		#-------------------------------------------------------------
		dndMbis1=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[0],limM[4],len(m_middle1), Masses = m_middle1)[1]
		dndMbis2=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]
		dndMbis3=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[2],limM[4],len(m_middle3), Masses = m_middle3)[1]
		dndMbis4=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[3],limM[4],len(m_middle4), Masses = m_middle4)[1]
		
		#~ dndM1 = np.nan_to_num(dndM1)
		#~ dndM2 = np.nan_to_num(dndM2)
		#~ dndM3 = np.nan_to_num(dndM3)
		#~ dndM4 = np.nan_to_num(dndM4)
		#~ dndMbis1 = np.nan_to_num(dndMbis1)
		#~ dndMbis2 = np.nan_to_num(dndMbis2)
		#~ dndMbis3 = np.nan_to_num(dndMbis3)
		#~ dndMbis4 = np.nan_to_num(dndMbis4)
###########################################
		
		bt1=np.empty(len(m_middle1),dtype=np.float64)
		bt2=np.empty(len(m_middle2),dtype=np.float64)
		bt3=np.empty(len(m_middle3),dtype=np.float64)
		bt4=np.empty(len(m_middle4),dtype=np.float64)
		#~ bst1=np.empty(len(m_middle1),dtype=np.float64)
		#~ bst2=np.empty(len(m_middle2),dtype=np.float64)
		#~ bst3=np.empty(len(m_middle3),dtype=np.float64)
		#~ bst4=np.empty(len(m_middle4),dtype=np.float64)
		#~ bsmt1=np.empty(len(m_middle1),dtype=np.float64)
		#~ bsmt2=np.empty(len(m_middle2),dtype=np.float64)
		#~ bsmt3=np.empty(len(m_middle3),dtype=np.float64)
		#~ bsmt4=np.empty(len(m_middle4),dtype=np.float64)
		#~ bm1=np.empty(len(m_middle1),dtype=np.float64)
		#~ bm2=np.empty(len(m_middle2),dtype=np.float64)
		#~ bm3=np.empty(len(m_middle3),dtype=np.float64)
		#~ bm4=np.empty(len(m_middle4),dtype=np.float64)
		
		for i in range(len(m_middle1)):
			bt1[i]=bias(kcamb,Pcamb,Omega_m,m_middle1[i],'Tinker')
			bt2[i]=bias(kcamb,Pcamb,Omega_m,m_middle2[i],'Tinker')
			bt3[i]=bias(kcamb,Pcamb,Omega_m,m_middle3[i],'Tinker')
			bt4[i]=bias(kcamb,Pcamb,Omega_m,m_middle4[i],'Tinker')
			#~ #-------------------------------------------
			#~ bst1[i]=bias(kcamb,Pcamb,Omega_m,m_middle1[i],'Crocce')
			#~ bst2[i]=bias(kcamb,Pcamb,Omega_m,m_middle2[i],'Crocce')
			#~ bst3[i]=bias(kcamb,Pcamb,Omega_m,m_middle3[i],'Crocce')
			#~ bst4[i]=bias(kcamb,Pcamb,Omega_m,m_middle4[i],'Crocce')
			#~ #------------------------------------------
			#~ bsmt1[i]=bias(kcamb,Pcamb,Omega_m,m_middle1[i],'SMT01')
			#~ bsmt2[i]=bias(kcamb,Pcamb,Omega_m,m_middle2[i],'SMT01')
			#~ bsmt3[i]=bias(kcamb,Pcamb,Omega_m,m_middle3[i],'SMT01')
			#~ bsmt4[i]=bias(kcamb,Pcamb,Omega_m,m_middle4[i],'SMT01')
			#-----------------------------------------
			#~ bm1[i]=bias(kcamb,Pcamb,Omega_m,m_middle1[i],'Mice')
			#~ bm2[i]=bias(kcamb,Pcamb,Omega_m,m_middle2[i],'Mice')
			#~ bm3[i]=bias(kcamb,Pcamb,Omega_m,m_middle3[i],'Mice')
			#~ bm4[i]=bias(kcamb,Pcamb,Omega_m,m_middle4[i],'Mice')
		
		#~ bt1 = np.nan_to_num(bt1)
		#~ bt2 = np.nan_to_num(bt2)
		#~ bt3 = np.nan_to_num(bt3)
		#~ bt4 = np.nan_to_num(bt4)
		#~ bm1 = np.nan_to_num(bm1)
		#~ bm2 = np.nan_to_num(bm2)
		#~ bm3 = np.nan_to_num(bm3)
		#~ bm4 = np.nan_to_num(bm4)
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle1)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (DndM1[m],DndM2[m],DndM3[m],DndM4[m]))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle1)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (dndMbis1[m], dndMbis2[m], dndMbis3[m], dndMbis4[m]))
		#~ fid_file.close()

		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle1)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (bt1[m],bt2[m],bt3[m],bt4[m]))
				
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bst[m]))
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bsmt[m]))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/bcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(k)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k[index_k], bias1[index_k], bias2[index_k], bias3[index_k], bias4[index_k]))
		#~ fid_file.close()
		
	
		
		### Crocce/Tinker hmf ############################################
		#~ bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		#~ bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
		#~ bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
		#~ #------------------------------
		#~ bias_eff0_t1=np.sum(dndM1*dm1*bt1)/np.sum(dm1*dndM1)
		#~ bias_eff0_t2=np.sum(dndM2*dm2*bt2)/np.sum(dm2*dndM2)
		#~ bias_eff0_t3=np.sum(dndM3*dm3*bt3)/np.sum(dm3*dndM3)
		#~ bias_eff0_t4=np.sum(dndM4*dm4*bt4)/np.sum(dm4*dndM4)
		#------------------------------
		#~ Bias_eff0_t1=np.sum(DndM1*dm1*bt1)/np.sum(dm1*DndM1)
		#~ Bias_eff0_t2=np.sum(DndM2*dm2*bt2)/np.sum(dm2*DndM2)
		#~ Bias_eff0_t3=np.sum(DndM3*dm3*bt3)/np.sum(dm3*DndM3)
		#~ Bias_eff0_t4=np.sum(DndM4*dm4*bt4)/np.sum(dm4*DndM4)
		#------------------------------
		#~ bias_eff0_st1=np.sum(dndMbis1*dm1*bst1)/np.sum(dm1*dndMbis1)
		#~ bias_eff0_st2=np.sum(dndMbis2*dm2*bst2)/np.sum(dm2*dndMbis2)
		#~ bias_eff0_st3=np.sum(dndMbis3*dm3*bst3)/np.sum(dm3*dndMbis3)
		#~ bias_eff0_st4=np.sum(dndMbis4*dm4*bst4)/np.sum(dm4*dndMbis4)
		#------------------------------
		#~ bias_eff0_smt1=np.sum(dndMbis1*dm1*bsmt1)/np.sum(dm1*dndMbis1)
		#~ bias_eff0_smt2=np.sum(dndMbis2*dm2*bsmt2)/np.sum(dm2*dndMbis2)
		#~ bias_eff0_smt3=np.sum(dndMbis3*dm3*bsmt3)/np.sum(dm3*dndMbis3)
		#~ bias_eff0_smt4=np.sum(dndMbis4*dm4*bsmt4)/np.sum(dm4*dndMbis4)
		#~ #------------------------------
		#~ bias_eff0_m1=np.sum(dndMbis1*dm1*bm1)/np.sum(dm1*dndMbis1)
		#~ bias_eff0_m2=np.sum(dndMbis2*dm2*bm2)/np.sum(dm2*dndMbis2)
		#~ bias_eff0_m3=np.sum(dndMbis3*dm3*bm3)/np.sum(dm3*dndMbis3)
		#~ bias_eff0_m4=np.sum(dndMbis4*dm4*bm4)/np.sum(dm4*dndMbis4)
		
		#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff0_t1,Bias_eff0_t2, Bias_eff0_t3, Bias_eff0_t4))
		#~ fid_file.close()
		#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff_t1,Bias_eff_t2, Bias_eff_t3, Bias_eff_t4))
		#~ fid_file.close()
		
		#~ print bias_eff0_m1
		#~ print bias_eff0_m2
		#~ print bias_eff0_m3
		#~ print bias_eff0_m4
		#~ print bias_eff0_t1
		#~ print bias_eff0_t2
		#~ print bias_eff0_t3
		#~ print bias_eff0_t4
		#~ kill
########################################################################################################################################
########################################################################################################################################
#########################################################################################################################################

	if Mnu == 0.15:

		#----------------------------------------------------------------
		#----------Tinker, Crocce param bias ----------------------------
		#----------------------------------------------------------------

		#~ #compute tinker stuff
		##~ limM = [5e11,1e12,3e12,1e13, 3.2e15]
		limM = [4.2e11,1e12,3e12,1e13, 5e15]
		loglim = [ 11.623, 12., 12.477, 13., 15.698]
		camb = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb = camb[:,0]
		Pcamb = camb[:,1]

		#~ #### get the mass function from simulation
		massf = np. loadtxt('/home/david/codes/Paco/data2/0.15eV/hmf_z='+str(z[j])+'.txt')
		m_middle = massf[:,10]
		dm = massf[:,11]
		hmf_temp = np.zeros((len(m_middle),10))
		for i in xrange(0,10):
			hmf_temp[:,i]= massf[:,i]
		hmf = np.mean(hmf_temp[:,0:11], axis=1)
##########################
		#-------------------------------------------------------
		#~ dndM1=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[0],limM[4],len(m_middle1), Masses = m_middle1)[1]
		#~ dndM2=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]
		#~ dndM3=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[2],limM[4],len(m_middle3), Masses = m_middle3)[1]
		#~ dndM4=MFL.Tinker_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[3],limM[4],len(m_middle4), Masses = m_middle4)[1]
		#-------------------------------------------------------
		DndM1=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[0],limM[4],len(m_middle1), Masses = m_middle1)[1]
		DndM2=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]
		DndM3=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[2],limM[4],len(m_middle3), Masses = m_middle3)[1]
		DndM4=MFL.Tinker_2010_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[3],limM[4],len(m_middle4), Masses = m_middle4)[1]
		#-------------------------------------------------------------
		dndMbis1=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[0],limM[4],len(m_middle1), Masses = m_middle1)[1]
		dndMbis2=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[1],limM[4],len(m_middle2), Masses = m_middle2)[1]
		dndMbis3=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[2],limM[4],len(m_middle3), Masses = m_middle3)[1]
		dndMbis4=MFL.Crocce_mass_function(kcamb,Pcamb,Omega_m,z[j],limM[3],limM[4],len(m_middle4), Masses = m_middle4)[1]
		
		#~ dndM1 = np.nan_to_num(dndM1)
		#~ dndM2 = np.nan_to_num(dndM2)
		#~ dndM3 = np.nan_to_num(dndM3)
		#~ dndM4 = np.nan_to_num(dndM4)
		#~ dndMbis1 = np.nan_to_num(dndMbis1)
		#~ dndMbis2 = np.nan_to_num(dndMbis2)
		#~ dndMbis3 = np.nan_to_num(dndMbis3)
		#~ dndMbis4 = np.nan_to_num(dndMbis4)
		
		
		#~ bt=np.empty(len(m_middle),dtype=np.float64)
		#~ bst=np.empty(len(m_middle),dtype=np.float64)
		#~ bsmt=np.empty(len(m_middle),dtype=np.float64)
		#~ for i in range(len(m_middle)):
			#~ bt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Tinker')
			#~ bst[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Crocce')
			#~ bsmt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'SMT01')
		
		
		
		#~ with open('/home/david/codes/Paco/data2/0.15eV/chmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (dndMbis[m]))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.15eV/thmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (dndM[m]))
		#~ fid_file.close()

		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/tlb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bt[m]))
				
		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/clb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bst[m]))
				
		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/clb2_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bsmt[m]))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/LS2_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Cb1,Cb2, Cb3, Cb4))
		#~ fid_file.close()
		
		
		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Tb1,Tb2, Tb3, Tb4))
		#~ fid_file.close()

		
		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/ccl_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (lb1,lb2, lb3, lb4))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/bcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(k)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k[index_k], bias1[index_k], bias2[index_k], bias3[index_k], bias4[index_k]))
		#~ fid_file.close()
		################################################################
		
		dndM = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/thmf_z='+str(z[j])+'.txt')
		dndMbis = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/chmf_z='+str(z[j])+'.txt')
		diff = hmf/dndM
		bin1 = np.where(m_middle > 5e11 )[0]
		bin2 = np.where(m_middle > 1e12 )[0]
		bin3 = np.where(m_middle > 3e12 )[0]
		bin4 = np.where(m_middle > 1e13 )[0]
			
		
		bias_0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/bcc_z='+str(z[j])+'_.txt')
		bias1_0ev = bias_0ev[:,1]
		bias2_0ev = bias_0ev[:,2]
		bias3_0ev = bias_0ev[:,3]
		bias4_0ev = bias_0ev[:,4]

		als = np.where(k < 0.5)[0]


		dndM0bis = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt')
		dndM0 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt')
		massf0 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf_z='+str(z[j])+'.txt')
		hmf_temp0 = np.zeros((len(m_middle),10))
		for i in xrange(0,10):
			hmf_temp0[:,i]= massf0[:,i]
		hmf0 = np.mean(hmf_temp0[:,0:11], axis=1)
		
		#~ ############################################### WITH HMF CROCCE
		#~ #### 0.0 eV ################
		bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
		bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
		#------------------------------
		Bias_eff0_t1=np.sum(dndM0bis[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
		Bias_eff0_t2=np.sum(dndM0bis[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
		Bias_eff0_t3=np.sum(dndM0bis[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
		Bias_eff0_t4=np.sum(dndM0bis[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
		#------------------------------
		Bias_eff0_st1=np.sum(dndM0bis[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
		Bias_eff0_st2=np.sum(dndM0bis[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
		Bias_eff0_st3=np.sum(dndM0bis[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
		Bias_eff0_st4=np.sum(dndM0bis[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
		#------------------------------
		Bias_eff0_smt1=np.sum(dndM0bis[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
		Bias_eff0_smt2=np.sum(dndM0bis[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
		Bias_eff0_smt3=np.sum(dndM0bis[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
		Bias_eff0_smt4=np.sum(dndM0bis[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
		
		#### 0.15 eV ################
		bt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		bst = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb1_z='+str(z[j])+'.txt')
		bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb2_z='+str(z[j])+'.txt')
		#------------------------------
		Bias_eff_t1=np.sum(dndMbis[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*dndMbis[bin1])
		Bias_eff_t2=np.sum(dndMbis[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*dndMbis[bin2])
		Bias_eff_t3=np.sum(dndMbis[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*dndMbis[bin3])
		Bias_eff_t4=np.sum(dndMbis[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*dndMbis[bin4])
		#------------------------------
		Bias_eff_st1=np.sum(dndMbis[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*dndMbis[bin1])
		Bias_eff_st2=np.sum(dndMbis[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*dndMbis[bin2])
		Bias_eff_st3=np.sum(dndMbis[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*dndMbis[bin3])
		Bias_eff_st4=np.sum(dndMbis[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*dndMbis[bin4])
		#------------------------------
		Bias_eff_smt1=np.sum(dndMbis[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*dndMbis[bin1])
		Bias_eff_smt2=np.sum(dndMbis[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*dndMbis[bin2])
		Bias_eff_smt3=np.sum(dndMbis[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*dndMbis[bin3])
		Bias_eff_smt4=np.sum(dndMbis[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*dndMbis[bin4])
		
		
		#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff0_t1,Bias_eff0_t2, Bias_eff0_t3, Bias_eff0_t4))
		#~ fid_file.close()
		#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff_t1,Bias_eff_t2, Bias_eff_t3, Bias_eff_t4))
		#~ fid_file.close()


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################
	#### plot all different bias test
	#~ if j == z[0]:
		#~ fig2 = plt.figure()
	#~ J = j + 1
	
	#~ if len(z) == 1:
		#~ ax2 = fig2.add_subplot(1, len(z), J)
	#~ elif len(z) == 2:
		#~ ax2 = fig2.add_subplot(1, 2, J)
	#~ elif len(z) > 2:
		#~ ax2 = fig2.add_subplot(2, 2, J)
	############################################################
	#~ ax2.scatter(m_middle1, hmf1*m_middle1**2, marker='.', color='C0', label='Sim')
	#~ ax2.scatter(m_middle2, hmf2*m_middle2**2, marker='.', color='C1')
	#~ ax2.scatter(m_middle3, hmf3*m_middle3**2, marker='.', color='C2')
	#~ ax2.scatter(m_middle4, hmf4*m_middle4**2, marker='.', color='C3')
	#~ h1, = ax2.plot(m_middle1, dndM1*m_middle1**2, color='C0', linestyle=':')
	#~ h2, = ax2.plot(m_middle1, DndM1*m_middle1**2, color='C2', linestyle='-.')
	#~ h3, = ax2.plot(m_middle1, dndMbis1*m_middle1**2, color='C3', linestyle='--')
	#~ ax2.plot(m_middle2, dndM2*m_middle2**2, color='C0', linestyle=':')
	#~ ax2.plot(m_middle2, DndM2*m_middle2**2, color='C2', linestyle='-.')
	#~ ax2.plot(m_middle2, dndMbis2*m_middle2**2, color='C3', linestyle='--')
	#~ ax2.plot(m_middle3, dndM3*m_middle3**2, color='C0', linestyle=':')
	#~ ax2.plot(m_middle3, DndM3*m_middle3**2, color='C2', linestyle='-.')
	#~ ax2.plot(m_middle3, dndMbis3*m_middle3**2, color='C3', linestyle='--')
	#~ ax2.plot(m_middle4, dndM4*m_middle4**2, color='C0', linestyle=':')
	#~ ax2.plot(m_middle4, DndM4*m_middle4**2, color='C2', linestyle='-.')
	#~ ax2.plot(m_middle4, dndMbis4*m_middle4**2, color='C3', linestyle='--')
	#~ ax2.set_yscale('log')
	#~ ax2.set_xlim(1e11, 4e15)
	#~ ax2.set_ylim(1e7, 1e10)
	#~ plt.figlegend( (h1,h2, h3), (r'Tinker 2008 w/ $P_{cc}$',r'Tinker 2010 w/ $P_{cc}$',r'Crocce w/ $P_{cc}$'), \
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
	################################################################
	#~ ax2.set_ylim(0.8,1.2)
	#~ ax2.set_xlim(8e-3,0.1)
	#~ r1,  =ax2.plot(k,rbefft)
	#~ r2,  =ax2.plot(k,rbeffst)
	#~ r3,  =ax2.plot(k,rbeffsmt)
	#~ ax2.fill_between(k,rbefft - errbefft,rbefft + errbefft, alpha=0.6)
	#~ ax2.fill_between(k,rbeffst - errbeffst,rbeffst + errbeffst, alpha=0.6)
	#~ ax2.fill_between(k,rbeffsmt - errbeffsmt,rbeffsmt + errbeffsmt, alpha=0.6)
	#~ ax2.axhline(1, color='k')
	#~ plt.figlegend( (r1,r2,r3), ('Tinker','ST','SMT'), \
	#####################################################################
	####### comparison bias and != models #############################
	#~ M1, = ax2.plot(k, bias1,label=r'$b_{sim}$')
	#~ M2, = ax2.plot(k, bias2)
	#~ M3, = ax2.plot(k, bias3)
	#~ M4, = ax2.plot(k, bias4)
	#~ #-----------------------------------------------
	#~ M1, = ax2.plot(k, bias1_0ev, linestyle = '--', color='C0', label=r'$b_{cc , M_{\nu} = 0.0eV} $')
	#~ M2, = ax2.plot(k, bias2_0ev, linestyle = '--', color='C1')
	#~ M3, = ax2.plot(k, bias3_0ev, linestyle = '--', color='C2')
	#~ M4, = ax2.plot(k, bias4_0ev, linestyle = '--', color='C3')
	#~ #-------------------------------------------------------------
	#~ bres, = ax2.plot(k, bias1_0ev*bias_eff_t1/bias_eff0_t1, linestyle = '--', color='C0')
	#~ ax2.plot(k, bias2_0ev*bias_eff_t2/bias_eff0_t2, linestyle = '--', color='C1')
	#~ ax2.plot(k, bias3_0ev*bias_eff_t3/bias_eff0_t3, linestyle = '--', color='C2')
	#~ ax2.plot(k, bias4_0ev*bias_eff_t4/bias_eff0_t4, linestyle = '--', color='C3')
	#-------------------------------------------------------------
	#~ ax2.plot(k, bias1_0ev*Bias_eff_t1/Bias_eff0_t1, linestyle = ':', color='C0', label=r'$b_{fiducial}$')
	#~ ax2.plot(k, bias2_0ev*Bias_eff_t2/Bias_eff0_t2, linestyle = ':', color='C1')
	#~ ax2.plot(k, bias3_0ev*Bias_eff_t3/Bias_eff0_t3, linestyle = ':', color='C2')
	#~ ax2.plot(k, bias4_0ev*Bias_eff_t4/Bias_eff0_t4, linestyle = ':', color='C3')
	#~ #--------------------------------------------------------
	#~ ax2.fill_between(k,bias1-errb1, bias1+errb1, alpha=0.6)
	#~ ax2.fill_between(k,bias2-errb2, bias2+errb2, alpha=0.6)
	#~ ax2.fill_between(k,bias3-errb3, bias3+errb3, alpha=0.6)
	#~ ax2.fill_between(k,bias4-errb4, bias4+errb4, alpha=0.6)
	#~ #---------------------------------------------------------
	#~ ax2.set_ylim(bias1[0]*0.8,bias4[0]*1.4)
	#~ ax2.set_xlim(8e-3,1)
	#~ plt.figlegend( (M1,M2,M3,M4, st2,st3), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$', 'Sim hmf + Tinker bias', 'rescaled effective bias'), \
	#~ plt.figlegend( (M1,M2,M3,M4), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$'), \
	####################################################################
	#~ ax2.scatter(k,bias1/bias1_0ev, color='b', marker='.')
	#~ ax2.scatter(k,bias2/bias2_0ev, color='r', marker='.')
	#~ ax2.scatter(k,bias3/bias3_0ev, color='g', marker='.')
	#~ ax2.scatter(k,bias4/bias4_0ev, color='c', marker='.')
	#~ ax2.plot(k,bias1/bias1_0ev, color='b', linestyle='--')
	#~ ax2.plot(k,bias2/bias2_0ev, color='r', linestyle='--')
	#~ ax2.plot(k,bias3/bias3_0ev, color='g', linestyle='--')
	#~ ax2.plot(k,bias4/bias4_0ev, color='c', linestyle='--')
	#~ #-------------------------------------
	#~ ax2.axhline(bias_eff_t1/bias_eff0_t1, color='b')
	#~ ax2.axhline(bias_eff_t2/bias_eff0_t2, color='r')
	#~ ax2.axhline(bias_eff_t3/bias_eff0_t3, color='g')
	#~ ax2.axhline(bias_eff_t4/bias_eff0_t4, color='c')
	#~ #--------------------------------------
	#~ ax2.axhline(Bias_eff_t1/Bias_eff0_t1, color='b', linestyle=':')
	#~ ax2.axhline(Bias_eff_t2/Bias_eff0_t2, color='r', linestyle=':')
	#~ ax2.axhline(Bias_eff_t3/Bias_eff0_t3, color='g', linestyle=':')
	#~ ax2.axhline(Bias_eff_t4/Bias_eff0_t4, color='c', linestyle=':')
	#~ #-------------------------------------
	#~ ax2.axhline(bias_eff_st1/bias_eff0_st1, color='b', linestyle=':')
	#~ ax2.axhline(bias_eff_st2/bias_eff0_st2, color='r', linestyle=':')
	#~ ax2.axhline(bias_eff_st3/bias_eff0_st3, color='g', linestyle=':')
	#~ ax2.axhline(bias_eff_st4/bias_eff0_st4, color='c', linestyle=':')
	#-------------------------------------
	#~ ax2.axhline(bias_eff_smt1/bias_eff0_smt1, color='b', linestyle='--')
	#~ ax2.axhline(bias_eff_smt2/bias_eff0_smt2, color='r', linestyle='--')
	#~ ax2.axhline(bias_eff_smt3/bias_eff0_smt3, color='g', linestyle='--')
	#~ ax2.axhline(bias_eff_smt4/bias_eff0_smt4, color='c', linestyle='--')
	#---------------------------------------
	#~ ax2.set_ylim(1.02,1.08)
	#~ ax2.set_xlim(0.008,0.1)
	
	#~ plt.figlegend( (bres, bres2,M2,M3,M4), ('Tinker bias + sim HMF', 'Tinker bias + Tinker HMF','$M_{2}$','$M_{3}$','$M_{4}$'), \
	#######################################
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', model BB ')
	#~ ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, ncol=3, fontsize=9)
	#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#~ ax2.set_xscale('log')
	#~ plt.rcParams.update({'font.size': 12})
	#~ #----------------------------
	#~ if j == 0 :
		#~ ax2.tick_params(bottom='off', labelbottom='off')
		#~ ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$')
		#~ ax2.set_ylabel(r'$b_{cc}$', fontsize = 12)
		#~ ax2.set_ylabel(r'$M^2 n(M)$')
		#~ #ax2.grid()
	#~ if j == 1 :
		#~ ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$')
		#~ ax2.set_ylabel(r'$b_{cc}$')
		#~ ax2.set_ylabel(r'$M^2 n(M)$')
		#~ ax2.yaxis.set_label_position("right")
		#~ #ax2.grid()
	#~ if j == 2 :
		#ax.tick_params(labelleft=True)
		#~ ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$')
		#~ ax2.set_ylabel(r'$b_{cc}$')
		#~ ax2.set_ylabel(r'$M^2 n(M)$')
		#~ ax2.set_xlabel('k [h/Mpc]')
		#~ ax2.set_xlabel(r'M [$h^{-1} M_{\odot}$]')
		#~ #ax2.grid()
	#~ if j == 3 :
		#~ ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		#~ ax2.set_xlabel('k [h/Mpc]')
		#~ ax2.set_xlabel(r'M [$h^{-1} M_{\odot}$]')
		#~ ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$')
		#~ ax2.set_ylabel(r'$b_{cc}$')
		#~ ax2.set_ylabel(r'$M^2 n(M)$')
		#~ ax2.yaxis.set_label_position("right")
		#~ #ax2.grid()
	#~ #ax2.set_xlim(8e-3,0.05)
	#~ if j == len(z) -1:
		#~ plt.show()

end = time()
print 'total time is '+str((end - start))
