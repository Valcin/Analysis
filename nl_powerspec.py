
from time import time
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
	
	########################################################################
	############# 	0.0 eV Masseless neutrino 


	



		#~ #----------------------------------------------------------------
		#~ #----------Tinker, Crocce param bias ----------------------------
		#~ #----------------------------------------------------------------

		#~ #compute tinker stuff
		#limM = [5e11,1e12,3e12,1e13, 3.2e15]
		limM = [4.2e11,1e12,3e12,1e13, 5e15]
		loglim = [ 11.623, 12., 12.477, 13., 15.698]
		camb2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cc_z='+str(z[j])+'00.txt')
		kcamb2 = camb2[:,0]
		Pcamb2 = camb2[:,1]

		#~ #### get the mass function from simulation
		massf = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf_z='+str(z[j])+'.txt')
		m_middle = massf[:,10]
		dm = massf[:,11]
		hmf_temp = np.zeros((len(m_middle),10))
		for i in xrange(0,10):
			hmf_temp[:,i]= massf[:,i]
		
		#~ M_middle=10**(0.5*(np.log10(m_middle[1:])+np.log10(m_middle[:-1]))) #center of the bin
		hmf = np.mean(hmf_temp[:,0:11], axis=1)
		
		#~ dndM=MFL.Tinker_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
		#~ with open('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (dndM[m]))
		#~ fid_file.close()
		#~ dndMbis=MFL.Crocce_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
		#~ with open('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (dndMbis[m]))
		#~ fid_file.close()
		
		#~ bt=np.empty(len(m_middle),dtype=np.float64)
		#~ bst=np.empty(len(m_middle),dtype=np.float64)
		#~ bsmt=np.empty(len(m_middle),dtype=np.float64)
		#~ for i in range(len(m_middle)):
			#~ bt[i]=bias(kcamb2,Pcamb2,Omega_m,m_middle[i],'Tinker')
			#~ bst[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Crocce')
			#~ bsmt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'SMT01')
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bt[m]))
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bst[m]))
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (bsmt[m]))
		#~ fid_file.close()
		
		#~ dndM = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt')
		#~ diff = hmf/dndM
		#~ bin2 = np.where(m_middle > 1e12 )[0]
		#~ bin3 = np.where(m_middle > 3e12 )[0]
		#~ bin4 = np.where(m_middle > 1e13 )[0]
		#~ sca1 = np.sum(hmf*dm*diff)/np.sum(hmf*dm)
		#~ sca2 = np.sum(hmf[bin2]*dm[bin2]*diff[bin2])/np.sum(hmf[bin2]*dm[bin2])
		#~ sca3 = np.sum(hmf[bin3]*dm[bin3]*diff[bin3])/np.sum(hmf[bin3]*dm[bin3])
		#~ sca4 = np.sum(hmf[bin4]*dm[bin4]*diff[bin4])/np.sum(hmf[bin4]*dm[bin4])
		#~ print 1/sca1, 1/sca2, 1/sca3, 1/sca4


		#### get tinker and crocce hmf
		#Tb1 = halo_bias('Tinker', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#Tb2 = halo_bias('Tinker', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#Tb3 = halo_bias('Tinker', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Tb4 = halo_bias('Tinker', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )

		#~ #print Tb1, Tb2, Tb3, Tb4


		#~ #Cb1 = halo_bias('Crocce', z[j], limM[0],limM[1], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Cb2 = halo_bias('Crocce', z[j], limM[1],limM[2], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Cb3 = halo_bias('Crocce', z[j], limM[2],limM[3], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Cb4 = halo_bias('Crocce', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb1 = halo_bias('Crocce', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb2 = halo_bias('Crocce', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb3 = halo_bias('Crocce', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb4 = halo_bias('Crocce', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )

		#~ #print Cb1,Cb2,Cb3, Cb4
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/bcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(k)):
				#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k[index_k], bias1[index_k], bias2[index_k], bias3[index_k], bias4[index_k]))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/LS2_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Cb1,Cb2, Cb3, Cb4))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Tb1,Tb2, Tb3, Tb4))
		#~ fid_file.close()
		
		#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/ccl_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (lb1,lb2, lb3, lb4))
		#~ fid_file.close()

		Tb0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt')
		Tb1 = Tb0ev[0]
		Tb2 = Tb0ev[1]
		Tb3 = Tb0ev[2]
		Tb4 = Tb0ev[3]
		#Cb1 = Tb0ev[4]
		#Cb2 = Tb0ev[5]
		#Cb3 = Tb0ev[6]
		#Cb4 = Tb0ev[7]

		ccl00 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/ccl_z='+str(z[j])+'_.txt')
		lb1 = ccl00[0]
		lb2 = ccl00[1]
		lb3 = ccl00[2]
		lb4 = ccl00[3]
		#~ print lb1, lb2, lb3, lb4
		
		### Simu hmf #############################################
		bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
		bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
		bin1 = np.where(m_middle > 5e11 )[0]
		bin2 = np.where(m_middle > 1e12 )[0]
		bin3 = np.where(m_middle > 3e12 )[0]
		bin4 = np.where(m_middle > 1e13 )[0]
		#------------------------------
		bias_eff0_t1=np.sum(hmf[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*hmf[bin1])
		bias_eff0_t2=np.sum(hmf[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*hmf[bin2])
		bias_eff0_t3=np.sum(hmf[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*hmf[bin3])
		bias_eff0_t4=np.sum(hmf[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*hmf[bin4])
		#------------------------------
		bias_eff0_st1=np.sum(hmf[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*hmf[bin1])
		bias_eff0_st2=np.sum(hmf[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*hmf[bin2])
		bias_eff0_st3=np.sum(hmf[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*hmf[bin3])
		bias_eff0_st4=np.sum(hmf[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*hmf[bin4])
		#------------------------------
		bias_eff0_smt1=np.sum(hmf[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*hmf[bin1])
		bias_eff0_smt2=np.sum(hmf[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*hmf[bin2])
		bias_eff0_smt3=np.sum(hmf[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*hmf[bin3])
		bias_eff0_smt4=np.sum(hmf[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*hmf[bin4])
		
		### Crocce hmf ############################################
		#~ bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		#~ bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
		#~ bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
		#~ #------------------------------
		#~ Bias_eff0_t1=np.sum(dndM0bis[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
		#~ Bias_eff0_t2=np.sum(dndM0bis[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
		#~ Bias_eff0_t3=np.sum(dndM0bis[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
		#~ Bias_eff0_t4=np.sum(dndM0bis[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
		#------------------------------
		#~ Bias_eff0_st1=np.sum(dndM0bis[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
		#~ Bias_eff0_st2=np.sum(dndM0bis[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
		#~ Bias_eff0_st3=np.sum(dndM0bis[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
		#~ Bias_eff0_st4=np.sum(dndM0bis[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
		#~ #------------------------------
		#~ Bias_eff0_smt1=np.sum(dndM0bis[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
		#~ Bias_eff0_smt2=np.sum(dndM0bis[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
		#~ Bias_eff0_smt3=np.sum(dndM0bis[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
		#~ Bias_eff0_smt4=np.sum(dndM0bis[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
		
		#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff0_t1,Bias_eff0_t2, Bias_eff0_t3, Bias_eff0_t4))
		#~ fid_file.close()
		#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff_t1,Bias_eff_t2, Bias_eff_t3, Bias_eff_t4))
		#~ fid_file.close()
		
		#~ bias_0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/bcc_z='+str(z[j])+'_.txt')
		#~ bias1_0ev = bias_0ev[:,1]
		#~ bias2_0ev = bias_0ev[:,2]
		#~ bias3_0ev = bias_0ev[:,3]
		#~ bias4_0ev = bias_0ev[:,4]
		
		#### mean and error residuals on linear bias
		#~ rbefft = np.mean(np.array([bias_eff0_t1/bias1, bias_eff0_t2/bias2, bias_eff0_t3/bias3, bias_eff0_t4/bias4]), axis=0)
		#~ rbeffst = np.mean(np.array([bias_eff0_st1/bias1, bias_eff0_st2/bias2, bias_eff0_st3/bias3, bias_eff0_st4/bias4]), axis=0)
		#~ rbeffsmt = np.mean(np.array([bias_eff0_smt1/bias1, bias_eff0_smt2/bias2, bias_eff0_smt3/bias3, bias_eff0_smt4/bias4]), axis=0)
		#~ #-----------------------------
		#~ errbefft = np.std(np.array([bias_eff0_t1/bias1, bias_eff0_t2/bias2, bias_eff0_t3/bias3, bias_eff0_t4/bias4]), axis=0)
		#~ errbeffst = np.std(np.array([bias_eff0_st1/bias1, bias_eff0_st2/bias2, bias_eff0_st3/bias3, bias_eff0_st4/bias4]), axis=0)
		#~ errbeffsmt = np.std(np.array([bias_eff0_smt1/bias1, bias_eff0_smt2/bias2, bias_eff0_smt3/bias3, bias_eff0_smt4/bias4]), axis=0)


		dndM = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt')
		dndMbis = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt')
		
		
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
		
		####### comparison bias and != models #############################
		#~ M1, = ax2.plot(k, bias1, label='$b_{cc}$')
		#~ M2, = ax2.plot(k, bias2)
		#~ M3, = ax2.plot(k, bias3)
		#~ M4, = ax2.plot(k, bias4)
		#-----------------------------------------------
		#~ M1, = ax2.plot(k, bias1_0ev, linestyle = '--', color='C0', label=r'$b_{cc , M_{\nu} = 0.0eV} $')
		#~ M2, = ax2.plot(k, bias2_0ev, linestyle = '--', color='C1')
		#~ M3, = ax2.plot(k, bias3_0ev, linestyle = '--', color='C2')
		#~ M4, = ax2.plot(k, bias4_0ev, linestyle = '--', color='C3')
		#~ #-----------------------------------------
		#~ st1 =ax2.axhline(Tb1, color='C0', linestyle=':')
		#~ ax2.axhline(Tb2, color='C1', linestyle=':')
		#~ ax2.axhline(Tb3, color='C2', linestyle=':')
		#~ ax2.axhline(Tb4, color='C3', linestyle=':')
		#---------------------------------------------------
		#~ st2 =ax2.axhline(lb1, color='C0', linestyle=':')
		#~ ax2.axhline(lb2, color='C1', linestyle=':')
		#~ ax2.axhline(lb3, color='C2', linestyle=':')
		#~ ax2.axhline(lb4, color='C3', linestyle=':')
		#---------------------------------------------------
		#~ st2 =ax2.axhline(bias_eff0_t1, color='C0', linestyle=':')
		#~ ax2.axhline(bias_eff0_t2, color='C1', linestyle=':')
		#~ ax2.axhline(bias_eff0_t3, color='C2', linestyle=':')
		#~ ax2.axhline(bias_eff0_t4, color='C3', linestyle=':')
		#---------------------------------------------------
		#~ st3 =ax2.axhline(bias_eff0_st1, color='C0', linestyle='--')
		#~ ax2.axhline(bias_eff0_st2, color='C1', linestyle='--')
		#~ ax2.axhline(bias_eff0_st3, color='C2', linestyle='--')
		#~ ax2.axhline(bias_eff0_st4, color='C3', linestyle='--')
		#---------------------------------------------------
		#~ st4 =ax2.axhline(bias_eff0_smt1, color='C0', linestyle='-.')
		#~ ax2.axhline(bias_eff0_smt2, color='C1', linestyle='-.')
		#~ ax2.axhline(bias_eff0_smt3, color='C2', linestyle='-.')
		#~ ax2.axhline(bias_eff0_smt4, color='C3', linestyle='-.')
		#---------------------------------------------------
		#~ st3 =ax2.axhline(lb1/rsc1, color='C0', linestyle=':')
		#~ ax2.axhline(lb2/rsc2, color='C1', linestyle=':')
		#~ ax2.axhline(lb3/rsc3, color='C2', linestyle=':')
		#~ ax2.axhline(lb4/rsc4, color='C3', linestyle=':')
		#-----------------------------------------------------
		#~ ax2.axhline(Cb1, color='C0', linestyle=':', label='Simu + ST')
		#~ ax2.axhline(Cb2, color='C1', linestyle=':')
		#~ ax2.axhline(Cb3, color='C2', linestyle=':')
		#~ ax2.axhline(Cb4, color='C3', linestyle=':')
		#-----------------------------------------------
		#~ ax2.axvline( kk1, color='C0', linestyle=':', label='shot noise = 80% of P(k)')
		#~ ax2.axvline( kk2, color='C1', linestyle=':')
		#~ ax2.axvline( kk3, color='C2', linestyle=':')
		#~ ax2.axvline( kk4, color='C3', linestyle=':')
		#~ ax2.fill_between(k,bias1-errb1, bias1+errb1, alpha=0.6)
		#~ ax2.fill_between(k,bias2-errb2, bias2+errb2, alpha=0.6)
		#~ ax2.fill_between(k,bias3-errb3, bias3+errb3, alpha=0.6)
		#~ ax2.fill_between(k,bias4-errb4, bias4+errb4, alpha=0.6)
		#~ ax2.set_ylim(bias1[0]*0.8,bias4[0]*1.2)
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
		#~ ######################################
		#~ ax2.scatter(m_middle, hmf*m_middle**2, marker='.', color='k', label='Sim')
		#~ h1, = ax2.plot(m_middle, dndM*m_middle**2, color='r')
		#~ h2, = ax2.plot(m_middle, dndM2*m_middle**2, color='r', linestyle=':')
		#~ h3, = ax2.plot(m_middle, dndMbis*m_middle**2, color='b')
		#~ h4, = ax2.plot(m_middle, dndM2bis*m_middle**2, color='b', linestyle=':')
		#~ ax2.set_yscale('log')
		#~ ax2.set_xlim(1e13, 4e15)
		#~ ax2.set_ylim(1e7, 1e10)
		#~ plt.figlegend( (h1,h3), (r'Tinker w/ $P_{cc}$',r'Crocce w/ $P_{cc}$'), \
		#######################################
		#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case II ')
		#~ ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, ncol=3, fontsize=9)
		#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
		#~ ax2.set_xscale('log')
		#----------------------------
		
		
		#~ if j == 0 :
			#~ ax2.tick_params(bottom='off', labelbottom='off')
			#~ ax2.set_ylabel(r'$b_{eff}$ / $b_{sim}$')
			#~ ax2.set_ylabel(r'$b_{cc}$')
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

		
		#~ kill
		
		####################################################################
		#~ ##### define the maximum scale for the fit 
		kstop1 = [0.16,0.2,0.25,0.35]
		kstop2 = [0.12,0.16,0.2,0.2]
		kstop3 = [0.15,0.15,0.15,0.15]
		
		#~ #### the case 
		case = 2
		
		if case == 1:
			kstop = kstop1[ind]
		elif case == 2:
			kstop = kstop2[ind]
		elif case == 3:
			kstop = kstop3[ind]
		
		#~ kstoplim = [0.5,0.5,0.5,0.4]
		#~ kstop = kstoplim[ind]
		print kstop
		
		#~ # put identation to the rest to loop over kstop
		#kstop_arr = np.logspace(np.log10(0.05),np.log10(0.6),20)
		#for kstop in kstop_arr:
		#	print kstop
		#~ ####################################################################
		#~ #Plin = Pclass
		#~ #klin = kclass
		Plin = Pcamb
		klin = kcamb
		#Plin = pks
		#~ #klin = ks

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
		
		# interpolate to have more points and create an evenly logged array
		kbis = np.logspace(np.log10(np.min(kclass)), np.log10(np.max(kclass)), 200)
		#kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 350)
		Plinbis = np.interp(kbis, k, Plin)
		lim = np.where((kbis < kstop))[0]



		#~ plt.figure()
		#~ plt.plot(kcamb,Pcamb)
		#~ plt.plot(kbis,Plinbis)
		#~ plt.plot(k,Plin)
		#~ plt.plot(k,Pmm)
		#~ plt.xscale('log')
		#~ plt.yscale('log')
		#~ plt.xlim(1e-3,10)
		#~ plt.ylim(1e-1,4e4)
		#~ plt.show()
		#~ kill

	#####################################################################################################################################
	######### 0.15 eV Massive neutrino 
	###########################################################################################################################################
	if Mnu == 0.15:
		hierarchy = 'degenerate' #'degenerate', 'normal', 'inverted'
		Mnu       = 0.15  #eV
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
		#-------------------------------------------------
		#---------------- Class ---------------------------
		#-------------------------------------------------
		Class = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/class/test_z'+str(j+1)+'_pk.dat')
		kclass = Class[:,0]
		Pclass = Class[:,1]
		Class_trans = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/class/test_z'+str(j+1)+'_tk.dat')

		ktrans = Class_trans[:,0]
		Tb = Class_trans[:,2]
		Tcdm = Class_trans[:,3]
		Tm = Class_trans[:,8]
		Tcb = (Omega_c * Tcdm + Omega_b * Tb)/(Omega_c + Omega_b)

		#-----------------------------------------------------------------------
		#-------- get the transfer function and Pcc ----------------------------
		#-----------------------------------------------------------------------
		Pcc = Pclass * (Tcb/Tm)**2
		Plin = Pcc
		klin = kclass
		with open('/home/david/codes/Paco/data2/0.15eV/Pcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
			for index_k in xrange(len(klin)):
				fid_file.write('%.8g %.8g\n' % ( klin[index_k], Plin[index_k]))
		fid_file.close()
		
		#-------------------------------------------------
		#---------------- Camb ---------------------------
		#-------------------------------------------------
		camb = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb = camb[:,0]
		Pcamb = camb[:,1]
		Plin = Pcamb
		klin = kcamb

	#~ #-----------------------------------------------------------------------
		#~ #---------------- matter neutrino Real space ---------------------------
		#~ #-----------------------------------------------------------------------
		d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/NCV1/analysis/Pk_c_z='+str(z[j])+'.txt')
		e = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/NCV2/analysis/Pk_c_z='+str(z[j])+'.txt')
		k1 = d[:,0]
		p1 = d[:,1]
		k2 = e[:,0]
		p2 = e[:,1]


		d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Pcc_realisation_'+str(Mnu)+'_z='+str(z[j])+'.txt')
		kmat = d[:,8]
		Pmat = np.zeros((len(kmat),10))
		for i in xrange(0,8):
			Pmat[:,i]= d[:,i]
		
		Pmat[:,8] = p1
		Pmat[:,9] = p2


		#-----------------------------------------------------------------------
		#---------------- halo neutrino Real space ---------------------------
		#-----------------------------------------------------------------------
		d1 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_0.15_z='+str(z[j])+'.txt')
		k = d1[:,19]
		Phh1 = np.zeros((len(k),10))
		Pshot1 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh1[:,i]= d1[:,pnum1[i]]
			Pshot1[i]= d1[0,pnum2[i]]
		# second mass range
		d2 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_0.15_z='+str(z[j])+'.txt')
		k = d2[:,19]
		Phh2 = np.zeros((len(k),10))
		Pshot2 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh2[:,i]= d2[:,pnum1[i]]
			Pshot2[i]= d2[0,pnum2[i]]
		# third mass range
		d3 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_0.15_z='+str(z[j])+'.txt')
		k = d3[:,19]
		Phh3 = np.zeros((len(k),10))
		Pshot3 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh3[:,i]= d3[:,pnum1[i]]
			Pshot3[i]= d3[0,pnum2[i]]
		# fourth mass range
		d4 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_0.15_z='+str(z[j])+'.txt')
		k = d4[:,19]
		Phh4 = np.zeros((len(k),10))
		Pshot4 = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Phh4[:,i]= d4[:,pnum1[i]]
			Pshot4[i]= d4[0,pnum2[i]]


		
		#-------------------------------------------------------------------
		#----remove shot noise, compute bias and bias variance -------------
		#-------------------------------------------------------------------
		bhh1 = np.zeros((len(k),10))
		bhh2 = np.zeros((len(k),10))
		bhh3 = np.zeros((len(k),10))
		bhh4 = np.zeros((len(k),10))
		for i in xrange(0,10):
			Phh1[:,i] = Phh1[:,i]-Pshot1[i]
			Phh2[:,i] = Phh2[:,i]-Pshot2[i]
			Phh3[:,i] = Phh3[:,i]-Pshot3[i]
			Phh4[:,i] = Phh4[:,i]-Pshot4[i]
			nul1 = np.where(Phh1[:,i] < 0)[0]
			nul2 = np.where(Phh2[:,i] < 0)[0]
			nul3 = np.where(Phh3[:,i] < 0)[0]
			nul4 = np.where(Phh4[:,i] < 0)[0]
			Phh1[nul1,i] = 0
			Phh2[nul2,i] = 0
			Phh3[nul3,i] = 0
			Phh4[nul4,i] = 0
			bhh1[:,i] = np.sqrt(Phh1[:,i]/Pmat[:,i])
			bhh2[:,i] = np.sqrt(Phh2[:,i]/Pmat[:,i])
			bhh3[:,i] = np.sqrt(Phh3[:,i]/Pmat[:,i])
			bhh4[:,i] = np.sqrt(Phh4[:,i]/Pmat[:,i])
			
			
		#~ ### do the mean over quantitites ###
		
		Pmm = np.mean(Pmat[:,0:11], axis=1)
		PH1 = np.mean(Phh1[:,0:11], axis=1)
		PH2 = np.mean(Phh2[:,0:11], axis=1)
		PH3 = np.mean(Phh3[:,0:11], axis=1)
		PH4 = np.mean(Phh4[:,0:11], axis=1)

		
		bias1 = np.mean(bhh1[:,0:11], axis=1)
		bias2 = np.mean(bhh2[:,0:11], axis=1)
		bias3 = np.mean(bhh3[:,0:11], axis=1)
		bias4 = np.mean(bhh4[:,0:11], axis=1)
		
		errb1 = np.std(bhh1[:,0:11], axis=1)
		errb2 = np.std(bhh2[:,0:11], axis=1)
		errb3 = np.std(bhh3[:,0:11], axis=1)
		errb4 = np.std(bhh4[:,0:11], axis=1)
		
		errPhh1 = np.std(Phh1[:,0:11], axis=1)
		errPhh2 = np.std(Phh2[:,0:11], axis=1)
		errPhh3 = np.std(Phh3[:,0:11], axis=1)
		errPhh4 = np.std(Phh4[:,0:11], axis=1)
		
		
		#-----------------------------------------------------------------------
		#---------------- halo neutrino Redshift space ---------------------------
		#-----------------------------------------------------------------------
		d1a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d1b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d1c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d1d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_0_0.15_z='+str(z[j])+'.txt')
		d2a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d2b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d2c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d2d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_1_0.15_z='+str(z[j])+'.txt')
		d3a = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh1_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
		d3b = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh2_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
		d3c = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh3_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')
		d3d = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/Phh4_realisation_red_axis_2_0.15_z='+str(z[j])+'.txt')


		kx1a = d1a[:,19]
		Px1a = np.zeros((len(kx1a),10))
		Px1b = np.zeros((len(kx1a),10))
		Px1c = np.zeros((len(kx1a),10))
		Px1d = np.zeros((len(kx1a),10))
		Pxshot1a = np.zeros((10))
		Pxshot1b = np.zeros((10))
		Pxshot1c = np.zeros((10))
		Pxshot1d = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Px1a[:,i]= d1a[:,pnum1[i]]
			Px1b[:,i]= d1b[:,pnum1[i]]
			Px1c[:,i]= d1c[:,pnum1[i]]
			Px1d[:,i]= d1d[:,pnum1[i]]
			Pxshot1a[i]= d1a[0,pnum2[i]]
			Pxshot1b[i]= d1b[0,pnum2[i]]
			Pxshot1c[i]= d1c[0,pnum2[i]]
			Pxshot1d[i]= d1d[0,pnum2[i]]
		kx2a = d2a[:,19]
		Px2a = np.zeros((len(kx2a),10))
		Px2b = np.zeros((len(kx2a),10))
		Px2c = np.zeros((len(kx2a),10))
		Px2d = np.zeros((len(kx2a),10))
		Pxshot2a = np.zeros((10))
		Pxshot2b = np.zeros((10))
		Pxshot2c = np.zeros((10))
		Pxshot2d = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Px2a[:,i]= d2a[:,pnum1[i]]
			Px2b[:,i]= d2b[:,pnum1[i]]
			Px2c[:,i]= d2c[:,pnum1[i]]
			Px2d[:,i]= d2d[:,pnum1[i]]
			Pxshot2a[i]= d2a[0,pnum2[i]]
			Pxshot2b[i]= d2b[0,pnum2[i]]
			Pxshot2c[i]= d2c[0,pnum2[i]]
			Pxshot2d[i]= d2d[0,pnum2[i]]
		kx3a = d3a[:,19]
		Px3a = np.zeros((len(kx3a),10))
		Px3b = np.zeros((len(kx3a),10))
		Px3c = np.zeros((len(kx3a),10))
		Px3d = np.zeros((len(kx3a),10))
		Pxshot3a = np.zeros((10))
		Pxshot3b = np.zeros((10))
		Pxshot3c = np.zeros((10))
		Pxshot3d = np.zeros((10))
		pnum1 = [0,2,4,6,8,10,12,14,16,18]
		pnum2 = [1,3,5,7,9,11,13,15,17,20]
		for i in xrange(0,10):
			Px3a[:,i]= d3a[:,pnum1[i]]
			Px3b[:,i]= d3b[:,pnum1[i]]
			Px3c[:,i]= d3c[:,pnum1[i]]
			Px3d[:,i]= d3d[:,pnum1[i]]
			Pxshot3a[i]= d3a[0,pnum2[i]]
			Pxshot3b[i]= d3b[0,pnum2[i]]
			Pxshot3c[i]= d3c[0,pnum2[i]]
			Pxshot3d[i]= d3d[0,pnum2[i]]
			
		for i in xrange(0,10):
			Px1a[:,i] = Px1a[:,i]-Pxshot1a[i]
			Px1b[:,i] = Px1b[:,i]-Pxshot1b[i]
			Px1c[:,i] = Px1c[:,i]-Pxshot1c[i]
			Px1d[:,i] = Px1d[:,i]-Pxshot1d[i]
			Px2a[:,i] = Px2a[:,i]-Pxshot2a[i]
			Px2b[:,i] = Px2b[:,i]-Pxshot2b[i]
			Px2c[:,i] = Px2c[:,i]-Pxshot2c[i]
			Px2d[:,i] = Px2d[:,i]-Pxshot2d[i]
			Px3a[:,i] = Px3a[:,i]-Pxshot3a[i]
			Px3b[:,i] = Px3b[:,i]-Pxshot3b[i]
			Px3c[:,i] = Px3c[:,i]-Pxshot3c[i]
			Px3d[:,i] = Px3d[:,i]-Pxshot3d[i]
			
			nul1a = np.where(Px1a[:,i] < 0)[0]
			Px1a[nul1a,i] = 0
			nul1b = np.where(Px1b[:,i] < 0)[0]
			Px1b[nul1b,i] = 0
			nul1c = np.where(Px1c[:,i] < 0)[0]
			Px1c[nul1c,i] = 0
			nul1d = np.where(Px1d[:,i] < 0)[0]
			Px1d[nul1d,i] = 0
			nul2a = np.where(Px2a[:,i] < 0)[0]
			Px2a[nul2a,i] = 0
			nul2b = np.where(Px2b[:,i] < 0)[0]
			Px2b[nul2b,i] = 0
			nul2c = np.where(Px2c[:,i] < 0)[0]
			Px2c[nul2c,i] = 0
			nul2d = np.where(Px2d[:,i] < 0)[0]
			Px2d[nul2d,i] = 0
			nul3a = np.where(Px3a[:,i] < 0)[0]
			Px3a[nul3a,i] = 0
			nul3b = np.where(Px3b[:,i] < 0)[0]
			Px3b[nul3b,i] = 0
			nul3c = np.where(Px3c[:,i] < 0)[0]
			Px3c[nul3c,i] = 0
			nul3d = np.where(Px3d[:,i] < 0)[0]
			Px3d[nul3d,i] = 0

		Pmono1temp = (Px1a + Px2a + Px3a)/3
		Pmono2temp = (Px1b + Px2b + Px3b)/3
		Pmono3temp = (Px1c + Px2c + Px3c)/3
		Pmono4temp = (Px1d + Px2d + Px3d)/3


		### do the mean and std over quantitites ###
		
		Pmono1 = np.mean(Pmono1temp[:,0:11], axis=1)
		Pmono2 = np.mean(Pmono2temp[:,0:11], axis=1)
		Pmono3 = np.mean(Pmono3temp[:,0:11], axis=1)
		Pmono4 = np.mean(Pmono4temp[:,0:11], axis=1)
		
		
		errPr1 = np.std(Pmono1temp[:,0:11], axis=1)
		errPr2 = np.std(Pmono2temp[:,0:11], axis=1)
		errPr3 = np.std(Pmono3temp[:,0:11], axis=1)
		errPr4 = np.std(Pmono4temp[:,0:11], axis=1)

		#-------------------------------------------------------------------
		#--- compute bias and bias variance -------------
		#-------------------------------------------------------------------
		#bredhh1 = np.zeros((len(k),10))
		#bredhh2 = np.zeros((len(k),10))
		#bredhh3 = np.zeros((len(k),10))
		#bredhh4 = np.zeros((len(k),10))
		#for i in xrange(0,10):
		#	bredhh1[:,i] = np.sqrt(Pmono1temp[:,i]/Pmat_r[:,i])
		#	bredhh2[:,i] = np.sqrt(Pmono2temp[:,i]/Pmat_r[:,i])
		#	bredhh3[:,i] = np.sqrt(Pmono3temp[:,i]/Pmat_r[:,i])
		#	bredhh4[:,i] = np.sqrt(Pmono4temp[:,i]/Pmat_r[:,i])
			
			
		#~ ### do the mean over quantitites ###


		
		#biasred1 = np.mean(bredhh1[:,0:11], axis=1)
		#biasred2 = np.mean(bredhh2[:,0:11], axis=1)
		#biasred3 = np.mean(bredhh3[:,0:11], axis=1)
		#biasred4 = np.mean(bredhh4[:,0:11], axis=1)
		
		#errbred1 = np.std(bredhh1[:,0:11], axis=1)
		#errbred2 = np.std(bredhh2[:,0:11], axis=1)
		#errbred3 = np.std(bredhh3[:,0:11], axis=1)
		#errbred4 = np.std(bredhh4[:,0:11], axis=1)
		
		#----------------------------------------------------------------
		#----------Tinker, Crocce param bias ----------------------------
		#----------------------------------------------------------------

		#~ #compute tinker stuff
		##~ limM = [5e11,1e12,3e12,1e13, 3.2e15]
		limM = [4.2e11,1e12,3e12,1e13, 5e15]
		loglim = [ 11.623, 12., 12.477, 13., 15.698]
		camb2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
		kcamb2 = camb2[:,0]
		Pcamb2 = camb2[:,1]

		#~ #### get the mass function from simulation
		massf = np. loadtxt('/home/david/codes/Paco/data2/0.15eV/hmf_z='+str(z[j])+'.txt')
		m_middle = massf[:,10]
		dm = massf[:,11]
		hmf_temp = np.zeros((len(m_middle),10))
		for i in xrange(0,10):
			hmf_temp[:,i]= massf[:,i]
		
		#~ M_middle=10**(0.5*(np.log10(m_middle[1:])+np.log10(m_middle[:-1]))) #center of the bin
		hmf = np.mean(hmf_temp[:,0:11], axis=1)
		################################################################
		
		#~ dndM=MFL.Tinker_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
		#~ with open('/home/david/codes/Paco/data2/0.15eV/thmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (dndM[m]))
		#~ fid_file.close()
		#~ dndMbis=MFL.Crocce_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
		#~ with open('/home/david/codes/Paco/data2/0.15eV/chmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
			#~ for m in xrange(0, len(m_middle)):
				#~ fid_file.write('%.8g\n' % (dndMbis[m]))
		#~ fid_file.close()
		
		#~ bt=np.empty(len(m_middle),dtype=np.float64)
		#~ bst=np.empty(len(m_middle),dtype=np.float64)
		#~ bsmt=np.empty(len(m_middle),dtype=np.float64)
		#~ for i in range(len(m_middle)):
			#~ bt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Tinker')
			#~ bst[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Crocce')
			#~ bsmt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'SMT01')
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
		
		
		#~ #### get tinker and crocce hmf
		#~ def linb(hmf,lim1, lim2):
			#~ halo_mass = np.logspace(lim1, lim2,100) #a vector of halo masses between 1e10 and 1e15
			#~ M_middle=10**(0.5*(np.log10(halo_mass[1:])+np.log10(halo_mass[:-1]))) #center of the bin
			#~ deltaM=halo_mass[1:]-halo_mass[:-1]
			#~ a = 1./(1+z[j])
			#~ hmf = np.interp(M_middle, m_middle,hmf)
			#~ #---------------------------------------
			#~ b=np.empty(99,dtype=np.float64)
			#~ for i in range(99):
				#~ b[i] = bias(klin,Plin,Omega_m,M_middle[i],'Tinker')
			#~ bias_eff=np.sum(hmf*deltaM*b)/np.sum(hmf*deltaM)
			#~ return bias_eff

		#~ lb1 = linb(hmf,loglim[0],loglim[4])
		#~ lb2 = linb(hmf,loglim[1],loglim[4])
		#~ lb3 = linb(hmf,loglim[2],loglim[4])
		#~ lb4 = linb(hmf,loglim[3],loglim[4])
		#~ print lb1, lb2, lb3, lb4

		#### get tinker and crocce hmf
		#Tb1 = halo_bias('Tinker', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#Tb2 = halo_bias('Tinker', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#Tb3 = halo_bias('Tinker', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#Tb4 = halo_bias('Tinker', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		
		#~ #Tb1 = halo_bias('Tinker', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Tb2 = halo_bias('Tinker', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Tb3 = halo_bias('Tinker', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ #Tb4 = halo_bias('Tinker', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		
		#~ Cb1 = halo_bias('Crocce', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb2 = halo_bias('Crocce', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb3 = halo_bias('Crocce', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
		#~ Cb4 = halo_bias('Crocce', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )

		
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
		
		
		
		Tb0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt')
		tb1 = Tb0ev[0]
		tb2 = Tb0ev[1]
		tb3 = Tb0ev[2]
		tb4 = Tb0ev[3]
		
		Tb15ev = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/LS_z='+str(z[j])+'_.txt')
		Tb1 = Tb15ev[0]
		Tb2 = Tb15ev[1]
		Tb3 = Tb15ev[2]
		Tb4 = Tb15ev[3]
		
		Cb0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/LS2_z='+str(z[j])+'_.txt')
		cb1 = Cb0ev[0]
		cb2 = Cb0ev[1]
		cb3 = Cb0ev[2]
		cb4 = Cb0ev[3]
		
		Cb15ev = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/LS2_z='+str(z[j])+'_.txt')
		Cb1 = Cb15ev[0]
		Cb2 = Cb15ev[1]
		Cb3 = Cb15ev[2]
		Cb4 = Cb15ev[3]
		
		ccl00 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/ccl_z='+str(z[j])+'_.txt')
		Lb1 = ccl00[0]
		Lb2 = ccl00[1]
		Lb3 = ccl00[2]
		Lb4 = ccl00[3]
		
		ccl15 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/ccl_z='+str(z[j])+'_.txt')
		lb1 = ccl15[0]
		lb2 = ccl15[1]
		lb3 = ccl15[2]
		lb4 = ccl15[3]
		#~ print lb1, lb2, lb3, lb4
		
		
		
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
		
		############################################### WITH HMF SIMU
		#### 0.0 eV ################
		bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
		bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
		#------------------------------
		bias_eff0_t1=np.sum(hmf0[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*hmf0[bin1])
		bias_eff0_t2=np.sum(hmf0[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*hmf0[bin2])
		bias_eff0_t3=np.sum(hmf0[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*hmf0[bin3])
		bias_eff0_t4=np.sum(hmf0[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*hmf0[bin4])
		#------------------------------
		bias_eff0_st1=np.sum(hmf0[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*hmf0[bin1])
		bias_eff0_st2=np.sum(hmf0[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*hmf0[bin2])
		bias_eff0_st3=np.sum(hmf0[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*hmf0[bin3])
		bias_eff0_st4=np.sum(hmf0[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*hmf0[bin4])
		#------------------------------
		bias_eff0_smt1=np.sum(hmf0[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*hmf0[bin1])
		bias_eff0_smt2=np.sum(hmf0[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*hmf0[bin2])
		bias_eff0_smt3=np.sum(hmf0[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*hmf0[bin3])
		bias_eff0_smt4=np.sum(hmf0[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*hmf0[bin4])
		#### 0.15 eV ################
		bt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/tlb1_z='+str(z[j])+'.txt')
		bst = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb1_z='+str(z[j])+'.txt')
		bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb2_z='+str(z[j])+'.txt')
		#------------------------------
		bias_eff_t1=np.sum(hmf[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*hmf[bin1])
		bias_eff_t2=np.sum(hmf[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*hmf[bin2])
		bias_eff_t3=np.sum(hmf[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*hmf[bin3])
		bias_eff_t4=np.sum(hmf[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*hmf[bin4])
		#------------------------------
		bias_eff_st1=np.sum(hmf[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*hmf[bin1])
		bias_eff_st2=np.sum(hmf[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*hmf[bin2])
		bias_eff_st3=np.sum(hmf[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*hmf[bin3])
		bias_eff_st4=np.sum(hmf[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*hmf[bin4])
		#------------------------------
		bias_eff_smt1=np.sum(hmf[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*hmf[bin1])
		bias_eff_smt2=np.sum(hmf[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*hmf[bin2])
		bias_eff_smt3=np.sum(hmf[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*hmf[bin3])
		bias_eff_smt4=np.sum(hmf[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*hmf[bin4])
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
		
		
		
		#~ #######--------- mean and std of bias and ps ratio ------------#####
		if j == z[0]:
			fig2 = plt.figure()
		J = j + 1
		
		if len(z) == 1:
			ax2 = fig2.add_subplot(1, len(z), J)
		elif len(z) == 2:
			ax2 = fig2.add_subplot(1, 2, J)
		elif len(z) > 2:
			ax2 = fig2.add_subplot(2, 2, J)
		############################################################
		#~ ax2.scatter(m_middle, hmf*m_middle**2, marker='.', color='k', label='Sim')
		#~ h1, = ax2.plot(m_middle, dndM*m_middle**2, color='r')
		#~ h2, = ax2.plot(m_middle, dndM2*m_middle**2, color='r', linestyle=':')
		#~ h3, = ax2.plot(m_middle, dndMbis*m_middle**2, color='b')
		#~ h4, = ax2.plot(m_middle, dndM2bis*m_middle**2, color='b', linestyle=':')
		#~ ax2.set_yscale('log')
		#~ ax2.set_xlim(1e13, 4e15)
		#~ ax2.set_ylim(1e7, 1e10)
		#~ plt.figlegend( (h1,h3), (r'Tinker w/ $P_{cc}$',r'Crocce w/ $P_{cc}$'), \
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
		
		#~ ######################################
		#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+', case II ')
		#~ ax2.legend(loc = 'upper left', title='z = '+str(z[j]), fancybox=True, ncol=3, fontsize=9)
		#~ plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
		#~ ax2.set_xscale('log')
		#~ if j == 0 :
			#~ ax2.tick_params(bottom='off', labelbottom='off')
			#~ ax2.set_ylabel(r'$b_{cc}(k) / b_{sim}$')
			#~ ax2.set_ylabel(r'$b_{cc}^{sim} / b_{cc}^{analytic}$')
			#~ ax2.set_ylabel(r'$b_{cc}$')
		#~ if j == 1 :
			#~ ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
			#~ ax2.set_ylabel(r'$b_{cc}(k) / b_{sim}$')
			#~ ax2.set_ylabel(r'$b_{cc}^{sim} / b_{cc}^{analytic}$')
			#~ ax2.set_ylabel(r'$b_{cc}$')
			#~ ax2.yaxis.set_label_position("right")
		#~ if j == 2 :
			#ax.tick_params(labelleft=True)
			#~ ax2.set_ylabel(r'$b_{cc}(k) / b_{sim}$')
			#~ ax2.set_ylabel(r'$b_{cc}^{sim} / b_{cc}^{analytic}$')
			#~ ax2.set_ylabel(r'$b_{cc}$')
			#~ ax2.set_xlabel('k [h/Mpc]')
		#~ if j == 3 :
			#~ ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
			#~ ax2.set_xlabel('k [h/Mpc]')
			#~ ax2.set_ylabel(r'$b_{cc}(k) / b_{sim}$')
			#~ ax2.set_ylabel(r'$b_{cc}^{sim} / b_{cc}^{analytic}$')
			#~ ax2.set_ylabel(r'$b_{cc}$')
			#~ ax2.yaxis.set_label_position("right")
		#~ if j == len(z) -1:
			#~ plt.show()
		
		
		#~ kill

		#~ ####################################################################
		##### define the maximum scale for the fit 
		kstop1 = [0.16,0.2,0.25,0.35]
		kstop2 = [0.12,0.16,0.2,0.2]
		kstop3 = [0.15,0.15,0.15,0.15]
		
		#~ #### the case 
		case = 2
		
		if case == 1:
			kstop = kstop1[ind]
		elif case == 2:
			kstop = kstop2[ind]
		elif case == 3:
			kstop = kstop3[ind]
			
			
		
		
		# put identation to the rest to loop over kstop
		#~ #kstop_arr = np.logspace(np.log10(0.05),np.log10(0.6),20)
		#~ #for kstop in kstop_arr:
		#	print kstop
		#~ ####################################################################
		#Plin = Pclass
		#~ #klin = kclass
		Plin = Pcamb
		klin = kcamb
		#Plin = pks
		#klin = ks
		

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
		
		#~ # interpolate to have more points and create an evenly logged array
		#kbis = np.logspace(np.log10(np.min(k)), np.log10(np.max(k)), 350)
		

		kbis = np.logspace(np.log10(np.min(kclass)), np.log10(np.max(kclass)), 200)
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
	
	
	#~ plt.figure()
	#~ plt.plot(kbis, Pmono1bis, color='k')
	#~ plt.plot(kbis, PH1bis, color='b')
	#~ plt.plot(kbis, Pmono2bis, color='k', linestyle='--')
	#~ plt.plot(kbis, Pmono3bis, color='k', linestyle='-.')
	#~ plt.plot(kbis, Pmono4bis, color='k', linestyle=':')
	#~ plt.plot(kbis, Pmmbis, color='c')
	#~ plt.axvspan(kstop, 7, alpha=0.2, color='grey')
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(0.008,1.0)
	#~ plt.ylim(0.9,1.1)
	#~ plt.ylim(1e2,2e5)
	#~ plt.show()
	
	#~ kill
	####################################################################
	#### compute the one loop correction 


	# set the parameters for the power spectrum window and
	# Fourier coefficient window 
	#P_window=np.array([.2,.2])  
	C_window=0.95

	# padding length 
	nu=-2; n_pad=len(kbis)
	n_pad=int(0.5*len(kbis))
	to_do=['all']
					
	# initialize the FASTPT class 
	# including extrapolation to higher and lower k  
	# time the operation
	t1=time()
	fastpt=FPT.FASTPT(kbis,to_do=to_do,n_pad=n_pad, verbose=True) 
	t2=time()
		
	# calculate 1loop SPT (and time the operation) for density
	P_spt_dd=fastpt.one_loop_dd(Plinbis,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	# calculate 1loop SPT (and time the operation) for velocity
	P_spt_tt=fastpt.one_loop_tt(Plinbis,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	# calculate 1loop SPT (and time the operation) for velocity - density
	P_spt_dt=fastpt.one_loop_dt(Plinbis,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	#calculate tidal torque EE and BB P(k)
	#~ P_RSD=fastpt.RSD_components(P,1.0,C_window=C_window)	

	# update the power spectrum
	Pmod_dd=Plinbis+P_spt_dd[0]
	Pmod_dt=Plinbis+P_spt_dt[0]
	Pmod_tt=Plinbis+P_spt_tt[0]	
	A = P_spt_dd[2]
	B = P_spt_dd[3]
	C = P_spt_dd[4]
	D = P_spt_dd[5]
	E = P_spt_dd[6]
	F = P_spt_dd[7]
	G = P_spt_dt[2]
	H = P_spt_dt[3]
	

	#~ plt.figure()
	#~ plt.suptitle('z = '+str(z[j])+' ,expansion at 11th order, class h = 0.7, omega_b =0.05, omega_cdm = 0.25')
	#~ ax1=plt.subplot(311)
	#~ ax1.plot(kbis,Pmod_dd/Plinbis,label=r'$ \delta \delta FAST PT $', color='r')
	#~ ax1.plot(ksdd,psdd, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ ax2=plt.subplot(312)
	#~ ax2.plot(kbis,Pmod_dt/Plinbis,label=r'$ \delta \theta FAST PT $',color='r')
	#~ ax2.plot(ksdt,psdt, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ ax3=plt.subplot(313)
	#~ ax3.plot(kbis,Pmod_tt/Plinbis,label=r'$ \theta \theta FAST PT$', color='r')
	#~ ax3.plot(kstt,pstt, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.show()

	#~ plt.figure()
	#~ plt.plot(kbis,Pmod_dd)
	#~ plt.plot(kbis,Pmod_dt)
	#~ plt.plot(kbis,Pmod_tt)
	#~ plt.plot(kbis,Plinbis)
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(0.0008,10)
	#~ plt.ylim(3e3,4e4)
	#~ plt.ylim(1e-1,4e4)
	#~ plt.show()
	#~ kill
	
	
	
	####################################################################
	#### do the expected part
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/Pmod_dd_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], Pmod_dd[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/Pmod_dt_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], Pmod_dt[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/Pmod_tt_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], Pmod_tt[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/A_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], A[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/B_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], B[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/C_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], C[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/D_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], D[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/E_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], E[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/F_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], F[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/G_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], G[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/H_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], H[index_k]))
	#~ fid_file.close()
	
	
	####################################################################
	###### define bias fitting formulae for real space and redshift space
	def funcb(k, b1, b2, b3, b4):
		return b1 + b2 * k**2 + b3 * k**3 + b4 * k**4 
	def funcbis(k, b1, b2, b4):
		return b1 + b2 * k**2 + b4 * k**4 
	
	
	#~ # here kh because the simu scale
	popF1, pcovF1 = curve_fit(funcb, kbis, bias1bis, sigma = errb1bis,  check_finite=True, maxfev=500000)
	popF2, pcovF2 = curve_fit(funcb, kbis, bias2bis, sigma = errb2bis,  check_finite=True, maxfev=500000)
	popF3, pcovF3 = curve_fit(funcb, kbis, bias3bis, sigma = errb3bis,  check_finite=True, maxfev=500000)
	popF4, pcovF4 = curve_fit(funcb, kbis, bias4bis, sigma = errb4bis,  check_finite=True, maxfev=500000)

	popF1bis, pcovF1bis = curve_fit(funcbis, kbis, bias1bis, sigma = errb1bis,  check_finite=True, maxfev=500000)
	popF2bis, pcovF2bis = curve_fit(funcbis, kbis, bias2bis, sigma = errb2bis,  check_finite=True, maxfev=500000)
	popF3bis, pcovF3bis = curve_fit(funcbis, kbis, bias3bis, sigma = errb3bis,  check_finite=True, maxfev=500000)
	popF4bis, pcovF4bis = curve_fit(funcbis, kbis, bias4bis, sigma = errb4bis,  check_finite=True, maxfev=500000)

	
	########################################################################
	##### get fitted coefficient of the non linear bias  from the tidal model 
	#### (E.g Baldauf et el. Arxiv:1201.4827) given by Fast-PT bias function
	########################################################################
	


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
	#~ # odd power law----------------------------------------------------
	b1x1_mcmc, b2x1_mcmc, b3x1_mcmc, b4x1_mcmc = coeffit_pl(kstop, lb1, errlb1, popF1, kbis, bias1bis, errb1bis)
	b1x2_mcmc, b2x2_mcmc, b3x2_mcmc, b4x2_mcmc = coeffit_pl(kstop, lb2, errlb2, popF2, kbis, bias2bis, errb2bis)
	b1x3_mcmc, b2x3_mcmc, b3x3_mcmc, b4x3_mcmc = coeffit_pl(kstop, lb3, errlb3, popF3, kbis, bias3bis, errb3bis)
	b1x4_mcmc, b2x4_mcmc, b3x4_mcmc, b4x4_mcmc = coeffit_pl(kstop, lb4, errlb4, popF4, kbis, bias4bis, errb4bis)
	#~ # even power law ----------------------------------------------------------------------------------------
	#~ b1w1_mcmc, b2w1_mcmc, b4w1_mcmc = coeffit_pl2(kstop, lb1, errlb1, popF1bis, kbis, bias1bis, errb1bis)
	#~ b1w2_mcmc, b2w2_mcmc, b4w2_mcmc = coeffit_pl2(kstop, lb2, errlb2, popF2bis, kbis, bias2bis, errb2bis)
	#~ b1w3_mcmc, b2w3_mcmc, b4w3_mcmc = coeffit_pl2(kstop, lb3, errlb3, popF3bis, kbis, bias3bis, errb3bis)
	#~ b1w4_mcmc, b2w4_mcmc, b4w4_mcmc = coeffit_pl2(kstop, lb4, errlb4, popF4bis, kbis, bias4bis, errb4bis)
	# 2nd order bias ----------------------------------------------------------------------------------------------
	#~ b1y1_mcmc, b2y1_mcmc, bsy1_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb1,errlb1, pop1, kbis ,bias1bis ,errb1bis)
	#~ b1y2_mcmc, b2y2_mcmc, bsy2_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb2,errlb2, pop2, kbis ,bias2bis ,errb2bis)
	#~ b1y3_mcmc, b2y3_mcmc, bsy3_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb3,errlb3, pop3, kbis ,bias3bis ,errb3bis)
	#~ b1y4_mcmc, b2y4_mcmc, bsy4_mcmc = coeffit_exp1(kstop, Pmmbis, A, B, C, D, E, lb4,errlb4, pop4, kbis ,bias4bis ,errb4bis)
	#3rd order free -----------------------------------------------------------------------------------------------
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
		

	#~ cname1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_pl_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname1err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_pl_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname1bis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_ple_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname1errbis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_ple_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname2err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/err_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3bis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3errbis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ #------------------------------------------------------------------------------------------------
	#~ cname1 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_pl_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname1err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_pl_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname1bis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_ple_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname1errbis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_ple_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname2 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname2err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3 = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3err = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3bis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'
	#~ cname3errbis = '/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/case'+str(case)+'/err_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt'

	#~ with open(cname1, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x1_mcmc[0], b2x1_mcmc[0], b3x1_mcmc[0], b4x1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x2_mcmc[0], b2x2_mcmc[0], b3x2_mcmc[0], b4x2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x3_mcmc[0], b2x3_mcmc[0], b3x3_mcmc[0], b4x3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (b1x4_mcmc[0], b2x4_mcmc[0], b3x4_mcmc[0], b4x4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname1err, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1x1_mcmc[1], b2x1_mcmc[1], b3x1_mcmc[1], b4x1_mcmc[1]\
		#~ ,b1x1_mcmc[2], b2x1_mcmc[2], b3x1_mcmc[2], b4x1_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1x2_mcmc[1], b2x2_mcmc[1], b3x2_mcmc[1], b4x2_mcmc[1]\
		#~ ,b1x2_mcmc[2], b2x2_mcmc[2], b3x2_mcmc[2], b4x2_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1x3_mcmc[1], b2x3_mcmc[1], b3x3_mcmc[1], b4x3_mcmc[1]\
		#~ ,b1x3_mcmc[2], b2x3_mcmc[2], b3x3_mcmc[2], b4x3_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1x4_mcmc[1], b2x4_mcmc[1], b3x4_mcmc[1], b4x4_mcmc[1]\
		#~ ,b1x4_mcmc[2], b2x4_mcmc[2], b3x4_mcmc[2], b4x4_mcmc[2]))
	#~ fid_file.close()
	#~ with open(cname1bis, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1w1_mcmc[0], b2w1_mcmc[0], b4w1_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1w2_mcmc[0], b2w2_mcmc[0], b4w2_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1w3_mcmc[0], b2w3_mcmc[0], b4w3_mcmc[0]))
		#~ fid_file.write('%.8g %.8g %.8g\n' % (b1w4_mcmc[0], b2w4_mcmc[0], b4w4_mcmc[0]))
	#~ fid_file.close()
	#~ with open(cname1errbis, 'w') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1w1_mcmc[1], b2w1_mcmc[1], b4w1_mcmc[1]\
		#~ ,b1w1_mcmc[2], b2w1_mcmc[2], b4w1_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1w2_mcmc[1], b2w2_mcmc[1], b4w2_mcmc[1]\
		#~ ,b1w2_mcmc[2], b2w2_mcmc[2], b4w2_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1w3_mcmc[1], b2w3_mcmc[1], b4w3_mcmc[1]\
		#~ ,b1w3_mcmc[2], b2w3_mcmc[2], b4w3_mcmc[2]))
		#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g %.8g\n' % (b1w4_mcmc[1], b2w4_mcmc[1], b4w4_mcmc[1]\
		#~ ,b1w4_mcmc[2], b2w4_mcmc[2], b4w4_mcmc[2]))
	#~ fid_file.close()
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
	
	#~ M1pl, M2pl, M3pl, M4pl = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_pl_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1plbis, M2plbis, M3plbis, M4plbis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_ple_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1pt2, M2pt2, M3pt2, M4pt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1pt3, M2pt3, M3pt3, M4pt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1pt3bis, M2pt3bis, M3pt3bis, M4pt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/case'+str(case)+'/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#---------------------------------------------------------------------
	#~ M1pl, M2pl, M3pl, M4pl = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_pl_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1plbis, M2plbis, M3plbis, M4plbis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_ple_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1pt2, M2pt2, M3pt2, M4pt2 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_2exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1pt3, M2pt3, M3pt3, M4pt3 = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_3exp_'+str(Mnu)+'_z='+str(z[j])+'.txt')
	#~ M1pt3bis, M2pt3bis, M3pt3bis, M4pt3bis = np.loadtxt('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+\
	#~ str(Mnu)+'eV/coeff_3exp_fixed_'+str(Mnu)+'_z='+str(z[j])+'.txt')

	####################################################################
	#### get results from mcmc analysis to plots the different biases
	# power law odd ----------------------------------------------------------------------------
	biasF1 = b1x1_mcmc[0] + b2x1_mcmc[0] * kbis**2 + b3x1_mcmc[0] * kbis**3 + b4x1_mcmc[0] * kbis**4
	biasF2 = b1x2_mcmc[0] + b2x2_mcmc[1] * kbis**2 + b3x2_mcmc[0] * kbis**3 + b4x2_mcmc[0] * kbis**4
	biasF3 = b1x3_mcmc[0] + b2x3_mcmc[1] * kbis**2 + b3x3_mcmc[0] * kbis**3 + b4x3_mcmc[0] * kbis**4
	biasF4 = b1x4_mcmc[0] + b2x4_mcmc[1] * kbis**2 + b3x4_mcmc[0] * kbis**3 + b4x4_mcmc[0] * kbis**4
	
	#~ # power law even -------------------------------------------------------------------------------------------
	#~ biasF1bis = b1w1_mcmc[0] + b2w1_mcmc[0] * kbis**2 + b4w1_mcmc[0] * kbis**4
	#~ biasF2bis = b1w2_mcmc[0] + b2w2_mcmc[0] * kbis**2 + b4w2_mcmc[0] * kbis**4
	#~ biasF3bis = b1w3_mcmc[0] + b2w3_mcmc[0] * kbis**2 + b4w3_mcmc[0] * kbis**4
	#~ biasF4bis = b1w4_mcmc[0] + b2w4_mcmc[0] * kbis**2 + b4w4_mcmc[0] * kbis**4
	
	#~ # 2nd order ------------------------------------------------------------------ 
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
	#~ 1/2.*b2u1_mcmc[0]*bsu1_mcmc[0]*D + 1/4.*bsu1_mcmc[0]**2*E + 2*b1u1_mcmc[0]*B3nlTa*F)/Pmmbis)
	#~ bias3PTbis2 = np.sqrt((b1u2_mcmc[0]**2 * Pmmbis+ b1u2_mcmc[0]*b2u2_mcmc[0]*A + 1/4.*b2u2_mcmc[0]**2*B + b1u2_mcmc[0]*bsu2_mcmc[0]*C +\
	#~ 1/2.*b2u2_mcmc[0]*bsu2_mcmc[0]*D + 1/4.*bsu2_mcmc[0]**2*E + 2*b1u2_mcmc[0]*B3nlTb*F)/Pmmbis)
	#~ bias3PTbis3 = np.sqrt((b1u3_mcmc[0]**2 * Pmmbis+ b1u3_mcmc[0]*b2u3_mcmc[0]*A + 1/4.*b2u3_mcmc[0]**2*B + b1u3_mcmc[0]*bsu3_mcmc[0]*C +\
	#~ 1/2.*b2u3_mcmc[0]*bsu3_mcmc[0]*D + 1/4.*bsu3_mcmc[0]**2*E + 2*b1u3_mcmc[0]*B3nlTc*F)/Pmmbis)
	#~ bias3PTbis4 = np.sqrt((b1u4_mcmc[0]**2 * Pmmbis+ b1u4_mcmc[0]*b2u4_mcmc[0]*A + 1/4.*b2u4_mcmc[0]**2*B + b1u4_mcmc[0]*bsu4_mcmc[0]*C +\
	#~ 1/2.*b2u4_mcmc[0]*bsu4_mcmc[0]*D + 1/4.*bsu4_mcmc[0]**2*E + 2*b1u4_mcmc[0]*B3nlTd*F)/Pmmbis)
	
	
	#~ ### mean ####
	#~ B1 = np.array([bias2PT1/bias1bis, bias2PT2/bias2bis, bias2PT3/bias3bis, bias2PT4/bias4bis])
	#~ B1bis = np.array([bias3PT1/bias1bis, bias3PT2/bias2bis, bias3PT3/bias3bis, bias3PT4/bias4bis])
	#~ B1ter = np.array([bias3PTbis1/bias1bis, bias3PTbis2/bias2bis, bias3PTbis3/bias3bis, bias3PTbis4/bias4bis])
	B2 = np.array([bias1bis/bias1bis, bias2bis/bias2bis, bias3bis/bias3bis, bias4bis/bias4bis])
	B3 = np.array([biasF1/bias1bis, biasF2/bias2bis, biasF3/bias3bis, biasF4/bias4bis])
	#~ B3bis = np.array([biasF1bis/bias1bis, biasF2bis/bias2bis, biasF3bis/bias3bis, biasF4bis/bias4bis])
	#~ b1 = np.mean(B1,axis=0)
	#~ b1bis = np.mean(B1bis,axis=0)
	#~ b1ter = np.mean(B1ter,axis=0)
	b2 = np.mean(B2,axis=0)
	b3 = np.mean(B3,axis=0)
	#~ b3bis = np.mean(B3bis,axis=0)
	

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
	#~ ax2.set_ylim(0.9,1.1)
	#~ ax2.axhline(1, color='k', linestyle='--')
	#~ ax2.axhline(1.01, color='k', linestyle=':')
	#~ ax2.axhline(0.99, color='k', linestyle=':')
	#~ B3, = ax2.plot(kbis, b3)
	#~ B3bis, = ax2.plot(kbis, b3bis)
	#~ B2, = ax2.plot(kbis, b2, label='z = '+str(z[j]), color='k')
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
	ax2.set_ylim(0.9,1.1)
	ax2.set_yticks(np.linspace(0.9,1.1,5))
	ax2.axhline(1, color='k', linestyle='--')
	ax2.axhline(1.01, color='k', linestyle=':')
	ax2.axhline(0.99, color='k', linestyle=':')
	B3, = ax2.plot(kbis, b3,label=r'w/ $b_{sim}$', color='C0')
	#~ B1, = ax2.plot(kbis, b1, color='C1')	
	#~ B1bis, = ax2.plot(kbis, b1bis, color='C2')
	#~ B1ter, = ax2.plot(kbis, b1ter,  color='C3')
	B2, = ax2.plot(kbis, b2, color='k')
	
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

