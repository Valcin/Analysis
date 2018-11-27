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
#~ from load_data import ld_data
#~ from rescaling import rescal
#~ from loop_pt import pt_terms
#~ from polynomial import poly
#~ from perturbation import perturb


def interp_simu1(z, j, k,kcamb, Pcamb, Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H, way):

	if way == 1:
		
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file1.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], Pcamb[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file2.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], Pmod_dd[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file3.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], Pmod_dt[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file4.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], Pmod_tt[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file5.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], A[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file6.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], B[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file7.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], C[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file8.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], D[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file9.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], E[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file10.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], F[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file11.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], G[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file12.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], H[index_k]))
		fid_file.close()
				
				
		#~ exp2.expected(j)
		
		
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
		Pcamb = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
		Pmod_dd = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected3-'+str(z[j])+'.txt')
		Pmod_dt = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected4-'+str(z[j])+'.txt')
		Pmod_tt = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected5-'+str(z[j])+'.txt')
		A = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected6-'+str(z[j])+'.txt')
		B = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected7-'+str(z[j])+'.txt')
		C = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected8-'+str(z[j])+'.txt')
		D = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected9-'+str(z[j])+'.txt')
		E = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected10-'+str(z[j])+'.txt')
		F = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected11-'+str(z[j])+'.txt')
		G = pte[:,1]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected12-'+str(z[j])+'.txt')
		H = pte[:,1]
			
	if way == 2:
		Pcamb = np.interp(k, kcamb, Pcamb)
		Pmod_dd = np.interp(k, kcamb, Pmod_dd)
		Pmod_dt = np.interp(k, kcamb, Pmod_dt)
		Pmod_tt = np.interp(k, kcamb, Pmod_tt)
		A = np.interp(k, kcamb, A)
		B = np.interp(k, kcamb, B)
		C = np.interp(k, kcamb, C)
		D = np.interp(k, kcamb, D)
		E = np.interp(k, kcamb, E)
		F = np.interp(k, kcamb, F)
		G = np.interp(k, kcamb, G)
		H = np.interp(k, kcamb, H)
		
	return  Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H
	
	
def interp_simu2(z,j, k, kcamb, Pcamb, AB2_1,AB4_1,AB6_1,AB8_1,AB2_2,AB4_2,AB6_2,AB8_2,AB2_3,\
		AB4_3,AB6_3,AB8_3, AB2_4,AB4_4,AB6_4,AB8_4,AB2bis_1,AB4bis_1,AB6bis_1,AB8bis_1 ,AB2bis_2,\
		AB4bis_2,AB6bis_2,AB8bis_2, AB2bis_3,AB4bis_3,AB6bis_3,AB8bis_3, AB2bis_4,AB4bis_4,AB6bis_4,AB8bis_4, way):

	if way == 1:
		
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_1.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2_1[index_k],AB4_1[index_k],\
				AB6_1[index_k],AB8_1[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_2.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2_2[index_k],AB4_2[index_k],\
				AB6_2[index_k],AB8_2[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_3.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2_3[index_k],AB4_3[index_k],\
				AB6_3[index_k],AB8_3[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_4.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2_4[index_k],AB4_4[index_k],\
				AB6_4[index_k],AB8_4[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_5.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2bis_1[index_k],AB4bis_1[index_k],\
				AB6bis_1[index_k],AB8bis_1[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_6.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2bis_2[index_k],AB4bis_2[index_k],\
				AB6bis_2[index_k],AB8bis_2[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_7.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2bis_3[index_k],AB4bis_3[index_k],\
				AB6bis_3[index_k],AB8bis_3[index_k]))
		fid_file.close()
		with open('/home/david/codes/Paco/data2/0.0eV/exp/file_8.txt', 'w+') as fid_file:
			for index_k in xrange(len(kcamb)):
				fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( kcamb[index_k], AB2bis_4[index_k],AB4bis_4[index_k],\
				AB6bis_4[index_k],AB8bis_4[index_k]))
		fid_file.close()
				
				
		#~ expected_CF.expected(j)
		
		
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_1-'+str(z[j])+'.txt')
		AB2_1,AB4_1,AB6_1,AB8_1 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_2-'+str(z[j])+'.txt')
		AB2_2,AB4_2,AB6_2,AB8_2 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_3-'+str(z[j])+'.txt')
		AB2_3,AB4_3,AB6_3,AB8_3 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_4-'+str(z[j])+'.txt')
		AB2_4,AB4_4,AB6_4,AB8_4 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_5-'+str(z[j])+'.txt')
		AB2bis_1,AB4bis_1,AB6bis_1,AB8bis_1 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_6-'+str(z[j])+'.txt')
		AB2bis_2, AB4bis_2,AB6bis_2,AB8bis_2 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_7-'+str(z[j])+'.txt')
		AB2bis_3,AB4bis_3,AB6bis_3,AB8bis_3 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
		pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected_8-'+str(z[j])+'.txt')
		AB2bis_4,AB4bis_4,AB6bis_4,AB8bis_4 = pte[:,0], pte[:,1], pte[:,2], pte[:,3]
			
	if way == 2:
		### interpolate on simulation k
		AB2_1 = np.interp(k, kcamb, AB2_1); AB2bis_1 = np.interp(k, kcamb, AB2bis_1)
		AB4_1 = np.interp(k, kcamb, AB4_1); AB4bis_1 = np.interp(k, kcamb, AB4bis_1)
		AB6_1 = np.interp(k, kcamb, AB6_1); AB6bis_1 = np.interp(k, kcamb, AB6bis_1)
		AB8_1 = np.interp(k, kcamb, AB8_1); AB8bis_1 = np.interp(k, kcamb, AB8bis_1)
		AB2_2 = np.interp(k, kcamb, AB2_2); AB2bis_2 = np.interp(k, kcamb, AB2bis_2)
		AB4_2 = np.interp(k, kcamb, AB4_2); AB4bis_2 = np.interp(k, kcamb, AB4bis_2)
		AB6_2 = np.interp(k, kcamb, AB6_2); AB6bis_2 = np.interp(k, kcamb, AB6bis_2)
		AB8_2 = np.interp(k, kcamb, AB8_2); AB8bis_2 = np.interp(k, kcamb, AB8bis_2)
		AB2_3 = np.interp(k, kcamb, AB2_3); AB2bis_3 = np.interp(k, kcamb, AB2bis_3)
		AB4_3 = np.interp(k, kcamb, AB4_3); AB4bis_3 = np.interp(k, kcamb, AB4bis_3)
		AB6_3 = np.interp(k, kcamb, AB6_3); AB6bis_3 = np.interp(k, kcamb, AB6bis_3)
		AB8_3 = np.interp(k, kcamb, AB8_3); AB8bis_3 = np.interp(k, kcamb, AB8bis_3)
		AB2_4 = np.interp(k, kcamb, AB2_4); AB2bis_4 = np.interp(k, kcamb, AB2bis_4)
		AB4_4 = np.interp(k, kcamb, AB4_4); AB4bis_4 = np.interp(k, kcamb, AB4bis_4)
		AB6_4 = np.interp(k, kcamb, AB6_4); AB6bis_4 = np.interp(k, kcamb, AB6bis_4)
		AB8_4 = np.interp(k, kcamb, AB8_4); AB8bis_4 = np.interp(k, kcamb, AB8bis_4)
		
		
	return  AB2_1,AB4_1,AB6_1,AB8_1,AB2_2,AB4_2,AB6_2,AB8_2,AB2_3,AB4_3,AB6_3,AB8_3,\
			AB2_4,AB4_4,AB6_4,AB8_4, AB2bis_1,AB4bis_1,AB6bis_1,AB8bis_1 ,AB2bis_2,AB4bis_2,AB6bis_2,AB8bis_2,\
			AB2bis_3,AB4bis_3,AB6bis_3,AB8bis_3, AB2bis_4,AB4bis_4,AB6bis_4,AB8bis_4
			
			
def interp_simu3(z,j, k, kclass, Tm, Tcb, way):

	#~ if way == 1:
		#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file1.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(kcamb)):
				#~ fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], Pcamb[index_k]))
		#~ fid_file.close()
		#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/file2.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(kcamb)):
				#~ fid_file.write('%.8g %.8g\n' % ( kcamb[index_k], Pmod_dd[index_k]))
		#~ fid_file.close()
				
		#~ exp2.expected(j)
		
		
		#~ pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected1-'+str(z[j])+'.txt')
		#~ Pcamb = pte[:,1]
		#~ pte = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/exp/expected2-'+str(z[j])+'.txt')
		#~ Pmod_dd = pte[:,1]
		
	if way == 2:
		### interpolate on simulation k
		Tm= np.interp(k, kclass, Tm); Tcb= np.interp(k, kclass, Tcb)
		
	return  Tm, Tcb
