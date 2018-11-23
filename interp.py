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


def interp_simu(k,kcamb, Pcamb, Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H, way):

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
				
				
		expected_CF.expected(j)
		
		
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
