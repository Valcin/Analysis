import os
import numpy as np
import warnings
from numpy import newaxis as na
from math import exp, log, pi, log10
import time
import math
import matplotlib.pyplot as plt

dat_file_path1 = '/home/david/codes/montepython_public/chains/test/2019-02-08_10000__1.txt'
dat_file_path2 = '/home/david/codes/montepython_public/chains/test/2019-02-08_10000__2.txt'
dat_file_path3 = '/home/david/codes/montepython_public/chains/test/2019-02-08_10000__3.txt'
dat_file_path4 = '/home/david/codes/montepython_public/chains/test/2019-02-08_10000__4.txt'

#~ dat_file_path5 = '/home/david/codes/montepython_public/chains/test/2019-02-07_10000__1.txt'
#~ dat_file_path6 = '/home/david/codes/montepython_public/chains/test/2019-02-07_10000__2.txt'
#~ dat_file_path7 = '/home/david/codes/montepython_public/chains/test/2019-02-07_10000__3.txt'
#~ dat_file_path8 = '/home/david/codes/montepython_public/chains/test/2019-02-07_10000__4.txt'



#~ burn_in = 200
burn_in = 0
mnu = []
mnu2 = []
mnu3 = []
mnu4 = []
om_cdm = []
s8 = []
def read_col(name):
	
	fil = np.loadtxt(name)
	first_col = fil[:,0]
	size = len(first_col)
	arr_size = size-burn_in

	with open(name,'r') as f:
		line = f.readline()

		for index_k in range(arr_size):
			#~ mnu[index_k] = float(line.split()[3])
			mnu.append(float(line.split()[3])) 
			om_cdm.append(float(line.split()[2])) 
			s8.append(float(line.split()[8])) 
			line = f.readline()
	return 

om_cdm2 = []
def read_col2(name):
	
	fil = np.loadtxt(name)
	first_col = fil[:,0]
	size = len(first_col)
	arr_size = size-burn_in

	with open(name,'r') as f:
		line = f.readline()

		for index_k in range(arr_size):
			omega_m = float(line.split()[9])
			if 0.3170 < omega_m < 0.3180:
				mnu2.append(float(line.split()[3])) 
				om_cdm2.append(float(line.split()[2])) 
			line = f.readline()
	return
	
def read_col3(name):
	
	fil = np.loadtxt(name)
	first_col = fil[:,0]
	size = len(first_col)
	arr_size = size-burn_in

	with open(name,'r') as f:
		line = f.readline()

		for index_k in range(arr_size):
			sigma8 = float(line.split()[8])
			#~ print sigma8
			if 0.803 < sigma8 < 0.809:
				mnu3.append(float(line.split()[3])) 
			line = f.readline()
	return
	
def read_col4(name):
	
	fil = np.loadtxt(name)
	first_col = fil[:,0]
	size = len(first_col)
	arr_size = size-burn_in

	with open(name,'r') as f:
		line = f.readline()

		for index_k in range(arr_size):
			sigma8 = float(line.split()[8])
			omega_m = float(line.split()[9])
			#~ print sigma8
			if 0.3170 < omega_m < 0.3180 and 0.803 < sigma8 < 0.809:
				mnu4.append(float(line.split()[3])) 
			line = f.readline()
	return
	
read_col(dat_file_path1)
read_col(dat_file_path2)
read_col(dat_file_path3)
read_col(dat_file_path4)
#~ read_col(dat_file_path5)
#~ read_col(dat_file_path6)
#~ read_col(dat_file_path7)
#~ read_col(dat_file_path8)

read_col2(dat_file_path1)
read_col2(dat_file_path2)
read_col2(dat_file_path3)
read_col2(dat_file_path4)
#~ read_col2(dat_file_path5)
#~ read_col2(dat_file_path6)
#~ read_col2(dat_file_path7)
#~ read_col2(dat_file_path8)

read_col3(dat_file_path1)
read_col3(dat_file_path2)
read_col3(dat_file_path3)
read_col3(dat_file_path4)
#~ read_col3(dat_file_path5)
#~ read_col3(dat_file_path6)
#~ read_col3(dat_file_path7)
#~ read_col3(dat_file_path8)

read_col4(dat_file_path1)
read_col4(dat_file_path2)
read_col4(dat_file_path3)
read_col4(dat_file_path4)
#~ read_col4(dat_file_path5)
#~ read_col4(dat_file_path6)
#~ read_col4(dat_file_path7)
#~ read_col4(dat_file_path8)

#~ multidat = [mnu1, mnu2, mnu3, mnu4]
bins = np.linspace(0.06, 0.25,20)

print np.min(mnu), np.max(mnu)
print np.mean(mnu), np.std(mnu), np.std(mnu, ddof=1)
print np.mean(mnu2), np.std(mnu2), np.std(mnu2, ddof=1)
print np.mean(mnu3), np.std(mnu3), np.std(mnu3, ddof=1)
print np.mean(mnu4), np.std(mnu4), np.std(mnu4, ddof=1)

#~ lbl = [r'No constraints, $\bar{M_{\nu}}$ =  '+str(np.mean(mnu)),r' Constraints on $\Omega_m$',r' Constraints on $\sigma_8$',r' Constraints on $\Omega_m$ and $\sigma_8$']
lbl = [r'No constraints',r' Constraints on $\Omega_m$',r' Constraints on $\sigma_8$',r' Constraints on $\Omega_m$ and $\sigma_8$']
plt.hist([mnu,mnu2, mnu3, mnu4], bins=bins, edgecolor='k', label=lbl)  # arguments are passed to np.histogram
plt.legend(loc='upper right', fontsize =16)
plt.title("8 chains of 10000 steps with 200 points of burn-in in accepted steps")
plt.xlabel(r'$M_{\nu}$ [eV]', fontsize = 16)
plt.axvline(np.mean(mnu), c='C0', linestyle='--')
plt.axvline(np.mean(mnu2), c='C1', linestyle='--')
plt.axvline(np.mean(mnu3), c='C2', linestyle='--')
plt.axvline(np.mean(mnu4), c='C3', linestyle='--')
plt.axvline(0.15, c='k')
plt.show()


plt.figure()
plt.scatter(om_cdm, s8)
plt.xlabel(r'$\omega_{cdm}$', fontsize = 16)
plt.ylabel(r'$\sigma_8$', fontsize = 16)
plt.show()

plt.figure()
plt.scatter(mnu, s8)
plt.xlabel(r'$M_{\nu}$ [eV]', fontsize = 16)
plt.ylabel(r'$\sigma_8$', fontsize = 16)
plt.show()

plt.figure()
plt.scatter(om_cdm, mnu)
plt.xlabel(r'$\omega_{cdm} $', fontsize = 16)
plt.ylabel(r'$M_{\nu}$ [eV]', fontsize = 16)
plt.show()
