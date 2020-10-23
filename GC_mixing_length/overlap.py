import numpy as np
import scipy.optimize as op
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import emcee
import gc
import os
import sys
sys.path.append('/home/david/codes/isochrones')# folder where isochrones is installed
import math
import seaborn as sns
import pandas as pd
import time
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import stats
from astropy import stats as stt
from scipy import interpolate

from scipy import optimize
from scipy.signal import argrelextrema
from scipy.stats import linregress
#~ from tempfile import TemporaryFile
from multiprocessing import Pool
#~ from emcee.utils import MPIPool
#~ from emcee import PTSampler
# ~ from memory_profiler import profile

#~ from isochrones.dartmouth import Dartmouth_FastIsochrone
#~ from isochrones.mist import MIST_Isochrone
#~ mist = MIST_Isochrone()
#~ dar = Dartmouth_FastIsochrone()

########################################################################
########################################################################
#~ @profile
def cluster(nb):
	
	clus_name = ['arp2','ic4499','lynga7','ngc0104','ngc0288','ngc0362','ngc1261','ngc1851','ngc2298','ngc2808','ngc3201',
	'ngc4147','ngc4590','ngc4833','ngc5024','ngc5053','ngc5139','ngc5272','ngc5286','ngc5466','ngc5904','ngc5927','ngc5986',
	'ngc6093','ngc6101','ngc6121','ngc6144','ngc6171','ngc6205','ngc6218','ngc6254','ngc6304','ngc6341','ngc6352','ngc6362',
	'ngc6366','ngc6388','ngc6397','ngc6426','ngc6441','ngc6496','ngc6535','ngc6541','ngc6584','ngc6624','ngc6637','ngc6652',
	'ngc6656','ngc6681','ngc6715','ngc6717','ngc6723','ngc6752','ngc6779','ngc6809','ngc6838','ngc6934','ngc6981','ngc7006',  
	'ngc7078','ngc7089','ngc7099','palomar1','palomar12','palomar15','pyxis','ruprecht106','terzan7','terzan8']


	with open('/home/david/codes/Analysis/GC_mixing_length/Harris2010.dat',"r") as f:
		lines=f.readlines()[1:]
	f.close()
	harris_clus=[]
	for x in lines:
		harris_clus.append(x.split(' ')[0])

	with open('/home/david/codes/Analysis/GC_mixing_length/dotter2010.dat',"r") as f:
		lines=f.readlines()[3:]
	f.close()
	dotter_clus=[]
	for x in lines:
		dotter_clus.append(x.split(' ')[0])

	with open('/home/david/codes/Analysis/GC_mixing_length/roediger2014.dat',"r") as f:
		lines=f.readlines()[5:]
	f.close()
	roediger_clus=[]
	for x in lines:
		roediger_clus.append(x.split(' ')[0])

	clus_nb = clus_name[nb]
	
	# find acs initial values in different caltalogs
	index1 = harris_clus.index(clus_nb)
	
	with open('/home/david/codes/Analysis/GC_mixing_length/Harris2010.dat',"r") as f:
		lines=f.readlines()[1:]
	f.close()
	dist_mod=[]
	Abs=[]
	metal=[]
	for x in lines:
		dist_mod.append(x.split(' ')[5])
		Abs.append(x.split(' ')[3])
		metal.append(x.split(' ')[1])

	#select the value of the given cluster
	Abs = np.float64(Abs[index1])* 3.1 # 3.1 factor to compute absorption from extinction
	dist_mod = dist_mod[index1]
	metal = metal[index1]

	if clus_nb in dotter_clus :
		index2 = dotter_clus.index(clus_nb)
		with open('/home/david/codes/Analysis/GC_mixing_length/dotter2010.dat',"r") as f:
			lines=f.readlines()[3:]
		f.close()
		age_dotter=[]
		for x in lines:
			age_dotter.append(x.split(' ')[5])
		age = age_dotter[index2]
	elif clus_nb in roediger_clus:
		index2 = roediger_clus.index(clus_nb)
		with open('/home/david/codes/Analysis/GC_mixing_length/roediger2014.dat',"r") as f:
			lines=f.readlines()[5:]
		f.close()
		age_roediger=[]
		for x in lines:
			age_roediger.append(x.split(' ')[1])
		age = age_roediger[index2]
	else:
		age = 12.0


	if clus_nb in dotter_clus :
		index2 = dotter_clus.index(clus_nb)
		with open('/home/david/codes/Analysis/GC_mixing_length/dotter2010.dat',"r") as f:
			lines=f.readlines()[3:]
		f.close()
		afe_dotter=[]
		for x in lines:
			afe_dotter.append(x.split(' ')[2])
		afe_init = afe_dotter[index2]
	else:
		afe_init = 0.0
		
	### convert in isochrones units
	Age = np.log10(np.float32(age)*1e9)
	distance = 10**(np.float32(dist_mod)/5. +1)
	distplus = 10**((np.float32(dist_mod)+0.2)/5. +1)
	distmoins = 10**((np.float32(dist_mod)-0.2)/5. +1)
	print(dist_mod)
	Age = float(round(Age,3))
	metal = float(round(float(metal),3))
	distance = float(round(distance,3))
	Abs = float(round(Abs,3))
	afe_init = float(afe_init)

	extdutra = [0.22,0.03,0.03,0.01,0.22,0.06,0.14,0.21,0.10,0.37,0.41,0.52,0.07,0.23,0.41,0.11,0.11,0.063,0.25,0.08]
	extgc = [1,3,5,6,8,12,16,23,24,25,26,31,34,40,41,46,48,51,53,58]
	if glc in extgc:
		ind = extgc.index(glc)
		Abs = 3.1*extdutra[ind]

	return clus_nb, Age, metal, distance, Abs, afe_init, distplus, distmoins
#~ @profile
def photometry():
        
	files = np.loadtxt('/home/david/codes/data/GC_data/data_HST/hlsp_acsggct_hst_acs-wfc_'+clus_nb+'_r.rdviq.cal.adj.zpt', skiprows = 3)
	longueur = len(files)
	
	### magnitude cut --------------------------------------------------
	min_mag = np.min(files[:,3])
	mag_min = min_mag
	mag_max = 26
	
	#~ mg_cut = np.where(files[:,3] <= mag_max)[0]
	mg_cut = np.where(files[:,3] <50)[0]

	### limit on photmetric error ------------------------------------

	pv = np.zeros(len(files[mg_cut,4]))
	for i in range(len(files[mg_cut,4])):
		if files[i,11] == 1:
			pv[i] = files[i,4] * 3.5
		elif files[i,11] == 2:
			pv[i] = files[i,4] / 2.
		else:
			pv[i] = files[i,4]

	pr = np.zeros(len(files[mg_cut,8]))
	for i in range(len(files[mg_cut,8])):
		if files[i,12] == 1:
			pr[i] = files[i,8] * 3.5
		elif files[i,12] == 2:
			pr[i] = files[i,8] / 2.
		else:
			pr[i] = files[i,8]

	pcolor = np.sqrt(pr**2 + pv**2)
	#~ print(np.min(pcolor))
	#~ print(np.max(pcolor))

	lim606_right = np.percentile(pv, 95)
	lim814_right = np.percentile(pr, 95)
	

	### limit on pixel frame -----------------------------------------

	xstars = files[mg_cut,15]
	ystars = files[mg_cut,16]
	lim_x = np.percentile(xstars, 97.5)
	lim_y = np.percentile(ystars, 97.5)

	### filter the sample. bright stars have good photometry and the error is 0.
	### collapse later when divided by 0 so they are removed here
	filter_all = np.where((pv <= lim606_right)  & (pr <= lim814_right) 
	 & (xstars <= lim_x) & (ystars <= lim_y)  & (pcolor != 0.0))[0]


	#~ ### get the photmetry of the selected stars in both filters
	photo_v = files[mg_cut, 3][filter_all]
	photo_i = files[mg_cut, 7][filter_all]
	
	print(np.min(photo_v), np.min(photo_i))
	print(np.max(photo_v), np.max(photo_i))

	Color = files[mg_cut, 5][filter_all]
	err_Color = pcolor[filter_all]
	err_v = pv[filter_all]
	nmv = files[mg_cut,11][filter_all]
	nmi = files[mg_cut,12][filter_all]
	#~ photo_v = files[mg_cut, 3]
	#~ photo_i = files[mg_cut, 7]

	#~ Color = files[mg_cut, 5]
	#~ err_Color = pcolor
	#~ err_v = pv
	#~ nmv = files[mg_cut,11]
	#~ nmi = files[mg_cut,12]


	del files
	gc.collect()

	return photo_v, err_v, photo_i, Color, err_Color, nmv, nmi, longueur

########################################################################
########################################################################
### define global variables
start1 = time.time()
iteration = 0 
ncpu = 3 # number of cpu requested
#~ ncpu = len(os.sched_getaffinity(0)) # number of cpu requested aganice
#~ print(ncpu)
#~ kill
#~ ncpu = int(os.environ["cpu_num"]) # number of cpu requested hipatia

rescale = np.loadtxt('rescale_ig.csv',delimiter=',')
ind1 = np.loadtxt('ind_met15.txt')
ind2 = np.loadtxt('ind_met20.txt')


#rescale gc to put mstop at 0
chunkbot = rescale[:,5]
mstop = chunkbot[glc]

met = (input("what is the metallicity limit ? "))
if met == '-1.5':
	ind = ind1
elif met == '-2.0':
	ind = ind2

for g in ind[0]:
	glc = int(g)
	print(glc)
# ~glc = int(input("what is your cluster number? "))
# glc = int(os.environ["SLURM_ARRAY_TASK_ID"]) # aganice
#~ glc = int(os.environ["PBS_ARRAYID"])  # hipatia
	print("the chosen cluster is {}".format(glc))
	clus_nb, Age, metal, distance, Abs, afe_init, distplus, distmoins  = cluster(glc)
	print(clus_nb, Age, metal, distance, Abs, afe_init, distplus, distmoins)
	photo_v, err_v, photo_i, color, err_color, nmv, nmi, longueur = photometry()

# ~plt.figure()
# ~plt.scatter(color,photo_v, marker='.', s=10, color='grey', alpha=0.8)
	plt.scatter(color,photo_v, marker='.', s=10, alpha=0.8)
	plt.axhline(mstop)
plt.xlim(-0.5,3)
plt.ylim(27,10)
plt.tick_params(labelsize=16)
plt.subplots_adjust(bottom=0.16)
# ~lgnd = plt.legend(loc='best', fontsize = 24)
# ~lgnd.get_frame().set_edgecolor('k')
# ~lgnd.get_frame().set_linewidth(2.0)
plt.xlabel('F606W - F814W', fontsize = 24)
plt.ylabel('F606W', fontsize = 24)
plt.title('[Fe/H] < '+met+', '+str(len(ind))+' clusters', fontsize = 24)
plt.show() 
plt.close()
# ~kill
