
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib import colors
import emcee
import gc
import os
import corner
import sys
sys.path.append('/home/david/codes/isochrones')# folder where isochrones is installed
import math
import seaborn as sns
import pandas as pd
import time
import random
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import stats
from astropy import stats as stt
from scipy import interpolate
from scipy import optimize
# ~from scipy.misc import factorial

#~ from tempfile import TemporaryFile
# from multiprocessing import Pool
#~ from emcee.utils import MPIPool
#~ from emcee import PTSampler
#~ from memory_profiler import profile
#~ from isochrones.dartmouth import Dartmouth_FastIsochrone
#~ from isochrones.mist import MIST_Isochrone
#~ mist = MIST_Isochrone()
#~ dar = Dartmouth_FastIsochrone()


########################################################################
########################################################################

def cluster(nb):
	
	clus_name = ['arp2','ic4499','lynga7','ngc0104','ngc0288','ngc0362','ngc1261','ngc1851','ngc2298','ngc2808','ngc3201',
	'ngc4147','ngc4590','ngc4833','ngc5024','ngc5053','ngc5139','ngc5272','ngc5286','ngc5466','ngc5904','ngc5927','ngc5986',
	'ngc6093','ngc6101','ngc6121','ngc6144','ngc6171','ngc6205','ngc6218','ngc6254','ngc6304','ngc6341','ngc6352','ngc6362',
	'ngc6366','ngc6388','ngc6397','ngc6426','ngc6441','ngc6496','ngc6535','ngc6541','ngc6584','ngc6624','ngc6637','ngc6652',
	'ngc6656','ngc6681','ngc6715','ngc6717','ngc6723','ngc6752','ngc6779','ngc6809','ngc6838','ngc6934','ngc6981','ngc7006',  
	'ngc7078','ngc7089','ngc7099','palomar1','palomar12','palomar15','pyxis','ruprecht106','terzan7','terzan8']


	with open('/home/david/codes/Analysis/GC/Harris2010.dat',"r") as f:
		lines=f.readlines()[1:]
	f.close()
	harris_clus=[]
	for x in lines:
		harris_clus.append(x.split(' ')[0])

	with open('/home/david/codes/Analysis/GC/dotter2010.dat',"r") as f:
		lines=f.readlines()[3:]
	f.close()
	dotter_clus=[]
	for x in lines:
		dotter_clus.append(x.split(' ')[0])

	with open('/home/david/codes/Analysis/GC/roediger2014.dat',"r") as f:
		lines=f.readlines()[5:]
	f.close()
	roediger_clus=[]
	for x in lines:
		roediger_clus.append(x.split(' ')[0])

	clus_nb = clus_name[nb]
	
	# find acs initial values in different caltalogs
	index1 = harris_clus.index(clus_nb)
	
	with open('/home/david/codes/Analysis/GC/Harris2010.dat',"r") as f:
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
		with open('/home/david/codes/Analysis/GC/dotter2010.dat',"r") as f:
			lines=f.readlines()[3:]
		f.close()
		age_dotter=[]
		for x in lines:
			age_dotter.append(x.split(' ')[5])
		age = age_dotter[index2]
	elif clus_nb in roediger_clus:
		index2 = roediger_clus.index(clus_nb)
		with open('/home/david/codes/Analysis/GC/roediger2014.dat',"r") as f:
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
		with open('/home/david/codes/Analysis/GC/dotter2010.dat',"r") as f:
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
	# ~print(dist_mod)
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


	clus_name2 = ['Arp_2','IC_4499','Lynga_7','NGC_104','NGC_288','NGC_362','NGC_1261','NGC_1851','NGC_2298','NGC_2808','NGC_3201','NGC_4147','NGC_4590','NGC_4833','NGC_5024','NGC_5053','NGC_5139','NGC_5272','NGC_5286','NGC_5466','NGC_5904','NGC_5927','NGC_5986','NGC_6093','NGC_6101','NGC_6121','NGC_6144','NGC_6171','NGC_6205','NGC_6218','NGC_6254','NGC_6304','NGC_6341','NGC_6352','NGC_6362','NGC_6366','NGC_6388','NGC_6397','NGC_6426','NGC_6441','NGC_6496','NGC_6535','NGC_6541','NGC_6584','NGC_6624','NGC_6637','NGC_6652','NGC_6656','NGC_6681','NGC_6715','NGC_6717','NGC_6723','NGC_6752','NGC_6779','NGC_6809','NGC_6838','NGC_6934','NGC_6981','NGC_7006','NGC_7078','NGC_7089','NGC_7099','Pal_1','Pal_12','Pal_15','Pyxis','Rup_106','Ter_7','Ter_8']



	with open('Baumgardt.txt',"r") as f:
		lines=f.readlines()[2:]
	f.close()
	baum_clus=[]
	for x in lines:
		baum_clus.append(x.split(' ')[0])
	
	# find acs initial values in different caltalogs
	# ~index1 = harris_clus.index(clus_nb)

	# ~distance = np.zeros(len(clus_name2))
	# ~errdist = np.zeros(len(clus_name2))

	# ~for nb in range(len(clus_name2)):
	clus_nb2 = clus_name2[nb]

	index2 = baum_clus.index(clus_nb2)
	with open('Baumgardt.txt',"r") as f:
		lines=f.readlines()[2:]
	f.close()
	dist_baum=[]
	errdist_baum=[]
	for x in lines:
		dist_baum.append(x.split(' ')[5])
		errdist_baum.append(x.split(' ')[6])
	distance = dist_baum[index2]
	errdist = errdist_baum[index2]

	return clus_nb, Age, metal, float(distance)*1000., Abs, afe_init, float(errdist)*1000.

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
	
	# ~print(np.min(photo_v), np.min(photo_i))
	# ~print(np.max(photo_v), np.max(photo_i))

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

	
def iso_mag(Age, metal, distance, A, afe_val = None):


	if model == 'mist': 
		
		mag_v, eep_first = mist.mageep['F606W'](Age, metal, distance, A)
		mag_i, eep_first = mist.mageep['F814W'](Age, metal, distance, A)
		#~ mag_v = mist.mageep['F606'](Age, metal, distance, Abs)
		#~ mag_i = mist.mageep['F814'](Age, metal, distance, Abs)

	if model == 'dar': 
		if afe_val == -0.2:
			mag_v, eep_first = darm2.mageep['F606W'](Age, metal, distance, A)
			mag_i, eep_first = darm2.mageep['F814W'](Age, metal, distance, A)
		elif afe_val == 0.0:
			mag_v, eep_first = darp0.mageep['F606W'](Age, metal, distance, A)
			mag_i, eep_first = darp0.mageep['F814W'](Age, metal, distance, A)
		elif afe_val == 0.2:
			mag_v, eep_first = darp2.mageep['F606W'](Age, metal, distance, A)
			mag_i, eep_first = darp2.mageep['F814W'](Age, metal, distance, A)
		elif afe_val == 0.4:
			mag_v, eep_first = darp4.mageep['F606W'](Age, metal, distance, A)
			mag_i, eep_first = darp4.mageep['F814W'](Age, metal, distance, A)
		elif afe_val == 0.6:
			mag_v, eep_first = darp6.mageep['F606W'](Age, metal, distance, A)
			mag_i, eep_first = darp6.mageep['F814W'](Age, metal, distance, A)
		elif afe_val == 0.8:
			mag_v, eep_first = darp8.mageep['F606W'](Age, metal, distance, A)
			mag_i, eep_first = darp8.mageep['F814W'](Age, metal, distance, A)
		
	mag_i = mag_i[~np.isnan(mag_i)]# - Abs
	mag_v = mag_v[~np.isnan(mag_v)]# - Abs
	Color = (mag_v - mag_i)


	#make a magnitude cut
	#~ cut = np.where((mag_v < np.max(photo_v)) & (mag_v > np.min(photo_v)))[0]
	#~ if len(cut)<1:
		#~ print('ohlala problem with')
		#~ print(Age, metal, distance, Abs)
	#~ chosen = np.random.choice(cut, sample)
	
	#~ mag_v = mag_v[chosen]
	#~ mag_i = mag_i[chosen]
	#~ Color = Color[chosen]

	gc.collect()
	return mag_v, mag_i, Color, eep_first


def iso_mag2(mass, Age, metal, distance, A, afe_val = None):


	if model == 'mist': 
		
		mag_v = mist.mag['F606W'](Age, metal, distance, A)
		mag_i = mist.mag['F814W'](Age, metal, distance, A)
		#~ mag_v = mist.mag['F606'](Age, metal, distance, Abs)
		#~ mag_i = mist.mag['F814'](Age, metal, distance, Abs)

	if model == 'dar': 
		if afe_val == -0.2:
			mag_v = darm2.mag['F606W'](mass, Age, metal, distance, A)
			mag_i = darm2.mag['F814W'](mass, Age, metal, distance, A)
		elif afe_val == 0.0:
			mag_v = darp0.mag['F606W'](mass, Age, metal, distance, A)
			mag_i = darp0.mag['F814W'](mass, Age, metal, distance, A)
		elif afe_val == 0.2:
			mag_v = darp2.mag['F606W'](mass, Age, metal, distance, A)
			mag_i = darp2.mag['F814W'](mass, Age, metal, distance, A)
		elif afe_val == 0.4:
			mag_v = darp4.mag['F606W'](mass, Age, metal, distance, A)
			mag_i = darp4.mag['F814W'](mass, Age, metal, distance, A)
		elif afe_val == 0.6:
			mag_v = darp6.mag['F606W'](mass, Age, metal, distance, A)
			mag_i = darp6.mag['F814W'](mass, Age, metal, distance, A)
		elif afe_val == 0.8:
			mag_v = darp8.mag['F606W'](mass, Age, metal, distance, A)
			mag_i = darp8.mag['F814W'](mass, Age, metal, distance, A)
		
	mag_i = mag_i[~np.isnan(mag_i)]# - Abs
	mag_v = mag_v[~np.isnan(mag_v)]# - Abs
	Color = (mag_v - mag_i)


	#make a magnitude cut
	#~ cut = np.where((mag_v < np.max(photo_v)) & (mag_v > np.min(photo_v)))[0]
	#~ if len(cut)<1:
		#~ print('ohlala problem with')
		#~ print(Age, metal, distance, Abs)
	#~ chosen = np.random.choice(cut, sample)
	
	#~ mag_v = mag_v[chosen]
	#~ mag_i = mag_i[chosen]
	#~ Color = Color[chosen]

	gc.collect()
	return mag_v, mag_i, Color





#################################################################################################################
#################################################################################################################
t = 0
base_temp = 4
clus_name = ['arp2','ic4499','lynga7','ngc0104','ngc0288','ngc0362','ngc1261','ngc1851','ngc2298','ngc2808','ngc3201',
'ngc4147','ngc4590','ngc4833','ngc5024','ngc5053','ngc5139','ngc5272','ngc5286','ngc5466','ngc5904','ngc5927','ngc5986',
'ngc6093','ngc6101','ngc6121','ngc6144','ngc6171','ngc6205','ngc6218','ngc6254','ngc6304','ngc6341','ngc6352','ngc6362',
'ngc6366','ngc6388','ngc6397','ngc6426','ngc6441','ngc6496','ngc6535','ngc6541','ngc6584','ngc6624','ngc6637','ngc6652',
'ngc6656','ngc6681','ngc6715','ngc6717','ngc6723','ngc6752','ngc6779','ngc6809','ngc6838','ngc6934','ngc6981','ngc7006',  
'ngc7078','ngc7089','ngc7099','palomar1','palomar12','palomar15','pyxis','ruprecht106','terzan7','terzan8']

version = str(input("which version ? "))
#~ version = '176'
#~ version = '8'
#~ version = '10'
#~ version = 'mixte'

if version == '11': 
	ndim = 4
	nwalkers = 300
	ntemps = 1
	print(ntemps)
	garr = [68]
	model = 'mist'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '12':
	ndim = 4 
	nwalkers = 300
	ntemps = 1
	print(ntemps)
	garr = [14]
	model = 'dar'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '8':
	ndim = 4 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	model = 'mist'
	garr = [0,1,2,3,4,5,13,23,27,33,41,46,64,68]
	#~ garr = [13,23,27,33,41,46,64,68]
if version == '9':
	ndim = 5 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [0,1,2,3,4,5]
	model = 'dar'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '10':
	ndim = 4 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [0,1,2,3,4,5,13,23,27,33,41,46,64,68]
	model = 'mist'
	#~ garr =
if version == '15':
	ndim = 5
	nwalkers = 100
	ntemps = 1
	# ~print(ntemps)
	garr = [3,4,8,12,14,15,17,19,20,24,28,32,34,42,43,46,48,51,52,54,59,61]
	model = 'dar'
	#~ garr =
if version == '16':
	ndim = 5
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [3,4,8,12,14,15,17,19,20,24,28,32,34,42,43,46,48,51,52,54,59,61]
	model = 'dar'
	#~ garr =
if version == '17':
	ndim = 5
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [3,4,8,12,14,15,17,19,20,24,28,32,34,42,43,46,48,51,52,54,59,61]
	model = 'dar'
	#~ garr =
if version == '18':
	ndim = 5
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [3,4,8,12,14,15,17,19,20,24,28,32,34,42,43,46,48,51,52,54,59,61]
	model = 'dar'
	#~ garr =
if version == '0':
	ndim = 4 
	nwalkers = 300	
	version1 = '8' 
	model1 = 'mist'
	version2 = '9' 
	model2 = 'dar'

test = '1310'
	
if version == test:
	ndim = 5
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [3,4,8,12,14,15,17,19,20,24,28,32,34,42,43,46,48,51,52,54,59,61]
	model = 'dar'
	#~ garr =



################################################################################################ Plot the evolution of the chains as a function of the steps
################################################################################################

tot_age = []
tot_met = []

# ~for cn in [0,1,22,53]:
for cn in range(0,69):
# ~for cn in list(range(27))+ list(range(28,69)):
#~ for cn in garr: # 
	glc = cn
	clus_nb, Age0, metal0, distance0, Abs0, afe_init0, errdist  = cluster(glc)
	print(clus_nb, Age0, metal0, distance0, Abs0, afe_init0, errdist)
	photo_v, err_v, photo_i, color, err_color, nmv, nmi, longueur = photometry()
	
	#~ if glc==33 or glc ==36:
		#~ print('popo')
		#~ with open('/home/david/codes/GC/plots/data_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
			#~ fid_file.write('%s, %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (0, 0, 
			#~ 0, 0, 0, 0,0, 0, 0, 0, 0,
			#~ 0, 0))
		#~ fid_file.close()
		#~ continue
	#~ Age = []
	#~ Metal = []
	#~ Distance = []
	#~ AAbs = []
	#~ A = []
	#~ M = []
	#~ D = []
	#~ AA = []
	#~ with open('/home/david/codes/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt', 'r') as file:
		#~ for line in file:
			#~ array = line.split(' ')
			#~ Age.extend(array[0])
			#~ Metal.extend(array[1])
			#~ Distance.extend(array[2])
			#~ AAbs.extend(array[3])
			#~ A.append(array[0:nwalkers])
			#~ M.append(array[nwalkers:2*nwalkers])
			#~ D.append(array[2*nwalkers:3*nwalkers])
			#~ AA.append(array[3*nwalkers:4*nwalkers])
	#~ for j in range(nwalkers):
		#~ A = np.genfromtxt('/home/david/codes/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt', usecols=(j,), max_rows=2000)
		#~ M = np.genfromtxt('/home/david/codes/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt', usecols=(j+nwalkers,), max_rows=2000)
		#~ D = np.genfromtxt('/home/david/codes/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt', usecols=(j+nwalkers*2,), max_rows=2000)
		#~ AA = np.genfromtxt('/home/david/codes/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt', usecols=(j+nwalkers*3,), max_rows=2000)

	steps = 0
	if version in ['9','10','15','16','17','18',test]:
		files = np.loadtxt('/home/david/codes/data/GC_data/'+str(model)+'/data_1'+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt')
	else:
		files = np.loadtxt('/home/david/codes/Analysis/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt')

	Age = files[steps*nwalkers:,0]
	Metal = files[steps*nwalkers:,1]
	Distance = files[steps*nwalkers:,2]
	AAbs = files[steps*nwalkers:,3]
	# ~Age = files[steps*nwalkers:,0][prior]
	# ~Metal = files[steps*nwalkers:,1][prior]
	# ~Distance = files[steps*nwalkers:,2][prior]
	# ~AAbs = files[steps*nwalkers:,3][prior]
	if model == 'dar':
		Afe = files[steps*nwalkers:,4]
		# ~Afe = files[steps*nwalkers:,4][prior]

		helium_y = ''
		from isochrones.dartmouth import Dartmouth_FastIsochrone
		darm2 = Dartmouth_FastIsochrone(afe='afem2', y=helium_y)
		darp0 = Dartmouth_FastIsochrone(afe='afep0', y=helium_y)
		darp2 = Dartmouth_FastIsochrone(afe='afep2', y=helium_y)
		darp4 = Dartmouth_FastIsochrone(afe='afep4', y=helium_y)
		darp6 = Dartmouth_FastIsochrone(afe='afep6', y=helium_y)
		darp8 = Dartmouth_FastIsochrone(afe='afep8', y=helium_y)

		
	#~ files2 = np.loadtxt('/home/david/codes/Analysis/GC/plots/test/'+str(model)+'/data_2'+'_'+clus_nb+'_'+ version +'_'+str(model)+'.txt')
	#~ Age2 = files2[steps*nwalkers:,0]
	#~ Metal2 = files2[steps*nwalkers:,1]
	#~ Distance2 = files2[steps*nwalkers:,2]
	#~ AAbs2 = files2[steps*nwalkers:,3]
	#~ Age = files[:,0]
	#~ Metal = files[:,1]
	#~ Distance = files[:,2]
	#~ AAbs = files[:,3]

	# ~print(len(Age))
	step_walk = 20
	plt.suptitle('numero '+str(glc)+', '+clus_nb)
	ax1 = plt.subplot(231)
	ax1.set_title('Age')
	for i in range(0,int(len(Age)/nwalkers),step_walk):
		# ~ax1.plot(np.full(nwalkers, i), Age[i*nwalkers:(i+1)*nwalkers], c='k')
		ax1.scatter(i, np.median(Age[i*nwalkers:(i+1)*nwalkers]), c='c')
		ax1.errorbar(i, np.mean(Age[i*nwalkers:(i+1)*nwalkers]), yerr=np.std(Age[i*nwalkers:(i+1)*nwalkers]),fmt = '.', c='r', ecolor='k')
	ax1.axhline(Age0, color='r', linestyle='--')
	ax1.axhline(10.176, color='c')
	ax1.grid()
	ax2 = plt.subplot(232)
	ax2.set_title('metal')
	for i in range(0,int(len(Age)/nwalkers),step_walk):
		# ~ax2.plot(np.full(nwalkers, i), Metal[i*nwalkers:(i+1)*nwalkers], c='k')
		ax2.scatter(i, np.median(Metal[i*nwalkers:(i+1)*nwalkers]), c='c')
		ax2.errorbar(i, np.mean(Metal[i*nwalkers:(i+1)*nwalkers]), yerr=np.std(Metal[i*nwalkers:(i+1)*nwalkers]),fmt = '.', c='r', ecolor='k')
	ax2.axhline(metal0, color='r', linestyle='--')
	#ax2.set_ylim(-1.6, -1.2)
	ax2.grid()
	ax3 = plt.subplot(233)
	ax3.set_title('distance')
	for i in range(0,int(len(Age)/nwalkers),step_walk):
		# ~ax3.plot(np.full(nwalkers, i), Distance[i*nwalkers:(i+1)*nwalkers], c='k')
		ax3.scatter(i, np.median(Distance[i*nwalkers:(i+1)*nwalkers]), c='c')
		ax3.errorbar(i, np.mean(Distance[i*nwalkers:(i+1)*nwalkers]), yerr=np.std(Distance[i*nwalkers:(i+1)*nwalkers]),fmt = '.', c='r', ecolor='k')
	ax3.axhline(distance0, color='r', linestyle='--')
	#ax3.set_ylim(32500, 35000)
	ax3.grid()
	ax4 = plt.subplot(234)
	ax4.set_title('A1')
	for i in range(0,int(len(Age)/nwalkers),step_walk):
		# ~ax4.plot(np.full(nwalkers, i), AAbs[i*nwalkers:(i+1)*nwalkers], c='k')
		ax4.scatter(i, np.median(AAbs[i*nwalkers:(i+1)*nwalkers]), c='c')
		ax4.errorbar(i, np.mean(AAbs[i*nwalkers:(i+1)*nwalkers]), yerr=np.std(AAbs[i*nwalkers:(i+1)*nwalkers]),fmt = '.', c='r', ecolor='k')
	ax4.axhline(Abs0, color='r', linestyle='--')
	#ax4.set_ylim(0.30, 0.36)
	ax4.grid()
	#plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/chains'+'_'+clus_nb+'_'+ version +'_'+str(model)+'.png')
	ax5 = plt.subplot(235)
	ax5.set_title(r'$\alpha$')
	for i in range(0,int(len(Age)/nwalkers),step_walk):
		# ~ax5.plot(np.full(nwalkers, i), Afe[i*nwalkers:(i+1)*nwalkers], c='k')
		ax5.scatter(i, np.median(Afe[i*nwalkers:(i+1)*nwalkers]), c='c')
		ax5.errorbar(i, np.mean(Afe[i*nwalkers:(i+1)*nwalkers]), yerr=np.std(Afe[i*nwalkers:(i+1)*nwalkers]),fmt = '.', c='r', ecolor='k')
	ax5.axhline(afe_init0, color='r', linestyle='--')
	#ax5.set_ylim(0.30, 0.36)
	ax5.grid()
	#plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/chains'+'_'+clus_nb+'_'+ version +'_'+str(model)+'.png')
	plt.show()
	# ~plt.close()

	#~ step_min = 20
	#~ if len(age) > step_min:
	#~ if 200 > step_min:
		#~ Age = []
		#~ Metal = []
		#~ Distance = []
		#~ AAbs = []

		#~ t = 0
		#~ sf = [1, 1, 1, 1, 1, 1]
		#~ for j in range(nwalkers):
			#~ Age.extend(np.genfromtxt('/home/david/codes/Analysis/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ 
			#~ version +'_'+str(model)+'.txt',skip_header = step_min, usecols=(j,), max_rows=2000))
			#~ Metal.extend(np.genfromtxt('/home/david/codes/Analysis/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version 
			#~ +'_'+str(model)+'.txt',skip_header = step_min, usecols=(nwalkers+j,), max_rows=2000))
			#~ Distance.extend(np.genfromtxt('/home/david/codes/Analysis/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version 
			#~ +'_'+str(model)+'.txt',skip_header = step_min, usecols=(2*nwalkers + j,), max_rows=2000))
			#~ AAbs.extend(np.genfromtxt('/home/david/codes/Analysis/GC/plots/test/data_'+str(t)+'_'+clus_nb+'_'+ version 
			#~ +'_'+str(model)+'.txt',skip_header = step_min, usecols=(3*nwalkers + j,), max_rows=2000))
		#~ bins = 50
		#~ ax = plt.figure()
		#~ ax.set_facecolor('c')
		#~ plt.suptitle('numero '+str(glc)+', '+clus_nb)
		#~ ax1 = plt.subplot(221)
		#~ ax1.set_title('Age')
		#~ ax1.hist(Age, bins=bins)
		#~ ax2 = plt.subplot(222)
		#~ ax2.set_title('metal')
		#~ ax2.hist(Metal, bins = bins)
		#~ ax3 = plt.subplot(223)
		#~ ax3.set_title('distance')
		#~ ax3.hist(Distance, bins = bins)
		#~ ax4 = plt.subplot(224)
		#~ ax4.set_title('A1')
		#~ ax4.hist(AAbs, bins = bins)
		#~ plt.show()
	#~ Age3 = files[steps*nwalkers:,0]
	#~ Metal3 = files[steps*nwalkers:,1]
	#~ Distance3 = files[steps*nwalkers:,2]
	#~ AAbs3 = files[steps*nwalkers:,3]

################################################################################################### remove the burn in phase for the analysis
###################################################################################################

	# ~steps = int(input("What is the convergence step ? "))
	# ~if steps == 0:
		# ~pass
	# ~else:
	slim = 2000
	steps = slim
	if steps < slim:
		pass
	else:
		#~ print(np.mean(files[:,3]), np.median(files[:,3]))
		prior = np.where((files[steps*nwalkers:,2] > distance0 - errdist)&(files[steps*nwalkers:,2] < distance0 + errdist))[0]
		#~ files[steps*nwalkers:,1] < metal0 +0.2)&(files[steps*nwalkers:,2] > metal0 -0.2))[0]


		Age = files[steps*nwalkers:,0]
		Metal = files[steps*nwalkers:,1]
		Distance = files[steps*nwalkers:,2]
		AAbs = files[steps*nwalkers:,3]
		# ~Age = files[steps*nwalkers:,0][prior]
		# ~Metal = files[steps*nwalkers:,1][prior]
		# ~Distance = files[steps*nwalkers:,2][prior]
		# ~AAbs = files[steps*nwalkers:,3][prior]
		if model == 'dar':
			Afe = files[steps*nwalkers:,4]
			# ~Afe = files[steps*nwalkers:,4][prior]

	#--------------------------------------------------------------
	### MAIN SEQUENCE

		if model == 'dar':
			taille = min(len(list((filter(None, Age)))),len(list((filter(None, Metal)))),len(list((filter(None, Distance))))
			,len(list((filter(None, AAbs)))),len(list((filter(None, Afe)))))
		else:
			taille = min(len(list((filter(None, Age)))),len(list((filter(None, Metal)))),len(list((filter(None, Distance))))
			,len(list((filter(None, AAbs)))))
			
			
		if len(Age) == 0:
			print('popo')
			#~ with open('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
				#~ fid_file.write('%s, %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (0, 0, 
				#~ 0, 0, 0, 0,0, 0, 0, 0, 0,
				#~ 0, 0))
			#~ fid_file.close()
			continue
			
		
		data = np.zeros((taille, ndim))
		for i in range(taille):
			data[i, 0] = 10**Age[i] /1.e9
			data[i, 1] = Metal[i] 
			data[i, 2] = Distance[i] /1000.
			data[i, 3] = AAbs[i] 
			if model == 'dar':
				data[i, 4] = Afe[i]

		# ~b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc, b5_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(data, [16, 50, 84], axis=0)))
		# ~print(b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc, b5_mcmc)
	

		# ~afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8]

		# ~afe_max = afe_values[np.searchsorted(afe_values, b5_mcmc[0])]
		# ~afe_min = afe_values[np.searchsorted(afe_values, b5_mcmc[0])-1]
		

		# ~mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(np.log10(b1_mcmc[0]*1e9), b2_mcmc[0], b3_mcmc[0]*1000.0, b4_mcmc[0], afe_min)
		# ~mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(np.log10(b1_mcmc[0]*1e9), b2_mcmc[0], b3_mcmc[0]*1000.0, b4_mcmc[0], afe_max)
		# ~lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate

		# ~mag_v3 = (mag_v1_min[:lpp]*(afe_max - b5_mcmc[0]) + mag_v1_max[:lpp]*(b5_mcmc[0] - afe_min)) / (afe_max - afe_min)
		# ~Color_iso3 = (Color_iso1_min[:lpp]*(afe_max - b5_mcmc[0]) + Color_iso1_max[:lpp]*(b5_mcmc[0] - afe_min)) / (afe_max - afe_min)

		# ~plt.figure()
		# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
		# ~plt.plot(Color_iso3,mag_v3, c='b',  label='main sequence')
		# ~plt.xlim(-0.5,3)
		# ~plt.ylim(26,10)
		# ~plt.legend(loc='upper right', fontsize = 16)
		# ~plt.xlabel('F606W - F814W', fontsize = 16)
		# ~plt.ylabel('F606W', fontsize = 16)
		# ~plt.title(clus_nb, fontsize = 16)
		# ~plt.show()
		# ~plt.close()

		gc.collect()

	#--------------------------------------------------------------------
	### Red Giant BRANCH
		#~ taille2 = min(len(filter(None, Age2)),len(filter(None, Metal2)),len(filter(None, Distance2)),len(filter(None, AAbs2)))
		
		#~ if len(Age2) == 0:
			#~ print('popo')
			#~ with open('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
				#~ fid_file.write('%s, %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (0, 0, 
				#~ 0, 0, 0, 0,0, 0, 0, 0, 0,
				#~ 0, 0))
			#~ fid_file.close()
			#~ continue
			
		#~ print(len(Age2)/nwalkers)	
		
		#~ data2 = np.zeros((taille2, ndim))
		#~ for i in range(taille2):
			#~ data2[i, 0] = Age2[i] 
			#~ data2[i, 1] = Metal2[i] 
			#~ data2[i, 2] = Distance2[i] 
			#~ data2[i, 3] = AAbs2[i] 

	#--------------------------------------------------------------------
	### TOTAL

		#~ Age3 = np.concatenate((Age, Age2))
		#~ Metal3 = np.concatenate((Metal, Metal2))
		#~ Distance3 = np.concatenate((Distance, Distance2))
		#~ AAbs3 = np.concatenate((AAbs, AAbs2))
		
		#~ taille3 = min(len(filter(None, Age3)),len(filter(None, Metal3)),len(filter(None, Distance3)),len(filter(None, AAbs3)))

		
		#~ if len(Age3) == 0:
			#~ print('popo')
			#~ with open('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
				#~ fid_file.write('%s, %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (0, 0, 
				#~ 0, 0, 0, 0,0, 0, 0, 0, 0,
				#~ 0, 0))
			#~ fid_file.close()
			#~ continue
			
		#~ print(len(Age3)/nwalkers)	
		
		#~ data3 = np.zeros((taille3, ndim))
		#~ for i in range(taille3):
			#~ data3[i, 0] = Age3[i] 
			#~ data3[i, 1] = Metal3[i] 
			#~ data3[i, 2] = Distance3[i] 
			#~ data3[i, 3] = AAbs3[i] 
	#--------------------------------------------------------------------
		print('cluster numero '+ str(glc))
		print('number of steps = '+ str(len(Age)/nwalkers))

		binage = np.linspace(5,15,200)
		h1, bh1 = np.histogram(10**Age / 1.e9, bins=binage)
		h2, bh2 = np.histogram(10**Age / 1.e9, bins=binage, density=True)
		binfe = np.linspace(-2.5,0,200)
		m1, bm1 = np.histogram(Metal, bins=binfe)
		m2, bm2 = np.histogram(Metal, bins=binfe, density=True)
		binafe = np.linspace(-0.2,0.8,100)
		a1, ba1 = np.histogram(Afe, bins=binafe)
		a2, ba2 = np.histogram(Afe, bins=binafe, density=True)
		h1 = np.transpose([h1]).T
		h2 = np.transpose([h2]).T
		a1 = np.transpose([a1]).T
		a2 = np.transpose([a2]).T
		m1 = np.transpose([m1]).T
		m2 = np.transpose([m2]).T
		
		Metal_mean = (bm1[np.argmax(m1)] + bm1[np.argmax(m1)+1])/2.
		
		with open('/home/david/codes/Analysis/GC/plots/w_tot', 'ab+') as fid_file:
			np.savetxt(fid_file, h1)
		fid_file.close()	
		with open('/home/david/codes/Analysis/GC/plots/renom_tot', 'ab+') as fid_file:
			np.savetxt(fid_file, h2)
		fid_file.close()
		with open('/home/david/codes/Analysis/GC/plots/a_tot', 'ab+') as fid_file:
			np.savetxt(fid_file, a1)
		fid_file.close()	
		with open('/home/david/codes/Analysis/GC/plots/a_renom', 'ab+') as fid_file:
			np.savetxt(fid_file, a2)
		fid_file.close()
		with open('/home/david/codes/Analysis/GC/plots/m_tot', 'ab+') as fid_file:
			np.savetxt(fid_file, m1)
		fid_file.close()	
		with open('/home/david/codes/Analysis/GC/plots/m_renom', 'ab+') as fid_file:
			np.savetxt(fid_file, m2)
		fid_file.close()
		
		
		if Metal_mean <= -1.5:
			with open('/home/david/codes/Analysis/GC/plots/w_1', 'ab+') as fid_file:
				np.savetxt(fid_file, h1)
			fid_file.close()	
			with open('/home/david/codes/Analysis/GC/plots/renom_1', 'ab+') as fid_file:
				np.savetxt(fid_file, h2)
			fid_file.close()
			with open('/home/david/codes/Analysis/GC_mixing_length/ind_met15.txt', 'a+') as fid_file:
				fid_file.write(str(glc)+"\n")
			fid_file.close()
			
		if Metal_mean <= -2:
			with open('/home/david/codes/Analysis/GC/plots/w_2', 'ab+') as fid_file:
				np.savetxt(fid_file, h1)
			fid_file.close()	
			with open('/home/david/codes/Analysis/GC/plots/renom_2', 'ab+') as fid_file:
				np.savetxt(fid_file, h2)
			fid_file.close()	
			with open('/home/david/codes/Analysis/GC_mixing_length/int_met20.txt', 'a+') as fid_file:
				fid_file.write(str(glc)+"\n")
			fid_file.close()	
			

		nbins = 50

		binage = np.linspace(np.min(10**Age / 1.e9),np.max(10**Age / 1.e9),nbins)
		dage = np.diff(binage)[0]
		bincenter = (binage[:-1] + binage[1:]) / 2
		h1, bh1 = np.histogram(10**Age / 1.e9, bins=binage)


		binfe = np.linspace(np.min(Metal),np.max(Metal),nbins)
		dfe = np.diff(binfe)[0]
		bfcenter = (binfe[:-1] + binfe[1:]) / 2
		m1, bm1 = np.histogram(Metal, bins=binfe)

		bindis = np.linspace(np.min(Distance),np.max(Distance),nbins)
		ddis = np.diff(bindis)[0]
		bdcenter = (bindis[:-1] + bindis[1:]) / 2
		d1, bd1 = np.histogram(Distance, bins=bindis)

		
		binabs = np.linspace(np.min(AAbs),np.max(AAbs),nbins)
		dabs = np.diff(binabs)[0]
		bacenter = (binabs[:-1] + binabs[1:]) / 2
		ab1, bab1 = np.histogram(AAbs, bins=binabs)


		binafe = np.linspace(np.min(Afe),np.max(Afe),nbins)
		dafe = np.diff(binafe)[0]
		bcenter = (binafe[:-1] + binafe[1:]) / 2
		a1, ba1 = np.histogram(Afe, bins=binafe)
		
		
		def error_compute(dbins, histo, bhisto):
			amp = 1.0
			while amp > 0.0:
				integ = np.sum(dbins*histo)
				above = np.where(histo > amp*np.max(histo))[0]
				#~ print(above)
				tinteg = np.sum(dbins*histo[above])
				s = tinteg/integ
				#~ print('integral percentage is '+str(s))
				if s > 0.68:
					#~ print([np.min(above)])
					#~ print([np.max(above)])
					return bhisto[np.min(above)], bhisto[np.max(above)]
					break
				amp -= 0.01
				#~ print('percentage of the amplitude is '+str(amp))
		
		
		Age_low, Age_high = error_compute(dage, h1,bincenter)
		Metal_low, Metal_high = error_compute(dfe, m1,bfcenter)
		Distance_low, Distance_high = error_compute(ddis, d1,bdcenter)
		AAbs_low, AAbs_high = error_compute(dabs, ab1,bacenter)
		Afe_low, Afe_high = error_compute(dafe, a1,bcenter)


		Age_mean = (bh1[np.argmax(h1)] + bh1[np.argmax(h1)+1])/2.
		Metal_mean = (bm1[np.argmax(m1)] + bm1[np.argmax(m1)+1])/2.
		Distance_mean = (bd1[np.argmax(d1)] + bd1[np.argmax(d1)+1])/2.
		AAbs_mean = (bab1[np.argmax(ab1)] + bab1[np.argmax(ab1)+1])/2.
		Afe_mean = (ba1[np.argmax(a1)] + ba1[np.argmax(a1)+1])/2.

		
		#~ plt.hist(10**Age / 1.e9, bins=binage)
		#~ plt.axvline(Age_low, c='r')
		#~ plt.axvline(Age_high,c='g')
		#~ plt.axvline(Age_mean,c='k')
		#~ plt.axvline(10**np.percentile(Age,50)/1.e9,c='k', linestyle='--')
		#~ plt.show()
		#~ plt.close()
		#~ plt.hist(Metal, bins=binfe)
		#~ plt.axvline(Metal_low, c='r')
		#~ plt.axvline(Metal_high,c='g')
		#~ plt.axvline(Metal_mean,c='k')
		#~ plt.axvline(metal0,c='b')
		#~ plt.axvline(np.percentile(Metal,50),c='k', linestyle='--')
		#~ plt.show()
		#~ plt.close()
		#~ plt.hist(Distance, bins=bindis)
		#~ plt.axvline(Distance_low, c='r')
		#~ plt.axvline(Distance_high,c='g')
		#~ plt.axvline(Distance_mean,c='k')
		#~ plt.axvline(np.percentile(Distance,50),c='k', linestyle='--')
		#~ plt.xlim(np.min(Distance), np.max(Distance))
		#~ plt.show()
		#~ plt.close()
		#~ plt.hist(AAbs, bins=binabs)
		#~ plt.axvline(AAbs_low, c='r')
		#~ plt.axvline(AAbs_high,c='g')
		#~ plt.axvline(AAbs_mean,c='k')
		#~ plt.axvline(np.percentile(AAbs,50),c='k', linestyle='--')
		#~ plt.xlim(np.min(AAbs), np.max(AAbs))
		#~ plt.show()
		#~ plt.close()
		#~ plt.hist(Afe, bins=binafe)
		#~ plt.axvline(Afe_low, c='r')
		#~ plt.axvline(Afe_high,c='g')
		#~ plt.axvline(Afe_mean,c='k')
		#~ plt.axvline(np.percentile(Afe,50),c='k', linestyle='--')
		#~ plt.show()
		#~ plt.close()
		
		gc.collect()

			
		#~ kill

		if model == 'dar':
			fig = corner.corner(data,bins=50, range=[(np.min(data[:,0]),15.), (np.min(data[:,1]),np.max(data[:,1])),
			(np.min(data[:,2]),np.max(data[:,2])),(np.min(data[:,3]),np.max(data[:,3])),(np.min(data[:,4]),np.max(data[:,4]))],
			labels=["$Age$ [Gyr]", "$metallicity$", "$distance$ [kpc]", "$Absorption$", r"[$\alpha$/fe]"]
			, hist_kwargs={'fill':'True',"edgecolor":'k',"linewidth":"1.2"},
			plot_contours=True, label_kwargs={"fontsize":10}, color ='lightblue', plot_datapoints=False,
			levels=(1-np.exp(-0.5),0.6321,0.7769))
			plt.subplots_adjust(hspace=0.2, wspace=0.2, top = 0.95, left = 0.1, right=0.95)
			for ax in fig.get_axes():
				ax.tick_params(axis='both', labelsize=10)
			plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/corner'+'_'+clus_nb+'_'+ version +'_'+str(model)+'.png')
			# ~plt.show()
			plt.close()
		else:
			fig = corner.corner(data, range=[(np.min(data[:,0]),15.), (np.min(data[:,1]),np.max(data[:,1])),
			(np.min(data[:,2]),np.max(data[:,2])),(np.min(data[:,3]),np.max(data[:,3]))],
			labels=["$Age$ [Gyr]", "$metallicity$", "$distance$ [kpc]", "$Absorption$"]
			, hist_kwargs={'fill':'True',"edgecolor":'k',"linewidth":"2"}, labelpad = 50,
			plot_contours=True, label_kwargs={"fontsize":16}, color ='b', plot_datapoints=False,
			levels=(1-np.exp(-0.5),0.6321,0.7769))
			plt.subplots_adjust(hspace=0.2, wspace=0.2, top = 0.95, left = 0.1, right=0.95)
			for ax in fig.get_axes():
				ax.tick_params(axis='both', labelsize=12)
			# ~plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/corner'+'_'+clus_nb+'_'+ version +'_'+str(model)+'.png')
			# ~plt.show()
			plt.close()

		# ~kill
		#~ print(Age_low, Age_mean, Age_high, Metal_low, Metal_mean, Metal_high, Distance_low, Distance_mean, Distance_high, AAbs_low,
				#~ AAbs_mean, AAbs_high, Afe_low, Afe_mean, Afe_high)


		if model == 'dar':
			with open('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
				fid_file.write('%s %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (clus_nb, Age_low, 
				Age_mean, Age_high, Metal_low, Metal_mean, Metal_high, Distance_low, Distance_mean, Distance_high, AAbs_low,
				AAbs_mean, AAbs_high, Afe_low, Afe_mean, Afe_high))
			fid_file.close()
			with open('/home/david/codes/Analysis/GC/plots/table_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
				fid_file.write('%s & $%.2f^{+%.2f}_{%.2f}$ & $%.2f^{+%.2f}_{%.2f}$ & $%.2f^{+%.2f}_{%.2f}$ & $%.2f^{+%.2f}_{%.2f}$ & $%.2f^{+%.2f}_{%.2f}$\n' % (clus_nb, 
				Age_mean, (Age_high-Age_mean), (Age_low-Age_mean), Metal_mean, Metal_high-Metal_mean,
				Metal_low-Metal_mean,  Distance_mean/ 1000., (Distance_high-Distance_mean)/1000., (Distance_low-Distance_mean)/1000., 
				AAbs_mean, AAbs_high-AAbs_mean, AAbs_low-AAbs_mean, Afe_mean, Afe_high-Afe_mean, Afe_low-Afe_mean))
			fid_file.close()
		# ~else:
			# ~with open('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
				# ~fid_file.write('%s %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n' % (clus_nb, Age_low, 
				# ~Age_mean, Age_high, Metal_low, Metal_mean, Metal_high, Distance_low, Distance_mean, Distance_high, AAbs_low,
				# ~AAbs_mean, AAbs_high))
			# ~fid_file.close()
		

			#~ age_fin[cn] = Age_mean
			#~ pluserr_fin[cn] = Age_high
			#~ minuserr_fin[cn] = Age_high
			#~ metal_fin[cn] = Metal_mean
			#~ distance_fin[cn] = Distance_mean
			#~ Abs_fin[cn] = AAbs_mean
			#~ elem_fin = np.arange(len(age_fin))


			#~ def hide_current_axis(*args, **kwds):
				#~ plt.gca().set_visible(False)
				


		# ~helium_y = ''
		# ~from isochrones.dartmouth import Dartmouth_FastIsochrone
		# ~darm2 = Dartmouth_FastIsochrone(afe='afem2', y=helium_y)
		# ~darp0 = Dartmouth_FastIsochrone(afe='afep0', y=helium_y)
		# ~darp2 = Dartmouth_FastIsochrone(afe='afep2', y=helium_y)
		# ~darp4 = Dartmouth_FastIsochrone(afe='afep4', y=helium_y)
		# ~darp6 = Dartmouth_FastIsochrone(afe='afep6', y=helium_y)
		# ~darp8 = Dartmouth_FastIsochrone(afe='afep8', y=helium_y)
		### create a sample from best fit
		afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8] 

		afe_max = afe_values[np.searchsorted(afe_values, Afe_mean)]
		afe_min = afe_values[np.searchsorted(afe_values, Afe_mean)-1]
		# ~print(afe_min)
		# ~print(afe_max)

		print(Age_mean, Metal_mean, Distance_mean, AAbs_mean, Afe_mean)
		# ~print(np.log10(Age_mean*1.e9), Metal_mean, Distance_mean, AAbs_mean, afe_max)

		mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(np.log10(Age_mean*1.e9), Metal_mean, Distance_mean, AAbs_mean, afe_min)
		mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(np.log10(Age_mean*1.e9), Metal_mean, Distance_mean, AAbs_mean, afe_max)
		#mag_v1_min , mag_i1_min, Color_iso1_min = iso_mag2(mass,np.log10(Age_mean*1.e9), Metal_mean, Distance_mean, AAbs_mean, afe_min)
		#mag_v1_max , mag_i1_max, Color_iso1_max = iso_mag2(mass,np.log10(Age_mean*1.e9), Metal_mean, Distance_mean, AAbs_mean, afe_max)
		lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
		
		mag_v1 = (mag_v1_min[:lpp]*(afe_max - Afe_mean) + mag_v1_max[:lpp]*(Afe_mean - afe_min)) / (afe_max - afe_min)
		Color_iso1 = (Color_iso1_min[:lpp]*(afe_max - Afe_mean) + Color_iso1_max[:lpp]*(Afe_mean - afe_min)) / (afe_max - afe_min)


		#interv = np.linspace(np.min(mag_v1), np.max(photo_v),len(color) )
		#interv = np.linspace(np.min(mag_v1), np.max(photo_v),2000)
		#fmag = interpolate.interp1d(mag_v1, Color_iso1, 'nearest',fill_value="extrapolate")
		#fmag = interpolate.interp1d(mag_v1, Color_iso1, 'cubic')
		#xinterp = fmag(interv)


		# ~xinterp = np.interp(interv, mag_v1, Color_iso1)
		# ~print(xinterp)


		plt.figure()
		plt.scatter(color,photo_v, marker='.',s=10, color='grey', alpha=0.5,label='stars')
		# sc =plt.scatter(Color_iso2,mag_v2, marker='.',s=10,c='r',  label='best fit')
		sc =plt.scatter(Color_iso1,mag_v1, marker='.',s=10,c='r',  label='best fit')
		# sc =plt.scatter(Color_iso1 + noise,mag_v1, marker='.',s=30,c='b',  label='best fit')
		# sc =plt.scatter(xinterp+noise, interv, marker='.',s=30,c='b',  label='best fit')
		plt.xlim(-0.5,3)
		plt.ylim(25.75,10)
		# ~lgnd = plt.legend(loc='upper left', fontsize = 16)
		# ~lgnd.legendHandles[0]._sizes = [286]
		# ~lgnd.legendHandles[1]._sizes = [286]
		# ~plt.xlabel('F606W - F814W', fontsize = 16)
		# ~plt.ylabel('F606W', fontsize = 16)
		plt.tick_params(labelsize=16)
		# ~plt.title(clus_nb, fontsize = 16)
		plt.title(clus_nb+' age = '+str(Age_mean))
		# ~plt.title('IC4499', fontsize = 16)
		# ~plt.subplots_adjust(bottom=0.13)
		plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/fit_'+clus_nb+'_'+ version +'.png')
		# ~plt.show()
		plt.close()
		# ~kill

		gc.collect()
