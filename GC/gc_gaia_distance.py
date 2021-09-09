import numpy as np
import scipy.optimize as op
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import statistics
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
#~ @profile

def gauss(x,amp, mean, stddev):
	return amp*np.exp(-((x - mean) / 4 / stddev)**2)
def twogauss(x,amp1, mean1, stddev1,amp2, mean2, stddev2):
	return amp1*np.exp(-((x - mean1) / 4 / stddev1)**2) +amp2*np.exp(-((x - mean2) / 4 / stddev2)**2)
#~ def threegauss(x,amp1, mean1, stddev1,amp2, mean2, stddev2,amp3, mean3, stddev3):
	#~ return amp1*np.exp(-((x - mean1) / 4 / stddev1)**2) +amp2*np.exp(-((x - mean2) / 4 / stddev2)**2) \
	#~ +amp3*np.exp(-((x - mean3) / 4 / stddev3)**2)
def spread_color_x():


	lbin = len(bincenter)
	
	gauss_meanMS = np.zeros(lbin)
	gauss_dispMS = np.zeros(lbin)
	binmid = np.zeros(lbin)
	starnum = np.zeros(lbin)


	
	for ij in range(0,lbin):
		#~ print(bincenter[ij], top_y)

		inbin = np.digitize(photo_v, binMS)
		#~ inbin2 = np.digitize(ep_mag, binMS)
		ici = np.where(inbin == ij+1)[0]
		#~ ici2 = np.where(inbin2 == ij+1)[0]

		#~ gopini = []
		#~ nelem = 1000
		#~ for s in range(len(rgood)):
			#~ gopini.extend(np.random.normal(rgood[s], errgood[s], nelem))
		gopini = np.array(color)[ici]
	

		bincol = np.linspace(-0.5, 3, 700)
		#~ gop, bb = np.histogram(gopini, bins=bincol, weights=1./np.array(err_color)[ici])
		gop, bb = np.histogram(gopini, bins=bincol)
		#~ gop, bb = np.histogram(gopini, bins=100)
		x = (bb[:-1] + bb[1:]) / 2
		
		from scipy.signal import find_peaks
		#~ peaks, prop = find_peaks(gop, height=50, prominence = 200)
		#~ peaks, prop = find_peaks(gop, height=np.max(gop)*0.2, prominence = np.max(gop)*0.2)
		peaks, prop = find_peaks(gop, height=np.max(gop)*0.2, distance=10)
		#~ print(prop)
		#~ print(peaks.size)

		#~ print(len(ici))
		#~ if peaks.size == 1:
		popt,pcov = curve_fit(gauss,x,gop, p0 = [np.max(gop),x[np.argmax(gop)], np.std(err_color[ici])], maxfev = 950000)
		v0, v1, v2 = popt
		#~ else:
			#~ popt,pcov = curve_fit(twogauss,x,gop, p0 = [np.max(gop),x[np.argmax(gop)], np.std(err_color[ici]), 
			#~ np.max(gop)/2.,x[np.argmax(gop)]-0.1, np.std(err_color[ici])/2.], maxfev = 950000)
			#~ v0, v1, v2, v3, v4, v5 = popt

		#~ from scipy.stats import skew
		#~ print(np.std(gopini), v2, v2 + (v1-np.median(gopini)), v2*0.8)
		### compute the sine of the inclinationto rescale the error
		tp = np.where((np.array(photo_v)[ici] >= binMS[ij]) & (np.array(photo_v)[ici] <= binMS[ij]+step/4.))[0]
		tps = np.median(np.array(color)[ici][tp])
		#~ print(tps)
		bp = np.where((np.array(photo_v)[ici] <= binMS[ij+1]) & (np.array(photo_v)[ici] >= binMS[ij+1]-step/4.))[0]
		bps = np.median(np.array(color)[ici][bp])
		#~ print(bps)
		
		#~ vector_1 = [0, 1]
		vector_1 = [0,step]
		vector_2 = [bps-tps, step]
		unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
		unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
		dot_product = np.dot(unit_vector_1, unit_vector_2) # this is the cosine
		angle = np.arccos(dot_product)
		sine  = np.sqrt(1 - dot_product**2)
		#~ print((angle), dot_product)

		#~ kill
		starnum[ij] = len(ici)
		gauss_meanMS[ij] = v1
		gauss_dispMS[ij] = v2 * dot_product
		#~ gauss_dispMS[ij] = v2* dot_product /np.sqrt(len(ici))
		binmid[ij] = np.median(np.array(photo_v)[ici])

		#~ print(v1,v2,dot_product, sine,angle, np.degrees(angle))
		
		#plt.scatter(color, photo_v, label='data', c='k')
		#~ plt.scatter(color[ici], photo_v[ici], label='stars', c='grey')
		#~ plt.axvline(v1, c='k', label='abscissa of central value', lw=2)
		#~ plt.plot([tps,bps], [binMS[ij], binMS[ij+1]], c='r', label='orientation', lw=2)
		#~ # plt.scatter(ep_col[ici2], ep_mag[ici2], label='EEP',c='b')
		#~ # plt.plot(ep_col[ici2], slope*ep_col[ici2] + intercept, label='fit', c='k')
		#~ # plt.errorbar(abscisse, bincenter[ij], xerr=v2, c='k')
		#~ #plt.scatter(v1, bincenter[ij], label='bin center', c='r')
		#~ plt.xlabel('F606W - F814W', fontsize = 24)
		#~ plt.ylabel('F606W', fontsize = 24)
		#~ plt.title('IC4499', fontsize = 24)
		#~ plt.axhline(binMS[ij], label='bin edge', alpha=0.5)
		#~ plt.axhline(binMS[ij+1], alpha=0.5)
		#~ # for ji in binMS:
			#~ # plt.axhline(ji)
		#~ plt.ylim(binMS[ij+1]+0.02,binMS[ij]-0.02)
		#~ lgnd = plt.legend(loc='upper right', fontsize = 18)
		#~ lgnd.legendHandles[3]._sizes = [286]
		#~ lgnd.legendHandles[2]._sizes = [286]
		#~ lgnd.get_frame().set_edgecolor('k')
		#~ lgnd.get_frame().set_linewidth(2.0)
		#~ plt.tick_params(labelsize=16)
		#~ plt.show()
		#~ plt.close()
		#~ kill

		#~ plt.figure()
		#~ plt.hist(gopini, bins=bincol, weights=1./np.array(err_color)[ici])
		#~ plt.hist(gopini, bins=bincol, edgecolor='k')
		#~ plt.plot(x,gop,label='envelop', c='b')
		#~ #plt.plot(x,twogauss(x,v0, v1, v2, v3, v4, v5),label='fit', c='g')
		#~ plt.axvline(v1, c='r', label= r'$C_i^{data}$', linewidth=3)
		#~ plt.plot(x,gauss(x,v0, v1, v2),label='fit', c='k', linewidth=3)
		#~ #for i in peaks:
			#~ #plt.axvline(x[i],c='g')
		#~ plt.xlabel('F606W - F814W', fontsize = 24)
		#~ plt.ylabel('number of stars', fontsize = 24)
		#~ plt.legend(loc='upper right', fontsize = 24)
		#~ plt.title('IC4499', fontsize = 24)
		#~ plt.tick_params(labelsize=16)
		#~ plt.xlim(0.5,0.9)
		#~ plt.show()
		#~ plt.close()
		#~ kill
		
	# ~zero = np.where(gauss_meanMS == 0)[0]
	# ~plt.figure()
	# ~plt.scatter(color,photo_v, marker='.', s=10, color='grey', label='stars')
	# ~#plt.scatter(Color_iso,mag_v, marker='.', s=10, color='b', label='initial guess')
	# ~plt.errorbar(gauss_meanMS,bincenter, xerr=gauss_dispMS, capsize=2, c='k', linewidth=2, fmt='none', label=r'$\sigma_i^{data}$')
	# ~plt.scatter(gauss_meanMS,bincenter, marker='o', s=30, color='r', label=r'$C_i^{data}$')
	# ~#plt.axhline(top_y)
	# ~#plt.axvline(top_y, c='c')
	# ~plt.axhline(binMS[0], label='bin edges',alpha=0.5)
	# ~for ji in binMS[1:]:
		# ~plt.axhline(ji,alpha=0.5)
	# ~for ii in zero:
		# ~if ii == zero[0]:
			# ~plt.fill_between(np.linspace(0,2), binMS[ii], binMS[ii+1], color='grey', label='bins ignored')
		# ~else:
			# ~plt.fill_between(np.linspace(0,2), binMS[ii], binMS[ii+1], color='grey')
	# ~plt.xlim(0,2)
	# ~plt.ylim(27,10)
	# ~plt.tick_params(labelsize=16)
	# ~lgnd = plt.legend(loc='upper right', fontsize = 24)
	# ~lgnd.legendHandles[1]._sizes = [286]
	# ~lgnd.legendHandles[2]._sizes = [286]
	# ~lgnd.get_frame().set_edgecolor('k')
	# ~lgnd.get_frame().set_linewidth(2.0)
	# ~#plt.legend(loc='upper right', fontsize = 16)
	# ~plt.xlabel('F606W - F814W', fontsize = 24)
	# ~plt.ylabel('F606W', fontsize = 24)
	# ~plt.title('IC4499', fontsize = 24)

	# ~plt.show() 
	# ~gc.collect()
	# ~kill
	
	
	non_zero = np.where((gauss_meanMS > 0)&(gauss_meanMS < 3))[0]
	starnum = starnum[non_zero]
	gauss_meanMS = gauss_meanMS[non_zero]
	gauss_dispMS = gauss_dispMS[non_zero]
	binmid = bincenter[non_zero]
	
	starnum = starnum[~np.isnan(starnum)]
	gauss_meanMS = gauss_meanMS[~np.isnan(gauss_meanMS)]
	gauss_dispMS = gauss_dispMS[~np.isnan(gauss_dispMS)]
	binmid = binmid[~np.isnan(binmid)]
	
	return binmid, gauss_meanMS, gauss_dispMS, starnum, non_zero
	#~ return iva3_bin, ivs3_bin, starnum
#~ @profile	
def spread_color_y_rgb():

	def gauss(x,amp, mean, stddev):
		return amp*np.exp(-((x - mean) / 4 / stddev)**2)
		
	rangebinv = np.max(vgood) - np.min(vgood)
	rangebin = np.max(cgood) - np.min(cgood)
	nbins = int(round(rangebin /0.02))
	bingood = np.linspace(np.min(cgood), np.max(cgood),nbins)
	centergood = (bingood[:-1] + bingood[1:]) / 2 
	
	gauss_meanMS = np.zeros(lbin)
	gauss_dispMS = np.zeros(lbin)
	binmid = np.zeros(lbin)
	starnum = np.zeros(lbin)

	for ij in range(0,centergood):

		inbin = np.digitize(color, bingood)
		#~ inbin2 = np.digitize(ep_mag, binMS)
		ici = np.where(inbin == ij+1)[0]
		#~ ici2 = np.where(inbin2 == ij+1)[0]


		gopini = np.array(photo_v)[ici]
	

		bincol = np.linspace(rgb_lim, top_y, 100)
		gop, bb = np.histogram(gopini, bins=bincol)
		#~ gop, bb = np.histogram(gopini, bins=100)
		x = (bb[:-1] + bb[1:]) / 2
		
		from scipy.signal import find_peaks
		#~ peaks, prop = find_peaks(gop, height=50, prominence = 200)
		#~ peaks, prop = find_peaks(gop, height=np.max(gop)*0.2, prominence = np.max(gop)*0.2)
		peaks, prop = find_peaks(gop, height=np.max(gop)*0.2)
		#~ print(prop)
		#~ print(peaks.size)

		#~ print(len(ici))
		popt,pcov = curve_fit(gauss,x,gop, p0 = [np.max(gop),x[np.argmax(gop)], np.std(err_color[ici])], maxfev = 950000)
		v0, v1, v2 = popt


		#~ kill
		starnum[ij] = len(ici)
		gauss_meanMS[ij] = v1
		gauss_dispMS[ij] = v2
		binmid[ij] = np.median(np.array(photo_v)[ici])

	non_zero = np.where((gauss_meanMS > 0)&(gauss_meanMS < 3))[0]
	starnum = starnum[non_zero]
	gauss_meanMS = gauss_meanMS[non_zero]
	gauss_dispMS = gauss_dispMS[non_zero]
	binmid = binmid[non_zero]
	
	return gauss_meanMS, gauss_dispMS, starnum, centerhor
	#~ return iva3_bin, ivs3_bin, starnum

#~ @profile	
def spread_color_y( fin, debut=None):

	magv = []
	col = []
	errv = []
	errcol = []
	z = []

	if debut == None:
		dd = np.where(photo_v <= fin)[0]
	else:
		dd = np.where((photo_v <= fin)&(photo_v >= debut))[0]

	magv.extend(photo_v[dd])
	col.extend(color[dd])
	errv.extend(err_v[dd])
	errcol.extend(err_color[dd])
	
	
	#~ threshold = 3
	#~ z = np.abs(stats.zscore(col))
	#~ print(z)
	#~ plt.hist(z, bins=50)
	#~ plt.show()
	#~ kill
	#~ out = (np.where(z > threshold)[0])
	#~ col =  np.delete(col, out)
	#~ magv =  np.delete(magv, out)
	#~ errv =  np.delete(errv, out)
	#~ errcol =  np.delete(errcol, out)
	
	
	return magv, col, errv, errcol
	
	
	


	#~ fmean = interpolate.interp1d(bin_center, gauss_mean)
	#~ fdisp = interpolate.interp1d(bin_center, gauss_disp)

	#~ mu, sigma = f, fMad(mag_v) # mean and standard deviation
	#~ spread = np.random.normal(mu, sigma)
#~ @profile	
def binning_MS(debut, fin, photo_v, Color):
	#~ print(debut)
	rge = fin - debut
	new_bin1 = debut+step

	binarr = [debut]
	while new_bin1 < fin:
		#~ print(nbins)
		indGB = np.digitize(photo_v, binarr+[new_bin1]) 
		#~ print(indGB)
		#~ print(np.min(indGB), np.max(indGB))
		cbinGB = np.where(indGB == np.max(indGB)-1)[0]
		opsGB = Color[cbinGB]
		#~ print(opsGB.size, new_bin1) 
		#~ if glc in [62]:
			#~ starcount = 50
		#~ elif glc in [63]:
			#~ starcount = 100
		#~ else:
			#~ starcount = 50
		#~ starcount = len(photo_v)/40.
		starcount = 0
		if opsGB.size > starcount:
		#~ if opsGB.size > len(photo_v)*0.04:
			binarr.append(new_bin1)
			#~ print(binarr)
			#~ new_bin1 = new_bin2
		#~ print(new_bin1, new_bin2)
		new_bin1 = new_bin1 +step
	gc.collect()
	return binarr
#~ @profile	
def binning_SGB(debut, fin, photo_v, Color):
	rge = fin - debut
	new_bin1 = debut+0.01
	binarr = [debut]
	while new_bin1 < fin:
		#~ print(nbins)
		indGB = np.digitize(photo_v, binarr+[new_bin1]) 
		#~ print(indGB)
		#~ print(np.min(indGB), np.max(indGB))
		cbinGB = np.where(indGB == np.max(indGB)-1)[0]
		opsGB = Color[cbinGB]
		#~ print(opsGB.size, new_bin1) 
		#~ if glc in [62]:
			#~ starcount = 50
		#~ elif glc in [63]:
			#~ starcount = 100
		#~ else:
			#~ starcount = 50
		starcount = 20
		if opsGB.size > starcount:
		#~ if opsGB.size > len(photo_v)*0.04:
			binarr.append(new_bin1)
			#~ print(binarr)
			#~ new_bin1 = new_bin2
		#~ print(new_bin1, new_bin2)
		new_bin1 = new_bin1 +0.01
	gc.collect()
	return binarr
	
#~ @profile
def binning_GB(debut, fin, photo_v, Color):
	#~ step=0.2
	rge = fin - debut
	new_bin1 = debut+step
	binarr = [debut]
	while new_bin1 < fin:
		#~ print(nbins)
		indGB = np.digitize(photo_v, binarr+[new_bin1]) 
		#~ print(indGB)
		#~ print(np.min(indGB), np.max(indGB))
		cbinGB = np.where(indGB == np.max(indGB)-1)[0]
		opsGB = Color[cbinGB]
		#~ print(opsGB.size, new_bin1) 
		#~ if glc in [62]:
			#~ starcount = 50
		#~ elif glc in [63]:
			#~ starcount = 100
		#~ else:
			#~ starcount = 50
		starcount = 0
		if opsGB.size > starcount:
		#~ if opsGB.size > len(photo_v)*0.04:
			binarr.append(new_bin1)
			#~ print(binarr)
			#~ new_bin1 = new_bin2
		#~ print(new_bin1, new_bin2)
		new_bin1 = new_bin1 + step
	gc.collect()
	return binarr

#~ def binning_GB(debut, fin, photo_v, Color):
	#~ nbins = 40
	#~ gb = 0
	#~ gg = 0
	
	#~ while (nbins + gb) > gg:
		#~ gb = 0
		#~ gg = 0
		#~ binGB = np.linspace(debut,fin, nbins+1)
		#~ print(nbins)
		#~ indGB = np.digitize(photo_v, binGB) 
		#~ for n in range(1,nbins+1):
			#~ cbinGB = np.where(indGB == n)[0]
			#~ opsGB = Color[cbinGB]
			#~ print(opsGB.size)        
			#~ if opsGB.size ==0:
				#~ gb += 1
			#~ else:
				#~ gg += 1
		#~ print(nbins, gg,gb)
		#~ nbins = nbins - 1
	#~ gc.collect()
	#~ return nbins
#~ @profile	
def hist_compare(d1a, d1b, d2a, d2b, m1, weights):
	bins=250
	hist_1, xx, yy = np.histogram2d(d1a, d1b, bins=bins, range=[[-0.5, 3],[m1, mag_lim[0]]])
	hist_2, _, _ = np.histogram2d(d2a, d2b, bins=bins, range=[[-0.5, 3],[m1, mag_lim[0]]])

	#~ plt.hist2d(d1a, d1b, bins=bins, range=[[-0.5, 3],[m1, mag_lim[0]]])
	#~ plt.show()
	#~ kill
	#~ out = (np.where(z > threshold)[0])
	#~ opstot =  np.delete(opstot, out)
	#~ errtot =  np.delete(errtot, out)
	ln2 = 0
	inter = 0
	for i in range(bins):
		for j in range(bins):
			if hist_1[i,j] > 0:
				bindex = np.where((xx[i] <= d1a) & (d1a <= xx[i+1]) & (yy[j] <= d1b) & (d1b <= yy[j+1]))[0]
				mw =(np.mean(weights[bindex]))/len(bindex)
				ln2 -= mw*(hist_1[i,j] - hist_2[i,j]*np.log(hist_1[i,j]))
				#~ print(ln2)
				if hist_2[i,j] > 0:
					inter += 1
					#~ print(inter)
				#~ else:
					#~ inter +=(np.sum(weights[bindex]))
					#~ print(inter)
			else:
				ln2 -= 0
			
	if inter ==0:
		return -np.inf
	#~ print(inter)
	#~ return ln2 / inter
	return ln2
	
	
########################################################################
########################################################################
#~ @profile	
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
			#~ mag_v, eep_first = darp2.mageep['V'](Age, metal, distance, A)
			#~ mag_i, eep_first = darp2.mageep['I'](Age, metal, distance, A)
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




########################################################################
########################################################################
#~ @profile
def lnlike(theta):

	if model == 'mist': 
		
		age, FeH, dist, A1,falpha = theta
		
		corr = np.log10(0.694*(10**falpha) + 0.306)
		if age > 10.30 or  FeH + corr < -2.5 or  0.5 < FeH + corr:
		#~ if age > 10.30 or  FeH < -2.5 or  0.5 < FeH:
			print('nonon')
			return -np.inf
			

		
		#~ mag_v1, mag_i1, Color_iso1, eep_first1 = iso_mag(age, FeH+corr, dist, A1)
		mag_v1, mag_i1, Color_iso1, eep_first1 = iso_mag(age, FeH, dist, A1)


			
		#~ if len(argrelextrema(Color_iso1, np.less, order=20)[0][1:]) > 0 :
		if len(argrelextrema(Color_iso1, np.greater, order=4)[0]) > 0 :
			ct2prime = ([np.min(argrelextrema(Color_iso1, np.greater, order=4)[0])])
			ct2 = [605-int(eep_first1)]
			#~ print(ct2, ct2prime)
		else:
			print("List is empty")
			return -np.inf
			
		#~ nointerp = np.where(magv > np.min(mag_v1[:ct2[0]-1]))[0]
		#~ nointerp = np.where(magv > np.min(magv))[0]

		### compute the likelihood of the MS
		fmag = interpolate.interp1d(mag_v1[:ct2[0]-1], Color_iso1[:ct2[0]-1], 'nearest',fill_value="extrapolate")
		#~ fmag = interpolate.interp1d(mag_v1[:ct2[0]-1], Color_iso1[:ct2[0]-1], 'nearest')
		Color_new = fmag(bincenter)
		lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / gauss_disp**2)
		#~ lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp/np.sqrt(starnum))**2 ) 
		
		### compute the likelihood of the SGB
		#~ Color_new1 = fmag(gauss_mean_sgb)
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb)**2 )
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb/np.sqrt(starnum_sgb))**2 )
		
		#~ Color_new = fmag(full_mag)
		#~ lnl = -0.5*np.sum((full_col - Color_new)**2 / (full_err)**2 )
		### compute the likelihood of the RGB
		#~ print(magv2)
		#~ Color_new2 = fmag(magvbis)
		#~ lnl2 = -0.5*np.sum((np.array(colbis) - Color_new2)**2 / (np.array(errcolbis))**2 )

		Color_new2 = fmag(vcenter_rgb)
		lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / np.array(errcenter_rgb)**2 )
		#lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / (np.array(errcenter_rgb)/np.sqrt(sbin_rgb))**2 )

		#~ fmag_ver = interpolate.interp1d(Color_iso1[:ct2[0]-1][np.where(mag_v1[:ct2[0]-1] < mag_lim2)[0]],
		#~ mag_v1[:ct2[0]-1][np.where(mag_v1[:ct2[0]-1] < mag_lim2)[0]], 'nearest',fill_value="extrapolate")	
		#~ mag_new2 = fmag_ver(ccenter_rgb)
		#~ lnl2 = -0.5*np.sum((np.array(vcenter_rgb) - mag_new2)**2 / np.array(errcenterv_rgb)**2 )
		#~ topeep = np.where(mag_v1[:ct2[0]-1] < mag_lim2)[0]
		#~ fmag_ver = interpolate.interp1d(Color_iso1[:ct2[0]-1][topeep], mag_v1[:ct2[0]-1][topeep], 'nearest',fill_value="extrapolate")
		#~ Color_new2 = fmag_ver(ccenter_rgb)
		#~ lnl2 = -0.5*np.sum((np.array(vcenter_rgb) - Color_new2)**2 / (np.array(errcenterv_rgb))**2 )
		#~ with open('/home/david/codes/Analysis/GC/plots/starnum.txt', 'a+') as fid_file:
			#~ fid_file.write('%s %.8g %.8g %.2g %.8g %.2g\n' % (clus_nb, longueur, len(photo_v),
			#~ float(len(photo_v))/longueur*100, (np.sum(starnum)+ len(n1)), 
			#~ float((np.sum(starnum)+ len(n1)))/longueur*100))
		#~ fid_file.close()
		#~ kill
		#~ col_turn1 = [np.argmin((Color_iso1[:ct2[0]-1]))]
		#~ top_x1 = Color_iso1[col_turn1]
		#~ top_y1 = mag_v1[col_turn1]
		#~ diffx = (top_x1[0] - top_x)
		#~ diffy = (top_y1[0] - top_y)
		#~ lnl3 = -0.5*diffy**2/err_topy**2
		
		#~ print(age, FeH, dist, A1,lnl,lnl2, lnl+lnl2) 
		#~ print('lnl',age, FeH, dist, A1,lnl) 

	#-----------------------------------------------------------------------
	#-----------------------------------------------------------------------
	#-----------------------------------------------------------------------
	#-----------------------------------------------------------------------

	if model == 'dar': 
		
		age, FeH, dist, A1, afe = theta
		afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8]
		#~ afe_values=[-0.2, 0.0 , 0.2, 0.6, 0.8]  

		if age > 10.175 or FeH < -2.5 or FeH > 0.0:
			print('nonon')
			return -np.inf

		afe_max = afe_values[np.searchsorted(afe_values, afe)]
		afe_min = afe_values[np.searchsorted(afe_values, afe)-1]
		
		#~ print(afe, afe_min, afe_max)

		#~ mag_v1 , mag_i1, Color_iso1, eep_first = iso_mag(age, FeH, dist, A1, afe_init)
		mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(age, FeH, dist, A1, afe_min)
		mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(age, FeH, dist, A1, afe_max)
		lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
		
		mag_v1 = (mag_v1_min[:lpp]*(afe_max - afe) + mag_v1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
		Color_iso1 = (Color_iso1_min[:lpp]*(afe_max - afe) + Color_iso1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
	

		if len(mag_v1) < 2:
			print('nonon')
			return -np.inf
		#~ if np.min(mag_v1) > np.min(vcenter_rgb) or np.max(mag_v1) < np.max(bincenter):
			#~ print('oui')

		### compute the likelihood of the MS
		fmag = interpolate.interp1d(mag_v1, Color_iso1, 'nearest',fill_value="extrapolate")
		#~ fmag = interpolate.interp1d(mag_v1, Color_iso1, 'cubic')
		Color_new = fmag(bincenter)
		lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp)**2 )
		# ~lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp/np.sqrt(starnum))**2 )

		# ~fmag_ver1 = interpolate.interp1d(Color_iso1[np.where(mag_v1 > mag_lim2)[0]], mag_v1[np.where(mag_v1 > mag_lim2)[0]], 'nearest',fill_value="extrapolate")	
		# ~mag_new1 = fmag_ver1(gauss_mean)
		# ~lnl = -0.5*np.sum((np.array(bincenter) - mag_new1)**2 / np.array(errcenterv_gauss)**2 )
		
		### compute the likelihood of the SGB
		#~ Color_new1 = fmag(gauss_mean_sgb)
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb)**2 )
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb/np.sqrt(starnum_sgb))**2 )
		
		#~ Color_new = fmag(full_mag)
		#~ lnl = -0.5*np.sum((full_col - Color_new)**2 / (full_err)**2 )
		### compute the likelihood of the RGB
		#~ print(magv2)
		#~ Color_new2 = fmag(magvbis)
		#~ lnl2 = -0.5*np.sum((np.array(colbis) - Color_new2)**2 / (np.array(errcolbis))**2 )

		Color_new2 = fmag(vcenter_rgb)
		lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / np.array(errcenter_rgb)**2 )
		# ~lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / (np.array(errcenter_rgb)/np.sqrt(sbin_rgb))**2 )
	


		# ~col_turn1 = [np.argmin((Color_iso1))]
		# ~top_x1 = Color_iso1[col_turn1]
		# ~top_y1 = mag_v1[col_turn1]
		# ~diffy = (top_y1[0] - chunkbot[glc])
		# ~lnl3 = -0.5*diffy**2/(0.1)**2

		### compute the likelihood of the sgb vertical
		# ~fmag_ver = interpolate.interp1d(Color_iso1[np.where(mag_v1 < mag_lim2)[0]], mag_v1[np.where(mag_v1 < mag_lim2)[0]], 'nearest',fill_value="extrapolate")	
		# ~mag_new2 = fmag_ver(ccenter_rgb)
		# ~lnl3 = -0.5*np.sum((np.array(vcenter_rgb) - mag_new2)**2 / np.array(errcenterv_rgb)**2 )

		# ~plt.figure()
		# ~plt.plot(Color_iso1[np.where(mag_v1 > mag_lim2)[0]], mag_v1[np.where(mag_v1 > mag_lim2)[0]],c='b')
		# ~plt.plot(Color_iso1[np.where(mag_v1 < mag_lim2)[0]], mag_v1[np.where(mag_v1 < mag_lim2)[0]],c='r')
		# ~plt.scatter(ccenter_rgb,mag_new2, c='r')
		# ~plt.scatter(gauss_mean,mag_new1, c='b')
		# ~plt.scatter(gauss_mean,bincenter, c='k')
		# ~plt.scatter(ccenter_rgb,vcenter_rgb, c='k')
		# ~plt.xlim(-0.5,3)
		# ~plt.ylim(26,10)
		# ~plt.show()
		# ~plt.close()
		
		# ~print(lnl2) 
		# ~print(lnl,lnl2) 
		# ~print(len(Color_new),len(Color_new2))
		# ~print(lnl/len(Color_new),lnl2/len(Color_new2), lnl/len(Color_new)+lnl2/len(Color_new2))
		#~ print('lnl',age, FeH, dist, A1,lnl) 

	#~ plt.figure()
	#~ plt.scatter(color,photo_v, marker='.', s=10, color='grey', label='data')
	#~ plt.errorbar(ccenter_rgb, vcenter_rgb, xerr=errcenter_rgb,fmt = 'none', c='k')
	#~ plt.scatter(Color_new2, vcenter_rgb, c='r', marker='x', s=30)
	#~ plt.scatter(ccenter_rgb,vcenter_rgb, c='b', marker='o', s=30)
	#~ plt.scatter(Color_new,bincenter, c='r', marker='x', s=30)
	#~ plt.plot(Color_iso1[:ct2[0]-1], mag_v1[:ct2[0]-1])
	#~ plt.plot(Color_iso1, mag_v1)
	#~ plt.scatter(gauss_mean,bincenter, c='b', marker='o', s=30)
	#~ plt.xlim(-0.5,3)
	#~ plt.ylim(26,10)
	#~ plt.title(clus_nb+' '+str(glc), fontsize = 16)
	#~ plt.show() 
	#~ plt.close()
	
	
	if glc == 62:
		return (lnl+lnl2)
	else:
		# ~return lnl
		# ~return (lnl/len(Color_new)+lnl2/len(Color_new2))
		# ~print(lnl,lnl2,lnl+lnl2)
		gc.collect()
		return (lnl+lnl2)
		
def lnlike2(theta):

	if model == 'mist': 
		
		age, FeH, dist, A1 = theta
		
		mag_v1, mag_i1, Color_iso1, eep_first1 = iso_mag(age, FeH, dist, A1)

		if len(mag_v1) == 2:
			print('nonon')
			return -np.inf
			
		#~ if len(argrelextrema(Color_iso1, np.less, order=20)[0][1:]) > 0 :
		if len(argrelextrema(Color_iso1, np.greater, order=4)[0]) > 0 :
			ct2prime = ([np.min(argrelextrema(Color_iso1, np.greater, order=4)[0])])
			ct2 = [605-int(eep_first1)]
			#~ print(ct2, ct2prime)
		else:
			print("List is empty")
			return -np.inf
			
		#~ nointerp = np.where(magv > np.min(mag_v1[:ct2[0]-1]))[0]
		#~ nointerp = np.where(magv > np.min(magv))[0]

		### compute the likelihood of the MS
		fmag = interpolate.interp1d(mag_v1[:ct2[0]-1], Color_iso1[:ct2[0]-1], 'nearest',fill_value="extrapolate")
		Color_new = fmag(bincenter)
		lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / gauss_disp**2)
		#~ lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp/np.sqrt(starnum))**2 ) 
		
		### compute the likelihood of the SGB
		#~ Color_new1 = fmag(gauss_mean_sgb)
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb)**2 )
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb/np.sqrt(starnum_sgb))**2 )
		
		#~ Color_new = fmag(full_mag)
		#~ lnl = -0.5*np.sum((full_col - Color_new)**2 / (full_err)**2 )
		### compute the likelihood of the RGB
		#~ print(magv2)
		#~ Color_new2 = fmag(magvbis)
		#~ lnl2 = -0.5*np.sum((np.array(colbis) - Color_new2)**2 / (np.array(errcolbis))**2 )

		Color_new2 = fmag(vcenter_rgb)
		lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / np.array(errcenter_rgb)**2 )
		#~ #lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / (np.array(errcenter_rgb)/np.sqrt(sbin_rgb))**2 )


		#~ topeep = np.where(mag_v1[:ct2[0]-1] < mag_lim2)[0]
		#~ fmag_ver = interpolate.interp1d(Color_iso1[:ct2[0]-1][topeep], mag_v1[:ct2[0]-1][topeep], 'nearest',fill_value="extrapolate")
		#~ Color_new2 = fmag_ver(ccenter_rgb)
		#~ lnl2 = -0.5*np.sum((np.array(vcenter_rgb) - Color_new2)**2 / (np.array(errcenterv_rgb))**2 )
		#~ with open('/home/david/codes/Analysis/GC/plots/starnum.txt', 'a+') as fid_file:
			#~ fid_file.write('%s %.8g %.8g %.2g %.8g %.2g\n' % (clus_nb, longueur, len(photo_v),
			#~ float(len(photo_v))/longueur*100, (np.sum(starnum)+ len(n1)), 
			#~ float((np.sum(starnum)+ len(n1)))/longueur*100))
		#~ fid_file.close()
		#~ kill
		#~ col_turn1 = [np.argmin((Color_iso1[:ct2[0]-1]))]
		#~ top_x1 = Color_iso1[col_turn1]
		#~ top_y1 = mag_v1[col_turn1]
		#~ diffx = (top_x1[0] - top_x)
		#~ diffy = (top_y1[0] - top_y)
		#~ lnl3 = -0.5*diffy**2/err_topy**2

	#-----------------------------------------------------------------------
	#-----------------------------------------------------------------------
	#-----------------------------------------------------------------------
	#-----------------------------------------------------------------------

	if model == 'dar': 
		
		age, FeH, dist, A1, afe = theta
		afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8]
		#~ afe_values=[-0.2, 0.0 , 0.2, 0.6, 0.8]  


		afe_max = afe_values[np.searchsorted(afe_values, afe)]
		afe_min = afe_values[np.searchsorted(afe_values, afe)-1]
		
		#~ print(afe, afe_min, afe_max)

		#~ mag_v1 , mag_i1, Color_iso1, eep_first = iso_mag(age, FeH, dist, A1, afe_init)
		mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(age, FeH, dist, A1, afe_min)
		mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(age, FeH, dist, A1, afe_max)
		lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
		
		mag_v1 = (mag_v1_min[:lpp]*(afe_max - afe) + mag_v1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
		Color_iso1 = (Color_iso1_min[:lpp]*(afe_max - afe) + Color_iso1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
	

		if len(mag_v1) < 2:
			print('nonon')
			return -np.inf

		### compute the likelihood of the MS
		fmag = interpolate.interp1d(mag_v1, Color_iso1, 'nearest',fill_value="extrapolate")
		Color_new = fmag(bincenter)
		lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp)**2 )
		#~ lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp/np.sqrt(starnum))**2 )
		
		### compute the likelihood of the SGB
		#~ Color_new1 = fmag(gauss_mean_sgb)
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb)**2 )
		#~ lnl1 = -0.5*np.sum((centerhor - Color_new1)**2 / (gauss_disp_sgb/np.sqrt(starnum_sgb))**2 )
		
		#~ Color_new = fmag(full_mag)
		#~ lnl = -0.5*np.sum((full_col - Color_new)**2 / (full_err)**2 )
		### compute the likelihood of the RGB
		#~ print(magv2)
		#~ Color_new2 = fmag(magvbis)
		#~ lnl2 = -0.5*np.sum((np.array(colbis) - Color_new2)**2 / (np.array(errcolbis))**2 )

		Color_new2 = fmag(vcenter_rgb)
		lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / np.array(errcenter_rgb)**2 )
		#lnl2 = -0.5*np.sum((np.array(ccenter_rgb) - Color_new2)**2 / (np.array(errcenter_rgb)/np.sqrt(sbin_rgb))**2 )

		#~ col_turn1 = [np.argmin((Color_iso1))]
		#~ top_x1 = Color_iso1[col_turn1]
		#~ top_y1 = mag_v1[col_turn1]
		#~ diffx = (top_x1[0] - top_x)
		#~ diffy = (top_y1[0] - top_y)
		#~ lnl3 = -0.5*diffy**2/err_topy**2
	print(age, FeH, dist, A1,lnl,lnl2, lnl+lnl2) 
	#~ print('lnl',age, FeH, dist, A1,lnl) 

	#~ plt.figure()
	#~ plt.scatter(color,photo_v, marker='.', s=10, color='grey', label='data')
	#~ plt.errorbar(ccenter_rgb, vcenter_rgb, xerr=errcenter_rgb,fmt = 'none', c='k')
	#~ plt.scatter(ccenter_rgb, Color_new2, c='r', marker='x', s=30)
	#~ plt.scatter(ccenter_rgb,vcenter_rgb, c='b', marker='o', s=30)
	#~ plt.scatter(Color_new,bincenter, c='r', marker='x', s=30)
	#~ plt.plot(Color_iso1[:ct2[0]-1], mag_v1[:ct2[0]-1])
	#~ plt.scatter(gauss_mean,bincenter, c='b', marker='o', s=30)
	#~ plt.xlim(-0.5,3)
	#~ plt.ylim(26,10)
	#~ plt.title(clus_nb+' '+str(glc), fontsize = 16)
	#~ plt.show() 
	#~ plt.close()
	
	
	if glc == 62:
		return lnl
	else:
		return lnl
		#~ return (lnl+lnl2)


#~ @profile
def lnprior(theta):
	if model == 'mist':
		#~ print(theta)
		age, FeH, dist, A1, falpha = theta
		###gaussian prior on absorption
		abs_mu = Abs
		A1_mu = A1
		#~ abs_sigma = 1/3. * A1 #mag
		abs_sigma = 0.06 #mag
		lnl_abs = np.log(1.0/(np.sqrt(2*np.pi)*abs_sigma))-0.5*(A1_mu-abs_mu)**2/abs_sigma**2
		###gaussian prior on distance modulus
		dm_mu = (np.log10(distance) -1)*5
		dist_mu = (np.log10(dist) -1)*5
		dm_sigma = 0.5 #mag
		#~ dm_sigma = 0.15 #mag
		lnl_dm = np.log(1.0/(np.sqrt(2*np.pi)*dm_sigma))-0.5*(dist_mu-dm_mu)**2/dm_sigma**2
		###gaussian prior on metallicity
		fe_mu = metal
		me_mu = float(FeH)
		me_sigma = 0.2 #mag
		lnl_me = (math.log(1.0/(math.sqrt(2*math.pi)*me_sigma))-0.5*(me_mu-fe_mu)**2/me_sigma**2)
			#~ #flat priors on age, FeH, Av
		if 9 < age < 10.176 and -2.5 < FeH < 0.5  and 0.0 < dist and 0 < A1 < 3 and -0.2 <= falpha <= 0.8:
			return lnl_me +lnl_dm + lnl_abs + lnl_abu
			#~ return lnl_abs
		return -np.inf
		#if 9 < age < 10.30  and -4 < FeH < 0.5 and 0.0 < dist and 0 < A1 < 3.0:
		#	return lnl_me

	if model == 'dar':
		age, FeH, dist, A1, afe = theta
		###gaussian prior on absorption
		abs_mu = Abs
		A1_mu = A1
		# ~abs_sigma = 1/3. * A1 #mag
		abs_sigma = 0.2 #mag
		# ~lnl_abs = np.log(1.0/(np.sqrt(2*np.pi)*abs_sigma))-0.5*(A1_mu-abs_mu)**2/abs_sigma**2
		lnl_abs = -0.5*((A1_mu-abs_mu)**2)/(abs_sigma**2)
		###gaussian prior on distance
		# ~dm_mu = (np.log10(distance) -1)*5
		# ~dist_mu = (np.log10(dist) -1)*5
		# ~dm_sigma = 0.5 #mag	
		dm_mu = distance
		dist_mu = dist
		dm_sigma = errdist #mag
		# ~dm_sigma = 500 #mag
		# ~lnl_dm = np.log(1.0/(np.sqrt(2*np.pi)*dm_sigma))-0.5*(dist_mu-dm_mu)**2/dm_sigma**2
		lnl_dm = -0.5*((dist_mu-dm_mu)**2)/(dm_sigma**2)
		###gaussian prior on metallicity
		fe_mu = metal
		me_mu = float(FeH)
		me_sigma = 0.2 #mag
		# ~lnl_me = (math.log(1.0/(math.sqrt(2*math.pi)*me_sigma))-0.5*(me_mu-fe_mu)**2/me_sigma**2)
		lnl_me = -0.5*((me_mu-fe_mu)**2)/(me_sigma**2)
		###gaussian prior on abundance
		afe_mu = afe_init
		abu_mu = afe
		abu_sigma = 0.2
		# ~lnl_abu = (math.log(1.0/(math.sqrt(2*math.pi)*abu_sigma))-0.5*(abu_mu-afe_mu)**2/abu_sigma**2)
		lnl_abu = -0.5*((abu_mu-afe_mu)**2)/(abu_sigma**2)
			#~ #flat priors on age, FeH, Av
		if 9 < age < 10.175 and -2.5 < FeH < 0  and 0.0 < dist and 0 < A1 < 3.0 and -0.2 <= afe <= 0.8:
			# ~return 0.0
			# ~print('met: '+str(lnl_me),'dist: '+str(lnl_dm),'abs: '+str(lnl_abs),'afe: '+str(lnl_abu), 'TOTAL = '+str(lnl_me + lnl_dm + lnl_abs + lnl_abu))
			return lnl_me + lnl_dm + lnl_abs + lnl_abu
		return -np.inf
		
#~ @profile
def lnprior2(theta):
	if model == 'mist':
		#~ print(theta)
		age, FeH, dist, A1 = theta
		###gaussian prior on absorption
		abs_mu = b4_mcmc_fit[0]
		A1_mu = A1
		abs_sigma = np.abs(b4_mcmc_fit[2] - b4_mcmc_fit[1]) 
		#~ abs_sigma = 1/6. * A1 #mag
		lnl_abs = np.log(1.0/(np.sqrt(2*np.pi)*abs_sigma))-0.5*(A1_mu-abs_mu)**2/abs_sigma**2
		###gaussian prior on distance modulus
		dm_mu = (np.log10(b3_mcmc_fit[0]) -1)*5
		dist_mu = (np.log10(dist) -1)*5
		dm_sigma = np.abs((np.log10(b3_mcmc_fit[2]) -1)*5 - b3_mcmc_fit[1]) 
		#~ dm_sigma = 0.15 #mag
		lnl_dm = np.log(1.0/(np.sqrt(2*np.pi)*dm_sigma))-0.5*(dist_mu-dm_mu)**2/dm_sigma**2
		###gaussian prior on metallicity
		fe_mu = b2_mcmc_fit[0]
		me_mu = float(FeH)
		me_sigma = np.abs(b2_mcmc_fit[2] - b2_mcmc_fit[1]) 
		lnl_me = (math.log(1.0/(math.sqrt(2*math.pi)*me_sigma))-0.5*(me_mu-fe_mu)**2/me_sigma**2)
			#~ #flat priors on age, FeH, Av
		if 9 < age < 10.176 and -2.5 < FeH < 0.5  and 0.0 < dist and 0 < A1 < 3.0 :
			return lnl_me +lnl_dm + lnl_abs
		return -np.inf
	if model == 'dar':
		#~ print(theta)
		age, FeH, dist, A1, afe = theta
		###gaussian prior on absorption
		abs_mu = b4_mcmc_fit[0]
		A1_mu = A1
		abs_sigma = np.abs(b4_mcmc_fit[2] - b4_mcmc_fit[1]) 
		#~ abs_sigma = 1/6. * A1 #mag
		lnl_abs = np.log(1.0/(np.sqrt(2*np.pi)*abs_sigma))-0.5*(A1_mu-abs_mu)**2/abs_sigma**2
		###gaussian prior on distance modulus
		dm_mu = (np.log10(b3_mcmc_fit[0]) -1)*5
		dist_mu = (np.log10(dist) -1)*5
		dm_sigma = np.abs((np.log10(b3_mcmc_fit[2]) -1)*5 - b3_mcmc_fit[1]) 
		#~ dm_sigma = 0.15 #mag
		lnl_dm = np.log(1.0/(np.sqrt(2*np.pi)*dm_sigma))-0.5*(dist_mu-dm_mu)**2/dm_sigma**2
		###gaussian prior on metallicity
		fe_mu = b2_mcmc_fit[0]
		me_mu = float(FeH)
		me_sigma = np.abs(b2_mcmc_fit[2] - b2_mcmc_fit[1]) 
		lnl_me = (math.log(1.0/(math.sqrt(2*math.pi)*me_sigma))-0.5*(me_mu-fe_mu)**2/me_sigma**2)
		###gaussian prior on abundance
		fe_mu = b5_mcmc_fit[0]
		me_mu = afe
		abu_sigma = np.abs(b5_mcmc_fit[2] - b5_mcmc_fit[1]) 
		lnl_me = (math.log(1.0/(math.sqrt(2*math.pi)*abu_sigma))-0.5*(abu_mu-afe_mu)**2/abu_sigma**2)
			#~ #flat priors on age, FeH, Av
		if 9 < age < 10.176 and -2.5 < FeH < 0  and 0.0 < dist and 0 < A1 < 3.0 and 0 < afe < 0.8:
			return lnl_me +lnl_dm + lnl_abs
		return -np.inf


#~ @profile
def lnprob(theta):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	# ~print(lp,lnlike(theta),lp + lnlike(theta))
	return lp + lnlike(theta)
	
def lnprob2(theta):
	lp = lnprior2(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike2(theta)

	
#~ @profile
def rem_outliers(photo_v, err_v, color, err_color):

	#~ print()
	bp = np.where((np.array(photo_v)< 24) & (np.array(photo_v) > 24-0.05))[0]
	bps = np.median(np.array(color)[bp])
	# ~print(bps)
	
	from numpy import ones,vstack
	from numpy.linalg import lstsq
	points = [(top_x1[0], top_y1[0]),(bps, 24)]
	x_coords, y_coords = zip(*points)
	A = vstack([x_coords,ones(len(x_coords))]).T
	m, c = lstsq(A, y_coords)[0]
	#~ print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
	step = 0.2
	low =1-step
	high=1+step
	inside = np.where((photo_v > m*color+c*low)&(photo_v<m*color+c*high))[0]
	binum=100
	hist1,bined1 = np.histogram(color, bins=binum)
	hist2,bined2 = np.histogram(photo_v[inside], bins=binum)

	x_cut =(bined1[np.argmax(np.abs(np.diff(hist1)))+1])
	y_cut = m*x_cut + c
	x1 = (bined1[np.argmax(np.abs(np.diff(hist1)))-1]- c)/m
	x2 = (bined1[np.argmax(np.abs(np.diff(hist1)))+1]- c)/m
	err_xcut = x2-x1
	err_ycut = bined1[np.argmax(np.abs(np.diff(hist1)))+2] - bined1[np.argmax(np.abs(np.diff(hist1)))]
	# ~print(x_cut, y_cut, err_ycut, err_xcut)
	#~ stragglers=np.where(color < c_cut)[0]
	#~ nono = np.where(photo_v[inside] < m_cut)[0]
	
	#~ plt.hist(m*color[inside]+c, bins=binum)
	#~ plt.axvline(y_cut, c='r')
	#~ plt.axvline(top_y[0], c='k')
	#~ plt.show()
	#~ plt.close()
	#~ plt.hist(photo_v[inside],bins=binum)
	#~ plt.axvline(x_cut, c='r')
	#~ plt.axvline(top_y[0]-0.5, c='g')
	#~ plt.show()
	#~ plt.close()
	#~ kill
	
	#~ plt.figure()
	#~ plt.scatter(color,photo_v, marker='.', s=10, color='grey', label='data')
	#~ plt.plot(np.linspace(-0.5,3), m*np.linspace(-0.5,3)+c)
	#~ plt.plot(np.linspace(-0.5,3), m*np.linspace(-0.5,3)+c*low, c='r')
	#~ plt.plot(np.linspace(-0.5,3), m*np.linspace(-0.5,3)+c*high, c='b')
	#~ plt.xlim(-0.5,3)
	#~ plt.ylim(26,10)
	#~ plt.title(clus_nb+' '+str(glc), fontsize = 16)
	#~ plt.scatter(x_cut, y_cut)
	#~ plt.axvline(y_cut)
	#~ plt.axhline(x_cut)
	#~ plt.axhline(top_y1)
	#~ plt.show() 
	#~ plt.close()
	#~ kill
	#~ print(inside,nono)


	return x_cut, y_cut, err_ycut, err_xcut
	
#~ @profile
def logli(ep_mag2, ep_col2):

	ln = 0
	if len(ep_mag2) == 0:
		print('zero')
		return -np.inf
	
	for c in range(0,len(bincenter)):
		#~ inbin = np.where((vgood < ep_mag2[c-1])&(vgood > ep_mag2[c+1]))[0]
		inbin2 = np.digitize(ep_mag2, binMS)
		ici2 = np.where(inbin2 == c+1)[0]
		if len(ici2) == 0:
			return -np.inf
		#~ print(len(ici))
		
		slope, intercept, _, _, _ = linregress(ep_col2[ici2], ep_mag2[ici2])
		
		abscisse = (bincenter[c] - intercept)/ slope
		#~ plt.scatter(ep_col2[ici2], ep_mag2[ici2])
		#~ plt.scatter(gauss_mean[c], bincenter[c])
		#~ plt.scatter(abscisse, bincenter[c])
		#~ plt.plot(ep_col2[ici2], slope*ep_col2[ici2] + intercept, c='r')
		#~ plt.show()
		#~ kill
		
		# compute the perpendicular distance
		#~ rgood = abs((slope*gauss_mean[c] - bincenter[c] + intercept)) / (math.sqrt(slope**2 + 1))
		#~ ln -= 0.5 * np.sum((rgood)**2 / gauss_disp**2)
		
		ln -= 0.5 * (abscisse - gauss_mean[c])**2 / gauss_disp[c]**2
		#~ print(ln)
	if ln == 0:
		return -np.inf
	else:
		#~ print(ici, ici2)
		return ln
#~ @profile		
def logli1(ep_mag2, ep_col2):

	ln = 0
	if len(ep_mag2) == 0:
		print('zero')
		return -np.inf
	
	for c in range(0,len(bincentersgb)):
		#~ inbin = np.where((vgood < ep_mag2[c-1])&(vgood > ep_mag2[c+1]))[0]
		inbin2 = np.digitize(ep_mag2, binSGB)
		ici2 = np.where(inbin2 == c+1)[0]
		if len(ici2) == 0:
			return -np.inf
		#~ print(len(ici))
		
		slope, intercept, _, _, _ = linregress(ep_col2[ici2], ep_mag2[ici2])
		
		abscisse = (bincentersgb[c] - intercept)/ slope
		#~ plt.scatter(ep_col2[ici2], ep_mag2[ici2])
		#~ plt.scatter(gauss_mean[c], bincenter[c])
		#~ plt.scatter(abscisse, bincenter[c])
		#~ plt.plot(ep_col2[ici2], slope*ep_col2[ici2] + intercept, c='r')
		#~ plt.show()
		#~ kill
		
		# compute the perpendicular distance
		#~ rgood = abs((slope*gauss_mean[c] - bincenter[c] + intercept)) / (math.sqrt(slope**2 + 1))
		#~ ln -= 0.5 * np.sum((rgood)**2 / gauss_disp**2)
		
		ln -= 0.5 * (abscisse - gauss_mean_sgb[c])**2 / gauss_disp_sgb[c]**2
		#~ print(ln)
	if ln == 0:
		return -np.inf
	else:
		#~ print(ici, ici2)
		return ln
		
#~ @profile		
def way(vgood, cgood, errgood, errgoodv, step = None):

	#remove duplicate
	for i, j in zip(vgood, cgood):
		dup = np.where((vgood == i)&(cgood==j))[0]
		if len(dup) > 1:
			vgood = np.delete(np.array(vgood), dup[1:])
			cgood = np.delete(np.array(cgood), dup[1:])

	#~ print(bingood)
	#~ bingood = np.linspace(np.min(vgood), np.max(vgood), nsplit+1)

	#~ bingood = np.array(binning_GB(np.min(vgood), np.max(vgood), ep_mag2, ep_col2))
	#~ print(np.min(vgood), np.max(vgood))
	#~ print(np.min(ep_mag2), np.max(ep_mag2))

	step = 0.15
	#~ nbins = 20
	rangebin = np.max(vgood) - np.min(vgood)
	if step is not None:
		nbins = int(round(rangebin/step))
	else:
		nbins = int(round(rangebin/0.2))
	#~ print(rangebin)
	#~ kill

	#~ spacegood = np.geomspace(1, rangebin+1,nbins)
	#~ bingood = np.max(vgood) - (spacegood-1)
	#~ bingood = np.flipud(bingood) 
	#~ bingood = np.geomspace(np.min(vgood), np.max(vgood),nbins) 
	bingood = np.linspace(np.min(vgood), np.max(vgood),nbins)
	#~ bingood = np.array(binning_GB(np.min(vgood), np.max(vgood), np.array(vgood), np.array(cgood)))
	#~ bingood = np.append(bingood, mag_lim2)
	centergood = (bingood[:-1] + bingood[1:]) / 2 
	
	vcenter = np.zeros(len(centergood))
	ccenter = np.zeros(len(centergood))
	errcenter = np.zeros(len(centergood))
	errcenterv = np.zeros(len(centergood))
	size_bin = np.zeros(len(centergood))




	#~ print(bingood)
	#~ print(centergood)

	for c in range(0,len(centergood)):
		inbin = np.digitize(vgood, bingood)
		ici = np.where(inbin == c+1)[0]

		# ~print(errgood[ici])
		
		# ~apmstop = np.where(np.array(cgood)[ici] > top_x-0.05)[0]

		# ~# wrt median
		threshold = 3
		med = np.median(np.array(cgood)[ici])
		diff_med = np.abs(np.array(cgood)[ici] - med)
		errmed = np.median(diff_med) # multiply by 1.486 for notmal distribution
		scoremad = errmed* 1.4826 # multiply by 1.4826 for notmal distribution
		z = diff_med / scoremad
		out = (np.where(z > threshold)[0])
		zcol =  np.delete(np.array(cgood)[ici], out)
		zmagv =  np.delete(np.array(vgood)[ici], out)
		ecol =  np.delete(np.array(errgood)[ici], out)
		ecolv =  np.delete(np.array(errgoodv)[ici], out)
		# ~print(scoremad)

		times=0
		while times < 5:
			threshold = 3
			med = np.median(zcol)
			diff_med = np.abs(zcol - med)
			errmed = np.median(diff_med) # multiply by 1.486 for notmal distribution
			scoremad = errmed* 1.4826 # multiply by 1.4826 for notmal distribution
			# ~print(scoremad)
			z = diff_med / scoremad
			out2 = (np.where(z > threshold)[0])
			zcol =  np.delete(zcol, out2)
			zmagv =  np.delete(zmagv, out2)
			ecol =  np.delete(ecol, out2)
			ecolv =  np.delete(ecolv, out2)
			times=times+1


		#wrt mean
		# ~threshold = 3
		# ~z = np.abs(stats.zscore(np.array(cgood)[ici]))
		# ~out = (np.where(z > threshold)[0])
		# ~zcol =  np.delete(np.array(cgood)[ici], out)
		# ~zmagv =  np.delete(np.array(vgood)[ici], out)

		# ~times = 0
		# ~while times < 5:
			# ~z = np.abs(stats.zscore(zcol))
			# ~out = (np.where(z > threshold)[0])
			# ~zcol =  np.delete(zcol, out)
			# ~zmagv =  np.delete(zmagv, out)
			# ~ecol =  np.delete(ecol, out)
			# ~ecolv =  np.delete(ecolv, out)
			# ~times = times + 1
			# ~scoremad = np.std(zcol)
			# ~print(scoremad)

		if len(ici) > 2:
			tp = np.where((zmagv >= np.min(zmagv)) & (zmagv <= np.min(zmagv)+step/4.))[0]
			tps = np.median(zcol[tp])
			bp = np.where((zmagv <= np.max(zmagv)) & (zmagv >= np.max(zmagv)-step/4.))[0]
			bps = np.median(zcol[bp])
			
			x_axis_real_length = 21.7 #cm
			x_axis_lim = 3.5 #mag 
			y_axis_real_length = 11 #cm
			y_axis_lim = 16 #mg

			# ~#get the cm equivalent to 1 mag in x or y
			norm_x = x_axis_real_length/x_axis_lim
			norm_y = y_axis_real_length/y_axis_lim

			lat = step * norm_y
			lon = (bps - tps) * norm_x
			# ~lat = step
			# ~lon = (bps - tps)
			dist = np.sqrt((lat)**2 + (lon)**2)
			cos_ver = lat/dist
			cos_hor = lon/dist
			# ~angle = np.degrees(math.acos(cos))
			# ~print(c,len(ici), cos, angle)
		else:
			cos_ver = 1.
			cos_hor = 1.




		if len(ici) == 1:
			ccenter[c] =np.median(zcol)
			# ~vcenter[c] =centergood[c]
			# ~ccenter[c] =np.mean(zcol)
			errcenter[c] =np.array(errgood)[ici]
			errcenterv[c] =np.array(errgoodv)[ici]
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
		elif len(ici) == 2:
			ccenter[c] =np.median(zcol)
			# ~errcenter[c] = np.std(zcol)
			# ~errcenter[c] = scoremad*cos_ver
			# ~vcenter[c] =centergood[c]
			# ~errcenter[c] =1.2533*np.std(zcol)*cos
			# ~ccenter[c] =np.mean(zcol)
			errcenter[c] = np.std(zcol)*cos_ver
			errcenterv[c] = np.std(zmagv)*cos_hor
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
		elif len(ici) > 2 and len(ici) < 50:
			ccenter[c] =np.median(zcol)
			# ~errcenter[c] =1.2533*np.std(zcol)*cos
			errcenter[c] = scoremad*cos_ver
			# ~vcenter[c] =centergood[c]
			# ~ccenter[c] =np.mean(zcol)
			if np.std(zcol) == 0:
				errcenter[c] =np.mean(np.array(errgood)[ici])
				print('std nul')
			else:
				# ~errcenter[c] =np.std(zcol)*cos_ver
				errcenter[c] = scoremad * cos_ver
				errcenterv[c] =np.std(zmagv)*cos_hor
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
			# ~ccenter[c] =np.mean(zcol)
			# ~errcenter[c] = np.std(zcol)
			# ~vcenter[c] =np.median(zmagv)
			# ~vcenter[c] =np.median(zmagv)		
		elif len(ici) >= 50:
			# ~plt.scatter(cgood[ici], vgood[ici], label='stars', c='grey')
			# ~plt.scatter(zcol, zmagv, label='stars', c='lightblue')
			# ~plt.plot([tps,bps], [np.min(zmagv), np.max(zmagv)], c='r', label='orientation', lw=2)
			# ~plt.scatter(np.mean(zcol), np.mean(zmagv), c='b', marker='o')
			# ~plt.xlabel('F606W - F814W', fontsize = 24)
			# ~plt.ylabel('F606W', fontsize = 24)
			# ~plt.title('IC4499', fontsize = 24)
			# ~plt.axhline(bingood[c], label='bin edge', alpha=0.5)
			# ~plt.axhline(bingood[c+1], alpha=0.5)
			# ~plt.ylim(bingood[c+1]+0.02,bingood[c]-0.02)
			# ~plt.tick_params(labelsize=16)
			# ~plt.show()
			# ~plt.close()

			
			ccenter[c] =np.median(zcol)
			# ~errcenter[c] = np.std(zcol)
			# ~errcenter[c] = scoremad * cos
			# ~vcenter[c] =np.median(zmagv)
			# ~errcenter[c] = scoremad*cos
			# ~errcenter[c] =1.2533*np.std(zcol)*cos
			# ~vcenter[c] =centergood[c]
			# ~vcenter[c] =np.median(zmagv)
			# ~errcenter[c] =1.2533*np.std(zcol)	
			# ~ccenter[c] =np.mean(zcol)
			if np.std(zcol) == 0:
				errcenter[c] =np.mean(np.array(errgood)[ici])
				print('std nul')
			else:
				# ~errcenter[c] =np.std(zcol)*cos_ver
				errcenter[c] = scoremad*cos_ver
				errcenterv[c] =np.std(zmagv)*cos_hor
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)

		# ~plt.figure()
		# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
		# ~plt.scatter(np.array(cgood)[ici],np.array(vgood)[ici], marker='.',s=10, color='b', label='data')
		# ~plt.scatter(zcol,zmagv, marker='.',s=10, color='r', label='data')
		# ~plt.scatter(zcol,zmagv, marker='.',s=10, color='r', label='data')
		# ~plt.axvline(np.median(np.array(cgood)[ici]),c='r')
		# ~plt.errorbar(ccenter[c], vcenter[c], xerr=errcenter[c], capsize= 2, linewidth=2,fmt = 'x', c='k', alpha=0.5)
		# ~plt.xlim(-0.5,3)
		# ~plt.ylim(26,10)
		# ~plt.legend(loc='upper right', fontsize = 16)
		# ~plt.xlabel('F606W - F814W', fontsize = 16)
		# ~plt.ylabel('F606W', fontsize = 16)
		# ~plt.title(clus_nb, fontsize = 16)
		# ~plt.show()
		# ~plt.close()
			# ~kill						

			# ~from scipy.stats import skewnorm
			# ~amp2, moy2, dev2 = skewnorm.fit(zcol, 1, loc=0.5, scale=0.05)
			# ~amp2, moy2, dev2 = skewnorm.fit(zcol, 10, loc=0.5, scale=0.05)
			# ~print(amp2, moy2, dev2)		

			# ~x = np.linspace(np.min(zcol), np.max(zcol), 50)
			# ~p = stats.skewnorm.pdf(x,amp2, moy2, dev2)#.rvs(100)
			# ~mean_skew = stats.skewnorm.mean(amp2, moy2, dev2)#.rvs(100)
			# ~median_skew = stats.skewnorm.median(amp2, moy2, dev2)#.rvs(100)
			# ~err_skew = stats.skewnorm.std(amp2, moy2, dev2)#.rvs(100)
			# ~ccenter[c] = moy2

			# ~med = np.median(zcol)
			# ~diff_med = np.abs(zcol - med)
			# ~errmed = np.median(diff_med) # multiply by 1.486 for notmal distribution
			# ~scoremad = errmed* 1.4826 # multiply by 1.4826 for notmal distribution
			# ~z = diff_med / scoremad
			# ~out2 = (np.where(z > threshold)[0])
			# ~zcol =  np.delete(zcol, out2)
			# ~zmagv =  np.delete(zmagv, out2)
			# ~times=times+1


			# ~plt.figure()
			# ~plt.hist(zcol, bins=50, density=True,color='grey')
			# ~plt.plot(x, p, 'k', linewidth=2)
			# ~plt.axvline(moy2, c='y')
			# ~plt.axvline(mean_skew, c='k')
			# ~plt.axvline(median_skew, c='b', linestyle='--')
			# ~plt.axvline(np.median(np.array(cgood)[ici]),c='r')
			# ~plt.axvline(np.mean(np.array(cgood)[ici]),c='c')
			# ~plt.show()
			# ~plt.close()
			# ~kill


			# ~print(len(np.where(zcol < med)[0]))
			# ~print(len(np.where(zcol > med)[0]))

			# ~plt.figure()
			# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
			# ~plt.scatter(np.array(cgood)[ici],np.array(vgood)[ici], marker='.',s=10, color='b', label='data')
			# ~plt.scatter(zcol,zmagv, marker='.',s=10, color='r', label='data')
			# ~plt.scatter(zcol[np.where(zcol < med)[0]],zmagv[np.where(zcol < med)[0]], marker='.',s=10, color='c', label='data')
			# ~plt.axvline(mean_skew, c='k')
			# ~plt.axvline(median_skew, c='b', linestyle='--')
			# ~plt.axvline(med, c='b')
			# ~plt.axvline(np.median(np.array(cgood)[ici]),c='r')
			# ~plt.axvline(np.mean(np.array(cgood)[ici]),c='c')
			# ~plt.xlim(-0.5,3)
			# ~plt.ylim(26,10)
			# ~plt.legend(loc='upper right', fontsize = 16)
			# ~plt.xlabel('F606W - F814W', fontsize = 16)
			# ~plt.ylabel('F606W', fontsize = 16)
			# ~plt.title(clus_nb, fontsize = 16)
			# ~plt.show()
			# ~plt.close()
			# ~kill

	return vcenter, ccenter, errcenter, size_bin, bingood, errcenterv
	
def way2(vgood, cgood, errgood, errgoodv, step = None):

	#remove duplicate
	for i, j in zip(vgood, cgood):
		dup = np.where((vgood == i)&(cgood==j))[0]
		if len(dup) > 1:
			vgood = np.delete(np.array(vgood), dup[1:])
			cgood = np.delete(np.array(cgood), dup[1:])

	#~ print(bingood)
	#~ bingood = np.linspace(np.min(vgood), np.max(vgood), nsplit+1)

	#~ bingood = np.array(binning_GB(np.min(vgood), np.max(vgood), ep_mag2, ep_col2))
	#~ print(np.min(vgood), np.max(vgood))
	#~ print(np.min(ep_mag2), np.max(ep_mag2))

	step = 0.2
	#~ nbins = 20
	rangebin = np.max(vgood) - np.min(vgood)
	if step is not None:
		nbins = int(round(rangebin/step))
	else:
		nbins = int(round(rangebin/0.2))
	#~ print(rangebin)
	#~ kill

	#~ spacegood = np.geomspace(1, rangebin+1,nbins)
	#~ bingood = np.max(vgood) - (spacegood-1)
	#~ bingood = np.flipud(bingood) 
	#~ bingood = np.geomspace(np.min(vgood), np.max(vgood),nbins) 
	bingood = np.linspace(np.min(vgood), np.max(vgood),nbins)
	#~ bingood = np.array(binning_GB(np.min(vgood), np.max(vgood), np.array(vgood), np.array(cgood)))
	#~ bingood = np.append(bingood, mag_lim2)
	centergood = (bingood[:-1] + bingood[1:]) / 2 
	
	vcenter = np.zeros(len(centergood))
	ccenter = np.zeros(len(centergood))
	errcenter = np.zeros(len(centergood))
	errcenterv = np.zeros(len(centergood))
	size_bin = np.zeros(len(centergood))




	#~ print(bingood)
	#~ print(centergood)

	for c in range(0,len(centergood)):
		inbin = np.digitize(vgood, bingood)
		ici = np.where(inbin == c+1)[0]

		# ~print(errgood[ici])
		
		# ~apmstop = np.where(np.array(cgood)[ici] > top_x-0.05)[0]

		# ~# wrt median
		threshold = 3
		med = np.median(np.array(cgood)[ici])
		diff_med = np.abs(np.array(cgood)[ici] - med)
		errmed = np.median(diff_med) # multiply by 1.486 for notmal distribution
		scoremad = errmed* 1.4826 # multiply by 1.4826 for notmal distribution
		z = diff_med / scoremad
		out = (np.where(z > threshold)[0])
		zcol =  np.delete(np.array(cgood)[ici], out)
		zmagv =  np.delete(np.array(vgood)[ici], out)
		ecol =  np.delete(np.array(errgood)[ici], out)
		ecolv =  np.delete(np.array(errgoodv)[ici], out)

		times=0
		while times < 5:
			med = np.median(zcol)
			diff_med = np.abs(zcol - med)
			errmed = np.median(diff_med) # multiply by 1.486 for notmal distribution
			scoremad = errmed* 1.4826 # multiply by 1.4826 for notmal distribution
			z = diff_med / scoremad
			out2 = (np.where(z > threshold)[0])
			zcol =  np.delete(zcol, out2)
			zmagv =  np.delete(zmagv, out2)
			ecol =  np.delete(ecol, out2)
			ecolv =  np.delete(ecolv, out2)
			times=times+1

		#wrt mean
		# ~threshold = 3
		# ~z = np.abs(stats.zscore(np.array(cgood)[ici]))
		# ~out = (np.where(z > threshold)[0])
		# ~zcol =  np.delete(np.array(cgood)[ici], out)
		# ~zmagv =  np.delete(np.array(vgood)[ici], out)

		# ~times = 0
		# ~while times < 1:
			# ~z = np.abs(stats.zscore(zcol))
			# ~out = (np.where(z > threshold)[0])
			# ~zcol =  np.delete(zcol, out)
			# ~zmagv =  np.delete(zmagv, out)
			# ~times = times + 1
			# ~scoremad = np.std(zcol)

		if len(ici) > 1:
			tp = np.where((zmagv >= np.min(zmagv)) & (zmagv <= np.min(zmagv)+step/4.))[0]
			tps = np.median(zcol[tp])
			bp = np.where((zmagv <= np.max(zmagv)) & (zmagv >= np.max(zmagv)-step/4.))[0]
			bps = np.median(zcol[bp])
			
			x_axis_real_length = 21.7 #cm
			x_axis_lim = 3.5 #mag 
			y_axis_real_length = 11 #cm
			y_axis_lim = 16 #mg

			# ~#get the cm equivalent to 1 mag in x or y
			norm_x = x_axis_real_length/x_axis_lim
			norm_y = y_axis_real_length/y_axis_lim

			lat = step * norm_y
			lon = (bps - tps) * norm_x
			# ~lat = step
			# ~lon = (bps - tps)
			dist = np.sqrt((lat)**2 + (lon)**2)
			cos = lat/dist
			angle = np.degrees(math.acos(cos))
			# ~print(c,len(ici), cos, angle)
		else:
			pass


		# ~plt.figure()
		# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
		# ~plt.scatter(np.array(cgood)[ici],np.array(vgood)[ici], marker='.',s=10, color='b', label='data')
		# ~plt.scatter(zcol,zmagv, marker='.',s=10, color='r', label='data')
		# ~plt.axvline(np.median(np.array(cgood)[ici]),c='r')
		# ~plt.axvline(np.mean(np.array(cgood)[ici]),c='c')
		# ~plt.xlim(-0.5,3)
		# ~plt.ylim(26,10)
		# ~plt.legend(loc='upper right', fontsize = 16)
		# ~plt.xlabel('F606W - F814W', fontsize = 16)
		# ~plt.ylabel('F606W', fontsize = 16)
		# ~plt.title(clus_nb, fontsize = 16)
		# ~plt.show()
		# ~plt.close()
		# ~kill

		if len(ici) == 1:
			ccenter[c] =np.median(zcol)
			# ~vcenter[c] =centergood[c]
			# ~ccenter[c] =np.mean(zcol)
			errcenter[c] =np.array(errgood)[ici]
			errcenterv[c] =np.array(errgoodv)[ici]
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
		elif len(ici) == 2:
			ccenter[c] =np.median(zcol)
			# ~errcenter[c] = np.std(zcol)
			errcenter[c] = scoremad*cos
			# ~vcenter[c] =centergood[c]
			# ~errcenter[c] =1.2533*np.std(zcol)*cos
			# ~ccenter[c] =np.mean(zcol)
			# ~errcenter[c] = np.std(zcol)*cos
			errcenterv[c] = np.std(zmagv)
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
		elif len(ici) > 2 and len(ici) < 50:
			ccenter[c] =np.median(zcol)
			# ~errcenter[c] =1.2533*np.std(zcol)*cos
			# ~errcenter[c] = scoremad*cos
			# ~vcenter[c] =centergood[c]
			# ~ccenter[c] =np.mean(zcol)
			if np.std(zcol) == 0:
				errcenter[c] =np.mean(np.array(errgood)[ici])
				print('std nul')
			else:
				# ~errcenter[c] =np.std(zcol)*cos
				errcenter[c] = scoremad * cos
				errcenterv[c] =1.2533*np.std(zmagv)
			vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
			# ~ccenter[c] =np.mean(zcol)
			# ~errcenter[c] = np.std(zcol)
			# ~vcenter[c] =np.median(zmagv)
			# ~vcenter[c] =np.median(zmagv)
								
		elif len(ici) >= 50:
			# ~plt.scatter(cgood[ici], vgood[ici], label='stars', c='grey')
			# ~plt.scatter(zcol, zmagv, label='stars', c='lightblue')
			# ~plt.plot([tps,bps], [np.min(zmagv), np.max(zmagv)], c='r', label='orientation', lw=2)
			# ~plt.scatter(np.mean(zcol), np.mean(zmagv), c='b', marker='o')
			# ~plt.xlabel('F606W - F814W', fontsize = 24)
			# ~plt.ylabel('F606W', fontsize = 24)
			# ~plt.title('IC4499', fontsize = 24)
			# ~plt.axhline(bingood[c], label='bin edge', alpha=0.5)
			# ~plt.axhline(bingood[c+1], alpha=0.5)
			# ~plt.ylim(bingood[c+1]+0.02,bingood[c]-0.02)
			# ~plt.tick_params(labelsize=16)
			# ~plt.show()
			# ~plt.close()

			
			ccenter[c] =np.median(zcol)
			# ~errcenter[c] = np.std(zcol)
			# ~errcenter[c] = scoremad * cos
			# ~vcenter[c] =np.median(zmagv)
			# ~errcenter[c] = scoremad*cos
			# ~errcenter[c] =1.2533*np.std(zcol)*cos
			# ~vcenter[c] =centergood[c]
			# ~vcenter[c] =np.median(zmagv)
			# ~errcenter[c] =1.2533*np.std(zcol)	
			# ~ccenter[c] =np.mean(zcol)
			if np.std(zcol) == 0:
				errcenter[c] =np.mean(np.array(errgood)[ici])
				print('std nul')
			else:
				# ~errcenter[c] =np.std(zcol)*cos
				errcenter[c] = scoremad*cos
				errcenterv[c] =1.2533*np.std(zmagv)
			vcenter[c] =np.mean(zmagv)
			size_bin[c] = len(ici)

						

			# ~from scipy.stats import skewnorm
			# ~amp2, moy2, dev2 = skewnorm.fit(zcol, 1, loc=0.5, scale=0.05)
			# ~amp2, moy2, dev2 = skewnorm.fit(zcol, 10, loc=0.5, scale=0.05)
			# ~print(amp2, moy2, dev2)		

			# ~x = np.linspace(np.min(zcol), np.max(zcol), 50)
			# ~p = stats.skewnorm.pdf(x,amp2, moy2, dev2)#.rvs(100)
			# ~mean_skew = stats.skewnorm.mean(amp2, moy2, dev2)#.rvs(100)
			# ~median_skew = stats.skewnorm.median(amp2, moy2, dev2)#.rvs(100)
			# ~err_skew = stats.skewnorm.std(amp2, moy2, dev2)#.rvs(100)
			# ~ccenter[c] = moy2

			# ~med = np.median(zcol)
			# ~diff_med = np.abs(zcol - med)
			# ~errmed = np.median(diff_med) # multiply by 1.486 for notmal distribution
			# ~scoremad = errmed* 1.4826 # multiply by 1.4826 for notmal distribution
			# ~z = diff_med / scoremad
			# ~out2 = (np.where(z > threshold)[0])
			# ~zcol =  np.delete(zcol, out2)
			# ~zmagv =  np.delete(zmagv, out2)
			# ~times=times+1


			# ~plt.figure()
			# ~plt.hist(zcol, bins=50, density=True,color='grey')
			# ~plt.plot(x, p, 'k', linewidth=2)
			# ~plt.axvline(moy2, c='y')
			# ~plt.axvline(mean_skew, c='k')
			# ~plt.axvline(median_skew, c='b', linestyle='--')
			# ~plt.axvline(np.median(np.array(cgood)[ici]),c='r')
			# ~plt.axvline(np.mean(np.array(cgood)[ici]),c='c')
			# ~plt.show()
			# ~plt.close()
			# ~kill


			# ~print(len(np.where(zcol < med)[0]))
			# ~print(len(np.where(zcol > med)[0]))

			# ~plt.figure()
			# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
			# ~plt.scatter(np.array(cgood)[ici],np.array(vgood)[ici], marker='.',s=10, color='b', label='data')
			# ~plt.scatter(zcol,zmagv, marker='.',s=10, color='r', label='data')
			# ~plt.scatter(zcol[np.where(zcol < med)[0]],zmagv[np.where(zcol < med)[0]], marker='.',s=10, color='c', label='data')
			# ~plt.axvline(mean_skew, c='k')
			# ~plt.axvline(median_skew, c='b', linestyle='--')
			# ~plt.axvline(med, c='b')
			# ~plt.axvline(np.median(np.array(cgood)[ici]),c='r')
			# ~plt.axvline(np.mean(np.array(cgood)[ici]),c='c')
			# ~plt.xlim(-0.5,3)
			# ~plt.ylim(26,10)
			# ~plt.legend(loc='upper right', fontsize = 16)
			# ~plt.xlabel('F606W - F814W', fontsize = 16)
			# ~plt.ylabel('F606W', fontsize = 16)
			# ~plt.title(clus_nb, fontsize = 16)
			# ~plt.show()
			# ~plt.close()
			# ~kill

	return vcenter, ccenter, errcenter, size_bin, bingood, errcenterv
	
def angle_correction(vcenter, ccenter, errcenter, size_bin):


	for i in range(len(vcenter)):
		# ~print(len(vcenter),i, size_bin[i])
		if size_bin[i] < 100:
			continue
			


		x_axis_real_length = 21.7 #cm
		x_axis_lim = 3.5 #mag 
		y_axis_real_length = 11 #cm
		y_axis_lim = 16 #mg

		#get the cm equivalent to 1 mag in x or y
		norm_x = x_axis_real_length/x_axis_lim
		norm_y = y_axis_real_length/y_axis_lim


		if i==0:
			lat = (vcenter[i+1] - vcenter[i]) * norm_y
			lon = (ccenter[i+1] - ccenter[i]) * norm_x
			dist = np.sqrt((lat)**2 + (lon)**2)
			cos = lat/dist
			angle = math.acos(cos)
			# ~print(np.degrees(angle), cos)
			# ~kill
		
			# ~plt.figure()
			# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
			# ~plt.scatter(ccenter[i:i+2],vcenter[i:i+2], marker='.',s=10, color='r', label='data')
			# ~plt.axvline(ccenter[i])
			# ~plt.plot(ccenter[[i,i+1]],vcenter[[i,i+1]], c='y')
			# ~plt.xlim(-0.5,3)
			# ~plt.ylim(26,10)
			# ~plt.legend(loc='upper right', fontsize = 16)
			# ~plt.xlabel('F606W - F814W', fontsize = 16)
			# ~plt.ylabel('F606W', fontsize = 16)
			# ~plt.title(clus_nb, fontsize = 16)
			# ~plt.show()
			# ~plt.close()
			# ~kill
			
		elif i>0 and i<len(vcenter)-1:
			lat = (vcenter[i+1] - vcenter[i-1]) * norm_y
			lon = (ccenter[i+1] - ccenter[i-1]) * norm_x
			dist = np.sqrt((lat)**2 + (lon)**2)
			cos = lat/dist
			angle = math.acos(cos)
			# ~print(np.degrees(angle), cos)
		
			# ~plt.figure()
			# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
			# ~plt.scatter(ccenter[i-1:i+2],vcenter[i-1:i+2], marker='.',s=10, color='r', label='data')
			# ~plt.axvline(ccenter[i-1])
			# ~plt.plot(ccenter[[i-1,i,i+1]],vcenter[[i-1,i,i+1]], c='y')
			# ~plt.xlim(-0.5,3)
			# ~plt.ylim(26,10)
			# ~plt.legend(loc='upper right', fontsize = 16)
			# ~plt.xlabel('F606W - F814W', fontsize = 16)
			# ~plt.ylabel('F606W', fontsize = 16)
			# ~plt.title(clus_nb, fontsize = 16)
			# ~plt.show()
			# ~plt.close()
			
		elif i == len(vcenter)-1:
			lat = (vcenter[i] - vcenter[i-2]) * norm_y
			lon = (ccenter[i] - ccenter[i-2]) * norm_x
			dist = np.sqrt((lat)**2 + (lon)**2)
			cos = lat/dist
			angle = math.acos(cos)
			# ~print(np.degrees(angle), cos)
		
			# ~plt.figure()
			# ~plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
			# ~plt.scatter(ccenter[i-1:i+2],vcenter[i-1:i+2], marker='.',s=10, color='r', label='data')
			# ~plt.axvline(ccenter[i-1])
			# ~plt.plot(ccenter[[i-1,i,i+1]],vcenter[[i-1,i,i+1]], c='y')
			# ~plt.xlim(-0.5,3)
			# ~plt.ylim(26,10)
			# ~plt.legend(loc='upper right', fontsize = 16)
			# ~plt.xlabel('F606W - F814W', fontsize = 16)
			# ~plt.ylabel('F606W', fontsize = 16)
			# ~plt.title(clus_nb, fontsize = 16)
			# ~plt.show()
			# ~plt.close()

		errcenter[i] = errcenter[i]*cos

	return vcenter, ccenter, errcenter, size_bin

#~ @profile
def default_beta_ladder(ndim, ntemps=None, Tmax=None):
	"""Returns a ladder of :math:`\beta \equiv 1/T` with temperatures
	geometrically spaced with spacing chosen so that a Gaussian
	posterior would have a 0.25 temperature swap acceptance rate.

	:param ndim:
		The number of dimensions in the parameter space.

	:param ntemps: (optional)
		If set, the number of temperatures to use.  If ``None``, the
		``Tmax`` argument must be given, and the number of
		temperatures is chosen so that the highest temperature is
		greater than ``Tmax``.

	:param Tmax: (optional)
		If ``ntemps`` is not given, this argument controls the number
		of temperatures.  Temperatures are chosen according to the
		spacing criteria until the maximum temperature exceeds
		``Tmax``

	"""
	tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
					  2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
					  2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
					  1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
					  1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
					  1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
					  1.51901, 1.50881, 1.49916, 1.49, 1.4813,
					  1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
					  1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
					  1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
					  1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
					  1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
					  1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
					  1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
					  1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
					  1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
					  1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
					  1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
					  1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
					  1.26579, 1.26424, 1.26271, 1.26121,
					  1.25973])
	dmax = tstep.shape[0]

	if ndim > dmax:
		# An approximation to the temperature step at large
		# dimension
		tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
	else:
		tstep = tstep[ndim-1]

	if ntemps is None and Tmax is None:
		raise ValueError('must specify one of ``ntemps`` and ``Tmax``')
	elif ntemps is None:
		ntemps = int(np.log(Tmax)/np.log(tstep)+2)

	return np.exp(np.linspace(0, -(ntemps-1)*np.log(tstep), ntemps))
	
########################################################################
########################################################################
### define global variables
start1 = time.time()
iteration = 0 
ncpu = 8 # number of cpu requested
#~ ncpu = len(os.sched_getaffinity(0)) # number of cpu requested aganice
#~ print(ncpu)
#~ kill
#~ ncpu = int(os.environ["cpu_num"]) # number of cpu requested hipatia
glc = int(input("what is your cluster number? "))
# glc = int(os.environ["SLURM_ARRAY_TASK_ID"]) # aganice
#~ glc = int(os.environ["PBS_ARRAYID"])  # hipatia
print("the chosen cluster is {}".format(glc))
clus_nb, Age, metal, distance, Abs, afe_init, errdist  = cluster(glc)
print(clus_nb, Age, metal, distance, Abs, afe_init, errdist)
photo_v, err_v, photo_i, color, err_color, nmv, nmi, longueur = photometry()
# ~kill

# ~print(np.mean(err_color))
# ~kill
#~ plt.figure()
#~ plt.hist(np.random.normal(Age, 1e-1,60), 50)
#~ plt.show()
#~ kill
#~ s= np.random.uniform(low=[10.0], high=[10.2], size=(500))
#~ count, bins, ignored = plt.hist(s, 15)
#~ plt.show()
#~ kill

### read file with all the coeefficients
file_coeff = np.loadtxt('coeff_gcpy.txt', skiprows=1)
# ~rescale = np.loadtxt('rescale_ig.csv',delimiter=',')
rescale = np.loadtxt('rescale_ig_v3.csv',delimiter=',')
maskbin = np.loadtxt('maskbin.csv',delimiter=',')

#--------------------------------------------------------------
#--------------------------------------------------------------
### DEFINE THE STELLAR MODEL and PARAMETERS FOR MCMC
thin = 1
T0 = 1
T1 = 1000
posnum = 1
ite = 10000
nwalkers = 30
# ~model = 'mist'
model = 'dar'
#~ #----------------
if model == 'mist':
	version = str(10)
	print(model)
	#~ pop = [Age, metal, distance, Abs]
	pop = [Age, metal, distance, Abs,0.0]
	ndim = len(pop)
	from isochrones.mist import MIST_Isochrone
	mist = MIST_Isochrone()
	lim_model = 5
	mag_v, mag_i, Color_iso, eep_first = iso_mag(Age, metal, distance, Abs)
	#~ mag_v, mag_i, Color_iso, eep_first = iso_mag(Age, metal+0.22, 4430, 0.093)


	
	# mag_vbf, mag_i, Color_isobf = iso_mag(Age_bf[glc], metal_bf[glc], distance_bf[glc], Abs_bf[glc])
	#print(Age, Age_bf[glc])

	ct = [605-int(eep_first)]

	# ~plt.figure()
	# ~plt.plot(Color_iso[:ct[0]-1], mag_v[:ct[0]-1])
	# ~plt.xlim(0.5,1.9)
	# ~plt.ylim(15,30)
	# ~plt.gca().invert_yaxis()
	# ~plt.show()
	# ~plt.close()
	# ~kill

	fmag_ini = interpolate.interp1d(mag_v[:ct[0]-1], Color_iso[:ct[0]-1], 'nearest',fill_value="extrapolate")

	if Abs-0.1>0 and distance-2000 >0:
		pos = np.random.uniform(low=[Age -0.1, metal-0.1, distance-2000, Abs-0.1, afe_init-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1, afe_init+0.1],
		size=(nwalkers, ndim))
	elif Abs-0.1 < 0 and distance-2000 >0:
		pos = np.random.uniform(low=[Age -0.1, metal-0.1, distance-2000, 0.001, afe_init-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1,  afe_init+0.1],
		size=(nwalkers, ndim))
	elif Abs-0.1>0 and distance-2000 < 0:
		pos = np.random.uniform(low=[Age -0.1, metal-0.1, 0.001, Abs-0.1,afe_init-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1, afe_init+0.1],
		size=(nwalkers, ndim))
	else:
		pos = np.random.uniform(low=[Age -0.1, metal-0.1, 0, 0.001, afe_init-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1,  afe_init+0.1],
		size=(nwalkers, ndim))
	#~ if Abs-0.1>0 and distance-2000 >0:
		#~ pos = np.random.uniform(low=[Age -0.1, metal-0.1, distance-2000, Abs-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1],
		#~ size=(nwalkers, ndim))
	#~ elif Abs-0.1 < 0 and distance-2000 >0:
		#~ pos = np.random.uniform(low=[Age -0.1, metal-0.1, distance-2000, 0.001], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1],
		#~ size=(nwalkers, ndim))
	#~ elif Abs-0.1>0 and distance-2000 < 0:
		#~ pos = np.random.uniform(low=[Age -0.1, metal-0.1, 0.001, Abs-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1],
		#~ size=(nwalkers, ndim))
	#~ else:
		#~ pos = np.random.uniform(low=[Age -0.1, metal-0.1, 0, 0.001], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1],
		#~ size=(nwalkers, ndim))


	#~ binc = 200
	#~ niso = int((1525-600)/25)
	#~ col = np.zeros((binc,niso))
	#~ mag = np.zeros((binc,niso))
	#~ magtest= np.linspace(10,26,binc)
	#~ import cmasher as cmr
	#~ cm = cmr.ember
	#~ norm = colors.Normalize(vmin=6,vmax=15)
	#~ s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	#~ s_m.set_array([])
	#~ import matplotlib.gridspec as gridspec
	#~ gs_in = gridspec.GridSpec(2, 2,hspace=0.3,height_ratios=[10,1],
	#~ width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
	#~ ax = plt.subplot(gs_in[0,0])
	#~ ax1 = plt.subplot(gs_in[0,1])
	#~ for ind,a in enumerate(range(600,1525,25)):
		#~ a = a/100. 
		#~ ag= np.log10(a*1e9 )  

		#~ print(ag, Age)
		#~ mag_v, mag_i, Color_iso, eep_first = iso_mag(float(ag), metal, distance, Abs)
		#~ fmag_ini = interpolate.interp1d(mag_v[:ct[0]-1], Color_iso[:ct[0]-1], 'nearest',fill_value="extrapolate")
		#~ mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(np.log10(15e9 ) , metal, distance, Abs)
		#~ fmag_iniref = interpolate.interp1d(mag_vref[:ct[0]-1], Color_isoref[:ct[0]-1], 'nearest',fill_value="extrapolate")

		#~ col[:,ind]= fmag_ini(magtest)
		#~ mag[:,ind]= magtest

		#~ ax.plot(fmag_ini(magtest),magtest, color=s_m.to_rgba(a),lw=2)
		#~ ax1.plot(fmag_iniref(magtest)/fmag_ini(magtest),magtest, color=s_m.to_rgba(a),lw=2)
		
	#~ ax.set_xlim(-0.50,3)
	#~ ax.set_ylim(30,10)
	#~ ax1.set_ylim(30,10)
	#~ ax1.set_yticklabels([])
	#~ ax.set_xlabel('F606W - F814W', fontsize = 16)
	#~ ax.set_ylabel('F606W', fontsize = 16)
	#~ ax1.set_xlabel(r'$\mathrm{Color_{15Gyr}}$ / $\mathrm{Color_{Age}}$', fontsize = 16)
	#~ ax.set_xticks([0.4,0.6,0.8,1.0,1.2])
	#~ ax.set_xticklabels(['0.4','0.6','0.8','1.0','1.2'])
	#~ ax1.set_xticks([1,1.25,1.5,2])
	#~ ax1.set_xticklabels(['1','1.25','1.5','2'])
	#~ ax.text(0.5,16,r'Age',va='center',fontsize=23,alpha=1.,
	#~ bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
	#~ cbar = plt.colorbar(s_m,cax=plt.subplot(gs_in[1,:]),orientation='horizontal')
	#~ cbar.ax.set_xlabel('Age [Gyr]', fontsize = 16)
	#~ ax.grid()
	#~ ax1.grid()
	#~ plt.show()  
	#~ plt.close()


	#~ binc = 100
	#~ niso = int((-55 -(-250))/5)
	#~ magtest= np.linspace(10,26,binc)
	#~ col = np.zeros((binc,niso))
	#~ mag = np.zeros((binc,niso))
	#~ import cmasher as cmr
	#~ cm = cmr.ember
	#~ norm = colors.Normalize(vmin=-2.5,vmax=-0.5)
	#~ s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	#~ s_m.set_array([])
	#~ import matplotlib.gridspec as gridspec
	#~ gs_in = gridspec.GridSpec(2, 2,hspace=0.3,height_ratios=[10,1],
	#~ width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
	#~ ax = plt.subplot(gs_in[0,0])
	#~ ax1 = plt.subplot(gs_in[0,1])
	#~ for ind,a in enumerate(range(-250, -55, 5)):
		#~ met= a/100. 

		#~ print(met)
		#~ mag_v, mag_i, Color_iso, eep_first = iso_mag(Age, met, distance, Abs)
		#~ fmag_ini = interpolate.interp1d(mag_v[:ct[0]-1], Color_iso[:ct[0]-1], 'nearest',fill_value="extrapolate")
		#~ mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(Age , -2.5, distance, Abs)
		#~ fmag_iniref = interpolate.interp1d(mag_vref[:ct[0]-1], Color_isoref[:ct[0]-1], 'nearest',fill_value="extrapolate")

		#~ col[:,ind]= fmag_ini(magtest)
		#~ mag[:,ind]= magtest
		#~ ax.plot(fmag_ini(magtest),magtest, color=s_m.to_rgba(met))
		#~ ax1.plot(fmag_ini(magtest)/fmag_iniref(magtest),magtest, color=s_m.to_rgba(met),lw=2)

	#~ ax.set_xlim(-0.50,3)
	#~ ax.set_ylim(30,10)
	#~ ax1.set_ylim(30,10)
	#~ ax1.set_yticklabels([])
	#~ ax.set_xlabel('F606W - F814W', fontsize = 16)
	#~ ax.set_ylabel('F606W', fontsize = 16)
	#~ ax1.set_xlabel(r'$  \mathrm{Color_{Metallicity}}$ / $\mathrm{Color_{[Fe/H]=-2.5}}$', fontsize = 16)
	#~ ax.set_xticks([0.6,0.8,1.0,1.2])
	#~ ax.set_xticklabels(['0.6','0.8','1.0','1.2'])
	#~ ax1.set_xticks([1,1.25,1.5])
	#~ ax1.set_xticklabels(['1','1.25','1.5'])
	#~ ax.text(0.56,17,r'Metallicity',va='center',fontsize=23,alpha=1.,
	#~ bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
	#~ cbar = plt.colorbar(s_m,cax=plt.subplot(gs_in[1,:]),orientation='horizontal')
	#~ cbar.ax.set_xlabel('Metallicity', fontsize = 16)
	#~ ax.grid()
	#~ ax1.grid()
	#~ plt.show()  
	#~ plt.close()
	#~ kill
#--------------
elif model == 'dar':
	version = str(15)
	helium_y = ''
	print(model)
	pop = [Age, metal, distance, Abs, afe_init]
	ndim = len(pop)
	ntemps = 1
	lim_model = 5
	betas = default_beta_ladder(ndim,ntemps=1)
	#~ #betas = default_beta_ladder(ndim,Tmax=T1)
	#~ #ntemps = int(len(default_beta_ladder(ndim,Tmax=T1)))
	from isochrones.dartmouth import Dartmouth_FastIsochrone
	darm2 = Dartmouth_FastIsochrone(afe='afem2', y=helium_y)
	darp0 = Dartmouth_FastIsochrone(afe='afep0', y=helium_y)
	darp2 = Dartmouth_FastIsochrone(afe='afep2', y=helium_y)
	darp4 = Dartmouth_FastIsochrone(afe='afep4', y=helium_y)
	darp6 = Dartmouth_FastIsochrone(afe='afep6', y=helium_y)
	darp8 = Dartmouth_FastIsochrone(afe='afep8', y=helium_y)

	# ~mag_v, mag_i, Color_iso, eep_first = iso_mag(Age, metal, distance, Abs, afe_init)
	afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8]
	bf = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_15_dar.txt', usecols=(2,5,8,11,14))
	if glc == 27:
		mag_v, mag_i, Color_iso, eep_first = iso_mag(np.log10(12.8e9),-1.54, 19680, 0.64, 0.0)
	elif glc < 27:
		afe_max = afe_values[np.searchsorted(afe_values, bf[glc,4])]
		afe_min = afe_values[np.searchsorted(afe_values, bf[glc,4])-1]
		mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(np.log10(bf[glc,0]*1e9), bf[glc,1], bf[glc,2], bf[glc,3], afe_min)
		mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(np.log10(bf[glc,0]*1e9), bf[glc,1], bf[glc,2], bf[glc,3], afe_max)
		lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
		mag_v = (mag_v1_min[:lpp]*(afe_max - bf[glc,4]) + mag_v1_max[:lpp]*(bf[glc,4] - afe_min)) / (afe_max - afe_min)
		Color_iso = (Color_iso1_min[:lpp]*(afe_max - bf[glc,4]) + Color_iso1_max[:lpp]*(bf[glc,4] - afe_min)) / (afe_max - afe_min)
	elif glc > 27:
		afe_max = afe_values[np.searchsorted(afe_values, bf[glc-1,4])]
		afe_min = afe_values[np.searchsorted(afe_values, bf[glc-1,4])-1]
		mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(np.log10(bf[glc-1,0]*1e9), bf[glc-1,1], bf[glc-1,2], bf[glc-1,3], afe_min)
		mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(np.log10(bf[glc-1,0]*1e9), bf[glc-1,1], bf[glc-1,2], bf[glc-1,3], afe_max)
		lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
		mag_v = (mag_v1_min[:lpp]*(afe_max - bf[glc-1,4]) + mag_v1_max[:lpp]*(bf[glc-1,4] - afe_min)) / (afe_max - afe_min)
		Color_iso = (Color_iso1_min[:lpp]*(afe_max - bf[glc-1,4]) + Color_iso1_max[:lpp]*(bf[glc-1,4] - afe_min)) / (afe_max - afe_min)
	mag_vy, mag_iy, Color_isoy, eep_firsty = iso_mag(np.log10((bf[glc-1,0])*1e9), bf[glc-1,1]-0.1, bf[glc-1,2], bf[glc-1,3], 0.2)
	mag_vz, mag_iz, Color_isoz, eep_firstz = iso_mag(np.log10((bf[glc-1,0])*1e9), bf[glc-1,1]+0.1, bf[glc-1,2], bf[glc-1,3], 0.2)

	# ~print(afe_min, afe_max, bf[glc,4])
	# ~plt.figure()
	# ~plt.plot(Color_iso, mag_v)
	# ~plt.xlim(0.5,1.9)
	# ~plt.ylim(15,30)
	# ~plt.gca().invert_yaxis()
	# ~plt.show()
	# ~plt.close()
	# ~kill

	
	
	fmag_ini = interpolate.interp1d(mag_v, Color_iso, 'nearest',fill_value="extrapolate")


	#~ binc = 200
	#~ niso = int((1510-600)/10)
	#~ col = np.zeros((binc,niso))
	#~ mag = np.zeros((binc,niso))
	#~ magtest= np.linspace(14,26,binc)
	#~ import cmasher as cmr
	#~ cm = cmr.ember
	#~ norm = colors.Normalize(vmin=6,vmax=15)
	#~ s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	#~ s_m.set_array([])
	#~ import matplotlib.gridspec as gridspec
	#~ gs_in = gridspec.GridSpec(2, 2,hspace=0.5,height_ratios=[10,1],
	#~ width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
	#~ ax = plt.subplot(gs_in[0,0])
	#~ ax1 = plt.subplot(gs_in[0,1])
	#~ for ind,a in enumerate(range(600,1510,10)):
		#~ a = a/100. 
		#~ ag= np.log10(a*1e9 )  

		#~ print(ag, Age)
		#~ mag_v, mag_i, Color_iso, eep_first = iso_mag(float(ag), metal, distance, Abs, afe_init)
		#~ fmag_ini = interpolate.interp1d(mag_v, Color_iso, 'nearest',fill_value="extrapolate")
		#~ mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(np.log10(15e9 ) , metal, distance, Abs, afe_init)
		#~ fmag_iniref = interpolate.interp1d(mag_vref, Color_isoref, 'nearest',fill_value="extrapolate")

		#~ col[:,ind]= fmag_ini(magtest)
		#~ mag[:,ind]= magtest

		#~ ax.plot(fmag_ini(magtest),magtest, color=s_m.to_rgba(a),lw=2)
		#~ ax1.plot(fmag_iniref(magtest)/fmag_ini(magtest),magtest, color=s_m.to_rgba(a),lw=2)
		
	#~ ax.set_xlim(0.4,1.5)
	#~ ax.set_ylim(26,16)
	#~ ax1.set_xlim(0.9,1.7)
	#~ ax1.set_ylim(26,16)
	#~ ax1.set_yticklabels([])
	#~ ax.set_xlabel('F606W - F814W', fontsize = 16)
	#~ ax.set_ylabel('F606W', fontsize = 16)
	#~ #ax1.set_xlabel(r'$\mathrm{Color_{15Gyr}}$ / $\mathrm{Color_{Age}}$', fontsize = 16)
	#~ ax1.set_xlabel(r'$\frac{\mathrm{(F606W-F814W)(Age=15 Gyr)}}{\mathrm{(F606W-F814W)(Age)}}$', fontsize = 18)
	#~ ax.set_xticks([0.4,0.6,0.8,1.0,1.2])
	#~ ax.set_xticklabels(['0.4','0.6','0.8','1.0','1.2'])
	#~ ax1.set_xticks([1,1.25,1.5,1.7])
	#~ ax1.set_xticklabels(['1','1.25','1.5','1.7'])
	#~ ax.tick_params(labelsize=16)
	#~ ax1.tick_params(labelsize=16)
	#~ ax.text(0.56,17,r'Age',va='center',fontsize=23,alpha=1.,
	#~ bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
	#~ cbar = plt.colorbar(s_m,cax=plt.subplot(gs_in[1,:]),orientation='horizontal')
	#~ cbar.ax.set_xlabel('Age [Gyr]', fontsize = 16)
	#~ cbar.ax.tick_params(labelsize=16)
	#~ ax.grid()
	#~ ax1.grid()
	#~ plt.show()  
	#~ plt.close()
	#~ kill

	#~ binc = 200
	#~ niso = int((-48 -(-248))/2)
	#~ magtest= np.linspace(14,26,binc)
	#~ col = np.zeros((binc,niso))
	#~ mag = np.zeros((binc,niso))
	#~ import cmasher as cmr
	#~ cm = cmr.ember
	#~ norm = colors.Normalize(vmin=-2.5,vmax=-0.5)
	#~ s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	#~ s_m.set_array([])
	#~ import matplotlib.gridspec as gridspec
	#~ gs_in = gridspec.GridSpec(2, 2,hspace=0.5,height_ratios=[10,1],
	#~ width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
	#~ ax = plt.subplot(gs_in[0,0])
	#~ ax1 = plt.subplot(gs_in[0,1])
	#~ for ind,a in enumerate(range(-248, -48, 2)):
		#~ met= a/100. 

		#~ print(met)
		#~ mag_v, mag_i, Color_iso, eep_first = iso_mag(Age, met, distance, Abs, afe_init)
		#~ fmag_ini = interpolate.interp1d(mag_v, Color_iso, 'nearest',fill_value="extrapolate")
		#~ mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(Age , -2.48, distance, Abs, afe_init)
		#~ mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(Age , -0.5, distance, Abs, afe_init)
		#~ fmag_iniref = interpolate.interp1d(mag_vref, Color_isoref, 'nearest',fill_value="extrapolate")

		#~ col[:,ind]= fmag_ini(magtest)
		#~ mag[:,ind]= magtest
		#~ ax.plot(fmag_ini(magtest),magtest, color=s_m.to_rgba(met))
		#~ ax1.plot(fmag_iniref(magtest)/fmag_ini(magtest),magtest, color=s_m.to_rgba(met),lw=2)


	#~ ax.set_xlim(0.4,1.5)
	#~ ax1.set_xlim(0.9,1.7)
	#~ ax.set_ylim(26,16)
	#~ ax1.set_ylim(26,16)
	#~ ax1.set_yticklabels([])
	#~ ax.set_xlabel('F606W - F814W', fontsize = 16)
	#~ ax.set_ylabel('F606W', fontsize = 16)
	#ax1.set_xlabel(r'$  \mathrm{Color_{Metallicity}}$ / $\mathrm{Color_{[Fe/H]=-2.5}}$', fontsize = 16)
	#~ ax1.set_xlabel(r'$\frac{\mathrm{(F606W-F814W)([Fe/H]=-0.5))}}{\mathrm{(F606W-F814W)([Fe/H])}}$', fontsize = 18)
	#~ ax.set_xticks([0.4,0.6,0.8,1.0,1.2])
	#~ ax.set_xticklabels(['0.4','0.6','0.8','1.0','1.2'])
	#~ ax1.set_xticks([1,1.25,1.5,1.7])
	#~ ax1.set_xticklabels(['1','1.25','1.5','1.7'])
	#~ ax.tick_params(labelsize=16)
	#~ ax1.tick_params(labelsize=16)
	#~ ax.text(0.56,17,r'[Fe/H]',va='center',fontsize=23,alpha=1.,
	#~ bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
	#~ cbar = plt.colorbar(s_m,cax=plt.subplot(gs_in[1,:]),orientation='horizontal')
	#~ cbar.ax.set_xlabel('[Fe/H]', fontsize = 16)
	#~ cbar.ax.tick_params(labelsize=16)
	#~ ax.grid()
	#~ ax1.grid()
	#~ plt.show()  
	#~ plt.close()
	#~ kill
	
	#~ binc = 200
	#~ niso = int((85+20)/1)
	#~ magtest= np.linspace(14,26,binc)
	#~ col = np.zeros((binc,niso))
	#~ mag = np.zeros((binc,niso))
	#~ import cmasher as cmr
	#~ cm = cmr.ember
	#~ norm = colors.Normalize(vmin=-0.2,vmax=0.8)
	#~ s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
	#~ s_m.set_array([])
	#~ import matplotlib.gridspec as gridspec
	#~ gs_in = gridspec.GridSpec(2, 2,hspace=0.5,height_ratios=[10,1],
	#~ width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
	#~ ax = plt.subplot(gs_in[0,0])
	#~ ax1 = plt.subplot(gs_in[0,1])
	#~ for ind,a in enumerate(range(-20, 81, 1)):
		#~ afe = a/100.

		#~ afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8] 
		#~ afe_max = afe_values[np.searchsorted(afe_values, afe)]
		#~ afe_min = afe_values[np.searchsorted(afe_values, afe)-1]
		
		#~ mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(Age, metal, distance, Abs, afe_min)
		#~ mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(Age, metal, distance, Abs, afe_max)
		#~ lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
		
		#~ mag_v = (mag_v1_min[:lpp]*(afe_max - afe) + mag_v1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
		#~ Color_iso = (Color_iso1_min[:lpp]*(afe_max - afe) + Color_iso1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)

		#~ fmag_ini = interpolate.interp1d(mag_v, Color_iso, 'nearest',fill_value="extrapolate")
		#~ mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(Age , metal, distance, Abs, 0.8)
		#~ fmag_iniref = interpolate.interp1d(mag_vref, Color_isoref, 'nearest',fill_value="extrapolate")

		#~ col[:,ind]= fmag_ini(magtest)
		#~ mag[:,ind]= magtest
		#~ ax.plot(fmag_ini(magtest),magtest, color=s_m.to_rgba(afe))
		#~ ax1.plot(fmag_iniref(magtest)/fmag_ini(magtest),magtest, color=s_m.to_rgba(afe),lw=2)

	#~ ax.set_xlim(0.4,1.5)
	#~ ax1.set_xlim(0.9,1.7)
	#~ ax.set_ylim(26,16)
	#~ ax1.set_ylim(26,16)
	#~ ax1.set_yticklabels([])
	#~ ax.set_xlabel('F606W - F814W', fontsize = 16)
	#~ ax.set_ylabel('F606W', fontsize = 16)
	#ax1.set_xlabel(r'$  \mathrm{Color_{abundance}}$ / $\mathrm{Color_{[\alpha/Fe]=0.0}}$', fontsize = 16)
	#~ ax1.set_xlabel(r'$\frac{\mathrm{(F606W-F814W)([\alpha/Fe]=0.8)}}{\mathrm{(F606W-F814W)([\alpha/Fe]})}$', fontsize = 18)
	#~ ax.set_xticks([0.4,0.6,0.8,1.0,1.2])
	#~ ax.set_xticklabels(['0.4','0.6','0.8','1.0','1.2'])
	#~ ax1.set_xticks([1,1.25,1.5,1.7])
	#~ ax1.set_xticklabels(['1','1.25','1.5','1.7'])
	#~ ax1.tick_params(labelsize=16)
	#~ ax.tick_params(labelsize=16)
	#~ ax1.tick_params(labelsize=16)
	#~ ax.text(0.56,17,r'[$\alpha$/Fe]',va='center',fontsize=23,alpha=1.,
	#~ bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
	#~ cbar = plt.colorbar(s_m,cax=plt.subplot(gs_in[1,:]),orientation='horizontal')
	#~ cbar.ax.set_xlabel(r'[$\alpha$/Fe]', fontsize = 16)
	#~ cbar.ax.tick_params(labelsize=16)
	#~ ax.grid()
	#~ ax1.grid()
	#~ plt.show()  
	#~ plt.close()
	#~ kill


	# ~pos = np.random.uniform(low=[Age -0.01, -2.4, distance-1000, 0.01, 0.01], high=[Age +0.01, 0.1, distance+1000, 2.99, 0.78],
	# ~size=(nwalkers, ndim))
	if Abs-0.05>0 and distance-500 >0:
		pos = np.random.uniform(low=[Age-0.01, metal-0.05, distance-500, Abs-0.05, afe_init-0.05], high=[Age+0.01, metal+0.05, distance+500, Abs+0.05, afe_init+0.05],
		size=(nwalkers, ndim))
	elif Abs-0.05 < 0 and distance-500 >0:
		pos = np.random.uniform(low=[Age-0.01, metal-0.05, distance-500, 0.001, afe_init-0.05], high=[Age+0.01, metal+0.05, distance+500, Abs+0.05,  afe_init+0.05],
		size=(nwalkers, ndim))
	elif Abs-0.05>0 and distance-500 < 0:
		pos = np.random.uniform(low=[Age-0.01, metal-0.05, 0.001, Abs-0.05,afe_init-0.05], high=[Age+0.01, metal+0.05, distance+500, Abs+0.05, afe_init+0.05],
		size=(nwalkers, ndim))
	else:
		pos = np.random.uniform(low=[Age-0.01, metal-0.05, 0, 0.001, afe_init-0.05], high=[Age+0.01, metal+0.05, distance+500, Abs+0.05,  afe_init+0.05],
		size=(nwalkers, ndim))


# ~kill

#--------------------------------------------------------------
#--------------------------------------------------------------

### find the plateau in color which corresponds to the turnoff point
if model=='dar':
	col_turn = [np.argmin((Color_iso))]
	#~ col_turn = [np.min(argrelextrema(Color_iso, np.less, order=20))]
else:
	#~ col_turn = np.min(argrelextrema(Color_iso, np.less, order=20))
	col_turn = [np.argmin((Color_iso[:ct[0]]))]
	#~ col_turn = [454-int(eep_first)]

#~ #ct1 = np.min(argrelextrema(Color_iso, np.greater, order=4))
top_x1 = Color_iso[col_turn]
top_y1 = mag_v[col_turn]
top_x = top_x1[0]
top_y = top_y1[0]
err_topy = (mag_v[col_turn[0]+1] - mag_v[col_turn[0]-1])/2.

#~ dd = np.where((photo_v > mag_v[col_turn[0]+1])&(photo_v < mag_v[col_turn[0]-1]))[0]
#~ top_x = np.median(color[dd])
#~ err_topx = np.std(color[dd])
### remove blue stragglers
#~ top_x, top_y, err_topy, err_topx = rem_outliers(photo_v, err_v, color, err_color)
#~ print(top_y)

#~ plt.figure()
#~ plt.scatter(color,photo_v, marker='.', s=10, color='grey', label='data')
#~ plt.scatter(Color_iso[:ct[0]-1],mag_v[:ct[0]-1], marker='.', s=10, color='b', label='initial guess MIST')
#~ plt.scatter(Color_iso,mag_v, marker='.', s=10, color='r', label='initial guess DSED')
#~ plt.xlim(-0.5,3)
#~ plt.ylim(26,10)
#~ plt.axvline(top_x, c='r')
#~ plt.axhline(top_y, c='r')
#~ plt.title(clus_nb+' '+str(glc), fontsize = 16)
#~ plt.grid()
#~ plt.show() 
#~ plt.close()
#~ kill

chunkbot = rescale[:,5]
#~ print((chunkbot[glc]))
if glc in [62]:
	step = 0.4
else:
	step = 0.2

rgb_lim = np.min(photo_v)
#~ rgb_lim = 16.5
# ~mag_lim1 = chunkbot[glc]-2
# ~mag_lim2 = chunkbot[glc] -1
mag_lim2 = chunkbot[glc]
mag_lim3 = min(chunkbot[glc] + lim_model, 26)
# ~mag_lim3 = 26

#~ above = np.where(photo_v < mag_lim3)[0]
#~ with open('/home/david/codes/GC/plots/table_number_'+ version +'_'+str(model)+'.txt', 'a+') as fid_file:
	#~ fid_file.write('%s & $ %d $ %d $ & $ %d $\n' % (clus_nb, len(photo_v), len(photo_v[above]),
	#~ len(photo_v[above])/len(photo_v)*100))
#~ fid_file.close()

#~ kill
### MAIN SEQUENCE
#binMS = np.geomspace(mag_lim2, mag_lim3,msbins)
#binMS = np.linspace(mag_lim2, mag_lim3, msbins)
binMS = np.array(binning_MS(mag_lim2, mag_lim3, photo_v, color))
bincenter = (binMS[:-1] + binMS[1:]) / 2 
#~ ### make a gaussian fit for ms stars
# ~binmid, gauss_mean, gauss_disp, starnum, non_zero = spread_color_x()
# ~bincenter = binmid
magv1, col1, errv1, errcol1 = spread_color_y(mag_lim3, mag_lim2)
magvuno = np.array(magv1)
if len(magvuno) > 0: 
	coluno = np.array(col1)
	errvuno = np.array(errv1)
	errcoluno = np.array(errcol1)
else:
	print("List is empty")

bincenter, gauss_mean, gauss_disp, starnum, bingood, errcenterv_gauss = way(magvuno, coluno, errcoluno, errvuno)

# ~bincenter, gauss_mean, gauss_disp, starnum = angle_correction(bincenter, gauss_mean, gauss_disp, starnum)

#~ plt.figure()
#~ ax1 = plt.subplot(211)
#~ ax1.scatter(color,photo_v, marker='.', s=10, color='grey', label='data')
#~ ax1.scatter(Color_iso,mag_v, marker='.',s=10, color='k', label='initial guess')
#~ ax1.scatter(Color_iso[ct], mag_v[ct], marker='o',s=15, c='r')
#~ ax1.scatter(Color_iso[454-int(eep_first)], mag_v[454-int(eep_first)], marker='o',s=15, c='g')
#~ ax1.scatter(Color_iso[col_turn], mag_v[col_turn], marker='o',s=15, c='c')
#~ ax1.set_xlim(-0.5,3)
#~ ax1.set_ylim(25,10)
#~ ax1.legend(loc='upper left', fontsize = 16)
#~ ax1.set_xlabel('F606W - F814W', fontsize = 16)
#~ ax1.set_ylabel('F606W', fontsize = 16)
#~ plt.title(clus_nb, fontsize = 16)
#~ ax2 = plt.subplot(212)
#~ ax2.axvline(ct, c='r')
#~ ax2.axvline([454-int(eep_first)], c='g')
#~ ax2.axvline(col_turn, c='c')
#~ ax2.plot(np.arange(mag_v.size), Color_iso, c='k')
#~ ax2.set_xlabel('EEP', fontsize =16)
#~ ax2.set_ylabel('Color', fontsize =16)
#~ plt.show()
#~ kill


bou1 = np.where(photo_v < mag_lim3)[0]
bou2 = np.where((photo_v < mag_lim2))[0]



values, base = np.histogram(photo_v, bins=100)
basecenter = (base[:-1] + base[1:]) / 2
lim1 = np.where(basecenter < mag_lim2)[0]
lim2 = np.where((basecenter < mag_lim3)&(basecenter > mag_lim2))[0]
lim3 = np.where((basecenter > mag_lim3))[0]


#~ plt.figure()
#~ cumulative = np.cumsum(values)
#~ print(len(base))
#~ print(len(values))
#~ print(len(cumulative))
#~ plt.plot(basecenter[:lim2[1]], cumulative[:lim2[1]], c='b', linewidth = 4)
#~ plt.plot(basecenter[lim2[0]:lim3[1]], cumulative[lim2[0]:lim3[1]], c='r', linewidth = 4)
#~ plt.plot(basecenter[lim3], cumulative[lim3], c='grey', linewidth = 4)
#~ plt.axvline(basecenter[lim2[0]], c='r', label='magnitude of MSTOP', linestyle='--', linewidth = 2)
#~ plt.axvline(basecenter[lim3[0]], c='k', label='magnitude cut', linestyle = '--', linewidth = 2)
#~ plt.ylabel('number of stars', fontsize = 24)
#~ plt.xlabel('F606W', fontsize = 24)
#~ plt.title('IC4499', fontsize = 24)
#~ plt.legend(loc='upper left', fontsize = 16)
#~ plt.tick_params(labelsize=16)
#~ plt.show()
#~ plt.close()
#~ kill

#~ plt.figure()
#~ plt.scatter(color,photo_v, marker='.', s=30, color='grey', label= r'< $\mathrm{m_{cut}}$', alpha=0.2)
#~ plt.scatter(color[bou1],photo_v[bou1], marker='.', s=30, color='r', label='MS', alpha=0.2)
#~ plt.scatter(color[bou2],photo_v[bou2], marker='.', s=30, color='b', label='UB', alpha=0.2)
#~ plt.axhline(mag_lim2, c='r', linestyle='--', linewidth = 3)
#~ plt.axhline(mag_lim3, c='k', linestyle='--', linewidth = 3)
#~ plt.xlim(-0.5,3)
#~ plt.ylim(28,10)
#~ plt.tick_params(labelsize=16)
#~ lgnd = plt.legend(loc='upper right', fontsize = 24)
#~ for handle in lgnd.legendHandles:
    #~ handle.set_sizes([286.0])
#~ plt.xlabel('F606W - F814W', fontsize = 24)
#~ plt.ylabel('F606W', fontsize = 24)
#~ plt.title('IC4499', fontsize = 24)
#~ plt.show() 
#~ kill
gc.collect()
#--------------------------------------------------------------
#--------------------------------------------------------------

### get all stars above the turnoff point
magv2, col2, errv2, errcol2 = spread_color_y(mag_lim2, rgb_lim)
magvbis = np.array(magv2)
if len(magvbis) > 0: 
	colbis = np.array(col2)
	errvbis = np.array(errv2)
	errcolbis = np.array(errcol2)
else:
	print("List is empty")
	
col_dr = top_x
dr = np.where(colbis > col_dr)[0]
if glc == 2:
	vcenter, ccenter, errcenter, sbin, bingood, errcenterv = way(magvbis[dr], colbis[dr], errcolbis[dr], errvbis[dr])
else:
	# ~vcenter, ccenter, errcenter, sbin, bingood, errcenterv = way(magvbis, colbis, errcolbis, errvbis)
	vcenter, ccenter, errcenter, sbin, bingood, errcenterv = way(magvbis[dr], colbis[dr], errcolbis[dr], errvbis[dr])
#~ vcenter, ccenter, errcenter, sbin, bingood, rangebinv, errcenterv = way(magvbis, colbis, errcolbis, errvbis)

# ~vcenter, ccenter, errcenter, sbin = angle_correction(vcenter, ccenter, errcenter, sbin)


# ~threshold = 1
# ~z = np.abs(stats.zscore(errcenter))
# ~out = (np.where(z > threshold)[0])
# ~print(out)
# ~nempty = np.where((sbin > 1)&(errcenter > 0)&(z < threshold))[0]

nempty = np.where((sbin > 0)&(errcenter > 0))[0]
vcenter = vcenter[nempty]
ccenter = ccenter[nempty]
errcenter = errcenter[nempty]
errcenterv = errcenterv[nempty]
sbin = sbin[nempty]

print(errcenter)

# ~if model == 'dar':
	# ~indice = np.abs((np.array(ccenter) - (fmag_ini(vcenter))))	
	# ~maybe = np.where((indice > 0.05*rescale[glc,0])&(vcenter < top_y))[0]
	# ~#~ indice = np.abs((np.array(colbis) - (fmag_ini(magvbis))))	
	# ~#~ maybe = np.where((indice > 0.05*rescale[glc,0]))[0]

# ~elif model == 'mist':
	# ~indice = np.abs((np.array(ccenter) - (fmag_ini(vcenter))))	
	# ~maybe = np.where((indice > 0.05*rescale[glc,0])&(vcenter < top_y))[0]
	#~ indice = np.abs((np.array(colbis) - (fmag_ini(magvbis))))	
	#~ maybe = np.where((indice > 0.05*rescale[glc,0]))[0]

#~ import cmasher as cmr
#~ cm = cmr.ember
#~ norm = colors.Normalize(vmin=np.min((indice)),vmax=np.max((indice)))
#~ s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
#~ s_m.set_array([])
#~ plt.figure()
#~ plt.scatter(color,photo_v, marker='.', s=10, color='grey', label='data')
#~ plt.errorbar(ccenter, vcenter, yerr=errcenterv,fmt = 'none', c='k')
#~ plt.scatter(ccenter,vcenter, marker='.',s=20, c='b')
#~ for x,y,z in zip(ccenter, vcenter, np.arange(len(ccenter))):
	#~ plt.text(x,y, str(z), color='b', label='selected points')
#~ plt.colorbar(s_m)
#~ plt.xlim(-0.5,3)
#~ plt.ylim(26,10)
#~ plt.title(clus_nb+' '+str(glc), fontsize = 16)
#~ plt.show() 
#~ plt.close()
#~ kill

#~ vcenter, ccenter, errcenter, sbin, bingood, rangebinv, errcenterv = way2(np.delete(magvbis,maybe), np.delete(colbis,maybe)
#~ , np.delete(errcolbis,maybe), np.delete(errvbis,maybe))
#~ vcenter_rgb = magvbis
#~ ccenter_rgb = colbis
#~ errcenter_rgb = errcolbis
#~ vcenter_rgb = np.delete(magvbis,maybe)
#~ ccenter_rgb = np.delete(colbis,maybe)
#~ errcenter_rgb = np.delete(errcolbis,maybe)
# ~sgb = np.where(vcenter > mag_lim2-1)[0]
# ~rgb = np.where(vcenter < mag_lim2-1)[0]
# ~vcenter_sgb = vcenter[sgb]
# ~ccenter_sgb = ccenter[sgb]
# ~errcenter_sgb = errcenter[sgb]
# ~errcenterv_sgb = errcenterv[sgb]
# ~vcenter_rgb = vcenter[rgb]
# ~ccenter_rgb = ccenter[rgb]
# ~errcenter_rgb = errcenter[rgb]
# ~errcenterv_rgb = errcenterv[rgb]
vcenter_rgb = vcenter
ccenter_rgb = ccenter
errcenter_rgb = errcenter
errcenterv_rgb = errcenterv
sbin_rgb = sbin



# ~maybe=np.array([9,10,11,12])
# ~vcenter_rgb = np.delete(vcenter,maybe)
# ~ccenter_rgb = np.delete(ccenter,maybe)
# ~errcenter_rgb = np.delete(errcenter,maybe)
# ~errcenterv_rgb = np.delete(errcenterv,maybe)

ecart = np.abs(ccenter - fmag_ini(vcenter))/errcenter
print(ecart)
base = np.where((ecart>1)|(vcenter < np.min(mag_v)))[0]
vcenter_rgb = np.delete(vcenter,base)
ccenter_rgb = np.delete(ccenter,base)
errcenter_rgb = np.delete(errcenter,base)
errcenterv_rgb = np.delete(errcenterv,base)
sbin_rgb = np.delete(sbin,base)

# ~base = []
# ~# base.extend(np.where((vcenter > mag_lim2 -2)&(vcenter < mag_lim2 -1))[0])
# ~base.extend(np.arange(int(rescale[glc,6]),int(rescale[glc,7])+1))
# ~lg = len(np.where(rescale[glc,:] < 100)[0])
# ~for ind in range(8,lg):
	# ~base.append(int(rescale[glc,ind]))
# ~#~ vcenter_rgb = vcenter[base]
# ~#~ ccenter_rgb = ccenter[base]
# ~#~ errcenter_rgb = errcenter[base]
# ~#~ errcenterv_rgb = errcenterv[base]
# ~#~ sbin_rgb = sbin[base]
# ~vcenter_rgb = np.delete(vcenter,base)
# ~ccenter_rgb = np.delete(ccenter,base)
# ~errcenter_rgb = np.delete(errcenter,base)
# ~errcenterv_rgb = np.delete(errcenterv,base)
# ~sbin_rgb = np.delete(sbin,base)



#~ ff = np.loadtxt('iso.txt', skiprows = 9)
#~ f606 = ff[:,10] + 5*np.log10(distance) - 5
#~ f814 = ff[:,15] + 5*np.log10(distance) - 5
#~ coliso = f606-f814
#~ f6063 = ff[:,10] + 5*np.log10(distance) - 5 + Abs*0.94
#~ f8143 = ff[:,15] + 5*np.log10(distance) - 5 + Abs*0.61
#~ coliso3 = f6063-f8143
#~ ff1 = np.loadtxt('iso1.txt', skiprows = 13)
#~ f6061 = ff1[:,14] + 5*np.log10(distance) - 5
#~ f8141 = ff1[:,19] + 5*np.log10(distance) - 5
#~ coliso1 = f6061-f8141
#~ ff2 = np.loadtxt('iso2.txt', skiprows = 13)
#~ f6062 = ff2[:,14] + 5*np.log10(distance) - 5
#~ f8142 = ff2[:,19] + 5*np.log10(distance) - 5
#~ coliso2 = f6062-f8142
# ~print(errcenter_rgb)

plt.figure()
plt.clf()
plt.plot(Color_iso, mag_v,c='y', alpha=0.5)
# ~plt.plot(Color_isoy, mag_vy,c='m', alpha=0.5)
# ~plt.plot(Color_isoz, mag_vz,c='c', alpha=0.5)
plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='stars')
plt.scatter(gauss_mean,bincenter, marker='o', s=10, color='r', label=r'$C_i^{data}$', alpha=0.5)
plt.errorbar(gauss_mean,bincenter, xerr=gauss_disp, yerr=errcenterv_gauss, c='k', linewidth=2, fmt='none', label=r'$\sigma_i^{data}$', alpha=0.5)
plt.scatter(ccenter,vcenter, marker='x',s=10, color='k')
plt.errorbar(ccenter, vcenter, xerr=errcenter, yerr=errcenterv, capsize= 2, linewidth=2,fmt = 'none', c='k', alpha=0.5)
plt.errorbar(ccenter_rgb, vcenter_rgb, xerr=errcenter_rgb,fmt = '.', c='c', ecolor='k', alpha=0.5)
plt.scatter(ccenter_rgb, vcenter_rgb , marker='o', s=10, color='c', label='selected points')
for x,y,z in zip(ccenter, vcenter, np.arange(len(ccenter))):
	plt.text(x+0.2,y, str(z), color='b', label='selected points', ha='left', va='center', fontsize=9)
# ~#~ if model == 'mist':
	# ~#~ plt.scatter(Color_iso[:ct[0]-1],mag_v[:ct[0]-1], marker='.', s=10, color='b')
# ~#~ elif model == 'dar':
	# ~#~ plt.scatter(Color_iso,mag_v, marker='.', s=10, color='b', label='isochrone')
# ~plt.axhline(binMS[0], label='bin edges',alpha=0.5)
# ~for ji in binMS[1:]:
	# ~plt.axhline(ji,alpha=0.5)
#~ plt.axvline(curve_x)
#~ plt.axhline(curve_y, c='r')
# ~for ig in bingood:
	# ~plt.axhline(ig, alpha=0.5)
#~ for ii in base:
	#~ if ii == base[0]:
		#~ plt.fill_between(np.linspace(-0.5,3), bingood[nempty][ii], bingood[nempty][ii+1], color='darkblue', label='masked bins', alpha=0.4)
	#~ else:
		#~ plt.fill_between(np.linspace(-0.5,3), bingood[nempty][ii], bingood[nempty][ii+1], color='darkblue', alpha=0.4)

plt.xlim(0.4,1.2)
plt.ylim(22,16)
plt.tick_params(labelsize=16)

plt.axvline(col_dr, c='r')
plt.xlim(-0.5,3)
plt.ylim(26,10)
plt.legend(loc='upper right', fontsize = 16)
plt.xlabel('F606W - F814W', fontsize = 16)
plt.ylabel('F606W', fontsize = 16)
plt.title(clus_nb, fontsize = 16)
plt.show()
# ~plt.close()
# ~kill
# ~plt.close()

print('coucou')
gc.collect()

# ~#~ #----------------------------------------------
#----------------------------------------------


#~ with Pool() as pool:
	#~ sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

	#~ f=open('asd.dat','ab')

	#~ for i, (results) in enumerate(sampler.sample(pos, iterations=ite)):
		#~ print(i)


#~ samples = sampler.chain[:,:, :].reshape((-1, ndim))
#~ if model == 'mist':
	#~ b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
	#~ zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	#~ b1_mcmc_fit = b1_mcmc 
	#~ b2_mcmc_fit = b2_mcmc 
	#~ b3_mcmc_fit = b3_mcmc 
	#~ b4_mcmc_fit = b4_mcmc 
	
	#~ mag_v3, mag_i3, Color_iso3, eep_first3 = iso_mag(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0])

	#~ ct3 = [605-int(eep_first3)]
	#~ plt.figure()
	#~ plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
	#~ plt.scatter(gauss_mean,bincenter, marker='o', s=10, color='k', label='fit at bin center')
	#~ plt.scatter(ccenter_rgb, vcenter_rgb , marker='o', s=30, color='k', label='selected points')
	#~ plt.plot(Color_iso3[:ct3[0]-1],mag_v3[:ct3[0]-1], c='b',  label='main sequence')
	#~ plt.xlim(-0.5,3)
	#~ plt.ylim(25,10)
	#~ plt.legend(loc='upper right', fontsize = 16)
	#~ plt.xlabel('F606W - F814W', fontsize = 16)
	#~ plt.ylabel('F606W', fontsize = 16)
	#~ plt.title(clus_nb, fontsize = 16)
	#~ plt.show()
	#~ plt.close()
	
#~ elif model == 'dar':
	#~ b1_mcmc_fit, b2_mcmc_fit, b3_mcmc_fit, b4_mcmc_fit, b5_mcmc_fit = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
	#~ zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	#~ b1_mcmc_fit = b1_mcmc 
	#~ b2_mcmc_fit = b2_mcmc 
	#~ b3_mcmc_fit = b3_mcmc 
	#~ b4_mcmc_fit = b4_mcmc 
	#~ b5_mcmc_fit = b5_mcmc 
	

	#~ afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8] 
	#~ afe_max = afe_values[np.searchsorted(afe_values, b5_mcmc[0])]
	#~ afe_min = afe_values[np.searchsorted(afe_values, b5_mcmc[0])-1]

	#~ mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0], afe_min)
	#~ mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0], afe_max)
	#~ lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
	
	#~ mag_v3 = (mag_v1_min[:lpp]*(afe_max - b5_mcmc[0]) + mag_v1_max[:lpp]*(b5_mcmc[0] - afe_min)) / (afe_max - afe_min)
	#~ Color_iso3 = (Color_iso1_min[:lpp]*(afe_max - b5_mcmc[0]) + Color_iso1_max[:lpp]*(b5_mcmc[0] - afe_min)) / (afe_max - afe_min)

	#~ plt.figure()
	#~ plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
	#~ plt.scatter(gauss_mean,bincenter, marker='o', s=30, color='r', label='fit at bin center')
	#~ plt.scatter(ccenter_rgb, vcenter_rgb , marker='o', s=30, color='k', label='selected points')
	#~ plt.plot(Color_iso3,mag_v3, c='b',  label='main sequence')
	#~ plt.xlim(-0.5,3)
	#~ plt.ylim(25,10)
	#~ plt.legend(loc='upper right', fontsize = 16)
	#~ plt.xlabel('F606W - F814W', fontsize = 16)
	#~ plt.ylabel('F606W', fontsize = 16)
	#~ plt.title(clus_nb, fontsize = 16)
	#~ plt.show()
	#~ plt.close()
#~ kill
#----------------------------------------------
#----------------------------------------------
# ~from multiprocessing import set_start_method
# ~set_start_method('spawn', force=True)




# ~def your_func():
# ~with get_context("spawn").Pool() as pool:
# ~sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=6, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),],)
with Pool() as pool:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
# ~sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

	for i, (results) in enumerate(zip(sampler.sample(pos, iterations=ite))):
		print(i)
		if (i+1) % 200 == 0:
			ind = int((i+1)/1)
	# 		with open('test2_'+str(clus_nb)+'_'+str(model)+'.txt', 'a+') as fid_file:
			print("first phase is at {0:.1f}%\n".format(100 * float(i) /ite))
	# 		fid_file.close()

			
			print(sampler.acceptance_fraction)
			print(np.mean(sampler.acceptance_fraction))
			#~ print(np.mean(sampler2.acceptance_fraction))
			#~ samples = sampler.chain[:,:, :].reshape((-1, ndim))
			#~ samples2 = sampler2.chain[:,:, :].reshape((-1, ndim))

			#~ print(np.shape(samples), np.shape(samples2), np.shape(samples3))
			#~ kill
			
			if model == 'mist':
				#~ plt.figure()
				#~ ax1 = plt.subplot(231)
				#~ ax1.set_title('Age')
				#~ for ii in range(0,nwalkers):
					#~ ax1.plot(np.arange(i), sampler.chain[ii,:i,0], c='k')
					#~ ax1.plot(np.arange(i), sampler2.chain[ii,:i,0], c='r')
				#~ ax1.axhline(Age, color='k', linestyle='--')
				#~ ax1.grid()
				#~ ax1.axhline(10.176, color='c')
				#~ ax2 = plt.subplot(232)
				#~ ax2.set_title('metal')
				#~ for ii in range(0,nwalkers):
					#~ ax2.plot(np.arange(i), sampler.chain[ii,:i,1], c='k')
					#~ ax2.plot(np.arange(i), sampler2.chain[ii,:i,1], c='r')
				#~ ax2.axhline(metal, color='k', linestyle='--')
				#~ ax2.grid()
				#~ ax2.axhline(b2_ml, color='k')
				#~ ax3 = plt.subplot(233)
				#~ ax3.set_title('distance')
				#~ for ii in range(0,nwalkers):
					#~ ax3.plot(np.arange(i), sampler.chain[ii,:i,2], c='k')
					#~ ax3.plot(np.arange(i), sampler2.chain[ii,:i,2], c='r')
				#~ ax3.axhline(distance, color='k', linestyle='--')
				#~ ax3.grid()
				#~ ax3.axhline(b3_ml, color='k')
				#~ ax4 = plt.subplot(234)
				#~ ax4.set_title('A1')
				#~ for ii in range(0,nwalkers):
					#~ ax4.plot(np.arange(i), sampler.chain[ii,:i,3], c='k')
					#~ ax4.plot(np.arange(i), sampler2.chain[ii,:i,3], c='r')
				#~ ax4.axhline(Abs, color='k', linestyle='--')
				#~ ax4.grid()
				#~ ax5 = plt.subplot(235)
				#~ ax5.set_title(r'$\alpha$')
				#~ for ii in range(0,nwalkers):
					#~ ax5.plot(np.arange(i), sampler.chain[ii,:i,4], c='k')
					#~ ax5.plot(np.arange(i), sampler2.chain[ii,:i,4], c='r')
				#~ ax5.axhline(afe_init, color='k', linestyle='--')
				#~ ax5.grid()
				#~ plt.show()
				#~ plt.close()
				plt.hist(10**sampler.chain[:,:i,0].flatten() / 1.e9, bins=np.linspace(7.5,15,100),histtype='step')
				plt.show()
				plt.close()
				
								
				import corner
				fig = corner.corner(sampler.chain[:,:i, :].reshape((-1, ndim)), bins=100, labels=["$log10(Age)$", "$metallicity$", "$distance$", "$A1$", "$afe$"], 
		truths=[Age, metal,distance,Abs, afe_init], title_fmt='.3f', plot_contours=False, show_titles=True, title_args={"fontsize":12}, color='k')
				#~ corner.corner(sampler2.chain[:,:i, :2].reshape((-1, 2)), labels=["$log10(Age)$", "$metallicity$"], 
		#~ truths=[Age, metal], title_fmt='.3f', plot_contours=False, show_titles=True, title_args={"fontsize":12}, fig=fig, color='r')
				fig.suptitle('numero '+str(glc)+', '+clus_nb)
				plt.show()
				plt.close()
				
				samples = sampler.chain[:,:, :].reshape((-1, ndim))
				#~ samples2 = sampler2.chain[:,:, :].reshape((-1, ndim))

				b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc, b5_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
				zip(*np.percentile(samples, [16, 50, 84], axis=0)))
				#~ b1_mcmc2, b2_mcmc2, b3_mcmc2, b4_mcmc2, b5_mcmc2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
				#~ zip(*np.percentile(samples2, [16, 50, 84], axis=0)))

			
				corr = np.log10(0.64*10**b5_mcmc[0] + 0.36)
				mag_v3 , mag_i3, Color_iso3, eep_first3 = iso_mag(b1_mcmc[0], b2_mcmc[0]+corr, b3_mcmc[0], b4_mcmc[0], b5_mcmc[0])
				#~ mag_v3 , mag_i3, Color_iso3, eep_first3 = iso_mag(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0], b5_mcmc[0])


				ct3 = [605-int(eep_first3)]
				plt.figure()
				plt.scatter(color,photo_v, marker='.',s=30, color='grey', label='data')
				plt.scatter(gauss_mean,bincenter, marker='o', s=10, color='k', label='fit at bin center')
				plt.errorbar(gauss_mean,bincenter, xerr=gauss_disp, c='k', fmt='none')
				plt.scatter(ccenter_rgb, vcenter_rgb , marker='o', s=30, color='k', label='selected points')
				plt.errorbar(ccenter_rgb, vcenter_rgb, xerr=errcenter_rgb,fmt = 'none', c='k')
				plt.plot(Color_iso3[:ct3[0]-1],mag_v3[:ct3[0]-1], c='b',  label='main sequence')
				plt.xlim(-0.5,3)
				plt.ylim(26,10)
				plt.legend(loc='upper right', fontsize = 16)
				plt.xlabel('F606W - F814W', fontsize = 16)
				plt.ylabel('F606W', fontsize = 16)
				plt.title(clus_nb, fontsize = 16)
				plt.show()
				plt.close()
				
			elif model == 'dar':
				
				# ~plt.hist(10**sampler.chain[:,:i,0].flatten() / 1.e9, bins=np.linspace(7.5,15,100),histtype='step')
				# ~plt.show()
				# ~plt.close()
				
				ax1 = plt.subplot(231)
				ax1.set_title('Age')
				for ii in range(0,nwalkers):
					ax1.plot(np.arange(i), sampler.chain[ii,:i,0], c='k')
					#ax1.plot(np.arange(i), sampler2.chain[ii,:i,0], c='r')
				ax1.axhline(Age, color='r', linestyle='--')
				ax1.grid()
				ax1.axhline(10.176, color='c')
				ax2 = plt.subplot(232)
				ax2.set_title('metal')
				for ii in range(0,nwalkers):
					ax2.plot(np.arange(i), sampler.chain[ii,:i,1], c='k')
					#ax2.plot(np.arange(i), sampler2.chain[ii,:i,1], c='r')
				ax2.axhline(metal, color='r', linestyle='--')
				ax2.grid()
				#ax2.axhline(b2_ml, color='k')
				ax3 = plt.subplot(233)
				ax3.set_title('distance')
				for ii in range(0,nwalkers):
					ax3.plot(np.arange(i), sampler.chain[ii,:i,2], c='k')
					#ax3.plot(np.arange(i), sampler2.chain[ii,:i,2], c='r')
				ax3.axhline(distance, color='r', linestyle='--')
				ax3.grid()
				#ax3.axhline(b3_ml, color='k')
				ax4 = plt.subplot(234)
				ax4.set_title('A1')
				for ii in range(0,nwalkers):
					ax4.plot(np.arange(i), sampler.chain[ii,:i,3], c='k')
					#ax4.plot(np.arange(i), sampler2.chain[ii,:i,3], c='r')
				ax4.axhline(Abs, color='r', linestyle='--')
				ax4.grid()
				ax5 = plt.subplot(235)
				ax5.set_title(r'$\alpha$')
				for ii in range(0,nwalkers):
					ax5.plot(np.arange(i), sampler.chain[ii,:i,4], c='k')
					#ax5.plot(np.arange(i), sampler2.chain[ii,:i,4], c='r')
				ax5.axhline(afe_init, color='r', linestyle='--')
				ax5.grid()
				plt.show()
				plt.close()

				import corner
				fig = corner.corner(sampler.chain[:,:i, :].reshape((-1, ndim)),bins=100,range=[(np.min(sampler.chain[:,:i, 0]),10.176), (np.min(sampler.chain[:,:i, 1]),np.max(sampler.chain[:,:i, 1])),
		(np.min(sampler.chain[:,:i, 2]),np.max(sampler.chain[:,:i, 2])),(np.min(sampler.chain[:,:i, 3]),np.max(sampler.chain[:,:i, 3])),(np.min(sampler.chain[:,:i, 4]),np.max(sampler.chain[:,:i, 4]))], labels=["$log10(Age)$", "$metallicity$", "$distance$", "$A1$", "$afe$"], 
		truths=[Age, metal,distance,Abs, afe_init], title_fmt='.3f', plot_contours=False, show_titles=True, title_args={"fontsize":8}, color='k')
				#~ corner.corner(sampler2.chain[:,:i, :2].reshape((-1, 2)), labels=["$log10(Age)$", "$metallicity$"], 
		#~ truths=[Age, metal], title_fmt='.3f', plot_contours=False, show_titles=True, title_args={"fontsize":12}, fig=fig, color='r')
				#~ fig.suptitle('numero '+str(glc)+', '+clus_nb)
				fig.subplots_adjust(top=0.911,bottom=0.14,left=0.085,right=0.966,hspace=0.05,wspace=0.05)
				plt.show()
				plt.close()

				samples = sampler.chain[:,:i, :].reshape((-1, ndim))
				#~ samples2 = sampler2.chain[:,:, :].reshape((-1, ndim))
				#~ samples3 = sampler3.chain[:,:, :].reshape((-1, ndim))

				b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc, b5_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
				zip(*np.percentile(samples, [16, 50, 84], axis=0)))
				#~ b1_mcmc2, b2_mcmc2, b3_mcmc2, b4_mcmc2, b5_mcmc2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
				#~ zip(*np.percentile(samples2, [16, 50, 84], axis=0)))
				

				afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8]
				#~ afe_values=[-0.2, 0.0 , 0.2, 0.6, 0.8]  


				afe_max = afe_values[np.searchsorted(afe_values, b5_mcmc[0])]
				afe_min = afe_values[np.searchsorted(afe_values, b5_mcmc[0])-1]
				#~ afe_maxbis = afe_values[np.searchsorted(afe_values, b5_mcmc2[0])]
				#~ afe_minbis = afe_values[np.searchsorted(afe_values, b5_mcmc2[0])-1]
				print(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0], afe_min)
				

				mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0], afe_min)
				mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(b1_mcmc[0], b2_mcmc[0], b3_mcmc[0], b4_mcmc[0], afe_max)
				lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
				#~ mag_v1_minbis , mag_i1_minbis, Color_iso1_minbis, eep_firstbis = iso_mag(b1_mcmc2[0], b2_mcmc2[0], b3_mcmc2[0], b4_mcmc2[0], afe_minbis)
				#~ mag_v1_maxbis , mag_i1_maxbis, Color_iso1_maxbis, eep_firstbis = iso_mag(b1_mcmc2[0], b2_mcmc2[0], b3_mcmc2[0], b4_mcmc2[0], afe_maxbis)
				#~ lppbis = (min(len(mag_v1_minbis), len(mag_v1_maxbis))) # get minimum length to interpolate
				#~ mag_v1_minter , mag_i1_minter, Color_iso1_minter, eep_firstter = iso_mag(b1_mcmc3[0], b2_mcmc3[0], b3_mcmc3[0], b4_mcmc3[0], afe_minter)
				#~ mag_v1_maxter , mag_i1_maxter, Color_iso1_maxter, eep_firstter = iso_mag(b1_mcmc3[0], b2_mcmc3[0], b3_mcmc3[0], b4_mcmc3[0], afe_maxter)
				#~ lppter = (min(len(mag_v1_minter), len(mag_v1_maxter))) # get minimum length to interpolate
				
				mag_v3 = (mag_v1_min[:lpp]*(afe_max - b5_mcmc[0]) + mag_v1_max[:lpp]*(b5_mcmc[0] - afe_min)) / (afe_max - afe_min)
				Color_iso3 = (Color_iso1_min[:lpp]*(afe_max - b5_mcmc[0]) + Color_iso1_max[:lpp]*(b5_mcmc[0] - afe_min)) / (afe_max - afe_min)
				#~ mag_v3ter = (mag_v1_minter[:lppter]*(afe_maxter - b5_mcmc3[0]) + mag_v1_maxter[:lppter]*(b5_mcmc3[0] - afe_minter)) / (afe_maxter - afe_minter)
				#~ Color_iso3ter = (Color_iso1_minter[:lppter]*(afe_maxter - b5_mcmc3[0]) + Color_iso1_maxter[:lppter]*(b5_mcmc3[0] - afe_minter)) / (afe_maxter - afe_minter)
				#~ mag_v3bis = (mag_v1_minbis[:lppbis]*(afe_maxbis - b5_mcmc2[0]) + mag_v1_maxbis[:lppbis]*(b5_mcmc2[0] - afe_minbis)) / (afe_maxbis - afe_minbis)
				#~ Color_iso3bis = (Color_iso1_minbis[:lppbis]*(afe_maxbis - b5_mcmc2[0]) + Color_iso1_maxbis[:lppbis]*(b5_mcmc2[0] - afe_minbis)) / (afe_maxbis - afe_minbis)



				plt.figure()
				plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
				plt.scatter(gauss_mean,bincenter, marker='o', s=30, color='r', label='fit at bin center')
				plt.errorbar(gauss_mean,bincenter, xerr=gauss_disp, yerr=errcenterv_gauss, c='k', fmt='none')
				plt.errorbar(ccenter_rgb, vcenter_rgb, xerr=errcenter_rgb, yerr=errcenterv_rgb,fmt = 'none', c='k')
				plt.scatter(ccenter_rgb, vcenter_rgb , marker='o', s=30, color='c', label='selected points')
				# ~plt.errorbar(ccenter_sgb, vcenter_sgb, yerr=errcenterv_sgb,fmt = 'none', c='k')
				# ~plt.scatter(ccenter_sgb, vcenter_sgb , marker='o', s=30, color='b', label='selected points')
				plt.plot(Color_iso3,mag_v3, c='b',  label='main sequence')
				#~ plt.plot(Color_iso3bis,mag_v3bis, c='r',  label='RGB')
				#~ plt.plot(Color_iso3ter,mag_v3ter, c='c',  label='total')
				plt.xlim(-0.5,3)
				plt.ylim(26,10)
				plt.legend(loc='upper right', fontsize = 16)
				plt.xlabel('F606W - F814W', fontsize = 16)
				plt.ylabel('F606W', fontsize = 16)
				plt.title(clus_nb, fontsize = 16)
				plt.show()
				plt.close()
				
				plt.figure()
				plt.scatter(color,photo_v, marker='.',s=10, color='grey', label='data')
				plt.plot(Color_iso3,mag_v3, c='b',  label='main sequence')
				#~ plt.plot(Color_iso3bis,mag_v3bis, c='r',  label='RGB')
				#~ plt.plot(Color_iso3ter,mag_v3ter, c='c',  label='total')
				plt.xlim(-0.5,3)
				plt.ylim(26,10)
				plt.legend(loc='upper right', fontsize = 16)
				plt.xlabel('F606W - F814W', fontsize = 16)
				plt.ylabel('F606W', fontsize = 16)
				plt.title(clus_nb, fontsize = 16)
				plt.show()
				plt.close()

		pass


# ~if __name__ == "__main__":
	# ~your_func()

#~ end = time.time()






########################################################################
########################################################################

#not used anymore
########################################################################
########################################################################




