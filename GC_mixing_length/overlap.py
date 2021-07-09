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
import mesa_reader as mr

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


	with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/Harris2010.dat',"r") as f:
		lines=f.readlines()[1:]
	f.close()
	harris_clus=[]
	for x in lines:
		harris_clus.append(x.split(' ')[0])

	with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dotter2010.dat',"r") as f:
		lines=f.readlines()[3:]
	f.close()
	dotter_clus=[]
	for x in lines:
		dotter_clus.append(x.split(' ')[0])

	with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/roediger2014.dat',"r") as f:
		lines=f.readlines()[5:]
	f.close()
	roediger_clus=[]
	for x in lines:
		roediger_clus.append(x.split(' ')[0])

	clus_nb = clus_name[nb]
	
	# find acs initial values in different caltalogs
	index1 = harris_clus.index(clus_nb)
	
	with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/Harris2010.dat',"r") as f:
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
		with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dotter2010.dat',"r") as f:
			lines=f.readlines()[3:]
		f.close()
		age_dotter=[]
		for x in lines:
			age_dotter.append(x.split(' ')[5])
		age = age_dotter[index2]
	elif clus_nb in roediger_clus:
		index2 = roediger_clus.index(clus_nb)
		with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/roediger2014.dat',"r") as f:
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
		with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dotter2010.dat',"r") as f:
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

	Color = files[mg_cut, 5][filter_all]
	err_Color = pcolor[filter_all]
	err_v = pv[filter_all]
	nmv = files[mg_cut,11][filter_all]
	nmi = files[mg_cut,12][filter_all]

	
	# ~photo_v = files[mg_cut, 3]
	# ~photo_i = files[mg_cut, 7]
	# ~Color = files[mg_cut, 5]
	# ~err_Color = pcolor
	# ~err_v = pv
	# ~nmv = files[mg_cut,11]
	# ~nmi = files[mg_cut,12]


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

def way(vgood, cgood, errgood, errgoodv, step = None):

	#remove duplicate
	for i, j in zip(vgood, cgood):
		dup = np.where((vgood == i)&(cgood==j))[0]
		if len(dup) > 1:
			vgood = np.delete(np.array(vgood), dup[1:])
			cgood = np.delete(np.array(cgood), dup[1:])
	
	#~ nbins = 20
	rangebin = np.max(vgood) - np.min(vgood)
	if step is not None:
		nbins = int(round(rangebin/step))
	else:
		nbins = int(round(rangebin/0.2))

	bingood = np.linspace(np.min(vgood), np.max(vgood),nbins)
	centergood = (bingood[:-1] + bingood[1:]) / 2 
	
	vcenter = np.zeros(len(centergood))
	ccenter = np.zeros(len(centergood))
	errcenter = np.zeros(len(centergood))
	errcenterv = np.zeros(len(centergood))
	size_bin = np.zeros(len(centergood))


	for c in range(0,len(centergood)):
		inbin = np.digitize(vgood, bingood)
		ici = np.where(inbin == c+1)[0]
		#~ print(np.min(np.array(cgood)[ici]), np.max(np.array(cgood)[ici]))
		
		# ~threshold = 3
		# ~z = np.abs(stats.zscore(np.array(cgood)[ici]))
		# ~out = (np.where(z > threshold)[0])
		# ~zcol =  np.delete(np.array(cgood)[ici], out)
		# ~zmagv =  np.delete(np.array(vgood)[ici], out)
		
		# ~z = np.abs(stats.zscore(zcol))
		# ~out = (np.where(z > threshold)[0])
		# ~zcol =  np.delete(zcol, out)
		# ~zmagv =  np.delete(zmagv, out)
		
		zcol =  np.array(cgood)[ici]
		zmagv =  np.array(vgood)[ici]


		if len(ici) == 1:
			ccenter[c] =np.median(zcol)
			errcenter[c] =np.array(errgood)[ici]
			errcenterv[c] =np.array(errgoodv)[ici]
			vcenter[c] =centergood[c]
			#~ vcenter[c] =np.array(vgood)[ici]
			size_bin[c] = len(ici)
		elif len(ici) == 2:
			ccenter[c] =np.median(zcol)
			errcenter[c] = np.std(zcol)
			errcenterv[c] = np.std(zmagv)
			vcenter[c] =centergood[c]
			#~ vcenter[c] =np.median(zmagv)
			size_bin[c] = len(ici)
		elif len(ici) >2:
			ccenter[c] =np.median(zcol)
			#~ ccenter[c] =np.mean(zcol)
			if np.std(zcol) == 0:
				errcenter[c] =np.mean(np.array(errgood)[ici])
				print('std nul')
			else:
				errcenter[c] =np.std(zcol)
				errcenterv[c] =np.std(zmagv)
			vcenter[c] =centergood[c]
			size_bin[c] = len(ici)
		
		#~ print(centergood[c], ccenter[c])	

	return vcenter, ccenter, errcenter, size_bin, bingood, errcenterv

def cut(p):
	V, R = p[:,9], p[:,10]
	const = np.min(V-R)
	min_pts = np.where(V-R == const)[0]
	return min_pts[0]

def cut2(h):
	V = h.abs_mag_F606W
	R = h.abs_mag_F814W
	const = np.max(V)
	min_pts = np.where(V == const)[0]
	mag = V[min_pts[0]:]
	col = V[min_pts[0]:] - R[min_pts[0]:]
	return col, mag, min_pts[0]

def alpha_distribution(col, mag, bestalpha):
	# ~dx, dy = [[],[],[],[],[],[],[],[]],  [[],[],[],[],[],[],[],[]]
	dist_alpha = []
	limi = min(bestalpha, 16-bestalpha)

	for a in range(1, limi+1):
		for i in range(len(col)):
			if (col[i] < finterp[bestalpha -a](mag[i]) and col[i] > finterp[bestalpha -(a-1)](mag[i])):
				# dx[a-1].append(col[i])
				# dy[a-1].append(mag[i])
				# dist_alpha.append(a/10.)
				xp = [np.float(finterp[bestalpha -(a-1)](mag[i])),np.float(finterp[bestalpha -a](mag[i]))]
				yp = [np.float(-(a-1))*0.1,np.float(-a)*0.1]
				dist_alpha.append(np.abs(np.interp(col[i], xp, yp)))
				
			if (col[i] < finterp[bestalpha +(a-1)](mag[i]) and col[i] > finterp[bestalpha +a](mag[i])):
				# dx[a-1].append(col[i])
				# dy[a-1].append(mag[i])
				# dist_alpha.append(a/10.)
				xp = [np.float(finterp[bestalpha +a](mag[i])),np.float(finterp[bestalpha +(a-1)](mag[i]))]
				yp = [np.float(a)*0.1,np.float(a-1)*0.1]
				dist_alpha.append(np.abs(np.interp(col[i], xp, yp)))
				dist_alpha.append(np.abs(np.interp(col[i], xp, yp)))
				
	si = np.sqrt(np.sum(np.power(dist_alpha, 2))/len(dist_alpha))
	# ~return np.mean(dist_alpha), dx[0], dy[0]
	# ~return si, dx[0], dy[0]
	return si
			
def error_compute(dbins, histo, bhisto):
		amp = 1.0
		while amp > 0.0:
			integ = np.sum(dbins*histo)
			above = np.where(histo > amp*np.max(histo))[0]
			tinteg = np.sum(dbins*histo[above])
			s = tinteg/integ
			print('integral percentage is '+str(s))
			if s > 0.68:
				# ~print([np.min(above)])
				# ~print([np.max(above)])
				return bhisto[np.min(above)], bhisto[np.max(above)]
				break
			amp -= 0.01
			#~ print('percentage of the amplitude is '+str(amp))

def interp_eep(tef, sf, tefs, sfs, grid):

	Ntef = len(tefs)
	Nsf = len(sfs)


	isf = searchsorted(sfs, Nsf, sf)
	itef = searchsorted(tefs, Ntef, tef)
	#~ if isf==0 or itef==0 or isf==Nsf or itef==Ntef:
		#~ return np.nan

	pts1 = np.zeros((4,3))
	pts2 = np.zeros((4,3))
	# ~vals = np.zeros(4)

	### construct box
	i_f = isf - 1
	i_a = itef - 1
	pts1[0, 0] = tefs[i_a]
	pts1[0, 1] = sfs[i_f]
	pts2[0, 0] = tefs[i_a]
	pts2[0, 1] = sfs[i_f]
	aqui = np.where((dat[:,0] == tefs[i_a])&(dat[:,1] == sfs[i_f])&(dat[:,3] == 0.00))[0]
	pts1[0, 2] = dat[aqui,10]
	pts2[0, 2] = dat[aqui,15]
	# ~pts1[0, 2] = dat[aqui,3]
	# ~pts2[0, 2] = dat[aqui,4]
	
	i_f = isf - 1
	i_a = itef
	pts1[1, 0] = tefs[i_a]
	pts1[1, 1] = sfs[i_f]
	pts2[1, 0] = tefs[i_a]
	pts2[1, 1] = sfs[i_f]
	aqui = np.where((dat[:,0] == tefs[i_a])&(dat[:,1] == sfs[i_f])&(dat[:,3] == 0.0))[0]
	pts1[1, 2] = dat[aqui,10]
	pts2[1, 2] = dat[aqui,15]


	i_f = isf
	i_a = itef - 1
	pts1[2, 0] = tefs[i_a]
	pts1[2, 1] = sfs[i_f]
	pts2[2, 0] = tefs[i_a]
	pts2[2, 1] = sfs[i_f]
	aqui = np.where((dat[:,0] == tefs[i_a])&(dat[:,1] == sfs[i_f])&(dat[:,3] == 0.0))[0]
	pts1[2, 2] = dat[aqui,10]
	pts2[2, 2] = dat[aqui,15]


	i_f = isf
	i_a = itef
	pts1[3, 0] = tefs[i_a]
	pts1[3, 1] = sfs[i_f]
	pts2[3, 0] = tefs[i_a]
	pts2[3, 1] = sfs[i_f]
	aqui = np.where((dat[:,0] == tefs[i_a])&(dat[:,1] == sfs[i_f])&(dat[:,3] == 0.0))[0]
	pts1[3, 2] = dat[aqui,10]
	pts2[3, 2] = dat[aqui,15]
	
	#~ magtu[ind] = interp_box(tef, sf, pts, vals)
	# ~print(tef, sf)
	bc1 = bilinear_interpolation(tef, sf, pts1)
	bc2 = bilinear_interpolation(tef, sf, pts2)
	
	#~ print(ind, pts)
	#~ f = scipy.interpolate.interp2d(pts[:,0], pts[:,1], vals)
	#~ znew = f(tef, sf)
	#~ magtu[ind] = znew[0]


	return bc1, bc2

def searchsorted(arr, N, x):
	"""N is length of arr
	"""
	L = 0
	R = N-1
	done = False
	m = (L+R)//2
	while not done:
		if arr[m] < x:
			L = m + 1
			#~ print('moins')
		elif arr[m] > x:
			R = m - 1
			#~ print('plus')
		elif arr[m] == x:
			L = m
			done = True
		m = (L+R)//2
		#~ print(arr[m], x, L, R, m)
		if L>R:
			done = True
	return L

def bilinear_interpolation(x, y, pts):

	#~ points = sorted(points)               # order points by x, then by y
	#~ (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
	x1 = pts[0,0]
	y1 = pts[0,1]
	q11 = pts[0,2]
	
	_x1 = pts[2,0]
	y2 = pts[2,1]
	q12 = pts[2,2]
	
	x2 = pts[1,0]
	_y1 = pts[1,1]
	q21 = pts[1,2]
	
	_x2 = pts[3,0]
	_y2 = pts[3,1]
	q22 = pts[3,2]
	
	#~ x1 = pts[0,0]
	#~ x2 = pts[3,0]
	#~ y1 = pts[0,1]
	#~ y2 = pts[3,1]
	
	#~ q11 = pts[0,2]
	#~ q12 = pts[1,2]
	#~ q21 = pts[2,2]
	#~ q22 = pts[3,2]
	
	# ~print(pts)
	# ~print(x1,x,x2)
	# ~print(y1,y,y2)

	if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
		raise ValueError('points do not form a rectangle')
	if not x1 <= x <= x2 or not y1 <= y <= y2:
		raise ValueError('(x, y) not within the rectangle')


	fxy1 = (x2-x)/(x2-x1)*q11 + (x-x1)/(x2-x1)*q21
	fxy2 = (x2-x)/(x2-x1)*q12 + (x-x1)/(x2-x1)*q22
	
	fxy = (y2-y)/(y2-y1)*fxy1 + (y-y1)/(y2-y1)*fxy2
	
	return fxy

def chi2(C, M):
	dis = np.zeros(17)
	dis[0] = np.sum(np.abs(C - finterp[0](M)))
	dis[1] = np.sum(np.abs(C - finterp[1](M)))
	dis[2] = np.sum(np.abs(C - finterp[2](M)))
	dis[3] = np.sum(np.abs(C - finterp[3](M)))
	dis[4] = np.sum(np.abs(C - finterp[4](M)))
	dis[5] = np.sum(np.abs(C - finterp[5](M)))
	dis[6] = np.sum(np.abs(C - finterp[6](M)))
	dis[7] = np.sum(np.abs(C - finterp[7](M)))
	dis[8] = np.sum(np.abs(C - finterp[8](M)))
	dis[9] = np.sum(np.abs(C - finterp[9](M)))
	dis[10] = np.sum(np.abs(C - finterp[10](M)))
	dis[11] = np.sum(np.abs(C - finterp[11](M)))
	dis[12] = np.sum(np.abs(C - finterp[12](M)))
	dis[13] = np.sum(np.abs(C - finterp[13](M)))
	dis[14] = np.sum(np.abs(C - finterp[14](M)))
	dis[15] = np.sum(np.abs(C - finterp[15](M)))
	dis[16] = np.sum(np.abs(C - finterp[16](M)))
	# ~dis[0] = np.sum(np.abs(C - finterp[0](M)**2))
	# ~dis[1] = np.sum(np.abs(C - finterp[1](M)**2))
	# ~dis[2] = np.sum(np.abs(C - finterp[2](M)**2))
	# ~dis[3] = np.sum(np.abs(C - finterp[3](M)**2))
	# ~dis[4] = np.sum(np.abs(C - finterp[4](M)**2))
	# ~dis[5] = np.sum(np.abs(C - finterp[5](M)**2))
	# ~dis[6] = np.sum(np.abs(C - finterp[6](M)**2))
	# ~dis[7] = np.sum(np.abs(C - finterp[7](M)**2))
	# ~dis[8] = np.sum(np.abs(C - finterp[8](M)**2))
	# ~dis[9] = np.sum(np.abs(C - finterp[9](M)**2))
	# ~dis[10] = np.sum(np.abs(C - finterp[10](M)**2))
	# ~dis[11] = np.sum(np.abs(C - finterp[11](M)**2))
	# ~dis[12] = np.sum(np.abs(C - finterp[12](M)**2))
	# ~dis[13] = np.sum(np.abs(C - finterp[13](M)**2))
	# ~dis[14] = np.sum(np.abs(C - finterp[14](M)**2))
	# ~dis[15] = np.sum(np.abs(C - finterp[15](M)**2))
	# ~dis[16] = np.sum(np.abs(C - finterp[16](M)**2))

	
	return dis, np.argmin(dis)
	
def chi2_data(C, M):
	dis = np.zeros(17)
	dis[0] = np.sum(np.abs(C - finterp[0](M)))
	dis[1] = np.sum(np.abs(C - finterp[1](M)))
	dis[2] = np.sum(np.abs(C - finterp[2](M)))
	dis[3] = np.sum(np.abs(C - finterp[3](M)))
	dis[4] = np.sum(np.abs(C - finterp[4](M)))
	dis[5] = np.sum(np.abs(C - finterp[5](M)))
	dis[6] = np.sum(np.abs(C - finterp[6](M)))
	dis[7] = np.sum(np.abs(C - finterp[7](M)))
	dis[8] = np.sum(np.abs(C - finterp[8](M)))
	dis[9] = np.sum(np.abs(C - finterp[9](M)))
	dis[10] = np.sum(np.abs(C - finterp[10](M)))
	dis[11] = np.sum(np.abs(C - finterp[11](M)))
	dis[12] = np.sum(np.abs(C - finterp[12](M)))
	dis[13] = np.sum(np.abs(C - finterp[13](M)))
	dis[14] = np.sum(np.abs(C - finterp[14](M)))
	dis[15] = np.sum(np.abs(C - finterp[15](M)))
	dis[16] = np.sum(np.abs(C - finterp[16](M)))
	return np.argmin(dis)
	
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
Age_t = np.log10(13.32e9)
distance_t = 0
Abs_t = 0
afe_init_t = 0.0
helium_y = ''
model='dar'


from isochrones.dartmouth import Dartmouth_FastIsochrone
darm2 = Dartmouth_FastIsochrone(afe='afem2', y=helium_y)
darp0 = Dartmouth_FastIsochrone(afe='afep0', y=helium_y)
darp2 = Dartmouth_FastIsochrone(afe='afep2', y=helium_y)
darp4 = Dartmouth_FastIsochrone(afe='afep4', y=helium_y)
darp6 = Dartmouth_FastIsochrone(afe='afep6', y=helium_y)
darp8 = Dartmouth_FastIsochrone(afe='afep8', y=helium_y)

#-----------------------------------------------------------------------
#file to be loaded
rescale = np.loadtxt('rescale_ig.csv',delimiter=',')
ind1 = np.loadtxt('ind_met15.txt')
ind2 = np.loadtxt('ind_met20.txt')
ind3 = np.loadtxt('ind_met175m.txt')
ind4 = np.loadtxt('ind_met175p.txt')

version2 = '15'
model = 'dar'
model2 = 'dar'

Age_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(2,))
metal_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(5,))
distance_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(8,))
Abs_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(11,))
Afe_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(14,))

chunkbot = rescale[:,5]

Zsun = 0.0134
Fe_h = -2.0
c = 2.9979e8 # in meters
L0 = 3.0128e28 #zero-point luminosity in W
Lsun = 3.828e26 #solar luminosity in W
rsun = 6.957e+10	# in centimeters
msun = 1.989e+33 #g




#-----------------------------------------------------------------------
# plot total start

name = ['a100','a125','a150','a175','a200']
name2 = ['a100','a120','a140','a160','a180','a200']

#-----------------------------------------------------------------------
# ~p0 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_fid.data')
# ~m0 = cut(p0) # cut pre main sequence
# ~V0, R0 = p0[m0:,9], p0[m0:,10]
# ~#-----------------------------------------------------------------------
#varying mixing length
# ~p2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a120.data')
# ~m2 = cut(p2) # cut pre main sequence
# ~V2, R2 = p2[m2:,9], p2[m2:,10]
# ~p3 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a140.data')
# ~m3 = cut(p3) # cut pre main sequence
# ~V3, R3 = p3[m3:,9], p3[m3:,10]
# ~p4 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a160.data')
# ~m4 = cut(p4) # cut pre main sequence
# ~V4, R4 = p4[m4:,9], p4[m4:,10]
# ~p5 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a180.data')
# ~m5 = cut(p5) # cut pre main sequence
# ~V5, R5 = p5[m5:,9], p5[m5:,10]
# ~p6 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a200.data')
# ~m6 = cut(p6) # cut pre main sequence
# ~V6, R6 = p6[m6:,9], p6[m6:,10]
# ~p7 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a220.data')
# ~m7 = cut(p7) # cut pre main sequence
# ~V7, R7 = p7[m7:,9], p7[m7:,10]
# ~p8 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a240.data')
# ~m8 = cut(p8) # cut pre main sequence
# ~V8, R8 = p8[m8:,9], p8[m8:,10]
# ~p9 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a260.data')
# ~m9 = cut(p9) # cut pre main sequence
# ~V9, R9 = p9[m9:,9], p9[m9:,10]
# ~p10 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a280.data')
# ~m10 = cut(p10) # cut pre main sequence
# ~V10, R10 = p10[m10:,9], p10[m10:,10]
#-----------------------------------------------------------------------
# varying mass
# ~p5 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_M070.data')
# ~m5 = cut(p0) # cut pre main sequence
# ~V5, R5 = p5[m5:,9], p5[m5:,10]
# ~p1 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_M090.data')
# ~m1 = cut(p1) # cut pre main sequence
# ~V1, R1 = p1[m1:,9], p1[m1:,10]
# ~#-----------------------------------------------------------------------
# ~# No ledoux parameter
# ~p9 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_ledoux.data')
# ~m9 = cut(p9) # cut pre main sequence
# ~V9, R9 = p9[m9:,9], p9[m9:,10]
#-----------------------------------------------------------------------
# COX MLT
# ~p10 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_cox.data')
# ~m10 = cut(p10) # cut pre main sequence
# ~V10, R10 = p10[m10:,9], p10[m10:,10]
# ~#-----------------------------------------------------------------------
# ~# varying helium
# ~p11 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_y020.data')
# ~m11 = cut(p11) # cut pre main sequence
# ~V11, R11 = p11[m11:,9], p11[m11:,10]
# ~p12 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_y028.data')
# ~m12 = cut(p12) # cut pre main sequence
# ~V12, R12 = p12[m12:,9], p12[m12:,10]
#-----------------------------------------------------------------------
# type2
# ~p13 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_type2.data')
# ~m13 = cut(p13) # cut pre main sequence
# ~V13, R13 = p13[m13:,9], p13[m13:,10]
# ~#-----------------------------------------------------------------------
# ~# rotational mixing
# ~p14 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_rotmix.data')
# ~m14 = cut(p14) # cut pre main sequence
# ~V14, R14 = p14[m14:,9], p14[m14:,10]
# ~#-----------------------------------------------------------------------
# ~# diffusion
# ~p15 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_diff.data')
# ~m15 = cut(p15) # cut pre main sequence
# ~V15, R15 = p15[m15:,9], p15[m15:,10]
# ~#-----------------------------------------------------------------------
# ~# rgb wind
# ~p16 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_nowind.data')
# ~m16 = cut(p16) # cut pre main sequence
# ~V16, R16 = p16[m16:,9], p16[m16:,10]
#-----------------------------------------------------------------------
# overshoot
# ~p17 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_overshoot.data')
# ~m17 = cut(p17) # cut pre main sequence
# ~V17, R17 = p17[m17:,9], p17[m17:,10]

########################################################################
########################################################################
#-----------------------------------------------------------------------
##~# create sample file ####
#-----------------------------------------------------------------------

# ~for glc in list(range(27))+list(range(28,69)):
	# ~print("the chosen cluster is {}".format(glc))
	# ~clus_nb, Age, Metal, distance, Abs, afe_init, distplus, distmoins  = cluster(glc)
	# ~print(clus_nb, Age, Metal, distance, Abs, afe_init, distplus, distmoins)
	# ~photo_v, err_v, photo_i, color, err_color, nmv, nmi, longueur = photometry()

	# ~if glc < 27:
		# ~age = Age_dar[glc]
		# ~metal = metal_dar[glc]
		# ~dist = distance_dar[glc]
		# ~Abso = Abs_dar[glc]
		# ~afe = Afe_dar[glc]
	# ~else:
		# ~age = Age_dar[glc-1]
		# ~metal = metal_dar[glc-1]
		# ~dist = distance_dar[glc-1]
		# ~Abso = Abs_dar[glc-1]
		# ~afe = Afe_dar[glc-1]
	# ~print(Metal, metal)
	# ~with open('/home/david/codes/Analysis/GC_mixing_length/metal_comp.txt', 'a+') as fid_file:
		# ~fid_file.write('%.3f %.3f \n' %(Metal, metal))
	# ~fid_file.close()

	# ~if metal <= -1.499 and metal >= -1.749:
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/ind_met175m.txt', 'a+') as fid_file:
			# ~fid_file.write('%s \n' %(str(glc)))
		# ~fid_file.close()
	# ~if metal <= -1.749 and metal >= -1.999:
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/ind_met175p.txt', 'a+') as fid_file:
			# ~fid_file.write('%s \n' %(str(glc)))
		# ~fid_file.close()
# ~kill


########################################################################
########################################################################
#-----------------------------------------------------------------------
##~# main code ####
#-----------------------------------------------------------------------
# ~string_met = 'Z00005'
# ~smass = ['M080']
# ~string_name1 = ['Z00020']
smass = ['M075']
# ~string_name1 = ['Z00005', 'Z00010', 'Z00015', 'Z00020']
string_name1 = ['Z00020']
string_label = ['0.00005', '0.00010', '0.00015', '0.00020']
string_name2 = ['Z00015', 'Z00020', 'Z00025', 'Z00030']
string_name3 = ['Z00025', 'Z00030', 'Z00035', 'Z00040']

met = (input("what is the metallicity limit ? "))
for string_mass in smass:
	#-----------------------------------------------------------------------
	# global variable
	if met == '-1.5':
		ind = ind1
	elif met == '-2.0':
		ind = ind2
		string_name = string_name1
		Gname ='G1'
	elif met == '-1.75p':
		ind = ind4
		string_name = string_name2
		Gname ='G2'
	if met == '-1.75m':
		ind = ind3
		string_name = string_name3
		Gname ='G3'

	exV = 0.9110638171893733
	exI = 0.5641590452038215

	ctot = []
	vtot = []
	ctot_sample = []
	vtot_sample = []
	ctot_sample2 = []
	vtot_sample2 = []
	ctot_sample3 = []
	vtot_sample3 = []
	errtot = []
	errtotv = []
	errtot_sample = []
	errtotv_sample = []
	mean_stop = []

	isov = np.zeros((len(ind),265))
	isoc = np.zeros((len(ind),265))

	# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dist_alpha.txt', 'a+') as fid_file:
		# ~fid_file.write('%s %s %s %s \n' %('#GC name', 'M0', 'M0 + 2', 'M0 + 4'))
	# ~fid_file.close()
	for indmet, string_met in enumerate(string_name):
		print(string_mass, string_met)
		# ~print(str(string_label[indmet]))
		if (string_met == 'Z00020'and string_mass == 'M080'):
			h32 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a165.data')
			col32,mag32, mp = cut2(h32)
			h33 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a235.data')
			col33,mag33, mp = cut2(h33)
			h30 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a196.data')
			col30,mag30, mp = cut2(h30)
			h31 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a204.data')
			col31,mag31, mp = cut2(h31)
			h1 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M070.data')
			col1,mag1, mp = cut2(h1)
			h2 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M090.data')
			col2,mag2, mp2 = cut2(h2)
		if (string_met == 'Z00005'and string_mass == 'M075'):
			h30 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M070.data')
			col30,mag30, mp = cut2(h30)
			h31 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M080.data')
			col31,mag31, mp = cut2(h31)
			h1 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M065.data')
			col1,mag1, mp = cut2(h1)
			h2 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M085.data')
			col2,mag2, mp2 = cut2(h2)
			h3 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_y020.data')
			col3,mag3, mp3 = cut2(h3)
			h4 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_y028.data')
			col4,mag4, mp4 = cut2(h4)
			h5 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_ledoux.data')
			col5,mag5, mp5 = cut2(h5)
			h6 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_cox.data')
			col6,mag6, mp6 = cut2(h6)
			h7 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_type2.data')
			col7,mag7, mp7 = cut2(h7)
			h8 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_overshoot.data')
			col8,mag8, mp8 = cut2(h8)
			h9 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_reimers1.data')
			col9,mag9, mp9 = cut2(h9)
			h10 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_diffusion.data')
			col10,mag10, mp10 = cut2(h10)
			h11 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_mixing.data')
			col11,mag11, mp11 = cut2(h11)
			# h23 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_alphafe.data')
			# col23,mag23, mp23 = cut2(h23)
		# ~h32 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a155.data')
		# ~col32,mag32, mp = cut2(h32)
		# ~h33 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a225.data')
		# ~col33,mag33, mp = cut2(h33)

#----------------------------------------------------------------------- MIST bC tables
		else:
			h12 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a120.data')
			col12,mag12, mp12 = cut2(h12)
			h13 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a140.data')
			col13,mag13, mp13 = cut2(h13)
			h14 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a160.data')
			col14,mag14, mp14 = cut2(h14)
			h15 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a180.data')
			col15,mag15, mp15 = cut2(h15)
			h17 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a200.data')
			col17,mag17, mp17 = cut2(h17)
			h19 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a220.data')
			col19,mag19, mp19 = cut2(h19)
			h20 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a240.data')
			col20,mag20, mp20 = cut2(h20)
			h21 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a260.data')
			col21,mag21, mp21 = cut2(h21)
			h22 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a280.data')
			col22,mag22, mp22 = cut2(h22)
#------------------------------------------------------------------------

			# ~h12 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a120.data')
			# ~col12,mag12, mp12 = cut2(h12)
			h28 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a130.data')
			col28,mag28, mp28 = cut2(h28)
			# ~h13 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a140.data')
			# ~col13,mag13, mp13 = cut2(h13)
			h26 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a150.data')
			col26,mag26, mp26 = cut2(h26)
			# ~h14 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a160.data')
			# ~col14,mag14, mp14 = cut2(h14)
			h24 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a170.data')
			col24,mag24, mp24 = cut2(h24)
			# ~h15 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a180.data')
			# ~col15,mag15, mp15 = cut2(h15)
			h16 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a190.data')
			col16,mag16, mp16 = cut2(h16)
			# ~h17 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a200.data')
			# ~col17,mag17, mp17 = cut2(h17)
			h18 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a210.data')
			col18,mag18, mp18 = cut2(h18)
			# ~h19 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a220.data')
			# ~col19,mag19, mp19 = cut2(h19)
			h25= mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a230.data')
			col25,mag25, mp25= cut2(h25)
			# ~h20 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a240.data')
			# ~col20,mag20, mp20 = cut2(h20)
			h27= mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a250.data')
			col27,mag27, mp27= cut2(h27)
			# ~h21 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a260.data')
			# ~col21,mag21, mp21 = cut2(h21)
			h29= mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a270.data')
			col29,mag29, mp29= cut2(h29)
			# ~h22 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a280.data')
			# ~col22,mag22, mp22 = cut2(h22)


		#-----------------------------------------------------------------------
		# interpolate the tracks
		finterp =np.zeros(17)

		f12 = interpolate.interp1d(mag12,col12, 'nearest',fill_value="extrapolate")
		f13 = interpolate.interp1d(mag28,col28, 'nearest',fill_value="extrapolate")
		f14 = interpolate.interp1d(mag13,col13, 'nearest',fill_value="extrapolate")
		f15 = interpolate.interp1d(mag26,col26, 'nearest',fill_value="extrapolate")
		f16 = interpolate.interp1d(mag14,col14, 'nearest',fill_value="extrapolate")
		f17 = interpolate.interp1d(mag24,col24, 'nearest',fill_value="extrapolate")
		f18 = interpolate.interp1d(mag15,col15, 'nearest',fill_value="extrapolate")
		f19 = interpolate.interp1d(mag16,col16, 'nearest',fill_value="extrapolate")
		f20 = interpolate.interp1d(mag17,col17, 'nearest',fill_value="extrapolate")
		f21 = interpolate.interp1d(mag18,col18, 'nearest',fill_value="extrapolate")
		f22 = interpolate.interp1d(mag19,col19, 'nearest',fill_value="extrapolate")
		f23 = interpolate.interp1d(mag25,col25, 'nearest',fill_value="extrapolate")
		f24 = interpolate.interp1d(mag20,col20, 'nearest',fill_value="extrapolate")
		f25 = interpolate.interp1d(mag27,col27, 'nearest',fill_value="extrapolate")
		f26 = interpolate.interp1d(mag21,col21, 'nearest',fill_value="extrapolate")
		f27 = interpolate.interp1d(mag29,col29, 'nearest',fill_value="extrapolate")
		f28 = interpolate.interp1d(mag22,col22, 'nearest',fill_value="extrapolate")


		alpha_mix = ['1.2','1.3','1.4','1.5','1.6','1.7','1.8','1.9','2.0','2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8']
		alpha_mix2 = np.array([1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
		finterp = [f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28]


		########################################################################
		########################################################################
		#-----------------------------------------------------------------------


		for ig, g in enumerate(ind):
			glc = int(g)
			# ~print(glc)
			mstop = chunkbot[glc]
		#-----------------------------------------------------------------------
		#get best fit for each GC
			print("the chosen cluster is {}".format(glc))
			clus_nb, Age, Metal, distance, Abs, afe_init, distplus, distmoins  = cluster(glc)
			print(clus_nb, Age, Metal, distance, Abs, afe_init, distplus, distmoins)
			photo_v, err_v, photo_i, color, err_color, nmv, nmi, longueur = photometry()
		#-----------------------------------------------------------------------
		#remove cluster 49 because of multiple populations
			if glc == 49:
				continue
		#-----------------------------------------------------------------------
		#rescale gc to compute absolute magnitude
			if glc < 27:
				age = Age_dar[glc]
				metal = metal_dar[glc]
				dist = distance_dar[glc]
				Abso = Abs_dar[glc]
				afe = Afe_dar[glc]
			else:
				age = Age_dar[glc-1]
				metal = metal_dar[glc-1]
				dist = distance_dar[glc-1]
				Abso = Abs_dar[glc-1]
				afe = Afe_dar[glc-1]

			dm = 5*np.log10(dist) - 5
			abV = Abso*exV
			abcol = Abso*exV - Abso*exI

			corr_mag = photo_v - dm - abV
			corr_col = color - abcol

			# ~print(Metal, metal)
		#-----------------------------------------------------------------------
		# compute isochrones for each GC
			helium_y = ''
			from isochrones.dartmouth import Dartmouth_FastIsochrone
			darm2 = Dartmouth_FastIsochrone(afe='afem2', y=helium_y)
			darp0 = Dartmouth_FastIsochrone(afe='afep0', y=helium_y)
			darp2 = Dartmouth_FastIsochrone(afe='afep2', y=helium_y)
			darp4 = Dartmouth_FastIsochrone(afe='afep4', y=helium_y)
			darp6 = Dartmouth_FastIsochrone(afe='afep6', y=helium_y)
			darp8 = Dartmouth_FastIsochrone(afe='afep8', y=helium_y)
			### create a sample from best fit
			afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8] 

			afe_max = afe_values[np.searchsorted(afe_values, afe)]
			afe_min = afe_values[np.searchsorted(afe_values, afe)-1]


			mag_v1_min , mag_i1_min, Color_iso1_min, eep_first = iso_mag(np.log10(age*1.e9), metal, dist, Abso, afe_min)
			mag_v1_max , mag_i1_max, Color_iso1_max, eep_first = iso_mag(np.log10(age*1.e9), metal, dist, Abso, afe_max)
			lpp = (min(len(mag_v1_min), len(mag_v1_max))) # get minimum length to interpolate
			
			mag_v1 = (mag_v1_min[:lpp]*(afe_max - afe) + mag_v1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
			Color_iso1 = (Color_iso1_min[:lpp]*(afe_max - afe) + Color_iso1_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)


			mag_v1 = mag_v1 - dm - abV
			Color_iso1 = Color_iso1 - abcol

			fiso = interpolate.interp1d(Color_iso1,mag_v1, 'nearest',fill_value="extrapolate")
			# ~isoc.append(Color_iso1)
			# ~isov.append(mag_v1)
			isoc[ig, :] = Color_iso1[:265]
			isov[ig, :] = mag_v1[:265]

		#-----------------------------------------------------------------------
		# remove hb, outliers stars and rgb stars
			fmag = interpolate.interp1d(mag_v1, Color_iso1, 'nearest',fill_value="extrapolate")
			Color_new = fmag(corr_mag)
			col_dist = np.abs(Color_new - corr_col)

			width = 0.06
			M0 = mstop - dm - abV - 1.5
			M1 = mstop - dm - abV - 3.5
			M2 = mstop - dm - abV - 5.5
			rgb = np.where(corr_mag < M0)[0]
			close = np.where(col_dist[rgb] < width)[0]
			rgb2 = np.where(corr_mag < M1)[0]
			close2 = np.where(col_dist[rgb2] < width)[0]
			rgb3 = np.where(corr_mag < M2)[0]
			close3 = np.where(col_dist[rgb3] < width)[0]

			# ~print(len(rgb))

			cocol = corr_col[rgb][close]
			comag = corr_mag[rgb][close]
			cocol2 = corr_col[rgb2][close2]
			comag2 = corr_mag[rgb2][close2]
			cocol3 = corr_col[rgb3][close3]
			comag3 = corr_mag[rgb3][close3]

			ctot.extend(corr_col)
			vtot.extend(corr_mag)
			ctot_sample.extend(cocol)
			vtot_sample.extend(comag)
			ctot_sample2.extend(cocol2)
			vtot_sample2.extend(comag2)
			ctot_sample3.extend(cocol3)
			vtot_sample3.extend(comag3)
			errtot.extend(err_color)
			errtotv.extend(err_v)
			errtot_sample.extend(err_color[rgb][close])
			errtotv_sample.extend(err_v[rgb][close])

			vcenter, ccenter, errcenter, sbin, bingood, errcenterv = way(corr_mag[rgb][close], corr_col[rgb][close], err_color[rgb][close], err_v[rgb][close])

			std = np.sqrt(np.sum(col_dist[rgb][close]**2)/len(col_dist[rgb][close]))
			std = np.sqrt(np.sum(col_dist[rgb][close]**2)/len(col_dist[rgb][close]))
			std2 = np.sqrt(np.sum(col_dist[rgb2][close2]**2)/len(col_dist[rgb2][close2]))
			# ~print(std, len(col_dist[rgb][close]))
			# ~if string_met == 'Z00005':
				# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dispersion/'+Gname+str(clus_nb)+'.txt', 'a+') as fid_file:
					# ~fid_file.write('%s \n' %(str(clus_nb)))
					# ~fid_file.write('%.4f \n' %(std))
				# ~fid_file.close()
				

		#-----------------------------------------------------------------------
		# compute alpha best fit and distribution

			# ~magpts = np.where(mag_v1 < M0)[0]
			magpts = np.where(mag_v1 < M1)[0]

			magpts = np.where(mag_v1 < 0)[0]
			dis, bestalpha = chi2(Color_iso1[magpts],mag_v1[magpts])
			print(alpha_mix[bestalpha])

			# ~abest = alpha_mix2[bestalpha +1]*(1-dis[bestalpha+1]/(dis[bestalpha-1]+dis[bestalpha+1])) +  alpha_mix2[bestalpha -1]*(dis[bestalpha-1]/(dis[bestalpha-1]+dis[bestalpha+1]))

			# ~print(abest)

			
			# ~histo, dx, dy = alpha_distribution(corr_col[rgb][close],corr_mag[rgb][close], bestalpha)
			# ~histo2, dx2, dy2 = alpha_distribution(corr_col[rgb2][close2],corr_mag[rgb2][close2], bestalpha)
			# ~histo3, dx3, dy3 = alpha_distribution(corr_col[rgb3][close3],corr_mag[rgb3][close3], bestalpha)
			histo = alpha_distribution(corr_col[rgb][close],corr_mag[rgb][close], bestalpha)
			histo2 = alpha_distribution(corr_col[rgb2][close2],corr_mag[rgb2][close2], bestalpha)
			histo3 = alpha_distribution(corr_col[rgb3][close3],corr_mag[rgb3][close3], bestalpha)

			# ~print(len(corr_mag[rgb2][close2]), len(corr_mag[rgb3][close3]))
			if len(corr_mag[rgb2][close2]) < 10:
				histo2 = 999
			if len(corr_mag[rgb3][close3]) < 10:
				histo3 = 999
			# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dispersion/'+Gname+'_'+string_met+'.txt', 'a+') as fid_file:
				# ~fid_file.write('%s %.4f %.4f %s %.2f %.2f %.2f \n' %(str(clus_nb),std,std2, alpha_mix[bestalpha], histo, histo2, histo3))
			# ~fid_file.close()


			print(np.mean(histo))
			print(np.mean(histo2))	
			print(np.mean(histo3))	


		plt.scatter(ctot,vtot, marker='.', s=10, alpha=0.8, color='grey')
		# ~plt.scatter(corr_col,corr_mag, marker='.', s=10, color='grey', alpha=0.8)
		# ~# plt.scatter(corr_col,corr_mag, marker='.', s=10, alpha=0.8)
		# ~# plt.scatter(corr_col[rgb][close],corr_mag[rgb][close], marker='.', s=10, color='r', alpha=0.8)
		# ~# plt.scatter(corr_col[rgb2][close2],corr_mag[rgb2][close2], marker='.', s=10, color='b', alpha=0.8)
		# ~# plt.scatter(corr_col[rgb3][close3],corr_mag[rgb3][close3], marker='.', s=10, color='r', alpha=0.8)
		# ~plt.plot(Color_iso1,mag_v1, label=r'$isochrone$ = 2.00', c='c', linewidth=2.)
		# ~# plt.scatter(dx,dy, marker='.', s=10, color='b', alpha=0.8)
		# ~# plt.scatter(dx2,dy2, marker='.', s=10, color='r', alpha=0.8)
		# ~# plt.scatter(dx3,dy3, marker='.', s=10, color='y', alpha=0.8)
		# ~#plt.scatter(ccenter,vcenter, marker='o', s=10, color='b', alpha=0.8)
		plt.plot(col12 , mag12, label=r'$\Delta_{\alpha}$ = 0.8', c='c')
		plt.plot(col13 , mag13, label=r'$\Delta_{\alpha}$ = 0.6', c='orange')
		# ~plt.plot(col26 , mag26, label=r'$\Delta_{\alpha}$ = 0.5', c='y')
		plt.plot(col14 , mag14, label=r'$\Delta_{\alpha}$ = 0.4', c='g')
		# ~plt.plot(col24 , mag24, label=r'$\Delta_{\alpha}$ = 0.3', c='m')
		plt.plot(col15 , mag15, label=r'$\Delta_{\alpha}$ = 0.2', c='b')
		# ~plt.plot(col16 , mag16, label=r'$\Delta_{\alpha}$ = 0.1', c='r')
		plt.plot(col17 , mag17, label=r'$\alpha_{MLT}$ = 2.00', c='k')
		# ~plt.plot(col18 , mag18, c='r')
		plt.plot(col19 , mag19, c='b')
		# ~plt.plot(col25 , mag25, c='m')
		plt.plot(col20 , mag20, c='g')
		# ~plt.plot(col27 , mag27, c='y')
		plt.plot(col21 , mag21, c='orange')
		plt.plot(col22 , mag22, c='c')
		# ~plt.axhline(M1)
		# ~plt.axhline(M2)
		# ~plt.xlim(-0.23,1.65)
		# ~plt.ylim(5,-5)
		# ~plt.gca().invert_yaxis()
		plt.legend(loc='best', fontsize = 12)
		plt.xlabel(' F606W - F814W', fontsize = 20)
		plt.ylabel(' F606W', fontsize = 20)
		plt.tick_params(labelsize=16)
		plt.show() 
		plt.close()
		kill

		########################################################################	
		########################################################################
		# FOR ALL GCs	
		#-----------------------------------------------------------------------
		# compute the mean isochrone and mean mstop

		iso_midc = np.mean(isoc, axis=0)
		iso_midv = np.mean(isov, axis=0)
		m_stop = np.mean(mean_stop)

		# ~plt.figure()
		# ~for i in range(len(ind)):
			# ~plt.plot(isoc[i], isov[i], color='grey')
		# ~plt.plot(iso_midc, iso_midv, color='r', label='mean')
		# ~plt.legend(loc='best', fontsize=20)
		# ~plt.xlim(-0.5,3)
		# ~plt.ylim(5,-5)
		# ~plt.tick_params(labelsize=16)
		# ~plt.xlabel('Rescaled color, F606W - F814W', fontsize = 20)
		# ~plt.ylabel('Rescaled magnitude, F606W', fontsize = 20)
		# ~plt.show() 
		# ~plt.close()

		#-----------------------------------------------------------------------
		# remove hb, outliers stars and rgb stars
		fmag_tot = interpolate.interp1d(iso_midv, iso_midc, 'nearest',fill_value="extrapolate")
		Color_new = fmag_tot(vtot_sample)
		col_dist_tot = np.abs(Color_new - ctot_sample)
		
		Color_new2 = fmag_tot(vtot_sample2)
		col_dist_tot2 = np.abs(Color_new2 - ctot_sample2)

		# rgb = np.where(corr_mag < m_stop - dm - abV - 0.5)[0]
		# vcentertot, ccentertot, errcentertot, sbintot, bingoodtot, errcentervtot = way(vtot_sample, ctot_sample, errtot_sample, errtotv_sample)
		# fmag_tot2 = interpolate.interp1d(vcentertot[2:], ccentertot[2:], 'nearest',fill_value="extrapolate")
		# lim_mag = np.where(vtot_sample > np.min(vcentertot[2:]))[0]
		# Color_new2 = fmag_tot(np.array(vtot_sample)[lim_mag])
		# col_dist_tot2 = np.abs(Color_new2 - np.array(ctot_sample)[lim_mag])

		std_tot = np.sqrt(np.sum(col_dist_tot**2)/len(col_dist_tot))

		std_tot2 = np.sqrt(np.sum(col_dist_tot2**2)/len(col_dist_tot2))
		# ~if string_met == 'Z00005':
			# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dispersion/All12.txt', 'a+') as fid_file:
				# ~fid_file.write('%s \n' %('All 12 GCs'))
				# ~fid_file.write('%.4f \n' %(std_tot))
			# ~fid_file.close()


		#-----------------------------------------------------------------------
		# ~# compute the alpha best fit and distribution
		magpts = np.where(iso_midv < np.max(vtot_sample2))[0]
		# magpts = np.where(iso_midv < 0)[0]
		dis,bestalpha = chi2(iso_midc[magpts],iso_midv[magpts])
		print(alpha_mix[bestalpha])
		# histotot, pxtot, pytot = alpha_distribution(ctot_sample,vtot_sample, bestalpha)
		# histotot2, pxtot2, pytot2 = alpha_distribution(ctot_sample2,vtot_sample2, bestalpha)
		# histotot3, pxtot3, pytot3 = alpha_distribution(ctot_sample3,vtot_sample3, bestalpha)
		histotot = alpha_distribution(ctot_sample,vtot_sample, bestalpha)
		histotot2 = alpha_distribution(ctot_sample2,vtot_sample2, bestalpha)
		histotot3 = alpha_distribution(ctot_sample3,vtot_sample3, bestalpha)

		if len(vtot_sample2) < 10:
			histotot2 = 999
		if len(vtot_sample3) < 10:
			histotot3 = 999

		# ~allgc = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dispersion/'+Gname+'_'+string_met+'.txt', usecols=(1,2,3,4,5,6))
		# ~besta = np.mean(allgc[:,2])
		# ~deltaa = np.std(allgc[:,2])
		# ~besthist = np.mean(allgc[:,3])
		# ~besthist2 = np.mean(allgc[:,4])
		# ~petit = np.where(allgc[:,5] == 999.00)[0]
		# ~besthist3 = np.mean(np.delete(allgc[:,5], petit))
		
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dispersion/'+Gname+'_'+string_met+'.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %s %.2f %.2f %.2f \n' %('All_12_GCs',std_tot, std_tot2, alpha_mix[bestalpha], histotot, histotot2, histotot3))
		# ~fid_file.close()


		# ~print(np.mean(histotot))
		# ~print(np.mean(histotot2))	
		# ~print(np.mean(histotot3))

		# ~print(np.max(vtot_sample))
		# ~print(np.max(vtot_sample2))
		# ~print(np.max(vtot_sample3))
		# ~kill

		########################################################################
		########################################################################

		# ~afe_values=[-0.2, 0.0 , 0.2, 0.4, 0.6, 0.8] 

		# ~afe_max = afe_values[np.searchsorted(afe_values, 0.2)]
		# ~afe_min = afe_values[np.searchsorted(afe_values, 0.2)-1]


		# ~mag_v0_min , mag_i0_min, Color_iso0_min, eep_first = iso_mag(np.log10(13.5e9), -2.0, 0.0, 0.0, afe_min)
		# ~mag_v0_max , mag_i0_max, Color_iso0_max, eep_first = iso_mag(np.log10(13.5e9), -2.0, 0.0, 0.0, afe_max)
		# ~lpp = (min(len(mag_v0_min), len(mag_v0_max))) # get minimum length to interpolate

		# ~mag_v0 = (mag_v0_min[:lpp]*(afe_max - afe) + mag_v0_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)
		# ~Color_iso0 = (Color_iso0_min[:lpp]*(afe_max - afe) + Color_iso0_max[:lpp]*(afe - afe_min)) / (afe_max - afe_min)

########################################################################
########################################################################
#-----------------------------------------------------------------------
### plot mixing length variation ###
#-----------------------------------------------------------------------
		
		# ~plt.figure()
		# plt.scatter(corr_col,corr_mag, marker='.', s=10, alpha=0.8)
		# ~plt.scatter(ctot,vtot, marker='.', s=10, color='grey', alpha=0.8)
		# ~# plt.scatter(ctot_sample,vtot_sample, marker='.', s=10, color='r', alpha=0.8)
		# ~# plt.scatter(ctot_sample2,vtot_sample2, marker='.', s=10, color='b', alpha=0.8)
		# ~# plt.scatter(ccentertot[2:],vcentertot[2:], marker='o', s=10, color='b', alpha=0.8)
		# plt.plot(iso_midc, iso_midv, c='c', label='mean of the 12 isochrones', linewidth=2.0)
		# plt.plot(V2-R2,V2, label=r'$\Delta_{\alpha}$ = 0.8', c='c')
		# plt.plot(V3-R3,V3, label=r'$\Delta_{\alpha}$ = 0.6', c='orange')
		# plt.plot(V4-R4,V4, label=r'$\Delta_{\alpha}$ = 0.4', c='r')
		# plt.plot(V5-R5,V5, label=r'$\Delta_{\alpha}$ = 0.2', c='b')
		# plt.plot(V6-R6,V6, label=r'$\alpha_{MLT}$ = 2.00', c='k')
		# ~# plt.plot(V7-R7,V7, c='b')
		# ~# plt.plot(V8-R8,V8, c='r')
		# ~# plt.plot(V9-R9,V9, c='orange')
		# ~# plt.plot(V10-R10,V10, c='c')
		#plt.scatter(pxtot,pytot, marker='.', s=10, color='b', alpha=0.8)
		#plt.scatter(pxtot2,pytot2, marker='.', s=10, color='r', alpha=0.8)
		# ~plt.plot(col32 , mag32, label=r'$\Delta_{\alpha}$ = 0.35', c='b', linestyle = ':')
		# ~plt.plot(col24 , mag24, label=r'$\Delta_{\alpha}$ = 0.2', c='b', linestyle = '--')
		# ~plt.plot(col15 , mag15, label=r'$\Delta_{\alpha}$ = 0.1', c='b')
		# plt.plot(col16 , mag16, label=r'$\Delta_{\alpha}$ = 0.1', c='b')
		# ~plt.plot(col17 , mag17, c='b')
		# ~plt.plot(col18 , mag18, c='b', linestyle = '--')
		# ~plt.plot(col33 , mag33, c='b', linestyle = ':')
#-----------------------------------
		# ~binc = 200
		# ~niso = int((-200 -(-236))/2)
		# ~niso = int((-172 -(-200))/2)
		# ~magtest= np.linspace(-5,5,binc)
		# ~col = np.zeros((binc,niso))
		# ~mag = np.zeros((binc,niso))
		# ~import cmasher as cmr
		# ~cm = cmr.ember
		# ~norm = colors.Normalize(vmin=-2.36,vmax=-2)
		# ~norm = colors.Normalize(vmin=-2.0,vmax=-1.75)
		# ~s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
		# ~s_m.set_array([])
		# ~import matplotlib.gridspec as gridspec
		# ~gs_in = gridspec.GridSpec(2, 2,hspace=0.5,height_ratios=[10,1],
		# ~width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
		# ~for ind,a in enumerate(range(-236, -200, 2)):
		# ~for ind,a in enumerate(range(-200, -172, 2)):
			# ~met= a/100. 

			# ~print(met)
			# ~mag_v, mag_i, Color_iso, eep_first = iso_mag(Age_t, met, distance_t, Abs_t, afe_init_t)
			# ~fmag_ini = interpolate.interp1d(mag_v, Color_iso, 'nearest',fill_value="extrapolate")
			
			# ~col[:,ind]= fmag_ini(magtest)
			# ~mag[:,ind]= magtest
			# ~if ind == 0:
				# ~plt.plot(Color_iso,mag_v, color=s_m.to_rgba(met), label='DSED isochrones')
			# ~if ind == 9:
			# ~if ind == 6:
				# ~plt.plot(Color_iso,mag_v, color=s_m.to_rgba(met), label='DSED isochrones')
			# ~if ind == 17:
			# ~if ind == 12:
				# ~plt.plot(Color_iso,mag_v, color=s_m.to_rgba(met), label='DSED isochrones')
			# ~else:
				# ~plt.plot(Color_iso,mag_v, color=s_m.to_rgba(met))
		# ~cbar = plt.colorbar(s_m)
		# ~cbar.ax.set_ylabel('Metallicity range', fontsize = 16)
#----------------------------------
		#plt.xlim(-1,2.5)
		# ~plt.xlim(-0.23,1.25)
		# ~plt.ylim(5,-5)
		# ~# plt.gca().invert_yaxis()
		# ~plt.tick_params(labelsize=16)
		# ~plt.subplots_adjust(bottom=0.15, top=0.89, right=0.930)
		# ~lgnd = plt.legend(loc='best', fontsize = 14)
		# lgnd.get_frame().set_edgecolor('k')
		# lgnd.get_frame().set_linewidth(2.0)
		# ~plt.xlabel(' F606W - F814W', fontsize = 20)
		# ~plt.ylabel(' F606W', fontsize = 20)
		# ~#plt.title('[Fe/H] < '+met+', '+str(len(ind))+' clusters', fontsize = 24)
		# ~plt.show() 
		# ~plt.close()

		# ~kill
########################################################################
########################################################################
#-----------------------------------------------------------------------
##~# plot convection configuration ####
#-----------------------------------------------------------------------
		# 
		# ~plt.figure()
		#plt.scatter(corr_col,corr_mag, marker='.', s=10, alpha=0.8)
		# ~plt.scatter(ctot,vtot, marker='.', s=10, color='grey', alpha=0.8)
		# ~plt.scatter(ctot_sample,vtot_sample, marker='.', s=10, color='k', alpha=0.8)
		# ~plt.axhline(np.max(vtot_sample), c='k', label=r'$\mathcal{M}_{0}$')
		# ~plt.axhline(np.max(vtot_sample2), c='r', label=r'$\mathcal{M}_{1}$')
		# ~plt.axhline(np.max(vtot_sample3), c='b', label=r'$\mathcal{M}_{2}$')
		# plt.scatter(ccentertot[2:],vcentertot[2:], marker='o', s=10, color='b', alpha=0.8)
		# plt.plot(iso_midc, iso_midv, c='r', label='mean of the 12 isochrones')
		#plt.plot(V0-R0,V0, label='fiducial' ,linewidth=2, c='k')

		# ~plt.plot(col17 , mag17, label='Fiducial', c='k')
		# ~plt.plot(col32 , mag32, label=r'$\Delta_{\alpha}$ = 0.35', c='k', linestyle='--')
		# ~plt.plot(col33 , mag33, c='k', linestyle='--')
		# ~plt.plot(col15 , mag15, label=r'$\Delta_{\alpha}$ = 0.2', c='k', linestyle=':')
		# ~plt.plot(col19 , mag19, c='k', linestyle=':')
		# ~plt.plot(col30 , mag30, label=r'$\Delta_{\alpha}$ = 0.04', c='k', linestyle=(0, (3, 1, 1, 1, 1, 1)))
		# ~plt.plot(col31 , mag31, c='k', linestyle=(0, (3, 1, 1, 1, 1, 1)))
		# ~plt.plot(col1 , mag1, label=r'$\Delta$M = 0.1', c='b')
		# ~plt.plot(col2 , mag2, c='b')
		# ~plt.plot(col3 , mag3, label=r'$\Delta$Y = 0.04', c='g')
		# ~plt.plot(col4 , mag4, c='g')
		# ~plt.plot(col5 , mag5, label='No Ledoux criterion', linestyle=':')
		# ~plt.plot(col6 , mag6, label='Cox MLT theory', linestyle=':')
		# ~plt.plot(col7 , mag7, label='No type 2 opacities', linestyle=':')
		# ~plt.plot(col8 , mag8, label='No overshoot', linestyle=':')
		# ~plt.plot(col9 , mag9, label= r'Reimers $\eta$ = 0.8', linestyle=':', c='c')
		# ~plt.plot(col10 , mag10, label='No element diffusion', linestyle=':', c='r')
		# ~plt.plot(col11 , mag11, label='No rotational mixing', linestyle=':')

		# ~plt.xlim(-0.5,2.5)
		# ~plt.xlim(-0.23,1.65)
		# ~plt.ylim(5,-5)
		# ~plt.xlim(0.5,1.4)
		# ~plt.ylim(-3.5, 0)
		# ~plt.xlim(0.68,0.83)
		# ~plt.ylim(-0.9, -0.3)
		# ~plt.gca().invert_yaxis()
		# ~plt.tick_params(labelsize=14)
		# ~plt.subplots_adjust(bottom=0.15, top=0.89)
		# ~lgnd = plt.legend(loc='best', fontsize = 12)
		# ~# lgnd.get_frame().set_edgecolor('k')
		# ~# lgnd.get_frame().set_linewidth(2.0)
		# ~plt.xlabel(' F606W - F814W', fontsize = 20)
		# ~plt.ylabel(' F606W', fontsize = 20)
		# ~#plt.title('[Fe/H] < '+met+', '+str(len(ind))+' clusters', fontsize = 24)
		# ~plt.show() 
		# ~plt.close()
		# ~kill
########################################################################
########################################################################
#-----------------------------------------------------------------------
### plot raul values ####
#-----------------------------------------------------------------------
		
		# ~plt.figure()
		# ~#plt.scatter(corr_col,corr_mag, marker='.', s=10, alpha=0.8)
		# ~plt.scatter(ctot,vtot, marker='.', s=10, color='grey', alpha=0.8)
		# ~c = ['g','b','c','orange','r']
		# ~name = ['a100','a125','a150','a175','a200']
		# ~name2 = [r'$\rm \Delta_{\alpha}(JimMacD) = -0.4$',r'$\rm \Delta_{\alpha}(JimMacD) = -0.15$',r'$\rm \Delta_{\alpha}(JimMacD) = +0.1$',r'$\rm \Delta_{\alpha}(JimMacD) = +0.35$',r'$\rm \Delta_{\alpha}(JimMacD) = +0.6$']
		# ~dat = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/fehm200.HST_ACSWF')
		# ~tefs = np.unique(dat[:,0])
		# ~sfs = np.unique(dat[:,1])
		# ~feH = np.unique(dat[:,2])
		# ~plt.plot(col32 , mag32, label=r'$\rm \Delta_{\alpha}(MESA) = \pm 0.35$', c='k', linestyle='--')
		# ~plt.plot(col33 , mag33, c='k', linestyle='--')
		# ~plt.plot(col16 , mag16, c='k', label=r'$\rm \alpha_{MESA} = 1.9$',linestyle=':')
		# ~plt.plot(col17 , mag17, c='k', label=r'$\rm \alpha_{MESA} = 2.0$')
		# ~for count,j in enumerate(name):	
			# ~raul = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/JimMacD/'+j+'.txt')
			# ~sf = raul[:,2]
			# ~teff = 10**(raul[:,1])
			# ~logR = raul[:,9]
			# ~safe = np.where((teff > np.min(tefs))&(teff < np.max(tefs))&(sf > np.min(sfs))&(sf < np.max(sfs)))[0]
			# ~teff = teff[safe]
			# ~sf = sf[safe]
			# ~Mbol = -2.5*np.log10((Lsun*10**raul[:,5])/L0)[safe]
			# ~M606 = np.zeros(len(Mbol)) 
			# ~M814 = np.zeros(len(Mbol))

			# ~for i in range(len(Mbol)):
				# ~bc606, bc814 = interp_eep(teff[i], sf[i], tefs, sfs, dat)
				# ~M606[i] = Mbol[i] - bc606 
				# ~M814[i] = Mbol[i] - bc814
			# ~fcurve = interpolate.interp1d(M606, M606-M814)
			# ~mpts = np.linspace(np.min(M606), np.max(M606), 20)
			
			# ~plt.plot(M606-M814, M606)
			# ~plt.plot(fcurve(mpts), mpts, label=name2[count], color=c[count])
		# ~plt.xlim(-0.5,2.5)
		# ~plt.xlim(-0.23,1.65)
		# ~plt.ylim(5,-5)
		# ~plt.tick_params(labelsize=14)
		# ~plt.subplots_adjust(bottom=0.15, top=0.89)
		# ~lgnd = plt.legend(loc='best', fontsize = 12)
		# ~# lgnd.get_frame().set_edgecolor('k')
		# ~# lgnd.get_frame().set_linewidth(2.0)
		# ~plt.xlabel(' F606W - F814W', fontsize = 20)
		# ~plt.ylabel(' F606W', fontsize = 20)
		# ~#plt.title('[Fe/H] < '+met+', '+str(len(ind))+' clusters', fontsize = 24)
		# ~plt.show() 
		# ~plt.close()
		# ~kill
########################################################################
########################################################################
#-----------------------------------------------------------------------
### plot cmetallicity variation ####
#-----------------------------------------------------------------------
	
		# ~if string_mass == 'M075':
			# ~plt.plot(col16 , mag16, label='Z = '+str(string_label[indmet]))
		# ~else:
			# ~plt.plot(col16 , mag16, linestyle='--')
########################################################################
########################################################################
#-----------------------------------------------------------------------
### plot cmetallicity variation ####
#-----------------------------------------------------------------------

		# ~plt.plot(col1 , mag1, label = r'$M = 0.65$')
		# ~plt.plot(col30 , mag30, label = r'$M = 0.70$')
		# ~plt.plot(col16 , mag16, label = r'$M = 0.75$')
		# ~plt.plot(col31 , mag31, label = r'$M = 0.80$')
		# ~plt.plot(col2 , mag2, label = r'$M = 0.85$')
			
		# ~#plt.scatter(corr_col,corr_mag, marker='.', s=10, alpha=0.8)
		# ~plt.scatter(ctot,vtot, marker='.', s=10, color='grey', alpha=0.8)
		#plt.xlim(-0.5,2.5)
		# ~plt.xlim(-0.23,1.65)
		# ~plt.ylim(5,-5)
		# ~plt.tick_params(labelsize=14)
		# ~plt.subplots_adjust(bottom=0.15, top=0.89)
		# ~lgnd = plt.legend(loc='best', fontsize = 12)
		# ~# plt.text(1.0,4,r'0.75 $M_{\odot}$, $\alpha$ = 1.9',va='center',fontsize=16,alpha=1.,
			# ~bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
		# ~plt.text(0.87,4,r'$Z = 0.00005, \alpha = 1.9$',va='center',fontsize=16,alpha=1.,
			# ~bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
		# ~# lgnd.get_frame().set_edgecolor('k')
		# ~# lgnd.get_frame().set_linewidth(2.0)
		# ~plt.xlabel(' F606W - F814W', fontsize = 20)
		# ~plt.ylabel(' F606W', fontsize = 20)
		# ~#plt.title('[Fe/H] < '+met+', '+str(len(ind))+' clusters', fontsize = 24)
		# ~plt.show() 
		# ~plt.close()
		# ~kill
########################################################################
########################################################################
#-----------------------------------------------------------------------
### plot overlap ####
#-----------------------------------------------------------------------
		# ~plt.figure()
		# ~plt.scatter(ctot,vtot, marker='.', s=10, alpha=0.8)
		# ~plt.plot(isoc,isov, c='k')
		# ~plt.xlim(-1.0,2.5)
		# ~# plt.xlim(-0.23,1.65)
		# ~plt.ylim(5,-5)
		# ~plt.tick_params(labelsize=14)
		# ~plt.subplots_adjust(bottom=0.15, top=0.89)
		# ~# lgnd = plt.legend(loc='best', fontsize = 12)
		# ~# lgnd.get_frame().set_edgecolor('k')
		# ~# lgnd.get_frame().set_linewidth(2.0)
		# ~plt.xlabel(' F606W - F814W', fontsize = 20)
		# ~plt.ylabel(' F606W', fontsize = 20)
		# ~plt.title('[Fe/H] < '+met+', '+str(len(ind))+' clusters', fontsize = 24)
		# ~plt.title('-2.0 < [Fe/H] < -1.75, '+str(len(ind))+' clusters', fontsize = 24)
		# ~plt.title('-1.75 < [Fe/H] < -1.5, '+str(len(ind)-1)+' clusters', fontsize = 24)
		# ~plt.show() 
		# ~plt.close()
		# ~kill
