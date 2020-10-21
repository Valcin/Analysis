
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
# ~from emcee import PTSampler
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

	Age = float(round(Age,3))
	metal = float(round(float(metal),3))
	distance = float(round(distance,3))
	Abs = float(round(Abs,3))
	afe_init = float(afe_init)

	extdutra = [0.22,0.03,0.03,0.01,0.22,0.06,0.14,0.21,0.10,0.37,0.41,0.52,0.07,0.23,0.41,0.11,0.11,0.063,0.25,0.08]
	extgc = [1,3,5,6,8,12,16,23,24,25,26,31,34,40,41,46,48,51,53,58]
	if nb in extgc:
		ind = extgc.index(nb)
		Abs = 3.1*extdutra[ind]

	return clus_nb, Age, metal, distance, Abs, afe_init, distplus, distmoins

def photometry():
        
	files = np.loadtxt('/home/david/codes/Analysis/GC/data/hlsp_acsggct_hst_acs-wfc_'+clus_nb+'_r.rdviq.cal.adj.zpt', skiprows = 3)
	longueur = len(files)
	#----------------------------------------------------------------------
	#----------------------------------------------------------------------
	# filter the photmetric error -----------------------------------------

	lim606_left = np.percentile(files[:,4], 5)
	lim606_right = np.percentile(files[:,4], 95)
	lim814_left = np.percentile(files[:,8], 5)
	lim814_right = np.percentile(files[:,8], 95)
	filter_photo = np.where((files[:,4] < lim606_right) & (files[:,4] > lim606_left) & (files[:,8] < lim814_right) 
	& (files[:,8] > lim814_left))[0]

	#----------------------------------------------------------------------
	#----------------------------------------------------------------------
	# filter the pixel frame -----------------------------------------------

	lim_x = np.percentile(files[:,15], 97.5)
	lim_y = np.percentile(files[:,16], 97.5)

	lim_x_left = np.percentile(files[:,15], 2.5)
	# since both lim left is 0, i only used lim right
	filter_xy = np.where((files[:,15] < lim_x)&(files[:,16] < lim_y))[0]

	#----------------------------------------------------------------------
	#----------------------------------------------------------------------
	# compute the turn-off point by selecting a band around the main sequence fit and analyse the drop
	nono = np.where((files[:,3] < 23) & (files[:,3] > 16) & (files[:,5] < 2) & (files[:,5] > 0))[0]

	slope, intercept, r_value, p_value, std_err = stats.linregress(files[nono,5],files[nono,3])
	x = np.linspace(0,2,200)
	f = slope*x + intercept

	titi = np.where((files[:,3] > (slope*files[:,5] + intercept) - 0.05*(slope*files[:,5] + intercept)) \
	 & (files[:,3] < (slope*files[:,5] + intercept) + 0.05*(slope*files[:,5] + intercept)))[0]
	hi, bind = np.histogram(files[titi,5], bins = 100)
	tp_x = np.argmax(np.diff(hi)) - 1

	#define the turn-off point coordinates given the drop
	top_x = bind[tp_x]
	top_y = slope*bind[tp_x] + intercept

	# select random data above and below the turn-off point (top)--------------
	### HERE I DON'T USE THE LEFT LIMIT BECAUSE IT REMOVES ALL THE BRIGHT STAR ABOVE THE TURN-OFF
	#~ filter_top = np.where((files[:,4] < lim606_right) & (files[:,4] > lim606_left) & (files[:,8] < lim814_right) 
	#~ & (files[:,8] > lim814_left) & (files[:,15] < lim_x) & (files[:,16] < lim_y) & (files[:,3] < top_y))[0]
	#~ filter_top = np.where((files[:,4] < lim606_right) &  (files[:,8] < lim814_right) 
	#~ & (files[:,15] < lim_x) & (files[:,16] < lim_y) & (files[:,3] < top_y))[0]

	min_mag = np.min(files[:,3])
	min_mag = np.min(files[:,3])
	mag_min = min_mag
	mag_max = 28

	#~ filter_bottom = np.where((files[:,4] < lim606_right) & (files[:,4] > lim606_left) & (files[:,8] < lim814_right) 
	#~ & (files[:,8] > lim814_left) & (files[:,15] < lim_x) & (files[:,16] < lim_y) & (files[:,3] < mag_max))[0]
	filter_bottom = np.where((files[:,4] < lim606_right)  & (files[:,8] < lim814_right) 
	 & (files[:,15] < lim_x) & (files[:,16] < lim_y) & (files[:,3] < mag_max))[0]

	#~ ltop = (len(filter_top))

	### bright stars have good photometry and the error is 0. collapse later when divided by 0 so they are removed here
	#~ too_good_t = np.where(files[filter_top,4] != 0.0)[0]
	too_good_b = np.where(files[filter_bottom,4] != 0.0)[0]


	#~ if ltop > 1500:
		#~ sel_top = np.random.choice(filter_top[too_good_t], 1500)
		#~ sel_bottom = np.random.choice(filter_bottom[too_good_b], 1500)
	#~ else:
		#~ sel_top = np.random.choice(filter_top[too_good_t], ltop)
		#~ sel_bottom = np.random.choice(filter_bottom[too_good_b], ltop)

	#~ ###----------------
	#~ sel_total = np.concatenate((sel_bottom,sel_top))
	#~ l_total = len(sel_total)
	###----------------
	
	### define the total filter
	mg_cut = filter_bottom[too_good_b]
	sample = mg_cut.size

	### get the photmetry of the selected stars in both filters
	photo_v = files[mg_cut,3]
	lv = len(photo_v)
	err_v = files[mg_cut,4]
	range_v = np.max(files[mg_cut,3]) - np.min(files[mg_cut,3])

	photo_i = files[mg_cut,7]
	li = len(photo_i)
	err_i = files[mg_cut,8]
	range_i = np.max(files[mg_cut,7]) - np.min(files[mg_cut,7])

	#~ Pfield = 1/(range_v * range_i)
	xpos = files[mg_cut,0]
	ypos = files[mg_cut,2]
	rpos = np.sqrt(xpos**2 + ypos**2)
	Color = files[mg_cut,5]
	err_Color = files[mg_cut,6]
	err_V = files[mg_cut,4]
	nmv = files[mg_cut,11]
	nmi = files[mg_cut,12]
	del files
	gc.collect()
	
	

	
	return photo_v, err_v, photo_i, Color, err_Color, sample, top_x, top_y, nmv, nmi, longueur

def iso_mag(Age, metal, distance, A):


	if model == 'mist': 
		mag_v = mist.mageep['F606W'](Age, metal, distance, A)
		mag_i = mist.mageep['F814W'](Age, metal, distance, A)
		#~ mag_v = mist.mageep['F606'](Age, metal, distance, Abs)
		#~ mag_i = mist.mageep['F814'](Age, metal, distance, Abs)

	if model == 'dar': 
		mag_v = dar.mageep['F606W'](Age, metal, distance, A)
		mag_i = dar.mageep['F814W'](Age, metal, distance, A)
		
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
ndim = 4
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
	nwalkers = 300
	ntemps = 1
	print(ntemps)
	garr = [68]
	model = 'mist'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '12': 
	nwalkers = 300
	ntemps = 1
	print(ntemps)
	garr = [14]
	model = 'dar'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '8': 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [13,23,27,33,41,46,64,68]
	model = 'mist'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '9': 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [0,1,2,3,4,5]
	model = 'dar'
	#~ garr = [5,7,8,9,16,20,21,23,26,27,28]
if version == '10': 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [2,68]
	model = 'mist'
	#~ garr =
if version == '13': 
	nwalkers = 100
	ntemps = 1
	print(ntemps)
	garr = [2,68]
	model = 'dar'
	#~ garr =
if version == '0': 
	nwalkers = 300	
	version1 = '10' 
	model1 = 'mist'
	# ~version2 = '9' 
	version2 = '15' 
	model2 = 'dar'


###################################################################################################
###################################################################################################
binage = np.linspace(5,15,200)
bincenter = (binage[:-1] + binage[1:]) / 2
wtot = np.loadtxt('/home/david/codes/Analysis/GC/plots/w_tot')
w1 = np.loadtxt('/home/david/codes/Analysis/GC/plots/w_1')
w2 = np.loadtxt('/home/david/codes/Analysis/GC/plots/w_2')
renom_tot = np.loadtxt('/home/david/codes/Analysis/GC/plots/renom_tot')
renom_1 = np.loadtxt('/home/david/codes/Analysis/GC/plots/renom_1')
renom_2 = np.loadtxt('/home/david/codes/Analysis/GC/plots/renom_2')

binafe = np.linspace(-0.2,0.8,100)
bcenter = (binafe[:-1] + binafe[1:]) / 2
atot = np.loadtxt('/home/david/codes/Analysis/GC/plots/a_tot')
a_renom = np.loadtxt('/home/david/codes/Analysis/GC/plots/a_renom')

binfe = np.linspace(-2.5,0,200)
bfcenter = (binfe[:-1] + binfe[1:]) / 2
mtot = np.loadtxt('/home/david/codes/Analysis/GC/plots/m_tot')
m_renom = np.loadtxt('/home/david/codes/Analysis/GC/plots/m_renom')

#-----------------------------------------------------
### Licia's code

#~ window,0
  #~ nclu=68
  #~ nage=199
#~ nn=nclu*199
#~ aa=fltarr(nn)
#~ openr,1,'w_tot.txt'
#~ readf,1,aa
#~ close,1

#~ post=fltarr(nclu,nage)
#~ spost=fltarr(nclu,nage)
#~ dage=(15.-5.)/float(nage-1.)
#~ age=findgen(nage)*dage+5.

#~ ii=0
 #~ for i=0,nclu-1 do begin
    #~ for j=0,nage-1 do begin
       #~ post(i,j)=aa(ii)
       #~ ii=ii+1
    #~ endfor
    #~ post(i,*)=post(i,*)/total(post(i,*))  ; normalize
 #~ spost(i,*)=gauss_smooth(post(i,*),2.)  ; gaussian smoothing
 #~ spost(i,nage-7:nage-1)=post(i,nage-7:nage-1) ; egde truncate the gaussian smoothing
 #~ spost(i,*)=spost(i,*)/total(spost(i,*))  ; renormalize, just in case
 #~ endfor
 
  #~ posttot=fltarr(nage)
 #~ spost(*,*)=(spost(*,*)+1.d-8)*50. ; make zeros non zero but small so can take the log and then renomralize so logs are not too negatives, results do not change if you change this number as long 10^temp is not
 #~ for i=0,nage-1 do begin
    #~ temp=total(alog10(spost(*,i)))  ; take the log and sum
    #~ posttot(i)=10.d0^temp           ; undo the log
#~ endfor
 #~ plot,age, posttot/max(posttot),xr=[12,14],xsty=1, charsize=1.8,ytitle='P/P!dmax!N', xtitle='age [Gyr]'  

#-----------------------------------------------------------

mnorm_tot = np.zeros((len(mtot), 199))
anorm_tot = np.zeros((len(atot), 99))
tnorm_tot = np.zeros((len(wtot), 199))
tnorm_1 = np.zeros((len(w1), 199))
tnorm_2 = np.zeros((len(w2), 199))

for j in range(len(wtot)):
	tnorm_tot[j,:] = (wtot[j,:]/np.sum( wtot[j,:]) + 1e-8)*50.
for j in range(len(w1)):
	tnorm_1[j,:] = (w1[j,:]/np.sum( w1[j,:]) + 1e-8)*50.
for j in range(len(w2)):
	tnorm_2[j,:] = (w2[j,:]/np.sum( w2[j,:]) + 1e-8)*50.


# ~ptotm = np.ones(len(mtot[0,:]))
# ~ptota = np.ones(len(atot[0,:]))
ptot = np.ones(len(wtot[0,:]))
p1 = np.zeros(len(wtot[0,:]))
p2 = np.zeros(len(wtot[0,:]))

for i in range(len(wtot[0,:])):
	ptot[i] = 10**(np.sum(np.log10(tnorm_tot[:,i])))
	p1[i] = 10**(np.sum(np.log10(tnorm_1[:,i])))
	p2[i] = 10**(np.sum(np.log10(tnorm_2[:,i])))

from scipy.ndimage import gaussian_filter
ptot = gaussian_filter(ptot, 2)
p2 = gaussian_filter(p1, 2)
p2 = gaussian_filter(p2, 2)


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

dage = np.diff(binage)[0]
maxi = bincenter[np.argmax(p1)]
sd, sh = error_compute(dage, p1,bincenter)
maxi2 = bincenter[np.argmax(p2)]
dage = np.diff(binage)[0]
sd2, sh2 = error_compute(dage, p2,bincenter)


print('for -1.5 the age is ' + str(maxi))
print('error max is +'+str(sh-maxi))
print('error min is -'+str(maxi-sd))

print('for -2 the age is ' + str(maxi2))
print('error max is +'+str(sh2-maxi2))
print('error min is -'+str(maxi2-sd2))

plt.figure()
plt.plot(bincenter, ptot/np.max(ptot), label='All GC',c='r', linewidth=4)
plt.plot(bincenter, p1/np.max(p1), label ='[Fe/H] < -1.5',c='b', linewidth=4)
plt.plot(bincenter, p2/np.max(p2), label ='[Fe/H] < -2',c='g', linewidth=4)
plt.xlim(11.5,14.5)
#~ plt.ylim(0,0.025)
plt.ylabel('P/Pmax', fontsize=24)
plt.xlabel('Age [Gyr]', fontsize=24)
plt.legend(loc ='upper left', fontsize=24)
plt.tick_params(labelsize=16)
# ~plt.show()
plt.close()
# ~kill


### compute dt of galaxy formation
def pdt(x):
	l = np.log10(x)
	l1 = np.log10(0.1155)
	l2 = np.log10(0.255)
	sig1 = 0.15
	sig1p = 0.17
	sig2 = 0.155

	if x <= 0.1155:
		F1 = np.exp(-1/2. * (l-l1)**2 / sig1**2)
	elif x >= 0.1155:
		F1 = np.exp(-1/2. * (l-l1)**2 / sig1p**2)
	F2 = np.exp(-1/2. * (l-l2)**2 / sig2**2)

	return 0.95*F1 + 0.45*F2


dt = np.linspace(0,0.8,100)
# ~dt = np.arange(0,0.8,dage)
Pdt  = np.zeros(len(dt))
for i in range(len(dt)):
	Pdt[i] = pdt(dt[i])

# ~plt.figure()
# ~plt.plot(dt, Pdt)
# ~plt.show()
# ~plt.close()

# ~print(dage)
# ~print(np.diff(dt)[0])

#planck
pmu = 13.801
psigma =0.024
x = np.linspace(12, 15, 1000)
delta = 0.1
# ~P1 = stats.norm.pdf(x, maxi, 0.1)/np.max(stats.norm.pdf(x, maxi, 0.1))
P1 = p1/np.max(p1)
P2 = stats.norm.pdf(x, pmu, psigma)/np.max(stats.norm.pdf(x, pmu, psigma))
from scipy import signal
Pconv = signal.convolve(P1,Pdt,'same')
# ~Pconv = np.convolve(P1,Pdt,'same')
# ~Pconv = (P1 * Pdt)
# ~Pconv = np.multiply(P1,Pdt)


# ~maxi3 = x[np.argmax(Pconv)]
# ~dage = np.diff(binage)[0]
# ~sd3, sh3 = error_compute(dage, Pconv,x)
# ~maxi3 = bincenter[np.argmax(Pconv)]
# ~dage = np.diff(binage)[0]
# ~sd3, sh3 = error_compute(dage, Pconv,bincenter)

# ~print('for the convolution the age is ' + str(maxi3))
# ~print('error max is +'+str(sh3-maxi3))
# ~print('error min is -'+str(maxi3-sd3))

# ~plt.figure()
# ~plt.plot(bincenter, Pconv/np.max(Pconv), label='convoluted age')
# ~plt.plot(bincenter, P1, label='GC age')
# ~plt.legend(loc='best')
# ~plt.show()
# ~plt.close



#~ X, Y = np.meshgrid(bincenter, bfcenter)
#~ combined_p = np.empty((199,199))
#~ for i in range(199):
	#~ for j in range(199):
		#~ combined_p[i,j]= ptot[j]+ptotm[i]
#~ plt.contourf(X,Y,combined_p, 30, cmap='jet')
#~ plt.colorbar();
#~ plt.xlim(8,15)
#~ plt.ylabel('Metallicity [Fe/H]', fontsize=24)
#~ plt.xlabel('Age [Gyr]', fontsize=24)
#~ plt.legend(loc ='upper left', fontsize=24)
#~ plt.tick_params(labelsize=16)
#~ plt.show()
#~ plt.close()
#~ kill

#~ Age_mean = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', usecols=(2,))
#~ Age_high = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', usecols=(3,))
#~ Age_low = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', usecols=(1,))
#~ metal_fin = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', usecols=(5,))
#~ distance_fin = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', usecols=(8,))
#~ AAbs_fin = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version +'_'+str(model)+'.txt', usecols=(11,))
#~ elem_fin = np.arange(len(Age_mean))

#~ ### compute the inverse variance weighted mean
#~ err_min = Age_mean - Age_low
#~ err_max = Age_high - Age_mean
#~ err_mean = (err_min + err_max)/2.
#~ ### all clusters
#~ ivwa_tot = np.sum((Age_mean)/(err_mean)**2) / np.sum(1/(err_mean)**2)
#~ ivwv_tot = 1/ np.sum(1/(err_mean)**2)
#~ mu_tot = ivwa_tot 
#~ sigma_tot = np.sqrt(ivwv_tot)
#~ ### clusters with fe/h < -1.5
#~ gc1 = np.where(metal_fin < -1.5)[0]
#~ ivwa_1 = np.sum((Age_mean[gc1])/(err_mean[gc1])**2) / np.sum(1/(err_mean[gc1])**2)
#~ ivwv_1 = 1/ np.sum(1/(err_mean[gc1])**2)
#~ mu_1 = ivwa_1 
#~ sigma_1 = np.sqrt(ivwv_1)
#~ ### clusters with fe/h < -2.0
#~ gc2 = np.where(metal_fin < -2.0)[0]
#~ ivwa_2 = np.sum((Age_mean[gc2])/(err_mean[gc2])**2) / np.sum(1/(err_mean[gc2])**2)
#~ ivwv_2 = 1/ np.sum(1/(err_mean[gc2])**2)
#~ mu_2 = ivwa_2 
#~ sigma_2 = np.sqrt(ivwv_2)

#---------------------------------------------
#MIST 8
Age_mean_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(2,))
Age_high_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(3,))
Age_low_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(1,))
metal_fin_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(5,))
metal_low_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(4,))
metal_high_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(6,))
distance_low_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(7,))
distance_fin_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(8,))
distance_high_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(9,))
AAbs_low_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(10,))
AAbs_fin_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(11,))
AAbs_high_mist = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version1 +'_'+str(model1)+'.txt', usecols=(12,))
elem_fin_mist = np.arange(len(Age_mean_mist))


### compute the inverse variance weighted mean
err_min_mist = Age_mean_mist - Age_low_mist
err_max_mist = Age_high_mist - Age_mean_mist
err_mean_mist = (err_min_mist + err_max_mist)/2.
### all clusters
ivwa_tot_mist = np.sum((Age_mean_mist)/(err_mean_mist)**2) / np.sum(1/(err_mean_mist)**2)
ivwv_tot_mist = 1/ np.sum(1/(err_mean_mist)**2)
mu_tot_mist = ivwa_tot_mist 
sigma_tot_mist = np.sqrt(ivwv_tot_mist)
### clusters with fe/h < -1.5
gc1_mist = np.where(metal_fin_mist < -1.5)[0]
ivwa_1_mist = np.sum((Age_mean_mist[gc1_mist])/(err_mean_mist[gc1_mist])**2) / np.sum(1/(err_mean_mist[gc1_mist])**2)
ivwv_1_mist = 1/ np.sum(1/(err_mean_mist[gc1_mist])**2)
mu_1_mist = ivwa_1_mist 
sigma_1_mist = np.sqrt(ivwv_1_mist)
### clusters with fe/h < -2.0
gc2_mist = np.where(metal_fin_mist < -2.0)[0]
ivwa_2_mist = np.sum((Age_mean_mist[gc2_mist])/(err_mean_mist[gc2_mist])**2) / np.sum(1/(err_mean_mist[gc2_mist])**2)
ivwv_2_mist = 1/ np.sum(1/(err_mean_mist[gc2_mist])**2)
mu_2_mist = ivwa_2_mist 
sigma_2_mist = np.sqrt(ivwv_2_mist)

#~ #--------------------------------------------------------------------------
#DAR 12
Age_mean_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(2,))
Age_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(3,))
Age_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(1,))
metal_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(5,))
metal_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(4,))
metal_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(6,))
distance_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(7,))
distance_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(8,))
distance_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(9,))
AAbs_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(10,))
AAbs_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(11,))
AAbs_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(12,))
Afe_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(13,))
Afe_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(14,))
Afe_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(15,))
elem_fin_dar = np.arange(len(Age_mean_dar))

#~ ### compute the inverse variance weighted mean
err_min_dar = Age_mean_dar - Age_low_dar
err_max_dar = Age_high_dar - Age_mean_dar
err_mean_dar = (err_min_dar + err_max_dar)/2.
### all clusters
ivwa_tot_dar = np.sum((Age_mean_dar)/(err_mean_dar)**2) / np.sum(1/(err_mean_dar)**2)
ivwv_tot_dar = 1/ np.sum(1/(err_mean_dar)**2)
mu_tot_dar = ivwa_tot_dar
sigma_tot_dar = np.sqrt(ivwv_tot_dar)
### clusters with fe/h < -1.5
gc1_dar = np.where(metal_fin_dar < -1.5)[0]
ivwa_1_dar = np.sum((Age_mean_dar[gc1_dar])/(err_mean_dar[gc1_dar])**2) / np.sum(1/(err_mean_dar[gc1_dar])**2)
ivwv_1_dar = 1/ np.sum(1/(err_mean_dar[gc1_dar])**2)
mu_1_dar = ivwa_1_dar
sigma_1_dar = np.sqrt(ivwv_1_dar)
### clusters with fe/h < -2.0
gc2_dar = np.where(metal_fin_dar < -2.0)[0]
ivwa_2_dar = np.sum((Age_mean_dar[gc2_dar])/(err_mean_dar[gc2_dar])**2) / np.sum(1/(err_mean_dar[gc2_dar])**2)
ivwv_2_dar = 1/ np.sum(1/(err_mean_dar[gc2_dar])**2)
mu_2_dar = ivwa_2_dar
sigma_2_dar = np.sqrt(ivwv_2_dar)
#~ #--------------------------------------------------------------------------
#~ plt.hist(Afe_fin_dar, edgecolor ='k')
#~ plt.xlabel(r'[$\alpha$ / Fe]', fontsize=24)
#~ plt.ylabel(r'number of GC', fontsize=24)
#~ plt.tick_params(labelsize=16)
#~ plt.legend(loc='upper right')
#~ plt.show()
#~ plt.close()
#~ kill
#-----------------------------------------------------------------------------
omalley = np.loadtxt('omalley.txt')


#~ plt.figure()
#~ plt.scatter(metal_fin_dar,Age_mean_dar, c='r')
#~ plt.scatter(omalley[:,4],omalley[:,2], c='b')
#~ plt.scatter(metal_fin_dar[(omalley[:,5]).astype(int)], Age_mean_dar[(omalley[:,5]).astype(int)], c='b')
#~ plt.ylim(10.5,14)
#~ plt.xlabel('[Fe/H]')
#~ plt.ylabel('Age')
#~ plt.show()
#~ plt.close()


### plot distributons

plt.figure()
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=10)
#plt.suptitle('numero '+str(glc)+', '+clus_nb)

#AGE PLOT
#~ gs = gridspec.GridSpec(6, 4)
#~ ax1 = plt.subplot(gs[:2, :2])
#ax1.set_title('Age')
#ax1.errorbar(elem_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
plt.errorbar(elem_fin_dar, Age_mean_dar, yerr =[Age_mean_dar - Age_low_dar ,
Age_high_dar  - Age_mean_dar ], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
plt.ylabel(r'Age/Gyr', fontsize=20)
plt.xlabel(r'GC id', fontsize=20)
plt.ylim(6,16)
plt.xlim(-3,77)
plt.fill_between(np.linspace(-3,77), 15, 16, color='darkblue', alpha=0.4)
plt.subplots_adjust(top = 0.98)
plt.tick_params(labelsize=20)
# ~plt.show()
plt.close()

#METALLICITY PLOT
plt.figure()
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=10)
#plt.set_title('Metallicity')
#plt.errorbar(metal_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
plt.errorbar(elem_fin_dar,metal_fin_dar, yerr =[metal_fin_dar  - metal_low_dar ,
metal_high_dar  - metal_fin_dar ], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
plt.xlabel(r'GC id', fontsize=20)
plt.ylabel(r' Metallicity [Fe/H]', fontsize=20)
#plt.legend(loc = 'lower left')
plt.xlim(-3,77)
plt.ylim(-2.7, -0.35)
#plt.legend(loc = 'lower left')
plt.tick_params(labelsize=20)
plt.fill_between(np.linspace(-3,77), -2.5, -2.7, color='darkblue', alpha=0.4)
plt.fill_betweenx(np.linspace(-1.7,-1.3), 71, 71.5, color='k', alpha=0.4)
plt.text(72, -1.5, 'prior width', rotation=90, va='center', fontsize = 20)
#~ plt.axvspan(72, 73, ymin=0.2, ymax=0.8, alpha=0.5, color='k')
plt.subplots_adjust(top = 0.98)
# ~plt.show()
plt.close()

#DISTANCE PLOT
fig, ax = plt.subplots()
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=10)
#plt.set_title('distance')
#plt.errorbar(distance_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
plt.errorbar(elem_fin_dar, distance_fin_dar/1000., yerr =[distance_fin_dar/1000.  - distance_low_dar/1000. ,
distance_high_dar/1000.  - distance_fin_dar/1000. ], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
plt.xlabel(r'GC id', fontsize=20)
ax.set_ylabel(r' Distance [kpc]', fontsize=20)
ax.set_yscale('log')
ax.set_ylim(1,65)
ax.set_xlim(-3,77)
ax.set_yticks([2,5,10,30,60])
ax.set_yticklabels([r'2',r'5',r'10',r'30',r'60'])
#~ #plt.legend(loc = 'lower left')
plt.tick_params(labelsize=20)
plt.fill_between(np.linspace(-3,77), 0, -1, color='darkblue', alpha=0.4)
plt.fill_betweenx(np.linspace(7.94,12.59), 71, 71.5, color='k', alpha=0.4)
plt.text(72, 10, 'prior width', rotation=90, va='center', fontsize = 20)
plt.subplots_adjust(top = 0.98)
# ~plt.show()
plt.close()

#ABSORPTION PLOT
plt.figure()
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=10)
#plt.title('Absorption')
#plt.errorbar(AAbs_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
plt.errorbar(elem_fin_dar, AAbs_fin_dar, yerr =[AAbs_fin_dar  - AAbs_low_dar ,
AAbs_high_dar  - AAbs_fin_dar], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
plt.xlabel(r'GC id', fontsize=20)
plt.ylabel(r' Absorption', fontsize=20)
plt.tick_params(labelsize=20)
plt.xlim(-3,77)
plt.ylim(-0.2,2.4)
plt.fill_between(np.linspace(-3,77), 0, -0.2, color='darkblue', alpha=0.4)
plt.fill_betweenx(np.linspace(0.97,1.03), 71, 71.5, color='k', alpha=0.4)
plt.text(72, 1, 'prior width', rotation=90, va='center', fontsize = 20)
plt.subplots_adjust(top = 0.98)
# ~plt.show()
plt.close()

#ALPHA PLOT
plt.figure()
plt.locator_params(axis='y', nbins=5)
plt.locator_params(axis='x', nbins=10)
#plt.title('Absorption')
#plt.errorbar(AAbs_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
plt.errorbar(elem_fin_dar, Afe_fin_dar , yerr =[Afe_fin_dar - Afe_low_dar ,
Afe_high_dar  - Afe_fin_dar], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
plt.xlabel(r'GC id', fontsize=20)
plt.ylabel(r' [$\alpha$/Fe]', fontsize=20)
plt.tick_params(labelsize=20)
plt.xlim(-3,77)
plt.ylim(-0.3,0.9)
plt.fill_between(np.linspace(-3,77), -0.2, -0.3, color='darkblue', alpha=0.4)
plt.fill_between(np.linspace(-3,77), 0.8, 0.9, color='darkblue', alpha=0.4)
plt.fill_betweenx(np.linspace(0.2,0.4), 71, 71.5, color='k', alpha=0.4)
plt.text(72, 0.3, 'prior width', rotation=90, va='center', fontsize = 20)
plt.subplots_adjust(top = 0.98, left=0.14)
# ~plt.show()
plt.close()





# COMPARE ALPHA VALUE TO SPECTROSCOPY

# ~af = np.zeros(69)
# ~for cn in list(range(27))+ list(range(28,69)):
	# ~_, _, _, _, _, af[cn], _, _  = cluster(cn)

# ~af = np.delete(af,27)

# ~plt.figure()
# ~plt.scatter(elem_fin_dar, af, marker='x', label='Dotter et al.')
# ~plt.errorbar(elem_fin_dar, Afe_fin_dar , yerr =[Afe_fin_dar - Afe_low_dar ,
# ~Afe_high_dar  - Afe_fin_dar], color='r', fmt='o', ecolor='k', markersize=5, label='Best fit')
# ~plt.legend(loc='best')
# ~plt.subplots_adjust(top = 0.98, left=0.14)
# ~plt.xlabel(r'GC id', fontsize=20)
# ~plt.ylabel(r' [$\alpha$/Fe]', fontsize=20)
# ~plt.show()
# ~plt.close()
# ~kill

#~ plt.figure()
#~ plt.subplots_adjust(hspace=0, top = 0.95, left = 0.1, right=0.95)
#~ ax2 = plt.subplot(411)
#~ ax2.errorbar(Age_mean_dar , metal_fin_dar, yerr =[metal_fin_dar- metal_low_dar,
#~ metal_high_dar - metal_fin_dar], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
#~ ax2.set_ylabel(r'[Fe/H]', fontsize=22)
#~ ax2.set_ylim(-2.5,0.5)
#~ #ax2.legend(loc = 'lower left')
#~ ax2.tick_params(labelsize=16)
#~ ax3 = plt.subplot(412, sharex = ax2)
#~ #ax3.set_title('distance')
#~ #ax3.errorbar(distance_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#~ #Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
#~ ax3.errorbar(Age_mean_dar , distance_fin_dar/1000., yerr =[distance_fin_dar/1000. - distance_low_dar/1000.,
#~ distance_high_dar/1000. - distance_fin_dar/1000.], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
#~ ax3.set_ylabel(r' Distance [kpc]', fontsize=18)
#~ ax3.tick_params(labelsize=16)
#~ ax4 = plt.subplot(413, sharex = ax2)
#~ #ax4.set_title('Absorption')
#~ #ax4.errorbar(AAbs_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#~ #Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
#~ ax4.errorbar(Age_mean_dar , AAbs_fin_dar, yerr =[AAbs_fin_dar - AAbs_low_dar,
#~ AAbs_high_dar - AAbs_fin_dar], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
#~ ax4.set_ylabel(r' $A_{V}$', fontsize=22)
#~ #ax4.legend(loc = 'lower left')
#~ ax4.tick_params(labelsize=16)
#~ ax5 = plt.subplot(414, sharex = ax2)
#~ #ax4.set_title('Absorption')
#~ #ax4.errorbar(AAbs_fin_mist, Age_mean_mist , yerr =[Age_mean_mist  - Age_low_mist ,
#~ #Age_high_mist  - Age_mean_mist ], color='gold', fmt='o', ecolor='k', markersize=5, label='MIST')
#~ ax5.errorbar(Age_mean_dar , Afe_fin_dar, yerr =[Afe_fin_dar - Afe_low_dar,
#~ Afe_high_dar - Afe_fin_dar], color='r', fmt='o', ecolor='k', markersize=5, label='DSED')
#~ ax5.set_ylabel(r' [$\alpha$/Fe]', fontsize=22)
#~ #ax4.legend(loc = 'lower left')
#~ ax5.tick_params(labelsize=16)
#~ plt.xlabel(r'Age/Gyr', fontsize=22)
#~ #plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/param_distrib'+'_'+ version +'_'+str(model)+'.png')
#~ plt.show()
#~ plt.close()
#~ kill

#~ ###----------------------------------------------------------------------
#~ ### comparisons with omalley

#~ xerr=omalley[:,3]
#~ gs = gridspec.GridSpec(4, 4)
#~ ax1 = plt.subplot(gs[:2, :2])
#~ x = np.linspace(10,14.5,20)
#~ ax1.errorbar(omalley[:,2], Age_mean_mist[(omalley[:,5]).astype(int)] , yerr =[Age_mean_mist[(omalley[:,5]).astype(int)] - Age_low_mist[(omalley[:,5]).astype(int)]  ,
#~ Age_high_mist[(omalley[:,5]).astype(int)]  - Age_mean_mist[(omalley[:,5]).astype(int)]  ], xerr = xerr 
#~ , color='r', ecolor='k', fmt='o', markersize=5, label=r' Age/Gyr ')
#~ ax1.errorbar(omalley[:,2], Age_mean_mist[(omalley[:,5]).astype(int)] , yerr =[Age_mean_mist[(omalley[:,5]).astype(int)] - Age_low_mist[(omalley[:,5]).astype(int)]  ,
#~ Age_high_mist[(omalley[:,5]).astype(int)]  - Age_mean_mist[(omalley[:,5]).astype(int)]  ] 
#~ , color='r', ecolor='k', fmt='o', markersize=5, label=r' Age/Gyr ')
#~ ax1.plot(x,x, c='b', label='x = y')
#~ ax1.set_xlabel(' Age/Gyr (O\'Malley et al. 2017)', fontsize=19)
#~ ax1.set_ylabel(' Age/Gyr (This work)', fontsize=19)
#~ ax1.set_xlim(9,15)
#~ ax1.set_ylim(9,15)
#~ ax1.tick_params(labelsize=16)
#~ ax2 = plt.subplot(gs[:2, 2:])
#~ x = np.linspace(-2.5,-0.5,20)
#~ ax2.errorbar(omalley[:,4], metal_fin_mist[(omalley[:,5]).astype(int)] , yerr =[metal_fin_mist[(omalley[:,5]).astype(int)] - metal_low_mist[(omalley[:,5]).astype(int)],
#~ metal_high_mist[(omalley[:,5]).astype(int)] - metal_fin_mist[(omalley[:,5]).astype(int)]], color='r', fmt='o', ecolor='k', markersize=5, label=r' Metallicity [Fe/H] ')
#~ ax2.plot(x,x, c='b', label='x = y')
#~ ax2.set_xlabel('[Fe/H] (O\'Malley et al. 2017)', fontsize=19)
#~ ax2.set_ylabel('[Fe/H] (This work)', fontsize=19)
#~ ax2.tick_params(labelsize=16)
#~ ax3 = plt.subplot(gs[2:4, 1:3])
#~ distom = (omalley[:,0]/5. + 1)
#~ distomerrp = ((omalley[:,0]+omalley[:,1])/5. + 1)
#~ distomerrm = ((omalley[:,0]-omalley[:,1])/5. + 1)
#~ errdistp = (distomerrp - distom)/1000.
#~ errdistm = (distom -distomerrm)/1000. 
#~ x = np.linspace(1,25,20)
#~ ax3.errorbar(distom/1000., distance_fin_mist[(omalley[:,5]).astype(int)] /1000., yerr =[distance_fin_mist[(omalley[:,5]).astype(int)] /1000. - distance_low_mist[(omalley[:,5]).astype(int)] /1000.,
#~ distance_high_mist[(omalley[:,5]).astype(int)] /1000. - distance_fin_mist[(omalley[:,5]).astype(int)] /1000.], xerr = [errdistm, errdistp],color='r', fmt='o', ecolor='k', markersize=5, label='Distance/kpc')
#~ ax3.plot(x,x, c='b', label='x = y')
#~ ax3.set_xlabel(' Distance/kpc (O\'Malley et al. 2017)', fontsize=19)
#~ ax3.set_ylabel(' Distance/kpc (This work)', fontsize=19)
#~ ax3.tick_params(labelsize=16)
#~ plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/param_comparison.png', orientation='portrait', dpi=250)
#~ plt.subplots_adjust(hspace=1.8, wspace=1.0, top=0.95)
#~ plt.show()
#~ plt.close()

#OMALLEY COMPARISON
xerr=omalley[:,3]
gs = gridspec.GridSpec(4, 4)
ax1 = plt.subplot(gs[:2, :2])
x = np.linspace(10,14.5,20)
ax1.errorbar(omalley[:,2], Age_mean_dar[(omalley[:,5]).astype(int)] , yerr =[Age_mean_dar[(omalley[:,5]).astype(int)] - Age_low_dar[(omalley[:,5]).astype(int)]  ,
Age_high_dar[(omalley[:,5]).astype(int)]  - Age_mean_dar[(omalley[:,5]).astype(int)]  ], xerr = xerr 
, color='r', ecolor='k', fmt='o', markersize=5, label=r' Age/Gyr ')
#~ ax1.errorbar(omalley[:,2], Age_mean_dar[(omalley[:,5]).astype(int)] , yerr =[Age_mean_dar[(omalley[:,5]).astype(int)] - Age_low_dar[(omalley[:,5]).astype(int)]  ,
#~ Age_high_dar[(omalley[:,5]).astype(int)]  - Age_mean_dar[(omalley[:,5]).astype(int)]  ] 
#~ , color='r', ecolor='k', fmt='o', markersize=5, label=r' Age/Gyr ')
ax1.plot(x,x, c='b', label='x = y')
ax1.set_xlabel(' Age/Gyr (O\'Malley et al. 2017)', fontsize=12)
ax1.set_ylabel(' Age/Gyr (This work)', fontsize=12)
#~ ax1.set_xlim(9,15)
#~ ax1.set_ylim(9,15)
ax1.tick_params(labelsize=14)
ax2 = plt.subplot(gs[:2, 2:])
x = np.linspace(-2.5,-0.5,20)
ax2.errorbar(omalley[:,4], metal_fin_dar[(omalley[:,5]).astype(int)] , yerr =[metal_fin_dar[(omalley[:,5]).astype(int)] - metal_low_dar[(omalley[:,5]).astype(int)],
metal_high_dar[(omalley[:,5]).astype(int)] - metal_fin_dar[(omalley[:,5]).astype(int)]], color='r', fmt='o', ecolor='k', markersize=5, label=r' Metallicity [Fe/H] ')
ax2.plot(x,x, c='b', label='x = y')
ax2.set_xlabel('[Fe/H] (O\'Malley et al. 2017)', fontsize=12)
ax2.set_ylabel('[Fe/H] (This work)', fontsize=12)
ax2.tick_params(labelsize=14)
ax3 = plt.subplot(gs[2:4, 1:3])
distom = 10**(omalley[:,0]/5. + 1)
distomerrp = 10**((omalley[:,0]+omalley[:,1])/5. + 1)
distomerrm = 10**((omalley[:,0]-omalley[:,1])/5. + 1)
errdistp = (distomerrp - distom)/1000.
errdistm = (distom -distomerrm)/1000. 
x = np.linspace(1,25,20)
ax3.errorbar(distom/1000., distance_fin_dar[(omalley[:,5]).astype(int)] /1000., yerr =[distance_fin_dar[(omalley[:,5]).astype(int)] /1000. - distance_low_dar[(omalley[:,5]).astype(int)] /1000.,
distance_high_dar[(omalley[:,5]).astype(int)] /1000. - distance_fin_dar[(omalley[:,5]).astype(int)] /1000.], xerr = [errdistm, errdistp],color='r', fmt='o', ecolor='k', markersize=5, label='Distance/kpc')
ax3.plot(x,x, c='b', label='x = y')
ax3.set_xlabel(' Distance/kpc (O\'Malley et al. 2017)', fontsize=12)
ax3.set_ylabel(r' Distance/kpc' "\n" r' (This work)', fontsize=12)
ax3.tick_params(labelsize=14)
plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/param_comparison.png', orientation='portrait', dpi=250)
plt.subplots_adjust(hspace=1.8, wspace=1.0, top=0.95)
# ~plt.show()
plt.close()


#~ ###------------------------------------------------------------------------
### comparison of the two models
#plt.suptitle('numero '+str(glc)+', '+clus_nb)
#~ ax1 = plt.subplot(221)
#~ #ax1.set_title('Age')
#~ ax1.errorbar(Age_mean_dar , Age_mean_mist , xerr =[Age_mean_dar  - Age_low_dar ,
#~ Age_high_dar  - Age_mean_dar ], yerr =[Age_mean_mist  - Age_low_mist ,
#~ Age_high_mist  - Age_mean_mist ], color='r', fmt='o', ecolor='k', markersize=5)
#~ ax1.set_xlabel(r'Age/Gyr (DAR)', fontsize=24)
#~ ax1.set_ylabel(r'Age/Gyr (MIST)', fontsize=24)
#~ ax1.plot(np.linspace(8,16), np.linspace(8,16), label=' y = x')
#~ ax1.set_xlim(8,16)
#~ ax1.set_ylim(8,16)
#~ ax1.legend(loc = 'lower right')
#~ ax2 = plt.subplot(222)
#ax2.set_title('Metallicity')
#~ ax2.errorbar(metal_fin_dar, metal_fin_mist, xerr =[metal_fin_dar - metal_low_dar,
#~ metal_high_dar- metal_fin_dar] , yerr =[metal_fin_mist - metal_low_mist,
#~ metal_high_mist - metal_fin_mist], color='r', fmt='o', ecolor='k', markersize=5)
#~ ax2.set_xlabel(r' Metallicity [Fe/H] (DAR)', fontsize=24)
#~ ax2.set_ylabel(r' Metallicity [Fe/H] (MIST)', fontsize=24)
#~ ax2.plot(np.linspace(-2.5,-0.5), np.linspace(-2.5,-0.5), label=' y = x')
#~ ax2.set_xlim(-2.5,-0.5)
#~ ax2.set_ylim(-2.5,-0.5)
#~ ax2.legend(loc = 'lower right')
#~ ax3 = plt.subplot(223)
#ax3.set_title('distance')
#~ ax3.errorbar(distance_fin_dar/1000., distance_fin_mist/1000., xerr =[distance_fin_dar/1000. - distance_low_dar/1000.,
#~ distance_high_dar/1000.- distance_fin_dar/1000.] , yerr =[distance_fin_mist/1000. - distance_low_mist/1000.,
#~ distance_high_mist/1000. - distance_fin_mist/1000.], color='r', fmt='o', ecolor='k', markersize=5)
#~ ax3.set_xlabel(r' distance/kpc (DAR)', fontsize=24)
#~ ax3.set_ylabel(r' distance/kpc (MIST)', fontsize=24)
#~ ax3.plot(np.linspace(1,25,20), np.linspace(1,25,20), label=' y = x')
#~ ax3.set_xlim(1,25,20)
#~ ax3.set_ylim(1,25,20)
#~ ax3.legend(loc = 'lower right')
#~ ax4 = plt.subplot(224)
#ax4.set_title('Absorption')
#~ ax4.errorbar(AAbs_fin_dar, AAbs_fin_mist, xerr =[AAbs_fin_dar - AAbs_low_dar,
#~ AAbs_high_dar- AAbs_fin_dar] , yerr =[AAbs_fin_mist - AAbs_low_mist,
#~ AAbs_high_mist - AAbs_fin_mist], color='r', fmt='o', ecolor='k', markersize=5)
#~ ax4.set_xlabel(r' Absorption (DAR)', fontsize=24)
#~ ax4.set_ylabel(r' Absorption (MIST)', fontsize=24)
#~ ax4.plot(np.linspace(0,3), np.linspace(0,3), label=' y = x')
#~ ax4.set_xlim(0,3)
#~ ax4.set_ylim(0,3)
#~ ax4.legend(loc = 'lower right')
#~ #plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/param_distrib'+'_'+ version +'_'+str(model)+'.png')
#~ plt.subplots_adjust(wspace=0.3, hspace=0.3)
#~ plt.show()
#~ plt.close()


#~ #-----------------------------------------------------------------------

Age_mean_mist = Age_mean_mist[Age_mean_mist != 0]
Age_high_mist = Age_high_mist[Age_high_mist != 0]
Age_low_mist = Age_low_mist[Age_low_mist != 0]
metal_fin_mist = metal_fin_mist[metal_fin_mist != 0]
distance_fin_mist = distance_fin_mist[distance_fin_mist != 0]
AAbs_fin_mist = AAbs_fin_mist[AAbs_fin_mist != 0]
elem_fin_mist = elem_fin_mist[elem_fin_mist != 0]

Age_mean_dar = Age_mean_dar[Age_mean_dar != 0]
Age_high_dar = Age_high_dar[Age_high_dar != 0]
Age_low_dar = Age_low_dar[Age_low_dar != 0]
metal_fin_dar = metal_fin_dar[metal_fin_dar != 0]
distance_fin_dar = distance_fin_dar[distance_fin_dar != 0]
AAbs_fin_dar = AAbs_fin_dar[AAbs_fin_dar != 0]
elem_fin_dar = elem_fin_dar[elem_fin_dar != 0]

### DAR without the empty elements
### compute the inverse variance weighted mean
err_min_dar = Age_mean_dar - Age_low_dar
err_max_dar = Age_high_dar - Age_mean_dar
err_mean_dar = (err_min_dar + err_max_dar)/2.
### all clusters
ivwa_tot_dar = np.sum((Age_mean_dar)/(err_mean_dar)**2) / np.sum(1/(err_mean_dar)**2)
ivwv_tot_dar = 1/ np.sum(1/(err_mean_dar)**2)
mu_tot_dar = ivwa_tot_dar 
sigma_tot_dar = np.sqrt(ivwv_tot_dar)
### clusters with fe/h < -1.5
gc1_dar = np.where(metal_fin_dar < -1.5)[0]
ivwa_1_dar = np.sum((Age_mean_dar[gc1_dar])/(err_mean_dar[gc1_dar])**2) / np.sum(1/(err_mean_dar[gc1_dar])**2)
ivwv_1_dar = 1/ np.sum(1/(err_mean_dar[gc1_dar])**2)
mu_1_dar = ivwa_1_dar 
sigma_1_dar = np.sqrt(ivwv_1_dar)
### clusters with fe/h < -2.0
gc2_dar = np.where(metal_fin_dar < -2.0)[0]
ivwa_2_dar = np.sum((Age_mean_dar[gc2_dar])/(err_mean_dar[gc2_dar])**2) / np.sum(1/(err_mean_dar[gc2_dar])**2)
ivwv_2_dar = 1/ np.sum(1/(err_mean_dar[gc2_dar])**2)
mu_2_dar = ivwa_2_dar 
sigma_2_dar = np.sqrt(ivwv_2_dar)


print(len(gc1_dar))
print(gc1_dar)
print(len(gc2_dar))
print(gc2_dar)


print(mu_tot_mist, sigma_tot_mist)
print(mu_1_mist, sigma_1_mist)
print(mu_2_mist, sigma_2_mist)
print(mu_tot_dar, sigma_tot_dar)
print(mu_1_dar, sigma_1_dar)
print(mu_2_dar, sigma_2_dar)

x = np.linspace(12.5, 14.5, 1000)
#~ plt.plot(x, stats.norm.pdf(x, mu_tot_mist, sigma_tot_mist), label='MIST All globular clusters', c='b')
#~ plt.plot(x, stats.norm.pdf(x, mu_1_mist, sigma_1_mist), label='MIST GC with [Fe/H] < -1.5 dex', c='r')
#~ plt.plot(x, stats.norm.pdf(x, mu_2_mist, sigma_2_mist), label='MIST GC with [Fe/H] < -2.0 dex',c='g')
plt.plot(x, stats.norm.pdf(x, mu_tot_dar, sigma_tot_dar), label='DSED All globular clusters', c='b')
plt.plot(x, stats.norm.pdf(x, mu_1_dar, sigma_1_dar), label='DSED GC with [Fe/H] < -1.5 dex', c='r')
plt.plot(x, stats.norm.pdf(x, mu_2_dar, sigma_2_dar), label='DSED GC with [Fe/H] < -2.0 dex',c='g')
plt.xlabel('Age/Gyr', fontsize=24)
plt.title('Inverse variance age distribution', fontsize = 16)
plt.legend(loc='upper right', fontsize = 16)
#plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/iva'+'_'+ version +'_'+str(model)+'.png')
#~ plt.show()
plt.close()

#~ x = np.linspace(12.8, 14, 1000)
#~ mu_tot = (mu_tot_mist+mu_tot_dar)/2
#~ mu_1 = (mu_1_mist+mu_1_dar)/2
#~ mu_2 = (mu_2_mist+mu_2_dar)/2

#~ sigma_tot = (sigma_tot_mist**2 + sigma_tot_dar**2)/4
#~ sigma_1 = (sigma_1_mist**2 + sigma_1_dar**2)/4
#~ sigma_2 = (sigma_2_mist**2 + sigma_2_dar**2)/4
#~ plt.plot(x, stats.norm.pdf(x, 13.8, 0.02), c='k', label='Planck18')
#~ plt.plot(x, stats.norm.pdf(x, mu_tot+0.2, sigma_tot+0.1), label='All GC + galaxy formation time', c='b')
#~ plt.plot(x, stats.norm.pdf(x, mu_1+0.2, sigma_1+0.1), label='[Fe/H] < -1.5', c='r', linewidth=2)
#~ plt.plot(x, stats.norm.pdf(x, mu_2+0.2, sigma_2+0.1), label='[Fe/H] < -2.0',c='g', linewidth=2)
#~ plt.xlabel('Age/Gyr', fontsize = 16)
#~ plt.title('Inverse variance age distribution', fontsize = 16)
#~ plt.legend(loc='upper left', fontsize = 16)
#~ #plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/iva'+'_'+ version +'_'+str(model)+'.png')
#~ plt.show()
#~ plt.close()

kill

################################################################################################
################################################################################################



