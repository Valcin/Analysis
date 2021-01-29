import numpy as np
import mesa_reader as mr
import matplotlib.pyplot as plt
import math
import matplotlib
import os
from scipy import interpolate
from matplotlib import colors
import sys
sys.path.append('/home/david/codes/isochrones')# folder where isochrones is installed
########################################################################
########################################################################
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
	m = (L+R)/2
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
		m = (L+R)/2
		#~ print(arr[m], x, L, R, m)
		if L>R:
			done = True
	return L

def col_teff(V, I):
	# for F606W
	F6 = np.zeros(len(V))
	F8 = np.zeros(len(I))
	for i in range(len(V)):
		if V[i]-I[i] < 0.4:
			print(i, V[i]-I[i])
			F6[i] = V[i] - 26.394 - 0.153*(V[i]-I[i]) - 0.096*(V[i]-I[i])**2 + 26.398
		else:
			F6[i] = V[i] - 26.331 - 0.340*(V[i]-I[i]) + 0.038*(V[i]-I[i])**2 + 26.398
			
	for j in range(len(I)):
		if V[i]-I[i] < 0.1:
			F8[i] = I[i] - 25.489 - 0.041*(V[i]-I[i]) + 0.093*(V[i]-I[i])**2 + 25.501
		else:
			F8[i] = I[i] - 25.496 + 0.014*(V[i]-I[i]) - 0.015*(V[i]-I[i])**2 + 25.501

	return F6,F8


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
	# ~V = h.log_L
	# ~R = h.log_Teff
	# ~const = np.min(V)
	# ~min_pts = np.where(V == const)[0]
	# ~mag = V[min_pts[0]:]
	# ~col = R[min_pts[0]:]
	return col, mag, min_pts[0]

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

	# ~gc.collect()
	return mag_v, mag_i, Color, eep_first

def dc_da(magval, bestalpha):
	coledge1 = np.zeros(8)
	coledge2 = np.zeros(8)
	limi = min(bestalpha, 16-bestalpha)
	# ~print(alpha_mix[bestalpha])
	for a in range(1, limi+1):
		# ~coledge1[a-1] = np.abs(finterp[bestalpha -a](magval) - finterp[bestalpha](magval))/(a*0.1)
		# ~coledge2[a-1] = np.abs(finterp[bestalpha](magval) - finterp[bestalpha +a](magval))/(a*0.1)
		coledge1[a-1] = np.abs(finterp[bestalpha -a](magval) - finterp[bestalpha](magval))
		coledge2[a-1] = np.abs(finterp[bestalpha](magval) - finterp[bestalpha +a](magval))
	return coledge1, coledge2

def coord_fid(bestalpha):
	cm2 = finterp[bestalpha](-2.0)
	cp0 = finterp[bestalpha](0.0)
	cp2 = finterp[bestalpha](2.0)
	return cm2,cp0,cp2
	
########################################################################
########################################################################
Zsun = 0.0134
Fe_h = -2.0
c = 2.9979e8 # in meters
L0 = 3.0128e28 #zero-point luminosity in W
Lsun = 3.828e26 #solar luminosity in W
rsun = 6.957e+10	# in centimeters
msun = 1.989e+33 #g

#-----------------------------------------------------------------------
### COLMAG
# ~directory = '/home/david/codes/Analysis/GC_mixing_length/catalogs/colmag'
# ~with open('/home/david/codes/Analysis/GC_mixing_length/bc_colmag.dat',"w") as fid:
	# ~fid.write('%s %s %s %s %s \n' %('#Teff', 'log_g', 'M_div_h', 'F606W', 'F814W'))
	# ~for filename in os.listdir(directory):
		# ~print(os.path.join(directory, filename))
		# ~dat = np.loadtxt(os.path.join(directory, filename))
		# ~tefs = dat[:,0]
		# ~sfs = dat[:,1]
		# ~feH = dat[:,2]
		# ~Fil1 = dat[:,4]
		# ~Fil2 = dat[:,5]
		# ~for i in range(len(tefs)):
			# ~fid.write('%.8g %.8g %.8g %.8g %.8g\n' %(tefs[i], sfs[i], feH[i], Fil1[i], Fil2[i]))
# ~fid.close()

# MIST
# ~directory = '/home/david/codes/data/GC_mixing_length/bol_corr/'
# ~with open('/home/david/codes/Analysis/GC_mixing_length/bc_MIST.dat',"w") as fid:
	# ~fid.write('%s %s %s %s %s \n' %('#Teff', 'log_g', 'M_div_h', 'F606W', 'F814W'))
	# ~for filename in os.listdir(directory):
		# ~print(os.path.join(directory, filename))
		# ~dat = np.loadtxt(os.path.join(directory, filename))
		# ~tefs = dat[:,0]
		# ~sfs = dat[:,1]
		# ~feH = dat[:,2]
		# ~Fil1 = dat[:,10]
		# ~Fil2 = dat[:,15]
		# ~for i in range(len(tefs)):
			# ~fid.write('%.8g %.8g %.8g %.8g %.8g\n' %(tefs[i], sfs[i], feH[i], Fil1[i], Fil2[i]))
# ~fid.close()

#BOLCORR
# ~filename = '/home/david/codes/bolometric-corrections/BCcodes/output.file.all'
# ~with open('/home/david/codes/Analysis/GC_mixing_length/bc_CORRp0.dat',"w") as fid:
	# ~fid.write('%s %s %s %s %s \n' %('#Teff', 'log_g', 'M_div_h', 'F606W', 'F814W'))
	# ~dat = np.loadtxt(filename, usecols = (1,2,3,5,6))
	# ~tefs = dat[:,2]
	# ~sfs = dat[:,0]
	# ~feH = dat[:,1]
	# ~Fil1 = dat[:,3]
	# ~Fil2 = dat[:,4]
	# ~for i in range(len(tefs)):
		# ~fid.write('%.8g %.8g %.8g %.8g %.8g\n' %(tefs[i], sfs[i], feH[i], Fil1[i], Fil2[i]))
# ~fid.close()

# ~kill
# ~gs = np.linspace(0.0,5.0,20)
# ~tefs = np.linspace(3000.0, 8000.0, 20)
# ~feH = np.linspace(-2.5, 0.5, 8)
# ~Ebv = 0.0
# ~tot = 0
# ~with open('/home/david/codes/Analysis/GC_mixing_length/temp.dat',"w") as fid:
	# ~for i in range(len(gs)):
		# ~for j in range(len(tefs)):
			# ~for k in range(len(feH)):
				# ~tot+= 1
				# ~fid.write('%s %.8g %.8g %.8g %.8g\n' %('#'+str(tot), gs[i], feH[k], tefs[j], Ebv))
# ~fid.close()


#-----------------------------------------------------------------------

# ~for j in al:
# ~# load the data
# ~h = mr.MesaData('/home/david/codes/data/GC_mixing_length/Z3alpha'+j+'/trimmed_history.data')
# ~h = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history.data')


# ~name2 = ['a100','a120','a140','a160','a180','a200']
# ~name2 = ['a100_M070','a100_M075','a100_M080']

# ~for j in name2:
#j = name[4]
	# ~h = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_'+j+'.data')
	# ~G = 6.67259e-12 #cm3 g-1 s-2
	# ~M = h.star_mass * msun
	# ~R = 10**(h.log_R) * rsun
	# ~sf = np.log10(G*M/ R**2)
	# ~teff = 10**(h.log_Teff)
	# ~safe = np.where((teff > np.min(tefs))&(teff < np.max(tefs))&(sf > np.min(sfs))&(sf < np.max(sfs)))[0]
	# ~teff = teff[safe]
	# ~sf = sf[safe]

	# ~Mbol = -2.5*np.log10((Lsun*10**h.log_L)/L0)[safe]
	# ~M606 = np.zeros(len(Mbol))
	# ~M814 = np.zeros(len(Mbol))

	# ~for i in range(len(Mbol)):
		# ~bc606, bc814 = interp_eep(teff[i], sf[i], tefs, sfs, dat)
		# ~M606[i] = Mbol[i] - bc606
		# ~M814[i] = Mbol[i] - bc814
		# ~print(bc606)
		# ~print(bc814)

	# ~np.savetxt('catalogs/Malpha_'+j+'.txt', (M606-M814, M606))
	# ~plt.plot(M606-M814, M606)
# ~plt.xlim(0.3, 1.3)
# ~plt.ylim(0, 7)
# ~plt.gca().invert_yaxis()
# ~plt.show()

# ~dat = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/fehm200.HST_ACSWF')
# ~dat = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/colmag.Castelli.HSTACSWFC3.Vega.M-2.txt')
# ~dat = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/bc_CORRp0.dat')
# ~tefs = np.unique(dat[:,0])
# ~sfs = np.unique(dat[:,1])
# ~feH = np.unique(dat[:,2])

# ~name = ['a100','a125','a150','a175','a200']
# ~name = ['alpha100','alpha125','alpha150','alpha175','alpha200']
# ~for j in name:	
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

	# ~np.savetxt('catalogs/alpha_'+j+'.txt', (M606-M814, M606))


	# ~print(M606[1:-1][::-1])
	# ~fcurve = interpolate.interp1d(M606, M606-M814)
	# ~fcurve = interpolate.CubicSpline(M606[1:-1][::-1], M606[1:-1][::-1]-M814[1:-1][::-1])
	# ~mpts = np.linspace(np.min(M606), np.max(M606), 40)
	
	# ~plt.scatter(M606-M814, M606)
	# ~plt.plot(fcurve(mpts), mpts)
	# ~plt.scatter(fcurve(mpts), mpts)
# ~plt.gca().invert_yaxis()
# ~plt.show()

# ~kill

#-----------------------------------------------------------------------

Age = np.log10(13.32e9)
distance = 0
Abs = 0
afe_init = 0.
helium_y = ''
model='dar'
# ~model='mist'
print(Age)

from isochrones.mist import MIST_Isochrone
mist = MIST_Isochrone()

from isochrones.dartmouth import Dartmouth_FastIsochrone
darm2 = Dartmouth_FastIsochrone(afe='afem2', y=helium_y)
darp0 = Dartmouth_FastIsochrone(afe='afep0', y=helium_y)
darp2 = Dartmouth_FastIsochrone(afe='afep2', y=helium_y)
darp4 = Dartmouth_FastIsochrone(afe='afep4', y=helium_y)
darp6 = Dartmouth_FastIsochrone(afe='afep6', y=helium_y)
darp8 = Dartmouth_FastIsochrone(afe='afep8', y=helium_y)

# ~binc = 200
# ~niso = int((-200 -(-236))/2)
# ~magtest= np.linspace(-5,5,binc)
# ~col = np.zeros((binc,niso))
# ~mag = np.zeros((binc,niso))
# ~import cmasher as cmr
# ~cm = cmr.ember
# ~norm = colors.Normalize(vmin=-2.5,vmax=-0.5)
# ~s_m = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
# ~s_m.set_array([])
# ~import matplotlib.gridspec as gridspec
# ~gs_in = gridspec.GridSpec(2, 2,hspace=0.5,height_ratios=[10,1],
# ~width_ratios=[8,4],wspace=0.,left=0.10,right=0.9,bottom=0.1,top=0.9)
# ~for ind,a in enumerate(range(-236, -200, 2)):
	# ~met= a/100. 

	# ~print(met)
	# ~mag_v, mag_i, Color_iso, eep_first = iso_mag(Age, met, distance, Abs, afe_init)
	# ~fmag_ini = interpolate.interp1d(mag_v, Color_iso, 'nearest',fill_value="extrapolate")
	# ~# mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(Age , -2.48, distance, Abs, afe_init)
	# ~# mag_vref, mag_iref, Color_isoref, eep_firstref = iso_mag(Age , -0.5, distance, Abs, afe_init)
	# ~# fmag_iniref = interpolate.interp1d(mag_vref, Color_isoref, 'nearest',fill_value="extrapolate")

	# ~col[:,ind]= fmag_ini(magtest)
	# ~mag[:,ind]= magtest
	# ~plt.plot(Color_iso,mag_v, color=s_m.to_rgba(met))
	#ax1.plot(fmag_iniref(magtest)/fmag_ini(magtest),magtest, color=s_m.to_rgba(met),lw=2)
	
string_mass = 'M075'
string_met = 'scatter_test'
htest1 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a1938.data')
coltest1,magtest1, mptest1 = cut2(htest1)
# ~htest1bis = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a1938bis.data')
htest1bis = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a1938bis.data')
coltest1bis,magtest1bis, mptest1bis = cut2(htest1bis)
htest2 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_z00045.data')
coltest2,magtest2, mptest2 = cut2(htest2)
htest3 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_z000142.data')
coltest3,magtest3, mptest3 = cut2(htest3)
htest4 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_m080.data')
coltest4,magtest4, mptest4 = cut2(htest4)
htest5 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a1888.data')
coltest5,magtest5, mptest5 = cut2(htest5)
htest6 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a1988.data')
coltest6,magtest6, mptest6 = cut2(htest6)
htest7 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a1838.data')
coltest7,magtest7, mptest7 = cut2(htest7)
htest8 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a2038.data')
coltest8,magtest8, mptest8 = cut2(htest8)


met1 = -2.0
met2 = -1.75
met3 = -1.5
# ~met1 = -2.087
# ~met2 = -1.839
# ~met3 = -1.586

mag_v1, mag_i1, Color_iso1, eep_first1 = iso_mag(Age, met1, distance, Abs, afe_init)
mag_v2, mag_i2, Color_iso2, eep_first2 = iso_mag(Age, met2, distance, Abs, afe_init)
mag_v3, mag_i3, Color_iso3, eep_first3 = iso_mag(Age, met3, distance, Abs, afe_init)


# ~plt.plot(htest1.log_Teff, htest1.log_L)
# ~plt.plot(htest1bis.log_Teff, htest1bis.log_L, linestyle=':')
# ~plt.gca().invert_xaxis()
# ~plt.legend(loc='best')
# ~plt.show()
# ~plt.close()
# ~kill


# ~bi = np.loadtxt('catalogs/basti_iso.txt')
# ~plt.plot(bi[:,7] - bi[:,10],bi[:,7])


# ~mist = np.loadtxt('catalogs/MIST_iso_601176d8da858.iso.cmd')
# ~ct = [605-int(mist[0,0])]
# ~plt.plot(mist[:ct[0]-1,14] - mist[:ct[0]-1,19], mist[:ct[0]-1,14])

# ~mist2 = np.loadtxt('catalogs/0007700M.track.eep.cmd')
# ~plt.plot(mist2[:ct[0]-1,10] - mist2[:ct[0]-1,15], mist2[:ct[0]-1,10])


# ~plt.plot(coltest2 , magtest2, label=r'MESA track: M = 0.77 $M_{\odot}$, Z = 0.00045', c='r')
# ~plt.plot(Color_iso3,mag_v3, c='r', label=r'Isochrone: Age = 13.32 Gyr, [Fe/H] = -1.5',linestyle=':')
plt.plot(coltest1bis , magtest1bis, label=r'test', c='c')
# ~plt.plot(coltest1 , magtest1, label=r'MESA track: M = 0.77 $M_{\odot}$, Z = 0.00025', c='k')
plt.plot(Color_iso2,mag_v2, c='k', label=r'Isochrone: Age = 13.32 Gyr, [Fe/H] = -1.75',linestyle=':')
# ~plt.plot(coltest3 , magtest3, label=r'MESA track: M = 0.77 $M_{\odot}$, Z = 0.000142', c='c')
# ~plt.plot(Color_iso1,mag_v1, c='c', label=r'Isochrone: Age = 13.32 Gyr, [Fe/H] = -2.0',linestyle=':')
plt.axhline(0.18, linestyle ='--', label=r'$\rm Magnitude \: cut \: M_1$', c='k')
plt.xlim(0.2,1.15)
plt.ylim(6,-5)
plt.tick_params(labelsize=14)
plt.subplots_adjust(bottom=0.15, top=0.89)
lgnd = plt.legend(loc='upper left', fontsize = 10)
plt.text(0.9,3.5,r'$\alpha$ = 1.938',va='center',fontsize=16,alpha=1.,
	bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
plt.xlabel(' F606W - F814W', fontsize = 20)
plt.ylabel(' F606W', fontsize = 20)
plt.show() 
plt.close()
kill

# ~plt.plot(coltest1 , magtest1, label=r'MESA track: M = 0.77 $M_{\odot}$, Z = 0.00025', c='k')
# ~plt.plot(coltest5 , magtest5, label=r'MESA track: $\rm \Delta_{\alpha} = 0.05$', c='r')
# ~plt.plot(coltest6 , magtest6, c='r')
# ~plt.plot(coltest7 , magtest7, label=r'MESA track: $\rm \Delta_{\alpha} = 0.1$', c='c')
# ~plt.plot(coltest8 , magtest8, c='c')
# ~plt.plot(Color_iso2,mag_v2, c='k', label=r'Isochrone: Age = 13.32 Gyr, [Fe/H] = -1.75',linestyle=':')
# ~plt.axhline(0.18, linestyle ='--', label=r'$\rm Magnitude \: cut \: M_1$', c='k')
# ~plt.xlim(0.4,1.15)
# ~plt.ylim(0.5,-4)
# ~plt.tick_params(labelsize=14)
# ~plt.subplots_adjust(bottom=0.15, top=0.89)
# ~lgnd = plt.legend(loc='upper left', fontsize = 10)
# ~plt.text(0.9,3.5,r'$\alpha$ = 1.938',va='center',fontsize=16,alpha=1.,
	# ~bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
# ~plt.xlabel(' F606W - F814W', fontsize = 20)
# ~plt.ylabel(' F606W', fontsize = 20)
# ~plt.show() 
# ~plt.close()
# ~kill

#-----------------------------------------------------------------------

# READ MESSA FILES
# ~string_met = 'Z00005'
string_mass = 'M075'
# ~string_name = ['Z00005', 'Z00010', 'Z00015', 'Z00020','Z00025', 'Z00030', 'Z00035', 'Z00040']
# ~string_name = ['Z00015', 'Z00020','Z00025', 'Z00030', 'Z00035', 'Z00040']
# ~string_name = ['Z00025', 'Z00030', 'Z00035', 'Z00040']
string_name = ['Z00040']

for string_met in string_name:
	# ~if (string_met == 'Z00020'and string_mass == 'M080'):
		# ~h1 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M070.data')
		# ~col1,mag1, mp = cut2(h1)
		# ~h2 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_M090.data')
	if (string_met == 'Z00005'and string_mass == 'M075'):
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
		# ~h23 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_alphafe.data')
		# ~col23,mag23, mp23 = cut2(h23)
	h12 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a120.data')
	col12,mag12, mp12 = cut2(h12)
	h28 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a130.data')
	col28,mag28, mp28 = cut2(h28)
	h13 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a140.data')
	col13,mag13, mp13 = cut2(h13)
	h26 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a150.data')
	col26,mag26, mp26 = cut2(h26)
	h14 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a160.data')
	col14,mag14, mp14 = cut2(h14)
	h24 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a170.data')
	col24,mag24, mp24 = cut2(h24)
	h15 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a180.data')
	col15,mag15, mp15 = cut2(h15)
	h16 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a190.data')
	col16,mag16, mp16 = cut2(h16)
	h17 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a200.data')
	col17,mag17, mp17 = cut2(h17)
	h18 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a210.data')
	col18,mag18, mp18 = cut2(h18)
	h19 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a220.data')
	col19,mag19, mp19 = cut2(h19)
	h25= mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a230.data')
	col25,mag25, mp25= cut2(h25)
	h20 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a240.data')
	col20,mag20, mp20 = cut2(h20)
	h27= mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a250.data')
	col27,mag27, mp27= cut2(h27)
	h21 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a260.data')
	col21,mag21, mp21 = cut2(h21)
	h29= mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a270.data')
	col29,mag29, mp29= cut2(h29)
	h22 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/'+string_met+'/history_a280.data')
	col22,mag22, mp22 = cut2(h22)


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
	finterp = [f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28]

	bestalpha = 8 # alpha = 2
	# ~bestalpha = 4 # alpha = 1.6
	# ~bestalpha = 12 # alpha = 2.4

	cm2,cp0,cp2 = coord_fid(bestalpha)

	# ~alvm2 = np.linspace(-2.1, -1.9, 20)
	# ~for valm2 in alvm2:
		# ~coledge1_m2, coledge2_m2 = dc_da(valm2, bestalpha)
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c1_m2.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(string_met, coledge1_m2[0],coledge1_m2[1],coledge1_m2[2],coledge1_m2[3],coledge1_m2[4],coledge1_m2[5],coledge1_m2[6],coledge1_m2[7]))
		# ~fid_file.close()
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c2_m2.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(string_met, coledge2_m2[0],coledge2_m2[1],coledge2_m2[2],coledge2_m2[3],coledge2_m2[4],coledge2_m2[5],coledge2_m2[6],coledge2_m2[7]))
		# ~fid_file.close()
		
	# ~alvp0 = np.linspace(-0.1, 0.1, 20)
	# ~for valp0 in alvp0:
		# ~coledge1_p0, coledge2_p0 = dc_da(valp0, bestalpha)
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c1_p0.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(string_met, coledge1_p0[0],coledge1_p0[1],coledge1_p0[2],coledge1_p0[3],coledge1_p0[4],coledge1_p0[5],coledge1_p0[6],coledge1_p0[7]))
		# ~fid_file.close()
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c2_p0.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(string_met, coledge2_p0[0],coledge2_p0[1],coledge2_p0[2],coledge2_p0[3],coledge2_p0[4],coledge2_p0[5],coledge2_p0[6],coledge2_p0[7]))
		# ~fid_file.close()
		
	# ~alvp2 = np.linspace(1.9, 2.1, 20)
	# ~for valp2 in alvp2:
		# ~coledge1_p2, coledge2_p2 = dc_da(valp2, bestalpha)
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c1_p2.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(string_met, coledge1_p2[0],coledge1_p2[1],coledge1_p2[2],coledge1_p2[3],coledge1_p2[4],coledge1_p2[5],coledge1_p2[6],coledge1_p2[7]))
		# ~fid_file.close()
		# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c2_p2.txt', 'a+') as fid_file:
			# ~fid_file.write('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(string_met, coledge2_p2[0],coledge2_p2[1],coledge2_p2[2],coledge2_p2[3],coledge2_p2[4],coledge2_p2[5],coledge2_p2[6],coledge2_p2[7]))
		# ~fid_file.close()

	
	# ~print(coledge1_m2, coledge2_m2)
	# ~print(coledge1_p0, coledge2_p0)
	# ~print(coledge1_p2, coledge2_p2)
	# ~print(cm2,cp0,cp2)
	# ~kill



	
	

	# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord_25_a24.txt', 'a+') as fid_file:
		# ~fid_file.write('%s %.4f %.4f %.4f \n' %(string_met, cm2,cp0,cp2))
	# ~fid_file.close()
	# ~kill

########################################################################
########################################################################
	# READ PAINTED FILES
	# ~p1 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_fid.data')
	# ~m1 = cut(p1) # cut pre main sequence
	# ~V1, R1 = p1[m1:,9], p1[m1:,10]
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

	# ~met='-2.0'
	# ~hist = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/distrib'+met+'.txt', usecols=(1,2,3,4,5,6))
	# ~hist2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/distrib2'+met+'.txt', usecols=(1,2,3,4,5,6))
	#-----------------------------------------------------------------------


	# ~plt.figure()
	# ~plt.plot(col12 , mag12, label='a = 1.20')
	# ~plt.plot(col28 , mag28, label='a = 1.30')
	# ~plt.plot(col13 , mag13, label='a = 1.40')
	# ~plt.plot(col26 , mag26, label='a = 1.50')
	# ~plt.plot(col14 , mag14, label='a = 1.60')
	# ~plt.plot(col24 , mag24, label='a = 1.70')
	# ~plt.plot(col15 , mag15, label='a = 1.80')
	# ~plt.plot(col16 , mag16, label='a = 1.90')
	# ~plt.plot(col17 , mag17, label='a = 2.00')
	# ~plt.plot(col18 , mag18, label='a = 2.10')
	# ~plt.plot(col19 , mag19, label='a = 2.20')
	# ~plt.plot(col25 , mag25, label='a = 2.30')
	# ~plt.plot(col20 , mag20, label='a = 2.40')
	# ~plt.plot(col27 , mag27, label='a = 2.50')
	# ~plt.plot(col21 , mag21, label='a = 2.60')
	# ~plt.plot(col29 , mag29, label='a = 2.70')
	# ~plt.plot(col22 , mag22, label='a = 2.80')
	# ~plt.plot(col1 , mag1, label='M = 0.70', c='b')
	# ~plt.plot(col2 , mag2, label='M = 0.90', c='b')
	# ~plt.plot(col3 , mag3, label='y = 0.2004', c='r')
	# ~plt.plot(col4 , mag4, label='y = 0.2804', c='g')
	# ~plt.plot(col5 , mag5, label='No ledoux', linestyle=':')
	# ~plt.plot(col6 , mag6, label='cox', linestyle=':')
	# ~plt.plot(col7 , mag7, label='No type 2', linestyle=':')
	# ~plt.plot(col8 , mag8, label='No overshoot', linestyle=':')
	# ~plt.plot(col9 , mag9, label='Reimers eta = 0.8', linestyle=':')
	# ~plt.plot(col10 , mag10, label='No element diffusion', linestyle=':')
	# ~plt.plot(col11 , mag11, label='No rotational mixing', linestyle=':')
	# plt.plot(h.abs_mag_F606W - h.abs_mag_F814W , h.abs_mag_F606W, label='test', linestyle=':')
	# ~plt.gca().invert_yaxis()
	# ~plt.legend(loc='best')
	# ~plt.show()
	# ~plt.close()
	# ~plt.xlabel(r'Log $\rm T_{eff}$', fontsize=16)
	# ~plt.ylabel(r'Log L/$\rm L_{\odot}$', fontsize=16)
	# ~plt.text(3.73,3.0,r'$\alpha$ = 2.8',va='center',fontsize=16,alpha=1.)
	# ~plt.text(3.62,2.5,r'$\alpha$ = 1.2',va='center',fontsize=16,alpha=1.)
	# ~plt.text(3.68,0.18,r'0.75 $M_{\odot}$,   Z = 0.0001',va='center',fontsize=16,alpha=1.,
	# ~bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
	# ~plt.subplots_adjust(bottom=0.15, top=0.89, right=0.930)
	# ~plt.tick_params(labelsize=16)
	# ~plt.gca().invert_xaxis()
	# ~plt.show()
	# ~plt.close()
	# ~kill
	# ~plt.figure()
	# ~plt.plot(h17.log_Teff, h17.log_L, label='fiducial', c='k')
	# ~plt.plot(h1.log_Teff, h1.log_L, label='M = 0.70', c='b')
	# ~plt.plot(h2.log_Teff, h2.log_L, label='M = 0.90', c='b')
	# ~plt.plot(h3.log_Teff, h3.log_L, label='y = 0.2004', c='g')
	# ~plt.plot(h4.log_Teff, h4.log_L, label='y = 0.2804', c='g')
	# ~plt.plot(h5.log_Teff, h5.log_L, label='No ledoux', linestyle=':')
	# ~plt.plot(h6.log_Teff, h6.log_L, label='cox', linestyle=':')
	# ~plt.plot(h7.log_Teff, h7.log_L, label='No type 2', linestyle=':')
	# ~plt.plot(h.log_Teff, h.log_L, label='test', c='r', linestyle=':')
	# ~plt.gca().invert_xaxis()
	# ~plt.legend(loc='best')
	# ~plt.show()
	# ~plt.close()

	# ~binmix = np.linspace(1.2, 2.4,7)
	# ~dmix = np.diff(binmix)[0]
	# ~bincenter = (binmix[:-1] + binmix[1:]) / 2
	# ~mix_low, mix_high = error_compute(dmix, histo,bincenter)
	# ~mix_mean = (binmix[np.argmax(histo)] + binmix[np.argmax(histo)+1])/2.

	# ~plt.figure()
	# ~for j in range(12):
		# ~print(j)
		# ~plt.bar(bincenter, hist[j,:], width=0.2, align='center', edgecolor='k', alpha=0.6)
	# ~plt.bar(bincenter, hist2[12,:], width=0.2, align='center', edgecolor='k', alpha=0.6, color='b')
	# ~plt.bar(bincenter, hist[12,:], width=0.2, align='center', edgecolor='k', alpha=0.6, color='r')
	# ~plt.xlim(1.1, 2.5)
	# ~plt.show()
	# ~plt.close()
	# ~kill


