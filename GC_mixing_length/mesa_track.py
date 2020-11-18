import numpy as np
import mesa_reader as mr
import matplotlib.pyplot as plt
import math
import os

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
########################################################################
########################################################################
Zsun = 0.0134
Fe_h = -2.0
c = 2.9979e8 # in meters
L0 = 3.0128e28 #zero-point luminosity in W
Lsun = 3.828e26 #solar luminosity in W
rsun = 6.957e+10	# in centimeters
msun = 1.989e+33 #g

# ~directory = '/home/david/codes/data/GC_mixing_length/bol_corr/'
# ~with open('/home/david/codes/Analysis/GC_mixing_length/bc_MIST.dat',"w") as fid:
	# ~fid.write('%s %s %s %s %s %s \n' %('#','Teff','logg',' M_div_H', 'F606W', 'F814W'))
	# ~for filename in os.listdir(directory):
		# ~print(os.path.join(directory, filename))
		# ~# dat = np.loadtxt('catalogs/fehm200.HST_ACSWF')
		# ~dat = np.loadtxt(os.path.join(directory, filename))
		# ~# tefs = np.unique(dat[:,0])
		# ~# sfs = np.unique(dat[:,1])
		# ~tefs = dat[:,0]
		# ~sfs = dat[:,1]
		# ~feH = dat[:,2]
		# ~Fil1 = dat[:,10]
		# ~Fil2 = dat[:,15]
		# ~for i in range(len(tefs)):
			# ~fid.write('%.8g %.8g %.8g %.8g %.8g\n' %(tefs[i], sfs[i], feH[i], Fil1[i], Fil2[i]))
# ~fid.close()
	


al = ['10','12','14','16','18','20']


# ~cmd = np.loadtxt('/home/david/codes/data/GC_mixing_length/iso/isochrones_a100.txt.HST_ACSWF')
# ~cmd = np.loadtxt('/home/david/codes/data/GC_mixing_length/iso/isochrones_a100.txt')
# ~V = cmd[:,15]
# ~I = cmd[:,20]
# ~T = cmd[:,8]
# ~L = cmd[:,9]


# ~print(V[0], I[0])
# ~print(T[0], L[0])
kill

h = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history.data')
# ~h1a = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a100_M070.data')
# ~h1b = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a100_M075.data')
# ~h1c = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a100_M080.data')
# ~h2 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a120.data')
# ~h3 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a140.data')
# ~h4 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a160.data')
# ~h5 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a180.data')
# ~h6 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a200.data')
# ~raul = np.loadtxt('/home/david/codes/data/GC_mixing_length/alpha200.txt')
# ~tf, l = raul[:,0], raul[:,1]
# ~raul1 = np.loadtxt('/home/david/codes/data/GC_mixing_length/alpha100.txt')
# ~tf1, l1 = raul1[:,0], raul1[:,1]
# ~raula = np.loadtxt('/home/david/codes/data/GC_mixing_length/a100.txt')
# ~tfa, la = raula[:,1], raula[:,5]
# ~raulb = np.loadtxt('/home/david/codes/data/GC_mixing_length/a200.txt')
# ~tfb, lb = raulb[:,1], raulb[:,5]

# ~plt.plot(h.abs_mag_F606W - h.abs_mag_F814W, h.abs_mag_F606W)
plt.plot(h.abs_mag_V - h.abs_mag_I, h.abs_mag_V)
# ~plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend(loc='best')
plt.show()
plt.close()

# ~plt.figure()
# ~plt.plot(T,L, c='r', label='13.5 Gyr isochrone')
# ~plt.plot(h.log_Teff, h.log_L, label='test', c='b')
# ~plt.plot(h1a.log_Teff, h1a.log_L, label='a = 1.0, M = 0.7', c='b', linestyle=':')
# ~plt.plot(h1b.log_Teff, h1b.log_L, label='a = 1.0, M = 0.75', c='b')
# ~plt.plot(h1c.log_Teff, h1c.log_L, label='a = 1.0, M = 0.8', c='b', linestyle='--')
# ~plt.plot(h2.log_Teff, h2.log_L, label='a = 1.2')
# ~plt.plot(h3.log_Teff, h3.log_L, label='a = 1.4')
# ~plt.plot(h4.log_Teff, h4.log_L, label='a = 1.6')
# ~plt.plot(h5.log_Teff, h5.log_L, label='a = 1.8')
# ~plt.plot(h6.log_Teff, h6.log_L, label='a = 2.0')
# ~plt.plot(tf,l, label='Raul plot with a = 2', c='b', linestyle='--')
# ~plt.plot(tf1,l1, label='Raul plot with a = 1', c='r', linestyle='--')
# ~plt.scatter(tfa,la, label='Raul plot with a = 1', c='g', marker='o')
# ~plt.scatter(tfb,lb, label='Raul plot with a = 2', c='g', marker='o')
# ~plt.gca().invert_xaxis()
# ~plt.legend(loc='best')
# ~plt.show()
# ~plt.close()

# ~kill

# ~for j in al:
# ~# load the data
# ~h = mr.MesaData('/home/david/codes/data/GC_mixing_length/Z3alpha'+j+'/trimmed_history.data')
# ~h = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history.data')


# ~name2 = ['a100','a120','a140','a160','a180','a200']
# ~name2 = ['a100_M070','a100_M075','a100_M080']

# ~for j in name2:
#j = name[4]
	# ~h = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_'+j+'.data')
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


# ~name = ['a100','a125','a150','a175','a200']
# ~for j in name:	
	# ~raul = np.loadtxt('/home/david/codes/data/GC_mixing_length/'+j+'.txt')
	# ~sf = raul[:,2]
	# ~teff = 10**(raul[:,1])

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
	# ~plt.scatter(M606-M814, M606)
# ~plt.gca().invert_yaxis()
# ~plt.show()

