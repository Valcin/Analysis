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


#-----------------------------------------------------------------------

# READ MESSA FILES

# ~h = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history.data')
# ~h1 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a100_M075_z0002.data')
# ~h2 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a120_M075_z0002.data')
# ~h3 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a140_M075_z0002.data')
# ~h4 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a160_M075_z0002.data')
# ~h5 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a180_M075_z0002.data')
# ~h6 = mr.MesaData('/home/david/codes/data/GC_mixing_length/initial_mesa_dir/LOGS/history_a200_M075_z0002.data')

# READ PAINTED FILES
# ~p1 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a100.data')
# ~m1 = cut(p1) # cut pre main sequence
# ~V1, R1 = p1[m1:,9], p1[m1:,10]
# ~p2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a120.data')
# ~m2 = cut(p2) # cut pre main sequence
# ~V2, R2 = p2[m2:,9], p2[m2:,10]
# ~p3 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a140.data')
# ~m3 = cut(p3) # cut pre main sequence
# ~V3, R3 = p3[m3:,9], p3[m3:,10]
p4 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a160.data')
m4 = cut(p4) # cut pre main sequence
V4, R4 = p4[m4:,9], p4[m4:,10]
p5 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a180.data')
m5 = cut(p5) # cut pre main sequence
V5, R5 = p5[m5:,9], p5[m5:,10]
# ~p6 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/painted_files/painted_a200.data')
# ~m6 = cut(p6) # cut pre main sequence
# ~V6, R6 = p6[m6:,9], p6[m6:,10]



#-----------------------------------------------------------------------

plt.figure()
# ~plt.plot(h.abs_mag_F606W - h.abs_mag_F814W , h.abs_mag_F606W, c='g', label='test')
# ~plt.plot(h1.abs_mag_F606W - h1.abs_mag_F814W , h1.abs_mag_F606W, label='a = 1.0, M = 0.75, Z =0.0002', c='b')
# ~plt.plot(h2.abs_mag_F606W - h2.abs_mag_F814W , h2.abs_mag_F606W, label='a = 1.2, M = 0.75, Z =0.0002', c='r')
# ~plt.plot(v-i, v, c='k')
plt.plot(V4-R4,V4, label='a = 1.6')
plt.plot(V5-R5,V5, label='a = 1.8')
plt.gca().invert_yaxis()
plt.legend(loc='best')
plt.show()
plt.close()

# ~plt.figure()
# ~plt.plot(h.log_Teff, h.log_L, label='test', c='b')
# ~plt.plot(h1.log_Teff, h1.log_L, label='a = 1.0, M = 0.75', c='b', linestyle=':')
# ~plt.plot(h2.log_Teff, h2.log_L, label='a = 1.2')
# ~plt.plot(h3.log_Teff, h3.log_L, label='a = 1.4')
# ~plt.plot(h4.log_Teff, h4.log_L, label='a = 1.6')
# ~plt.plot(h5.log_Teff, h5.log_L, label='a = 1.8')
# ~plt.plot(h6.log_Teff, h6.log_L, label='a = 2.0')
# ~plt.gca().invert_xaxis()
# ~plt.legend(loc='best')
# ~plt.show()
# ~plt.close()



