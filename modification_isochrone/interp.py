from numba import jit, float64
from math import sqrt
import numpy as np
import scipy.interpolate
import os
import re

@jit(nopython=True)
#~ def interp_box(x, y, z, box, values):
def interp_box(x, y, box, values):
	"""
	box is 8x3 array, though not really a box

	values is length-8 array, corresponding to values at the "box" coords

	TODO: should make power `p` an argument
	"""
	
	### MODIFIED FOR 2D INTERP

	# Calculate the distance to each vertex

	#~ print(box)
	#~ print(values)
	#~ print(x,y,z)

	val = 0
	norm = 0
	for i in range(4):
		# Inv distance, or Inv-dsq weighting
		#~ distance = sqrt((x-box[i,0])**2 + (y-box[i,1])**2 + (z-box[i, 2])**2)
		distance = sqrt((x-box[i,0])**2 + (y-box[i,1])**2)

		# If you happen to land on exactly a corner, you're done.
		if distance == 0:
			val = values[i]
			norm = 1.
			break

		w = 1./distance
		# w = 1./((x-box[i,0])*(x-box[i,0]) +
		#         (y-box[i,1])*(y-box[i,1]) +
		#         (z-box[i, 2])*(z-box[i, 2]))
		val += w * values[i]
		norm += w
	#~ print(val/norm)	
	return val/norm

@jit(nopython=True)
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

@jit(nopython=True)
def searchsorted_many(arr, values):
	N = len(arr)
	Nval = len(values)
	inds = np.zeros(Nval)
	for i in range(Nval):
		x = values[i]
		L = 0
		R = N-1
		done = False
		m = (L+R)//2
		while not done:
			if arr[m] < x:
				L = m + 1
			elif arr[m] > x:
				R = m - 1
			m = (L+R)//2
			if L>R:
				done = True
		inds[i] = L
	return inds


	
@jit(nopython=True)
def interp_values(mass_arr, age_arr, feh_arr, icol,
				 grid, eep_col, mass_col, ages, fehs, grid_Ns):
	"""mass_arr, age_arr, feh_arr are all arrays at which values are desired

	icol is the column index of desired value
	grid is nfeh x nage x max(nmass) x ncols array
	mass_col is the column index of mass
	ages is grid of ages
	fehs is grid of fehs
	grid_Ns keeps track of nmass in each slice (beyond this are nans)

	"""

	N = len(mass_arr)
	results = np.zeros(N)

	Nage = len(ages)
	Nfeh = len(fehs)

	for i in range(N):
		results[i] = interp_value(mass_arr[i], age_arr[i], feh_arr[i], icol,
								 grid,eep_col, mass_col, ages, fehs, grid_Ns, False)

		## Things are slightly faster if the below is used, but for consistency,
		## using above.
		# mass = mass_arr[i]
		# age = age_arr[i]
		# feh = feh_arr[i]

		# ifeh = searchsorted(fehs, Nfeh, feh)
		# iage = searchsorted(ages, Nage, age)
		# if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
		#     results[i] = np.nan
		#     continue

		# pts = np.zeros((8,3))
		# vals = np.zeros(8)

		# i_f = ifeh - 1
		# i_a = iage - 1
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[0, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[0, 1] = ages[i_a]
		# pts[0, 2] = fehs[i_f]
		# vals[0] = grid[i_f, i_a, imass, icol]
		# pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[1, 1] = ages[i_a]
		# pts[1, 2] = fehs[i_f]
		# vals[1] = grid[i_f, i_a, imass-1, icol]

		# i_f = ifeh - 1
		# i_a = iage
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[2, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[2, 1] = ages[i_a]
		# pts[2, 2] = fehs[i_f]
		# vals[2] = grid[i_f, i_a, imass, icol]
		# pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[3, 1] = ages[i_a]
		# pts[3, 2] = fehs[i_f]
		# vals[3] = grid[i_f, i_a, imass-1, icol]

		# i_f = ifeh
		# i_a = iage - 1
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[4, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[4, 1] = ages[i_a]
		# pts[4, 2] = fehs[i_f]
		# vals[4] = grid[i_f, i_a, imass, icol]
		# pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[5, 1] = ages[i_a]
		# pts[5, 2] = fehs[i_f]
		# vals[5] = grid[i_f, i_a, imass-1, icol]

		# i_f = ifeh
		# i_a = iage
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[6, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[6, 1] = ages[i_a]
		# pts[6, 2] = fehs[i_f]
		# vals[6] = grid[i_f, i_a, imass, icol]
		# pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[7, 1] = ages[i_a]
		# pts[7, 2] = fehs[i_f]
		# vals[7] = grid[i_f, i_a, imass-1, icol]

		# results[i] = interp_box(mass, age, feh, pts, vals)
	return results
	
@jit(nopython=True)
def interp_eeps(age_arr, feh_arr, icol,
				 grid, eep_col, mass_col, ages, fehs, grid_Ns):
	"""mass_arr, age_arr, feh_arr are all arrays at which values are desired

	icol is the column index of desired value
	grid is nfeh x nage x max(nmass) x ncols array
	mass_col is the column index of mass
	ages is grid of ages
	fehs is grid of fehs
	grid_Ns keeps track of nmass in each slice (beyond this are nans)

	"""

	N = len(age_arr)
	results = np.zeros(N)

	Nage = len(ages)
	Nfeh = len(fehs)

	for i in range(N):
		results[i] = interp_eep(age_arr[i], feh_arr[i], icol,
								 grid,eep_col, mass_col, ages, fehs, grid_Ns, False)

		## Things are slightly faster if the below is used, but for consistency,
		## using above.
		# mass = mass_arr[i]
		# age = age_arr[i]
		# feh = feh_arr[i]

		# ifeh = searchsorted(fehs, Nfeh, feh)
		# iage = searchsorted(ages, Nage, age)
		# if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
		#     results[i] = np.nan
		#     continue

		# pts = np.zeros((8,3))
		# vals = np.zeros(8)

		# i_f = ifeh - 1
		# i_a = iage - 1
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[0, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[0, 1] = ages[i_a]
		# pts[0, 2] = fehs[i_f]
		# vals[0] = grid[i_f, i_a, imass, icol]
		# pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[1, 1] = ages[i_a]
		# pts[1, 2] = fehs[i_f]
		# vals[1] = grid[i_f, i_a, imass-1, icol]

		# i_f = ifeh - 1
		# i_a = iage
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[2, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[2, 1] = ages[i_a]
		# pts[2, 2] = fehs[i_f]
		# vals[2] = grid[i_f, i_a, imass, icol]
		# pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[3, 1] = ages[i_a]
		# pts[3, 2] = fehs[i_f]
		# vals[3] = grid[i_f, i_a, imass-1, icol]

		# i_f = ifeh
		# i_a = iage - 1
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[4, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[4, 1] = ages[i_a]
		# pts[4, 2] = fehs[i_f]
		# vals[4] = grid[i_f, i_a, imass, icol]
		# pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[5, 1] = ages[i_a]
		# pts[5, 2] = fehs[i_f]
		# vals[5] = grid[i_f, i_a, imass-1, icol]

		# i_f = ifeh
		# i_a = iage
		# Nmass = grid_Ns[i_f, i_a]
		# imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
		# pts[6, 0] = grid[i_f, i_a, imass, mass_col]
		# pts[6, 1] = ages[i_a]
		# pts[6, 2] = fehs[i_f]
		# vals[6] = grid[i_f, i_a, imass, icol]
		# pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
		# pts[7, 1] = ages[i_a]
		# pts[7, 2] = fehs[i_f]
		# vals[7] = grid[i_f, i_a, imass-1, icol]

		# results[i] = interp_box(mass, age, feh, pts, vals)
	return results

@jit(nopython=True)
def interp_value(mass, age, feh, icol,
				 grid ,eep_col, mass_col, ages, fehs, grid_Ns, debug):
				 # return_box):
	"""mass, age, feh are *single values* at which values are desired

	icol is the column index of desired value
	grid is nfeh x nage x max(nmass) x ncols array
	mass_col is the column index of mass
	ages is grid of ages
	fehs is grid of fehs
	grid_Ns keeps track of nmass in each slice (beyond this are nans)

	TODO:  fix situation where there is exact match in age, feh, so we just
	interpolate along the track, not between...
	"""
	if np.isnan(mass) or np.isnan(age) or np.isnan(feh):
		return np.nan

	Nage = len(ages)
	Nfeh = len(fehs)
	
	ifeh = searchsorted(fehs, Nfeh, feh)
	iage = searchsorted(ages, Nage, age)
	if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
		return np.nan

	
	i_f = ifeh
	i_a = iage
	Nmass = grid_Ns[i_f, i_a]
	imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
	
	pts = np.zeros((8,3))
	pts2 = np.zeros((8,3))
	vals = np.zeros(8)
	


	i_f = ifeh - 1
	i_a = iage - 1
	Nmass = grid_Ns[i_f, i_a]
	imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
	#~ print(mass, Nmass, imass)
	pts[0, 0] = grid[i_f, i_a, imass, mass_col]
	pts[0, 1] = ages[i_a]
	pts[0, 2] = fehs[i_f]
	vals[0] = grid[i_f, i_a, imass, icol]
	pts[1, 0] = grid[i_f, i_a, imass-1, mass_col]
	pts[1, 1] = ages[i_a]
	pts[1, 2] = fehs[i_f]
	vals[1] = grid[i_f, i_a, imass-1, icol]
	#~ print(i_f, i_a, imass)
	#~ print(grid[i_f, i_a, imass, :])
	#~ print(grid[i_f, i_a, imass, icol-1])
	#~ print(pts)	
	#~ print(vals[0])
	#~ print(vals[1])

	
	i_f = ifeh - 1
	i_a = iage
	Nmass = grid_Ns[i_f, i_a]
	imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
	#~ print(mass, Nmass, imass)
	pts[2, 0] = grid[i_f, i_a, imass, mass_col]
	pts[2, 1] = ages[i_a]
	pts[2, 2] = fehs[i_f]
	vals[2] = grid[i_f, i_a, imass, icol]
	pts[3, 0] = grid[i_f, i_a, imass-1, mass_col]
	pts[3, 1] = ages[i_a]
	pts[3, 2] = fehs[i_f]
	vals[3] = grid[i_f, i_a, imass-1, icol]
	
	

	i_f = ifeh
	i_a = iage - 1
	Nmass = grid_Ns[i_f, i_a]
	imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
	#~ print(imass)
	pts[4, 0] = grid[i_f, i_a, imass, mass_col]
	pts[4, 1] = ages[i_a]
	pts[4, 2] = fehs[i_f]
	vals[4] = grid[i_f, i_a, imass, icol]
	pts[5, 0] = grid[i_f, i_a, imass-1, mass_col]
	pts[5, 1] = ages[i_a]
	pts[5, 2] = fehs[i_f]
	vals[5] = grid[i_f, i_a, imass-1, icol]
	
	
	i_f = ifeh
	i_a = iage
	Nmass = grid_Ns[i_f, i_a]
	imass = searchsorted(grid[i_f, i_a, :, mass_col], Nmass, mass)
	#~ print(mass, Nmass, imass)
	pts[6, 0] = grid[i_f, i_a, imass, mass_col]
	pts[6, 1] = ages[i_a]
	pts[6, 2] = fehs[i_f]
	vals[6] = grid[i_f, i_a, imass, icol]
	pts[7, 0] = grid[i_f, i_a, imass-1, mass_col]
	pts[7, 1] = ages[i_a]
	pts[7, 2] = fehs[i_f]
	vals[7] = grid[i_f, i_a, imass-1, icol]
	#~ print(pts)
	#~ print(vals)
	
	# if debug:
	#     result = np.zeros((8,4))
	#     for i in range(8):
	#         result[i, 0] = pts[i, 0]
	#         result[i, 1] = pts[i, 1]
	#         result[i, 2] = pts[i, 2]
	#         result[i, 3] = vals[i]
	#     return result
	# else:
	return interp_box(mass, age, feh, pts, vals)


#~ @jit(nopython=True)
def interp_eep(age, feh, icol,
				 grid ,eep_col, mass_col, ages, fehs, grid_Ns, debug):
				 # return_box):
	"""mass, age, feh are *single values* at which values are desired

	icol is the column index of desired value
	grid is nfeh x nage x max(nmass) x ncols array
	mass_col is the column index of mass
	ages is grid of ages
	fehs is grid of fehs
	grid_Ns keeps track of nmass in each slice (beyond this are nans)

	TODO:  fix situation where there is exact match in age, feh, so we just
	interpolate along the track, not between...
	"""
	#~ if np.isnan(age) or np.isnan(feh):
		#~ return np.nan

	Nage = len(ages)
	Nfeh = len(fehs)


	ifeh = searchsorted(fehs, Nfeh, feh)
	iage = searchsorted(ages, Nage, age)
	#~ if ifeh==0 or iage==0 or ifeh==Nfeh or iage==Nage:
		#~ return np.nan

	

	i_f = ifeh
	i_a = iage
	#~ print(i_f,i_a)
	#~ print(feh,age)
	eep1 = grid[i_f-1, i_a-1, :, eep_col]
	eep2 = grid[i_f-1, i_a, :, eep_col]
	eep3 = grid[i_f, i_a-1, :, eep_col]
	eep4 = grid[i_f, i_a, :, eep_col]

	eep1 = eep1[~np.isnan(eep1)]
	eep2 = eep2[~np.isnan(eep2)]
	eep3 = eep3[~np.isnan(eep3)]
	eep4 = eep4[~np.isnan(eep4)]
	#~ print(eep1, eep2, eep3, eep4)
	
	#~ print(icol)

	debut = np.max(np.array([np.min(eep1), np.min(eep2), np.min(eep3), np.min(eep4)]))
	fin = np.min(np.array([np.max(eep1), np.max(eep2), np.max(eep3), np.max(eep4)]))

	magnitude_range = fin-debut
	
	magtu = np.zeros(int(round(magnitude_range))+1)
	if magnitude_range < 2:
		#~ magnitude = np.zeros(2)
		#~ teff = np.zeros(2)
		#~ logg = np.zeros(2)
		return magtu, debut





	for ind,j in enumerate(range(int(debut),int(fin)+1)):
	#~ for j in range(1,2):
		#~ print(ind,j, magnitude_range)
		pts = np.zeros((4,3))
		vals = np.zeros(4)

		### construct box
		i_f = ifeh - 1
		i_a = iage - 1
		dif1 = int(debut-np.min(eep1))
		pts[0, 0] = ages[i_a]
		pts[0, 1] = fehs[i_f]
		pts[0, 2] = grid[i_f, i_a, ind+dif1, icol]
		vals[0] = grid[i_f, i_a, ind+dif1, icol]

		i_f = ifeh - 1
		i_a = iage
		dif2 = int(debut-np.min(eep2))
		pts[1, 0] = ages[i_a]
		pts[1, 1] = fehs[i_f]
		pts[1, 2] = grid[i_f, i_a, ind+dif2, icol]
		vals[1] = grid[i_f, i_a, ind+dif2, icol]
	
		i_f = ifeh
		i_a = iage - 1
		dif3 = int(debut-np.min(eep3))
		pts[2, 0] = ages[i_a]
		pts[2, 1] = fehs[i_f]
		pts[2, 2] = grid[i_f, i_a, ind+dif3, icol]
		vals[2] = grid[i_f, i_a, ind+dif3, icol]
	
		i_f = ifeh
		i_a = iage
		dif4 = int(debut-np.min(eep4))
		pts[3, 0] = ages[i_a]
		pts[3, 1] = fehs[i_f]
		pts[3, 2] = grid[i_f, i_a, ind+dif4, icol]
		vals[3] = grid[i_f, i_a, ind+dif4, icol]
		
		#~ magtu[ind] = interp_box(age, feh, pts, vals)
		# ~print(age, feh)
		magtu[ind] = bilinear_interpolation(age, feh, pts)
		
		#~ print(ind, pts)
		#~ f = scipy.interpolate.interp2d(pts[:,0], pts[:,1], vals)
		#~ znew = f(age, feh)
		#~ magtu[ind] = znew[0]

		
	# if debug:
	#     result = np.zeros((8,4))
	#     for i in range(8):
	#         result[i, 0] = pts[i, 0]
	#         result[i, 1] = pts[i, 1]
	#         result[i, 2] = pts[i, 2]
	#         result[i, 3] = vals[i]
	#     return result
	# else:

	return magtu, debut

def bolcorr(icol, tefff, loggg, feh, AV):
				 # return_box):
	"""mass, age, feh are *single values* at which values are desired

	icol is the column index of desired value
	grid is nfeh x nage x max(nmass) x ncols array
	mass_col is the column index of mass
	ages is grid of ages
	fehs is grid of fehs
	grid_Ns keeps track of nmass in each slice (beyond this are nans)

	TODO:  fix situation where there is exact match in age, feh, so we just
	interpolate along the track, not between...
	"""
	#~ if np.isnan(age) or np.isnan(feh):
		#~ return np.nan
	
		
	metalarr = [-4.0, -3.5, -3.0, -2.75, -2.50, -2.25, -2.0, -1.75, -1.50, -1.25, -1.0, -0.75,\
	-0.5, -0.25, 0.0, 0.25, 0.5, 0.75 ]
	metalstr = ['m400', 'm350', 'm300', 'm275', 'm250', 'm225', 'm200', 'm175', 'm150', 'm125', 'm100', 'm075',\
	'm050', 'm025', 'p000', 'p025', 'p050', 'p075' ]

	ind = searchsorted(metalarr, len(metalarr), feh)
	iarray1 = np.zeros(tefff.size)
	iarray2 = np.zeros(tefff.size)
	
	distance1 = np.abs(metalarr[ind]-feh)
	distance2 = np.abs(metalarr[ind-1]-feh)
	norm = np.abs(metalarr[ind]-metalarr[ind-1])
	print(distance1, distance2, norm)
	
	if distance1 == 0:
		metallicity = [metalstr[ind]]
	elif distance2 == 0:
		metallicity = [metalstr[ind-1]]
	else:
		metallicity = [metalstr[ind], metalstr[ind-1]]

	if len(metallicity) == 1:
		co = corr(icol, tefff, loggg, AV, metallicity[0], iarray1)
	elif len(metallicity) == 2:
		co1 = corr(icol,tefff, loggg, AV, metallicity[0],iarray1)
		co2 = corr(icol,tefff, loggg, AV, metallicity[1],iarray2)
		
		co = (co1*distance1 + co2*distance2)/norm
		
	print(co)
		
	# if debug:
	#     result = np.zeros((8,4))
	#     for i in range(8):
	#         result[i, 0] = pts[i, 0]
	#         result[i, 1] = pts[i, 1]
	#         result[i, 2] = pts[i, 2]
	#         result[i, 3] = vals[i]
	#     return result
	# else:

	return co

@jit(nopython=True)
def interp3d(age, feh, pts, vals, iarray, j):

	### make 3d interpolation
	x1 = pts[0,0]
	x0 = pts[1,0]
	x = pts[6, 0]
	#~ print(x0, x, x1)
	
	y0 = pts[0,1]
	y1 = pts[2,1]
	y = age
	#~ print(y0, y, y1)
	
	z0 = pts[0,2]
	z1 = pts[4,2]
	z = feh
	#~ print(z0, z, z1)
	
	#first step
	xd = (x-x0)/(x1-x0)
	yd = (y-y0)/(y1-y0)
	zd = (z-z0)/(z1-z0)
	#~ print(xd, yd, zd)
	
	#second step
	C000 = vals[0]
	C100 = vals[1]
	C001 = vals[2]
	C101 = vals[3]
	C010 = vals[4]
	C110 = vals[5]
	C011 = vals[6]
	C111 = vals[7]
			
	C00 = C000*(1-xd) + C100*xd
	C01 = C001*(1-xd) + C101*xd
	C10 = C010*(1-xd) + C110*xd
	C11 = C011*(1-xd) + C111*xd
	
	#~ print(C00, C01)
	
	#third step
	C0 = C00*(1-yd) + C10*yd
	C1 = C01*(1-yd) + C11*yd
	
	#final step
	C = C0*(1-zd) + C1*zd

	iarray[j-1] = C
	
	
	return iarray

@jit(nopython=True)
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
	#~ return (q11 * (x2 - x) * (y2 - y) +
			#~ q21 * (x - x1) * (y2 - y) +
			#~ q12 * (x2 - x) * (y - y1) +
			#~ q22 * (x - x1) * (y - y1)
		   #~ ) / ((x2 - x1) * (y2 - y1) + 0.0)
	
	
def corr(icol, tefff, loggg, AV, metallicity, iarray):

	### define the indices
	DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'data'))
	bolfile = os.path.join(DATADIR,'feh'+ str(metallicity)+'.HST_ACSWF')
	TF = []
	LOGG = []
	av = []
	f1corr = []
	if icol == 11:
		numcol = 10
	elif icol == 13:
		numcol = 15

	for line in open(bolfile,'r'):
		if re.match('#', line):
			continue
		line = line.split()
		TF.append(float(line[0]))
		LOGG.append(float(line[1]))
		av.append(float(line[3]))
		f1corr.append(float(line[numcol]))



	for j in range(len(iarray)):
	#~ for j in range(5):

		tef = np.unique(TF)
		lg = np.unique(LOGG)
		aav = np.unique(av)
		itef = searchsorted(tef, len(tef), 10**tefff[j])
		ilogg = searchsorted(lg, len(lg), loggg[j])
		iav = searchsorted(aav, len(aav), AV)
		
		i_t = tef[itef]; i_l = lg[ilogg]; i_a = aav[iav]
		i_tmin = tef[itef-1]; i_lmin = lg[ilogg-1]; i_amin = aav[iav-1]
		
		#~ print(i_t, i_l, i_a)
		#~ print(i_tmin, i_lmin, i_amin)


		pts = np.zeros((8,3))
		vals = np.zeros(8)

		#~ for line in open(bolfile,'r'):
			#~ if re.match('#', line):
				#~ continue
			#~ line = line.split()
	
		### construct box
		#~ i_f = ifeh - 1
		#~ i_a = iage - 1
		pts[0, 0] = i_a
		pts[0, 1] = i_lmin
		pts[0, 2] = i_tmin
		hola = np.where((TF == i_tmin)&(LOGG == i_lmin)& (av == i_a))
		vals[0] = np.array(f1corr)[hola]
		pts[1, 0] = i_amin
		pts[1, 1] = i_lmin
		pts[1, 2] = i_tmin
		hola = np.where((TF == i_tmin)&(LOGG == i_lmin)& (av == i_amin))
		vals[1] = np.array(f1corr)[hola]


		#~ i_f = ifeh - 1
		#~ i_a = iage
		pts[2, 0] = i_a
		pts[2, 1] = i_l
		pts[2, 2] = i_tmin
		hola = np.where((TF == i_tmin)&(LOGG == i_l)& (av == i_a))
		vals[2] = np.array(f1corr)[hola]
		pts[3, 0] = i_amin
		pts[3, 1] = i_l
		pts[3, 2] = i_tmin
		hola = np.where((TF == i_tmin)&(LOGG == i_l)& (av == i_amin))
		vals[3] = np.array(f1corr)[hola]



		#~ i_f = ifeh
		#~ i_a = iage - 1
		pts[4, 0] = i_a
		pts[4, 1] = i_lmin
		pts[4, 2] = i_t
		hola = np.where((TF == i_t)&(LOGG == i_lmin)& (av == i_a))
		vals[4] = np.array(f1corr)[hola]
		pts[5, 0] = i_amin
		pts[5, 1] = i_lmin
		pts[5, 2] = i_t
		hola = np.where((TF == i_t)&(LOGG == i_lmin)& (av == i_amin))
		vals[5] = np.array(f1corr)[hola]
	
		
		#~ i_f = ifeh
		#~ i_a = iage
		pts[6, 0] = i_a
		pts[6, 1] = i_l
		pts[6, 2] = i_t
		hola = np.where((TF == i_t)&(LOGG == i_l)& (av == i_a))
		vals[6] = np.array(f1corr)[hola]
		pts[7, 0] = i_amin
		pts[7, 1] = i_l
		pts[7, 2] = i_t
		hola = np.where((TF == i_t)&(LOGG == i_l)& (av == i_amin))
		vals[7] = np.array(f1corr)[hola]

		#~ print(pts)
		#~ print(vals)
		
		#~ print(j)
		
		correction = interp3dbis(AV, loggg[j], 10**tefff[j], pts, vals)
		#~ print(correction)
		iarray[j] = correction
		
	return iarray
