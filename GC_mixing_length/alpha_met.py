import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import stats
# ~import linfit
from sklearn.linear_model import LinearRegression
import mesa_reader as mr
########################################################################
########################################################################

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
########################################################################
########################################################################
#-----------------------------------------------------------------------
### compute z - Fe/H conversion ####
#-----------------------------------------------------------------------

# ~Zsun = 0.0142

# ~Z = 0.000142
# ~Z = np.array([0.000142, 0.00025251567, 0.00045])
# ~Y = 0.24 + 2*Z
# ~# Y = 0.2305
# ~Y1 = 0.245 + 1.5*Z
# ~Y2 = 0.249 + 1.5*Z
# ~X = 1.0 - Y -Z
# ~X1 = 1.0 - Y1 -Z
# ~X2 = 1.0 - Y2 -Z

# ~print(X1,Y1)
# ~Fe_H = np.log10(Z/Zsun)
# ~XZ_GS98 = np.log10(0.02288)
# ~XZ_ASO9 = np.log10(0.0178)
# ~fe_mesa = np.log10(Z/X) - XZ_ASO9
# ~fe_dsed = np.log10(Z/X1) - XZ_GS98
# ~fe_mist = np.log10(Z/X2) - XZ_ASO9

# ~print(fe_mesa, fe_dsed, fe_mist, Fe_H, Z)
# ~print(fe_mesa, Fe_H, Z)

# ~kill


########################################################################
########################################################################
#-----------------------------------------------------------------------
### plot metallicity scatter ####
#-----------------------------------------------------------------------

# ~string_mass = 'M075'
# ~version2 ='15'
# ~model2 = 'dar'
# ~omalley = np.loadtxt('catalogs/omalley.txt')
# ~ax2.errorbar(omalley[:,4], metal_fin_dar[(omalley[:,5]).astype(int)] , yerr =[metal_fin_dar[(omalley[:,5]).astype(int)] - metal_low_dar[(omalley[:,5]).astype(int)],
#DAR 12
# ~Age_mean_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(2,))
# ~Age_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(3,))
# ~Age_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(1,))
# ~metal_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(5,))
# ~metal_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(4,))
# ~metal_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(6,))
# ~distance_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(7,))
# ~distance_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(8,))
# ~distance_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(9,))
# ~AAbs_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(10,))
# ~AAbs_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(11,))
# ~AAbs_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(12,))
# ~Afe_low_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(13,))
# ~Afe_fin_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(14,))
# ~Afe_high_dar = np.loadtxt('/home/david/codes/Analysis/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(15,))
# ~elem_fin_dar = np.arange(len(Age_mean_dar))

# ~ind1 = np.loadtxt('ind_met15.txt')
# ~ind2 = np.loadtxt('ind_met20.txt')
# ~ind3 = np.loadtxt('ind_met175m.txt')
# ~ind4 = np.loadtxt('ind_met175p.txt')

# ~meta = np.loadtxt('metal_comp.txt')
# ~harris = meta[:,0]
# ~moi = meta[:,1]

# ~metarr = np.linspace(-2.5,0,50)



# ~plt.figure()
# ~plt.plot(metarr, metarr)
# ~plt.scatter(harris, moi)
# ~plt.xlim(-2.5,0)
# ~plt.ylim(-2.5,0)
# ~plt.show()

# ~plt.figure()
# ~plt.plot(metarr, metarr)
# ~plt.scatter(omalley[:,4], moi[(omalley[:,5]).astype(int)] )
# ~plt.xlim(-2.5,0)
# ~plt.ylim(-2.5,0)
# ~plt.show()

# ~sigma_harris = np.sqrt(np.sum((harris - moi)**2)/len(moi))
# ~sigma_omalley = np.sqrt(np.sum((omalley[:,4] - moi[(omalley[:,5]).astype(int)] )**2)/len(moi[(omalley[:,5]).astype(int)]))

# ~print(sigma_harris, len(moi))
# ~print(sigma_omalley, len(moi[(omalley[:,5]).astype(int)]))

# ~met = '-1.75m'
# ~if met == '-1.5':
	# ~ind = ind1
# ~elif met == '-2.0':
	# ~ind = ind2
# ~elif met == '-1.75p':
	# ~ind = ind4

# ~if met == '-1.75m':
	# ~ind = ind3


# ~res_ind = np.zeros(len(ind))
# ~for count,i in enumerate(ind):
	# ~if i < 27:
		# ~res_ind[count] = i
	# if i == 49:
		# ~res_ind[count] = 999
	# ~else:
		# ~res_ind[count] = i-1



# ~non = np.where(res_ind == 48)[0]
# ~res_ind = np.delete(res_ind, non)

# ~x= np.linspace(-2.5, -1.2, 50)
# ~y= np.linspace(-2.5, -1.2, 50)
# ~sigma_harris_s1 = np.sqrt(np.sum((np.array(harris)[(res_ind).astype(int)] - moi[(res_ind).astype(int)])**2)/len(moi[(res_ind).astype(int)]))
# ~print(met)
# ~print(sigma_harris_s1)
# ~kill

# ~m,b = np.polyfit(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)], 1, w = 1/(2*(metal_fin_dar[(res_ind).astype(int)] - metal_low_dar[(res_ind).astype(int)])))
# ~coef = np.polyfit(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)],1)
# ~poly1d_fn = np.poly1d(coef, ) 
# ~print(m,b)
# ~print(coef)


# ~m1, b1, r_value, p_value, std_err = stats.linregress(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)])
# ~print(m1, b1, r_value, p_value, std_err )

# ~with open('/home/david/fig2_data.txt', 'a+') as fid_file:
	# ~for i in range(len(np.array(harris)[(res_ind).astype(int)])):
		# ~fid_file.write('%.4f %.4f %.4f %.4f \n' %(np.array(harris)[(res_ind[i]).astype(int)], moi[(res_ind[i]).astype(int)],metal_fin_dar[(res_ind[i]).astype(int)] - metal_low_dar[(res_ind[i]).astype(int)], metal_high_dar[(res_ind[i]).astype(int)] - metal_fin_dar[(res_ind[i]).astype(int)]))
	# ~fid_file.close()

# ~plt.figure()
# ~plt.plot(metarr, metarr, c='b', label='x = y')
# ~plt.scatter(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)], label='metallicity best fits')
# ~# plt.plot(x,poly1d_fn(x), label='linear regression', c='r')
# ~plt.errorbar(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)], yerr =[metal_fin_dar[(res_ind).astype(int)] - metal_low_dar[(res_ind).astype(int)],
# ~metal_high_dar[(res_ind).astype(int)] - metal_fin_dar[(res_ind).astype(int)]], color='r', fmt='o', ecolor='k', markersize=5, label=r' Metallicity best fit ')
# ~plt.xlabel('[Fe/H] (Harris catalog. 2010)', fontsize=16)
# ~plt.ylabel('[Fe/H] (Valcin et al.)', fontsize=16)
# ~plt.legend(loc='best', fontsize=16)
# ~plt.subplots_adjust(bottom=0.135, left=0.12)
# ~plt.tick_params(labelsize=14)
# ~plt.xlim(-2.5,-1.2)
# ~plt.ylim(-2.5,-1.2)
# ~plt.show()

########################################################################
########################################################################
#-----------------------------------------------------------------------
### compute dcolor/dz ####
#-----------------------------------------------------------------------
# ~metx = [0.0, 0.00005, 0.00010]
# ~cm2 = np.zeros(8)
# ~cp0 = np.zeros(8)
# ~cp2 = np.zeros(8)
# ~cm2a16 = np.zeros(8)
# ~cp0a16 = np.zeros(8)
# ~cp2a16 = np.zeros(8)
# ~cm2[0] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00005_coord_cm2.txt'))
# ~cm2[1] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_coord_cm2.txt'))
# ~cm2[2] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_coord_cm2.txt'))
# ~cm2[3] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00020_coord_cm2.txt'))
# ~cm2[4] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_coord_cm2.txt'))
# ~cm2[5] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00030_coord_cm2.txt'))
# ~cm2[6] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00035_coord_cm2.txt'))
# ~cm2[7] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_coord_cm2.txt'))
# ~cp0[0] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00005_coord_cp0.txt'))
# ~cp0[1] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_coord_cp0.txt'))
# ~cp0[2] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_coord_cp0.txt'))
# ~cp0[3] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00020_coord_cp0.txt'))
# ~cp0[4] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_coord_cp0.txt'))
# ~cp0[5] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00030_coord_cp0.txt'))
# ~cp0[6] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00035_coord_cp0.txt'))
# ~cp0[7] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_coord_cp0.txt'))
# ~cp2[0] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00005_coord_cp2.txt'))
# ~cp2[1] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_coord_cp2.txt'))
# ~cp2[2] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_coord_cp2.txt'))
# ~cp2[3] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00020_coord_cp2.txt'))
# ~cp2[4] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_coord_cp2.txt'))
# ~cp2[5] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00030_coord_cp2.txt'))
# ~cp2[6] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00035_coord_cp2.txt'))
# ~cp2[7] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_coord_cp2.txt'))
# ~#---------------------------------------------------------------------------------------------------------------
# ~cm2a16[0] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00005_coord_cm2_16.txt'))
# ~cm2a16[1] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_coord_cm2_16.txt'))
# ~cm2a16[2] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_coord_cm2_16.txt'))
# ~cm2a16[3] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00020_coord_cm2_16.txt'))
# ~cm2a16[4] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_coord_cm2_16.txt'))
# ~cm2a16[5] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00030_coord_cm2_16.txt'))
# ~cm2a16[6] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00035_coord_cm2_16.txt'))
# ~cm2a16[7] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_coord_cm2_16.txt'))
# ~cp0a16[0] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00005_coord_cp0_16.txt'))
# ~cp0a16[1] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_coord_cp0_16.txt'))
# ~cp0a16[2] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_coord_cp0_16.txt'))
# ~cp0a16[3] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00020_coord_cp0_16.txt'))
# ~cp0a16[4] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_coord_cp0_16.txt'))
# ~cp0a16[5] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00030_coord_cp0_16.txt'))
# ~cp0a16[6] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00035_coord_cp0_16.txt'))
# ~cp0a16[7] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_coord_cp0_16.txt'))
# ~cp2a16[0] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00005_coord_cp2_16.txt'))
# ~cp2a16[1] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_coord_cp2_16.txt'))
# ~cp2a16[2] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_coord_cp2_16.txt'))
# ~cp2a16[3] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00020_coord_cp2_16.txt'))
# ~cp2a16[4] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_coord_cp2_16.txt'))
# ~cp2a16[5] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00030_coord_cp2_16.txt'))
# ~cp2a16[6] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00035_coord_cp2_16.txt'))
# ~cp2a16[7] = np.mean(np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_coord_cp2_16.txt'))
# ~# metx = [0.0, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035]
# ~coord20 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord_15.txt', usecols=(1,2,3))
# ~coord16 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord_15_a16.txt', usecols=(1,2,3))
# ~coord24 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord_15_a24.txt', usecols=(1,2,3))

# cm2,cp0,cp2 = coord[:,0], coord[:,1], coord[:,2]
# ~diffcm2 = np.zeros(len(metx))
# ~diffcp0 = np.zeros(len(metx))
# ~diffcp2 = np.zeros(len(metx))
# ~diffcm2a16 = np.zeros(len(metx))
# ~diffcp0a16 = np.zeros(len(metx))
# ~diffcp2a16 = np.zeros(len(metx))


# ~zc = 4
# ~for i in range(len(metx)):
	# ~print(i)
	# ~diffcm2[i] = cm2[i+zc]-cm2[zc-i]
	# ~diffcp0[i] = cp0[i+zc]-cp0[zc-i]
	# ~diffcp2[i] = cp2[i+zc]-cp2[zc-i]

# ~for i in range(len(metx)):
	# ~diffcm2a16[i] = cm2a16[i+zc]-cm2a16[zc-i]
	# ~diffcp0a16[i] = cp0a16[i+zc]-cp0a16[zc-i]
	# ~diffcp2a16[i] = cp2a16[i+zc]-cp2a16[zc-i]

# ~m1, b1, r_value, p_value, std_err = stats.linregress(metx,diffcm2)
# ~print(m1, b1, r_value, p_value, std_err )
# ~m1bis, b1bis, r_valuebis, p_valuebis, std_errbis = stats.linregress(metx,diffcm2a16)
# ~print(m1bis, b1bis, r_valuebis, p_valuebis, std_errbis)
# ~m2, b2, r_value2, p_value2, std_err2 = stats.linregress(metx,diffcp0)
# ~print(m2, b2, r_value2, p_value2, std_err2 )
# ~m3, b3, r_value3, p_value3, std_err3 = stats.linregress(metx,diffcp2)
# ~print(m3, b3, r_value3, p_value3, std_err3 )

# ~plt.figure()
# ~plt.plot(metx, diffcm2, label=r'$\rm \alpha = 2.0$', c='r')
# ~plt.plot(metx, diffcm2, label='F606W = -2.0', c='r')
# ~plt.plot(metx, diffcp0, label='F606W = 0.0', c='b')
# ~plt.plot(metx, diffcp2, label='F606W = 2.0', c='g')
# ~plt.plot(metx, diffcm2a16, linestyle='--', c='r', label=r'$\rm \alpha = 1.6$')
# ~plt.plot(metx, diffcp0a16, linestyle='--', c='b')
# ~plt.plot(metx, diffcp2a16, linestyle='--', c='g')
# ~# plt.plot(metx, diffcm2a24, linestyle=':', c='r', label=r'$\rm \alpha = 2.4$')
# ~# plt.plot(metx, diffcp0a24, linestyle=':', c='b')
# ~# plt.plot(metx, diffcp2a24, linestyle=':', c='g')
# ~plt.subplots_adjust(bottom = 0.13)
# plt.title(r'$\rm Z_{fid} = 0.00015$')
# ~lgnd = plt.legend(loc='best', fontsize=12)
# ~plt.xlabel(r'$\rm \Delta_{Z}$', fontsize=16)
# ~plt.ylabel(r'$\rm \Delta_{color}$', fontsize=16)
# ~plt.show()
# ~plt.close()
# ~kill

########################################################################
########################################################################
#-----------------------------------------------------------------------
### compute dcolor/dalpha ####
#-----------------------------------------------------------------------
x= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mx= [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
string_name = ['Z = 0.00005', 'Z = 0.00010', 'Z = 0.00015', 'Z = 0.00020','Z = 0.00025', 'Z = 0.00030', 'Z = 0.00035', 'Z = 0.00040']
string_val = [0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040]


# ~dc_da10 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00010_dcda2.txt')
# ~dc_da15 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_dcda2.txt')
# ~dc_da25 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_dcda2.txt')
# ~dc_da40 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_dcda2.txt')
# ~dc_da15 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00015_dcda.txt')
# ~dc_da25 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00025_dcda.txt')
# ~dc_da40 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/Z00040_dcda.txt')


# ~print('z = 0.00015 ----------------')
# ~m1, b1, r_value, p_value, std_err = stats.linregress(x[:2],dc_da15[2, :2])
# ~print(m1, b1, r_value, p_value, std_err )
# ~m2, b2, r_value2, p_value2, std_err2 = stats.linregress(x[:2],dc_da15[5, :2])
# ~print(m2, b2, r_value2, p_value2, std_err2 )
# ~m3, b3, r_value3, p_value3, std_err3 = stats.linregress(x[:2],dc_da15[8, :2])
# ~print(m3, b3, r_value3, p_value3, std_err3 )
# ~print('z = 0.00025 ----------------')
# ~m1, b1, r_value, p_value, std_err = stats.linregress(x[:2],dc_da25[2, :2])
# ~print(m1, b1, r_value, p_value, std_err )
# ~m2, b2, r_value2, p_value2, std_err2 = stats.linregress(x[:2],dc_da25[5, :2])
# ~print(m2, b2, r_value2, p_value2, std_err2 )
# ~m3, b3, r_value3, p_value3, std_err3 = stats.linregress(x[:2],dc_da25[8, :2])
# ~print(m3, b3, r_value3, p_value3, std_err3 )
# ~print('z = 0.00040 ----------------')
# ~m1, b1, r_value, p_value, std_err = stats.linregress(x[:2],dc_da40[2, :2])
# ~print(m1, b1, r_value, p_value, std_err )
# ~m2, b2, r_value2, p_value2, std_err2 = stats.linregress(x[:2],dc_da40[5, :2])
# ~print(m2, b2, r_value2, p_value2, std_err2 )
# ~m3, b3, r_value3, p_value3, std_err3 = stats.linregress(x[:2],dc_da40[8, :2])
# ~print(m3, b3, r_value3, p_value3, std_err3 )
# ~kill

# ~plt.figure()
# ~plt.plot(x[:2],dc_da15[2, :2], linestyle=':', c='b')
# ~plt.plot(x[:2],dc_da15[5, :2], linestyle='--', c='b')
# ~plt.plot(x[:2],dc_da15[8, :2], c='b', label='Z = 0.00015')
# ~plt.plot(x[:2],dc_da25[2, :2], linestyle=':', c='r')
# ~plt.plot(x[:2],dc_da25[5, :2], linestyle='--', c='r')
# ~plt.plot(x[:2],dc_da25[8, :2], c='r', label='Z = 0.00025')
# ~plt.plot(x[:2],dc_da40[2, :2], linestyle=':', c='g')
# ~plt.plot(x[:2],dc_da40[5, :2], linestyle='--', c='g')
# ~plt.plot(x[:2],dc_da40[8, :2], c='g', label='Z = 0.00040')
# ~plt.plot(x,dc_da15[2, :], linestyle=':', c='b')
# ~plt.plot(x,dc_da15[5, :], linestyle='--', c='b')
# ~plt.plot(x,dc_da15[8, :], c='b', label='Z = 0.00015')
# ~plt.plot(x,dc_da25[2, :], linestyle=':', c='r')
# ~plt.plot(x,dc_da25[5, :], linestyle='--', c='r')
# ~plt.plot(x,dc_da25[8, :], c='r', label='Z = 0.00025')
# ~plt.plot(x,dc_da40[2, :], linestyle=':', c='g')
# ~plt.plot(x,dc_da40[5, :], linestyle='--', c='g')
# ~plt.plot(x,dc_da40[8, :], c='g', label='Z = 0.00040')
# ~plt.legend(loc='best', fontsize=14)
# ~plt.subplots_adjust(bottom = 0.13)
# ~plt.xlabel(r'$\rm \Delta_{\alpha}$', fontsize=20)
# ~plt.ylabel(r'$\rm \Delta_{color} / \Delta_{\alpha}$', fontsize=20)
# ~plt.show()
# ~plt.close()
# ~kill

# ~color=iter(cm.rainbow(np.linspace(0,1,8)))
# ~plt.figure()
# ~for i in range(len(c1_m2)):
	# ~c=next(color)
	# ~plt.plot(mx[:2], c1_p2[i,:2], color=c, label=string_name[i])
	# ~plt.plot(x[:2], c2_p2[i,:2], color=c)
	# ~plt.plot(mx[:2], c1_p0[i,:2], linestyle='--', color=c)
	# ~plt.plot(x[:2], c2_p0[i,:2], linestyle='--', color=c)
	# ~plt.plot(mx[:2], c1_m2[i,:2], linestyle=':', color=c)
	# ~plt.plot(x[:2], c2_m2[i,:2], linestyle=':', color=c)

# ~print(c1_p2)
# ~print(np.mean(c1_p2[:,0]))
# ~print(np.mean(c1_p2, axis=0))
# ~print(c2_p2)
# ~print(np.mean(c2_p2[:,0]))
# ~print(np.mean(c2_p2, axis=0))
# ~diff_p2l = np.mean(c1_p2, axis=0)/mx
# ~diff_p2r = np.mean(c2_p2, axis=0)/x
# ~print(diff_p2l, diff_p2r)
# ~kill
########################################################################
########################################################################
#-----------------------------------------------------------------------
### compute dcolor/dalpha ####
#-----------------------------------------------------------------------

# ~string_met = 'Z00015'
# ~c1_m2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c1_m2.txt', usecols=(1,2,3,4,5,6,7,8))
# ~c2_m2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c2_m2.txt', usecols=(1,2,3,4,5,6,7,8))
# ~c1_p0 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c1_p0.txt', usecols=(1,2,3,4,5,6,7,8))
# ~c2_p0 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c2_p0.txt', usecols=(1,2,3,4,5,6,7,8))
# ~c1_p2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c1_p2.txt', usecols=(1,2,3,4,5,6,7,8))
# ~c2_p2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_c2_p2.txt', usecols=(1,2,3,4,5,6,7,8))

# ~diff_p2l = np.mean(c1_p2, axis=0)/mx
# ~diff_p2r = np.mean(c2_p2, axis=0)/x
# ~diff_p2t = (np.mean(c1_p2, axis=0) + np.mean(c2_p2, axis=0))/(np.multiply(x,2))
# ~diff_p0l = np.mean(c1_p0, axis=0)/mx
# ~diff_p0r = np.mean(c2_p0, axis=0)/x
# ~diff_p0t = (np.mean(c1_p0, axis=0) + np.mean(c2_p0, axis=0))/(np.multiply(x,2))
# ~diff_m2l = np.mean(c1_m2, axis=0)/mx
# ~diff_m2r = np.mean(c2_m2, axis=0)/x
# ~diff_m2t = (np.mean(c1_m2, axis=0) + np.mean(c2_m2, axis=0))/(np.multiply(x,2))


# ~with open('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/'+string_met+'_dcda2.txt', 'a+') as fid_file:
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_p2l[0],diff_p2l[1],diff_p2l[2],diff_p2l[3],diff_p2l[4],diff_p2l[5],diff_p2l[6],diff_p2l[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_p2r[0],diff_p2r[1],diff_p2r[2],diff_p2r[3],diff_p2r[4],diff_p2r[5],diff_p2r[6],diff_p2r[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_p2t[0],diff_p2t[1],diff_p2t[2],diff_p2t[3],diff_p2t[4],diff_p2t[5],diff_p2t[6],diff_p2t[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_p0l[0],diff_p0l[1],diff_p0l[2],diff_p0l[3],diff_p0l[4],diff_p0l[5],diff_p0l[6],diff_p0l[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_p0r[0],diff_p0r[1],diff_p0r[2],diff_p0r[3],diff_p0r[4],diff_p0r[5],diff_p0r[6],diff_p0r[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_p0t[0],diff_p0t[1],diff_p0t[2],diff_p0t[3],diff_p0t[4],diff_p0t[5],diff_p0t[6],diff_p0t[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_m2l[0],diff_m2l[1],diff_m2l[2],diff_m2l[3],diff_m2l[4],diff_m2l[5],diff_m2l[6],diff_m2l[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_m2r[0],diff_m2r[1],diff_m2r[2],diff_m2r[3],diff_m2r[4],diff_m2r[5],diff_m2r[6],diff_m2r[7]))
	# ~fid_file.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n' %(diff_m2t[0],diff_m2t[1],diff_m2t[2],diff_m2t[3],diff_m2t[4],diff_m2t[5],diff_m2t[6],diff_m2t[7]))
# ~fid_file.close()
# ~kill

# ~c='r'
# ~plt.plot(mx, np.mean(c1_p2, axis=0), color=c, label=string_met)
# ~plt.plot(x, np.mean(c2_p2, axis=0), color=c)
# ~plt.plot(mx, np.mean(c1_p0, axis=0), linestyle='--', color=c)
# ~plt.plot(x, np.mean(c2_p0, axis=0), linestyle='--', color=c)
# ~plt.plot(mx, np.mean(c1_m2, axis=0), linestyle=':', color=c)
# ~plt.plot(x, np.mean(c2_m2, axis=0), linestyle=':', color=c)
# ~plt.axvline(0.0, c='k', label=r'$\alpha = 2.0$')
# ~plt.subplots_adjust(bottom = 0.13)
# ~plt.title(string_met)
# ~lgnd = plt.legend(loc='best', fontsize=12)
# ~plt.xlabel(r'$\rm \Delta_{\alpha}$', fontsize=16)
# ~plt.ylabel(r'$\rm \Delta_{color}$', fontsize=16)
# ~plt.show()
# ~plt.close()
# ~kill

########################################################################
########################################################################
string_mass = 'M075'
string_met = 'Z00025'
h1 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_fid.data')
col1,mag1, mp1 = cut2(h1)
h2 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_m1sigma.data')
col2,mag2, mp2 = cut2(h2)
h3 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_p1sigma.data')
col3,mag3, mp3 = cut2(h3)


plt.figure()
if string_met == 'Z00015':
	h4 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a198.data')
	col4,mag4, mp4 = cut2(h4)
	h5 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a202.data')
	col5,mag5, mp5 = cut2(h5)
	h6 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a197.data')
	col6,mag6, mp6 = cut2(h6)
	h7 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a203.data')
	col7,mag7, mp7 = cut2(h7)

	plt.axhline(0.18, linestyle ='--', label=r'$\rm Magnitude \: cut \: \mathcal{M}_1$', c='g')
	plt.axhline(-2.0, c='grey')
	plt.plot(col3 , mag3, c='k', linestyle='--')
	plt.plot(col4 , mag4, label=r'$\rm \Delta_{\alpha}$ = 0.02', c='c')
	plt.plot(col5 , mag5, c='c')
	plt.plot(col6 , mag6, label=r'$\rm \Delta_{\alpha}$ = 0.03', c='r')
	plt.plot(col7 , mag7, c='r')
	plt.plot(col1 , mag1, label=r'$\rm [Fe/H] = -2.0$', c='k')
	plt.plot(col2 , mag2, label=r'$\rm \Delta_{[Fe/H]} = 0.09$', c='k', linestyle='--')
	plt.title('Z = 0.00015')

	
if string_met == 'Z00025':
	h4 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a197.data')
	col4,mag4, mp4 = cut2(h4)
	h5 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a203.data')
	col5,mag5, mp5 = cut2(h5)
	h6 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a196.data')
	col6,mag6, mp6 = cut2(h6)
	h7 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a204.data')
	col7,mag7, mp7 = cut2(h7)
	h8 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a195.data')
	col8,mag8, mp8 = cut2(h8)
	h9 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/'+string_met+'/history_a205.data')
	col9,mag9, mp9 = cut2(h9)

	plt.axhline(0.18, linestyle ='--', label=r'$\rm Magnitude \: cut \: \mathcal{M}_1$', c='g')
	plt.axhline(-2.0, c='grey')
	plt.plot(col3 , mag3, c='k', linestyle='--')
	plt.plot(col4 , mag4, label=r'$\rm \Delta_{\alpha}$ = 0.03', c='b')
	plt.plot(col5 , mag5, c='b')
	plt.plot(col6 , mag6, label=r'$\rm \Delta_{\alpha}$ = 0.04', c='r')
	plt.plot(col7 , mag7, c='r')
	plt.plot(col8 , mag8, label=r'$\rm \Delta_{\alpha}$ = 0.05', c='c')
	plt.plot(col9 , mag9, c='c')
	plt.plot(col1 , mag1, label=r'$\rm [Fe/H] = -1.75$', c='k')
	plt.plot(col2 , mag2, label=r'$\rm \Delta_{[Fe/H]} = 0.09$', c='k', linestyle='--')
	plt.title('Z = 0.00025')

plt.xlim(0.82, 0.9)
plt.ylim(-2.1, -1.9)
# ~plt.xlim(left=0.65)
# ~plt.ylim(top=0.5)
plt.gca().invert_yaxis()
plt.legend(loc='best', fontsize=13)
plt.tick_params(labelsize=16)
plt.subplots_adjust(bottom=0.15, top=0.89)
plt.xlabel(' F606W - F814W', fontsize = 20)
plt.ylabel(' F606W', fontsize = 20)
plt.show() 
plt.close()
