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

# ~Zsun = 0.0142
# ~Zsun = 0.017
# ~Zsun = 0.0134

# ~Z = 1.72E-4
# ~X = 0.7545
# ~Z = np.array([0.000142, 0.00025, 0.00045])
# ~Y = 0.24 + 2*Z
# ~Y1 = 0.245 + 1.5*Z
# ~X = 1.0 - Y -Z
# ~X1 = 1.0 - Y1 -Z
# ~print(X,Y,Z)
# ~Fe_H = np.log10(Z/Zsun)
# ~Fe_H = -2.0
# ~Z = 10**(Fe_H)*Zsun
# ~XZ_GS98 = np.log10(0.02288)
# ~XZ_GS93 = np.log10(0.0245)
# ~fe_mesa = np.log10(Z/X) - XZ_GS98
# ~fe_dsed = np.log10(Z/X1) - XZ_GS98

# ~print(fe_mesa, fe_dsed,Fe_H, Z)

# ~kill

string_mass = 'M075'
version2 ='15'
model2 = 'dar'
omalley = np.loadtxt('catalogs/omalley.txt')
# ~ax2.errorbar(omalley[:,4], metal_fin_dar[(omalley[:,5]).astype(int)] , yerr =[metal_fin_dar[(omalley[:,5]).astype(int)] - metal_low_dar[(omalley[:,5]).astype(int)],
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

ind1 = np.loadtxt('ind_met15.txt')
ind2 = np.loadtxt('ind_met20.txt')
ind3 = np.loadtxt('ind_met175m.txt')
ind4 = np.loadtxt('ind_met175p.txt')

meta = np.loadtxt('metal_comp.txt')
harris = meta[:,0]
moi = meta[:,1]

metarr = np.linspace(-2.5,0,50)

h1 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_fid.data')
col1,mag1, mp1 = cut2(h1)
h2 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_m1sigma.data')
col2,mag2, mp2 = cut2(h2)
h3 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_p1sigma.data')
col3,mag3, mp3 = cut2(h3)
h4 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_a180.data')
col4,mag4, mp4 = cut2(h4)
h5 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_a200.data')
col5,mag5, mp5 = cut2(h5)
h6 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_a185.data')
col6,mag6, mp6 = cut2(h6)
h7 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_a195.data')
col7,mag7, mp7 = cut2(h7)
h8 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_a1875.data')
col8,mag8, mp8 = cut2(h8)
h9 = mr.MesaData('/home/david/codes/Analysis/GC_mixing_length/catalogs/'+string_mass+'/scatter_test/history_a1925.data')
col9,mag9, mp9 = cut2(h9)

########################################################################
########################################################################

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

met = '-1.5'
if met == '-1.5':
	ind = ind1

res_ind = np.zeros(len(ind))
for count,i in enumerate(ind):
	if i < 27:
		res_ind[count] = i
	# ~if i == 49:
		# ~res_ind[count] = 999
	else:
		res_ind[count] = i-1



non = np.where(res_ind == 48)[0]
res_ind = np.delete(res_ind, non)

x= np.linspace(-2.5, -1.2, 50)
y= np.linspace(-2.5, -1.2, 50)
sigma_harris_s1 = np.sqrt(np.sum((np.array(harris)[(res_ind).astype(int)] - moi[(res_ind).astype(int)])**2)/len(moi[(res_ind).astype(int)]))
print(sigma_harris_s1)

# ~m,b = np.polyfit(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)], 1, w = 1/(2*(metal_fin_dar[(res_ind).astype(int)] - metal_low_dar[(res_ind).astype(int)])))
# ~coef = np.polyfit(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)],1)
# ~poly1d_fn = np.poly1d(coef, ) 
# ~print(m,b)
# ~print(coef)


# ~m1, b1, r_value, p_value, std_err = stats.linregress(np.array(harris)[(res_ind).astype(int)],moi[(res_ind).astype(int)])
# ~print(m1, b1, r_value, p_value, std_err )

c1_m2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/c1_m2.txt', usecols=(1,2,3,4,5,6,7,8))
c2_m2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/c2_m2.txt', usecols=(1,2,3,4,5,6,7,8))
c1_p0 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/c1_p0.txt', usecols=(1,2,3,4,5,6,7,8))
c2_p0 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/c2_p0.txt', usecols=(1,2,3,4,5,6,7,8))
c1_p2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/c1_p2.txt', usecols=(1,2,3,4,5,6,7,8))
c2_p2 = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/c2_p2.txt', usecols=(1,2,3,4,5,6,7,8))

# ~coord = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord.txt', usecols=(1,2,3))
# ~metx = [0.0, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035]
# ~coord = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord_15.txt', usecols=(1,2,3))
# ~metx = [0.0, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025]
coord = np.loadtxt('/home/david/codes/Analysis/GC_mixing_length/catalogs/dcolor/coord_25.txt', usecols=(1,2,3))
metx = [0.0, 0.00005, 0.00010, 0.00015]

cm2,cp0,cp2 = coord[:,0], coord[:,1], coord[:,2]
diffcm2 = np.zeros(len(metx))
diffcp0 = np.zeros(len(metx))
diffcp2 = np.zeros(len(metx))
for i in range(len(cm2)):
	diffcm2[i] = cm2[i]-cm2[0]
	diffcp0[i] = cp0[i]-cp0[0]
	diffcp2[i] = cp2[i]-cp2[0]


m1, b1, r_value, p_value, std_err = stats.linregress(metx,diffcm2)
print(m1, b1, r_value, p_value, std_err )
m2, b2, r_value2, p_value2, std_err2 = stats.linregress(metx,diffcp0)
print(m2, b2, r_value2, p_value2, std_err2 )
m3, b3, r_value3, p_value3, std_err3 = stats.linregress(metx,diffcp2)
print(m3, b3, r_value3, p_value3, std_err3 )

plt.figure()
plt.plot(metx, diffcm2, label='F606W = -2.0')
plt.plot(metx, diffcp0, label='F606W = 0.0')
plt.plot(metx, diffcp2, label='F606W = 2.0')
plt.subplots_adjust(bottom = 0.13)
# ~plt.title(r'$\rm Z_{fid} = 0.00025$')
lgnd = plt.legend(loc='best', fontsize=12)
plt.xlabel(r'$\rm \Delta_{Z}$', fontsize=16)
plt.ylabel(r'$\rm \Delta_{color}$', fontsize=16)
plt.show()
plt.close()
kill

# ~with open('/home/david/fig2_data.txt', 'a+') as fid_file:
	# ~for i in range(len(np.array(harris)[(res_ind).astype(int)])):
		# ~fid_file.write('%.4f %.4f %.4f %.4f \n' %(np.array(harris)[(res_ind[i]).astype(int)], moi[(res_ind[i]).astype(int)],metal_fin_dar[(res_ind[i]).astype(int)] - metal_low_dar[(res_ind[i]).astype(int)], metal_high_dar[(res_ind[i]).astype(int)] - metal_fin_dar[(res_ind[i]).astype(int)]))
	# ~fid_file.close() 

########################################################################
########################################################################
x= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
mx= [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
string_name = ['Z = 0.00005', 'Z = 0.00010', 'Z = 0.00015', 'Z = 0.00020','Z = 0.00025', 'Z = 0.00030', 'Z = 0.00035', 'Z = 0.00040']
string_val = [0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040]
color=iter(cm.rainbow(np.linspace(0,1,8)))
plt.figure()
for i in range(len(c1_m2)):
	c=next(color)
	plt.plot(mx[:2], c1_p2[i,:2], color=c, label=string_name[i])
	plt.plot(x[:2], c2_p2[i,:2], color=c)
	plt.plot(mx[:2], c1_p0[i,:2], linestyle='--', color=c)
	plt.plot(x[:2], c2_p0[i,:2], linestyle='--', color=c)
	plt.plot(mx[:2], c1_m2[i,:2], linestyle=':', color=c)
	plt.plot(x[:2], c2_m2[i,:2], linestyle=':', color=c)
	# ~plt.plot(mx, c1_p2[i,:], color=c, label=string_name[i])
	# ~plt.plot(x, c2_p2[i,:], color=c)
	# ~plt.plot(mx, c1_p0[i,:], linestyle='--', color=c)
	# ~plt.plot(x, c2_p0[i,:], linestyle='--', color=c)
	# ~plt.plot(mx, c1_m2[i,:], linestyle=':', color=c)
	# ~plt.plot(x, c2_m2[i,:], linestyle=':', color=c)
plt.axvline(0.0, c='k', label=r'$\alpha = 2.0$')
plt.subplots_adjust(bottom = 0.13)
# ~lgnd = plt.legend(loc='best', fontsize=12)
plt.xlabel(r'$\rm \Delta_{\alpha}$', fontsize=16)
plt.ylabel(r'$\rm \Delta_{color} / \Delta_{\alpha}$', fontsize=16)
plt.show()
plt.close()

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

# ~plt.figure()
# ~plt.plot(col1 , mag1, label=r'$\rm [Fe/H] = -1.75, \alpha = 1.9$', c='k')
# ~plt.plot(col2 , mag2, label=r'$\rm \Delta_{[Fe/H]}$', c='k', linestyle='--')
# ~plt.plot(col3 , mag3, c='k', linestyle='--')
# ~plt.plot(col4 , mag4, label=r'$\rm \Delta_{\alpha}$ = 0.1', c='b')
# ~plt.plot(col5 , mag5, c='b')
# ~plt.plot(col6 , mag6, label=r'$\rm \Delta_{\alpha}$ = 0.05', c='r')
# ~plt.plot(col7 , mag7, c='r')
# ~plt.plot(col8 , mag8, label=r'$\rm \Delta_{\alpha}$ = 0.025', c='c')
# ~plt.plot(col9 , mag9, c='c')
# ~plt.gca().invert_yaxis()
# ~plt.legend(loc='best', fontsize=16)
# ~plt.tick_params(labelsize=16)
# ~plt.subplots_adjust(bottom=0.15, top=0.89)
# ~plt.xlabel(' F606W - F814W', fontsize = 20)
# ~plt.ylabel(' F606W', fontsize = 20)
# ~plt.show() 
# ~plt.close()
