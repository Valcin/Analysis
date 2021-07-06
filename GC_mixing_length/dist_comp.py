import numpy as np
import matplotlib.pyplot as plt


########################################################################
########################################################################

def cluster():
	
	clus_name2 = ['Arp_2','IC_4499','Lynga_7','NGC_104','NGC_288','NGC_362','NGC_1261','NGC_1851','NGC_2298','NGC_2808','NGC_3201','NGC_4147','NGC_4590','NGC_4833','NGC_5024','NGC_5053','NGC_5139','NGC_5272','NGC_5286','NGC_5466','NGC_5904','NGC_5927','NGC_5986','NGC_6093','NGC_6101','NGC_6121','NGC_6144','NGC_6205','NGC_6218','NGC_6254','NGC_6304','NGC_6341','NGC_6352','NGC_6362','NGC_6366','NGC_6388','NGC_6397','NGC_6426','NGC_6441','NGC_6496','NGC_6535','NGC_6541','NGC_6584','NGC_6624','NGC_6637','NGC_6652','NGC_6656','NGC_6681','NGC_6715','NGC_6717','NGC_6723','NGC_6752','NGC_6779','NGC_6809','NGC_6838','NGC_6934','NGC_6981','NGC_7006','NGC_7078','NGC_7089','NGC_7099','Pal_1','Pal_12','Pal_15','Pyxis','Rup_106','Ter_7','Ter_8']#removed 'NGC_6171'



	with open('Baumgardt.txt',"r") as f:
		lines=f.readlines()[2:]
	f.close()
	baum_clus=[]
	for x in lines:
		baum_clus.append(x.split(' ')[0])
	
	# find acs initial values in different caltalogs
	# ~index1 = harris_clus.index(clus_nb)

	distance = np.zeros(len(clus_name2))
	errdist = np.zeros(len(clus_name2))

	for nb in range(len(clus_name2)):
		clus_nb = clus_name2[nb]

		index2 = baum_clus.index(clus_nb)
		with open('Baumgardt.txt',"r") as f:
			lines=f.readlines()[2:]
		f.close()
		dist_baum=[]
		errdist_baum=[]
		for x in lines:
			dist_baum.append(x.split(' ')[5])
			errdist_baum.append(x.split(' ')[6])
		distance[nb] = dist_baum[index2]
		errdist[nb] = errdist_baum[index2]

	return distance, errdist


########################################################################
########################################################################
distbaum, distbaumerr = cluster()

nwalkers = 300	
version2 = '15' 
model2 = 'dar'

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
#-----------------------------------------------------------------------------
omalley = np.loadtxt('omalley.txt')


# ~distom = 10**(omalley[:,0]/5. + 1)
# ~distomerrp = 10**((omalley[:,0]+omalley[:,1])/5. + 1)
# ~distomerrm = 10**((omalley[:,0]-omalley[:,1])/5. + 1)
# ~errdistp = (distomerrp - distom)/1000.
# ~errdistm = (distom -distomerrm)/1000. 
# ~x = np.linspace(1,25,20)
# ~plt.errorbar(distom/1000., distance_fin_dar[(omalley[:,5]).astype(int)] /1000., yerr =[distance_fin_dar[(omalley[:,5]).astype(int)] /1000. - distance_low_dar[(omalley[:,5]).astype(int)] /1000.,
# ~distance_high_dar[(omalley[:,5]).astype(int)] /1000. - distance_fin_dar[(omalley[:,5]).astype(int)] /1000.], xerr = [errdistm, errdistp],color='r', fmt='o', ecolor='k', markersize=5, label='Distance/kpc')
# ~plt.plot(x,x, c='b', label='x = y')
# ~plt.xlabel(' Distance [kpc] (O\'Malley et al. 2017)', fontsize=12)
# ~plt.ylabel(r' Distance [kpc]' "\n" r' (This work)', fontsize=12)
# ~plt.tick_params(labelsize=14)
# ~plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/param_comparison.png', orientation='portrait', dpi=250)
# ~plt.subplots_adjust(hspace=1.8, wspace=1.0, top=0.95)
# ~plt.show()
# ~plt.close()


n= len(distbaum)
x = np.linspace(1,25,20)
plt.errorbar(distbaum[(omalley[:,5]).astype(int)], distance_fin_dar[(omalley[:,5]).astype(int)] /1000., yerr =[distance_fin_dar[(omalley[:,5]).astype(int)] /1000. - distance_low_dar[(omalley[:,5]).astype(int)] /1000.,
distance_high_dar[(omalley[:,5]).astype(int)] /1000. - distance_fin_dar[(omalley[:,5]).astype(int)] /1000.], xerr = distbaumerr[(omalley[:,5]).astype(int)],color='r', fmt='o', ecolor='k', markersize=5, label='Distance/kpc')
# ~x = np.linspace(1,50,20)
# ~plt.errorbar(distbaum, distance_fin_dar /1000., yerr =[distance_fin_dar /1000. - distance_low_dar /1000.,
# ~distance_high_dar /1000. - distance_fin_dar /1000.], xerr = distbaumerr,color='r', fmt='o', ecolor='k', markersize=5, label='Distance/kpc')
plt.plot(x,x, c='b', label='x = y')
plt.xlabel(' Distance [kpc] (Baumgardt et al. 2021)', fontsize=12)
plt.ylabel(r' Distance [kpc]' "\n" r' (Valcin et al. 2020)', fontsize=12)
plt.tick_params(labelsize=14)
plt.savefig('/home/david/codes/Analysis/GC/plots/analysis/param_comparison.png', orientation='portrait', dpi=250)
plt.subplots_adjust(hspace=1.8, wspace=1.0, top=0.95)
plt.show()
plt.close()
