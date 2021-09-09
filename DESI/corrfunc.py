from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt
import gc
########################################################################
# define cosmology and edges
#---------------
# ~cosmo = cosmology.Cosmology()
cosmo = cosmology.Planck15
# ~edges = np.logspace(-1, np.log10(150), 100)
edges = np.linspace(0.1, 200, 500)

# ~z = data_cat.compute(data_cat['Z'])
# ~print(np.min(z))
# ~print("columns = ", data_cat.columns) # default Weight,Selection also present
# ~print("columns = ", rdn_cat.columns) # default Weight,Selection also present
# ~print("total size = ", data_cat.csize)


# filter redshifts
#---------------
ZMIN = 0.8
ZMAX = 1.05

########################################################################
# read catalogs
#---------------

# initialize the catalogs
fits_loc = '/home/david/codes/data/DESI/data/'
gal_type = ['BGS_ANY', 'ELG', 'LRG','QSO']
reg = ['','0_', '1_', '2_']
file_type = ['clustering.dat.fits','_clustering.ran.fits']
const = 'Alltiles_'

fits_name_elg = fits_loc+gal_type[1]+const+reg[0]+file_type[0]
fits_name_lrg = fits_loc+gal_type[2]+const+reg[0]+file_type[0]
fits_name_qso = fits_loc+gal_type[3]+const+reg[0]+file_type[0]

data_cat_elg = FITSCatalog(fits_name_elg)
data_cat_lrg = FITSCatalog(fits_name_elg)
data_cat_qso = FITSCatalog(fits_name_qso)

# slice the data
valid_elg = (data_cat_elg['Z'] > ZMIN)&(data_cat_elg['Z'] < ZMAX)
valid_lrg = (data_cat_lrg['Z'] > ZMIN)&(data_cat_lrg['Z'] < ZMAX)
valid_qso = (data_cat_qso['Z'] > ZMIN)&(data_cat_qso['Z'] < ZMAX)
data_cat_elg = data_cat_elg[valid_elg]
data_cat_lrg = data_cat_lrg[valid_lrg]
data_cat_qso = data_cat_qso[valid_qso]

Ntiles = 2
# create array to save corr function
corrval_elg = np.zeros((len(edges)-1,Ntiles))
corrval_lrg = np.zeros((len(edges)-1,Ntiles))
corrval_qso = np.zeros((len(edges)-1,Ntiles))

corrval_elg_lrg = np.zeros((len(edges)-1,Ntiles))
corrval_elg_qso = np.zeros((len(edges)-1,Ntiles))
corrval_lrg_qso = np.zeros((len(edges)-1,Ntiles))


# loop over the tiles
for i in range(Ntiles):

	fits_name_elg = fits_loc+gal_type[1]+const+str(i)+file_type[1]
	fits_name_lrg = fits_loc+gal_type[2]+const+str(i)+file_type[1]
	fits_name_qso = fits_loc+gal_type[3]+const+str(i)+file_type[1]
	print(fits_name_elg)
	print(fits_name_lrg)
	print(fits_name_qso)


	rdn_cat_elg = FITSCatalog(fits_name_elg)
	rdn_cat_lrg = FITSCatalog(fits_name_lrg)
	rdn_cat_qso = FITSCatalog(fits_name_qso)
	
	# slice the randoms
	valid_elg = (rdn_cat_elg['Z'] > ZMIN)&(rdn_cat_elg['Z'] < ZMAX)
	valid_lrg = (rdn_cat_lrg['Z'] > ZMIN)&(rdn_cat_lrg['Z'] < ZMAX)
	valid_qso = (rdn_cat_qso['Z'] > ZMIN)&(rdn_cat_qso['Z'] < ZMAX)

	rdn_cat_elg = rdn_cat_elg[valid_elg]
	rdn_cat_lrg = rdn_cat_lrg[valid_lrg]
	rdn_cat_qso = rdn_cat_qso[valid_qso]


	# compute 2pcf
	#---------------
	#auto 2pcf
	calc_elg = SurveyData2PCF(mode='1d',data1=data_cat_elg, randoms1=rdn_cat_elg, edges=edges, cosmo=cosmo, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )
	calc_lrg = SurveyData2PCF(mode='1d',data1=data_cat_lrg, randoms1=rdn_cat_lrg, edges=edges, cosmo=cosmo, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )
	calc_qso = SurveyData2PCF(mode='1d',data1=data_cat_qso, randoms1=rdn_cat_qso, edges=edges, cosmo=cosmo, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )

	rval = calc_elg.corr['r']
	corrval_elg[:,i] = calc_elg.corr['corr']
	corrval_lrg[:,i] = calc_lrg.corr['corr']
	corrval_qso[:,i] = calc_qso.corr['corr']

	#cross 2pcf
	calc_elg_lrg = SurveyData2PCF(mode='1d',data1=data_cat_elg, randoms1=rdn_cat_elg, edges=edges, cosmo=cosmo, data2=data_cat_lrg, randoms2=rdn_cat_lrg, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )
	calc_elg_qso = SurveyData2PCF(mode='1d',data1=data_cat_elg, randoms1=rdn_cat_elg, edges=edges, cosmo=cosmo, data2=data_cat_qso, randoms2=rdn_cat_qso, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )
	calc_lrg_qso = SurveyData2PCF(mode='1d',data1=data_cat_lrg, randoms1=rdn_cat_lrg, edges=edges, cosmo=cosmo, data2=data_cat_qso, randoms2=rdn_cat_qso, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )

	corrval_elg_lrg[:,i] = calc_elg_lrg.corr['corr']
	corrval_elg_qso[:,i] = calc_elg_qso.corr['corr']
	corrval_lrg_qso[:,i] = calc_lrg_qso.corr['corr']



	gc.collect()

########################################################################
# plots
#---------------
# auto corr
plt.figure()
plt.plot(rval, np.mean(corrval_elg,axis=1)*rval**2, c='b', label=gal_type[1])
plt.plot(rval, np.mean(corrval_lrg,axis=1)*rval**2, c='r', label=gal_type[2])
plt.plot(rval, np.mean(corrval_qso,axis=1)*rval**2, c='k', label=gal_type[3])
plt.errorbar(rval, np.mean(corrval_elg,axis=1)*rval**2, yerr=np.std(corrval_elg,axis=1)*rval**2,fmt = '.', c='b', ecolor='b', alpha=0.5)
plt.errorbar(rval, np.mean(corrval_lrg,axis=1)*rval**2, yerr=np.std(corrval_lrg,axis=1)*rval**2,fmt = '.', c='r', ecolor='r', alpha=0.5)
plt.errorbar(rval, np.mean(corrval_qso,axis=1)*rval**2, yerr=np.std(corrval_qso,axis=1)*rval**2,fmt = '.', c='k', ecolor='k', alpha=0.5)
plt.xlabel(r"$r$ $[h^{-1} \mathrm{Mpc}]$")
plt.ylabel(r"$r^2 \xi \ [h^{-2} \mathrm{Mpc}^2]$")
plt.title('Redshift '+str(ZMIN)+' < z < '+str(ZMAX))
plt.legend(loc='best')
plt.show()
plt.close()

# cross corr
plt.figure()
plt.plot(rval, np.mean(corrval_elg_lrg,axis=1)*rval**2, c='b', label=gal_type[1]+' vs '+gal_type[2])
plt.plot(rval, np.mean(corrval_elg_qso,axis=1)*rval**2, c='r', label=gal_type[1]+' vs '+gal_type[3])
plt.plot(rval, np.mean(corrval_lrg_qso,axis=1)*rval**2, c='k', label=gal_type[2]+' vs '+gal_type[3])
plt.errorbar(rval, np.mean(corrval_elg_lrg,axis=1)*rval**2, yerr=np.std(corrval_elg_lrg,axis=1)*rval**2,fmt = '.', c='b', ecolor='b', alpha=0.5)
plt.errorbar(rval, np.mean(corrval_elg_qso,axis=1)*rval**2, yerr=np.std(corrval_elg_qso,axis=1)*rval**2,fmt = '.', c='r', ecolor='r', alpha=0.5)
plt.errorbar(rval, np.mean(corrval_lrg_qso,axis=1)*rval**2, yerr=np.std(corrval_lrg_qso,axis=1)*rval**2,fmt = '.', c='k', ecolor='k', alpha=0.5)
plt.xlabel(r"$r$ $[h^{-1} \mathrm{Mpc}]$")
plt.ylabel(r"$r^2 \xi \ [h^{-2} \mathrm{Mpc}^2]$")
plt.title('Redshift '+str(ZMIN)+' < z < '+str(ZMAX))
plt.legend(loc='best')
plt.show()
plt.close()




