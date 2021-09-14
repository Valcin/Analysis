from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt
import gc
from nbodykit import CurrentMPIComm
# ~comm = CurrentMPIComm.get()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ZMIN = None
ZMAX = None
cosmo = None
nbins = None
Ntiles = None
fits_loc = None
galtype = None
reg = None
file_type = None
const = None
fits_name = None
delta_r = None

# broadcast param and allocate array on other ranks:
if rank == 0:
########################################################################
	# define cosmology and edges
	#---------------
	# ~cosmo = cosmology.Cosmology()
	cosmo = cosmology.Planck15
	# ~edges = np.logspace(-1, np.log10(150), 100)
	rmin = 0.1
	rmax = 200
	nbins = 40
	delta_r = (rmax- rmin)/nbins
	edges = np.linspace(0.1, rmax, nbins)

	# filter redshifts
	#---------------
	ZMIN = 0.8
	ZMAX = 1.05

	#number of tiles
	Ntiles = 4

	# initialize the catalogs
	fits_loc = '/home/david/codes/data/DESI/data/'
	gal_type = ['BGS_ANY', 'ELG', 'LRG','QSO']
	reg = ['','0_', '1_', '2_']
	file_type = ['clustering.dat.fits','_clustering.ran.fits']
	const = 'Alltiles_'

	galtype = gal_type[3]

	fits_name = fits_loc+galtype+const+reg[0]+file_type[0]

	
ZMIN = comm.bcast(ZMIN, root=0)
ZMAX = comm.bcast(ZMAX, root=0)
cosmo = comm.bcast(cosmo, root=0)
nbins = comm.bcast(nbins, root=0)
Ntiles = comm.bcast(Ntiles, root=0)
fits_loc = comm.bcast(fits_loc, root=0)
galtype = comm.bcast(galtype, root=0)
reg = comm.bcast(reg, root=0)
file_type = comm.bcast(file_type, root=0)
const = comm.bcast(const, root=0)
fits_name = comm.bcast(fits_name, root=0)
delta_r = comm.bcast(delta_r, root=0)

if rank != 0:    
    edges = np.empty(nbins, dtype='d')

comm.Bcast(edges, root=0)

########################################################################
	
# create array to save corr function
# read catalogs
#---------------
data_cat = FITSCatalog(fits_name)

z = data_cat.compute(data_cat['Z'])
# ~print(np.min(z))
# ~print("columns = ", data_cat.columns) # default Weight,Selection also present
# ~print("columns = ", rdn_cat.columns) # default Weight,Selection also present
# ~print("total size = ", data_cat.csize)
# ~kill

# slice the data
valid = (data_cat['Z'] > ZMIN)&(data_cat['Z'] < ZMAX)
data_cat = data_cat[valid]

   
corrval = np.empty((len(edges)-1,Ntiles))
pair_rdn = np.empty((len(edges)-1,Ntiles))

# ~corrval_lrg_elg = np.empty((len(edges)-1,Ntiles))


print('loop starts')

# loop over the tiles
for i in range(Ntiles):

	fits_name = fits_loc+galtype+const+str(i)+file_type[1]
	print(fits_name)

	rdn_cat = FITSCatalog(fits_name)

	if rank == 0:
		print('catalogs loaded')
	
	# slice the randoms
	valid = (rdn_cat['Z'] > ZMIN)&(rdn_cat['Z'] < ZMAX)

	rdn_cat = rdn_cat[valid]
	
	if rank == 0:
		print('redshift sliced')

	gc.collect()

	# compute 2pcf
	#---------------
	#auto 2pcf
	cf = SurveyData2PCF(mode='1d',data1=data_cat, randoms1=rdn_cat, edges=edges, cosmo=cosmo, ra='RA', dec='DEC',redshift='Z',show_progress=True )
	if rank == 0:
		print('first 2pcf computed')

	rval = cf.corr['r']
	corrval[:,i] = cf.corr['corr']

	rval2 = cf.R1R2['r']
	pair_rdn[:,i] = cf.R1R2['npairs']


	#cross 2pcf
	# ~cf_lrg = SurveyData2PCF(mode='1d',data1=data_cat, randoms1=rdn_cat, edges=edges, cosmo=cosmo, data2=data_cat_lrg, randoms2=rdn_cat_lrg, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )
	# ~cf_qso = SurveyData2PCF(mode='1d',data1=data_cat, randoms1=rdn_cat, edges=edges, cosmo=cosmo, data2=data_cat_qso, randoms2=rdn_cat_qso, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )
	# ~cf_lrg_qso = SurveyData2PCF(mode='1d',data1=data_cat_lrg, randoms1=rdn_cat_lrg, edges=edges, cosmo=cosmo, data2=data_cat_qso, randoms2=rdn_cat_qso, ra='RA', dec='DEC',redshift='Z',weight='WEIGHT',show_progress=True )

	# ~corrval_lrg[:,i] = cf_lrg.corr['corr']
	# ~corrval_qso[:,i] = cf_qso.corr['corr']
	# ~corrval_lrg_qso[:,i] = cf_lrg_qso.corr['corr']

	
	# power spectrum
	# ~scat = FKPCatalog(data_cat, rdn_cat)
	# ~ps = ConvolvedFFTPower(scat, [0,2], use_fkp_weights=False, P0_FKP=None)
	# ~if rank == 0:
		# ~print('first ps computed')



	gc.collect()

########################################################################
# plots
#---------------
if galtype == 'ELG':
	col = 'b'
elif galtype == 'LRG':
	col = 'r'
elif galtype == 'QSO':
	col = 'g'

if rank == 0:
	# auto corr
	plt.figure()
	
	for j in range(Ntiles):
		if j == 0:
			plt.plot(rval, corrval[:,j]*rval**2, c=col, label=galtype, alpha=0.5)
		else:
			plt.plot(rval, corrval[:,j]*rval**2, c=col, alpha=0.5)


	# ~plt.plot(rval, np.mean(corrval,axis=1)*rval**2, c='b', label=gal_type[1])
	plt.errorbar(rval, np.mean(corrval,axis=1)*rval**2, yerr=np.std(corrval,axis=1)*rval**2,fmt = 'o', c='k', ecolor='k')
	plt.xlabel(r"$r$ $[h^{-1} \mathrm{Mpc}]$", fontsize = 16)
	plt.ylabel(r"$r^2 \xi \ [h^{-2} \mathrm{Mpc}^2]$", fontsize = 16)
	plt.title('Redshift '+str(ZMIN)+' < z < '+str(ZMAX), fontsize=16)
	plt.legend(loc='best', fontsize=16)
	plt.tick_params(labelsize=16)
	plt.xlim(0,200)
	plt.ylim(-150,150)
	plt.show()
	plt.close()

	

	




