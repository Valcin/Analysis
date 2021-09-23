from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt
import gc
from scipy.interpolate import InterpolatedUnivariateSpline
from nbodykit import CurrentMPIComm
# ~comm = CurrentMPIComm.get()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ZMIN = None
ZMAX = None
cosmo = None
nbins = None
Ncat = None
Nmul = None
fits_loc = None
galtype = None
reg = None
file_type = None
const = None
fits_name = None
delta_k = None
FSKY = None

# broadcast param and allocate array on other ranks:
if rank == 0:
########################################################################
	# define cosmology and nbins
	#---------------
	# ~cosmo = cosmology.Cosmology()
	cosmo = cosmology.Planck15
	# ~nbins = np.logspace(-1, np.log10(150), 100)
	kmin = 0
	kmax = 0.25
	delta_k = 0.005
	nbins = int((kmax - kmin)/delta_k)

	# filter redshifts
	#---------------
	ZMIN = 0.8
	ZMAX = 1.05

	#number of catalog
	Ncat = 4
	
	#number of multipoles
	Nmul = 3

	# initialize the catalogs
	fits_loc = '/home/david/codes/data/DESI/data/'
	gal_type = ['BGS_ANY', 'ELG', 'LRG','QSO']
	reg = ['','0_', '1_', '2_']
	file_type = ['clustering.dat.fits','_clustering.ran.fits']
	const = 'Alltiles_'

	galtype = gal_type[3]

	FSKY = 0.00339369576


	fits_name = fits_loc+galtype+const+reg[0]+file_type[0]

	
ZMIN = comm.bcast(ZMIN, root=0)
ZMAX = comm.bcast(ZMAX, root=0)
cosmo = comm.bcast(cosmo, root=0)
nbins = comm.bcast(nbins, root=0)
Ncat = comm.bcast(Ncat, root=0)
Nmul = comm.bcast(Nmul, root=0)
fits_loc = comm.bcast(fits_loc, root=0)
galtype = comm.bcast(galtype, root=0)
reg = comm.bcast(reg, root=0)
file_type = comm.bcast(file_type, root=0)
const = comm.bcast(const, root=0)
fits_name = comm.bcast(fits_name, root=0)
delta_k = comm.bcast(delta_k, root=0)
FSKY = comm.bcast(FSKY, root=0)


########################################################################
	
# create array to save corr function
# read catalogs
#---------------
data_cat = FITSCatalog(fits_name)
# add Cartesian position column
data_cat['Position'] = transform.SkyToCartesian(data_cat['RA'], data_cat['DEC'], data_cat['Z'], cosmo=cosmo)

# ~z = data_cat.compute(data_cat['Z'])
# ~print(np.min(z))
# ~print("columns = ", data_cat.columns) # default Weight,Selection also present
# ~print("columns = ", rdn_cat.columns) # default Weight,Selection also present
# ~print("total size = ", data_cat.csize)
# ~kill

# slice the data
valid = (data_cat['Z'] > ZMIN)&(data_cat['Z'] < ZMAX)
data_cat = data_cat[valid]

   
ps = np.empty((nbins,Nmul,Ncat)) #power spectrum
sn = np.empty((nbins,Ncat)) #shot noise



print('loop starts')

# loop over the catalogs
for i in range(Ncat):

	fits_name = fits_loc+galtype+const+str(i)+file_type[1]
	print(fits_name)

	rdn_cat = FITSCatalog(fits_name)
	rdn_cat['Position'] = transform.SkyToCartesian(rdn_cat['RA'], rdn_cat['DEC'], rdn_cat['Z'], cosmo=cosmo)

	if rank == 0:
		print('catalogs loaded')
	
	# slice the randoms
	valid = (rdn_cat['Z'] > ZMIN)&(rdn_cat['Z'] < ZMAX)
	rdn_cat = rdn_cat[valid]

	if rank == 0:
		print('redshift sliced')

	# compute n(z) from the randoms
	zhist = RedshiftHistogram(rdn_cat, FSKY, cosmo, redshift='Z')

	# re-normalize to the total size of the data catalog
	alpha = 1.0 * data_cat.csize / rdn_cat.csize
	
	# add n(z) from randoms to the FKP source
	nofz = InterpolatedUnivariateSpline(zhist.bin_centers, alpha*zhist.nbar)

	# plot
	if rank == 0:
		plt.plot(zhist.bin_centers, alpha*zhist.nbar, label=galtype)
		plt.xlabel(r"$z$", fontsize=16)
		plt.ylabel(r"$n(z)$ $[h^{3} \mathrm{Mpc}^{-3}]$", fontsize=16)
		plt.legend(loc='best', fontsize=16)
		plt.tick_params(labelsize=16)
		plt.show()
		plt.close()
		# ~kill

	# add the n(z) columns to the the Catalogs
	rdn_cat['NZ'] = nofz(rdn_cat['Z'])
	data_cat['NZ'] = nofz(data_cat['Z'])

	# initialize the FKP source
	fkp = FKPCatalog(data_cat, rdn_cat)

	# add fkp weights
	fkp['data/FKPWeight'] = 1.0 / (1 + fkp['data/NZ'] * 1e4)
	fkp['randoms/FKPWeight'] = 1.0 / (1 + fkp['randoms/NZ'] * 1e4)

	if rank == 0:
		print('weight and density added')
	gc.collect()

	mesh1 = fkp.to_mesh(Nmesh=256, nbar='NZ', comp_weight='WEIGHT', fkp_weight='FKPWeight',window='cic')
	# ~mesh2 = fkp.to_mesh(Nmesh=256, nbar='NZ', comp_weight='WEIGHT', fkp_weight='FKPWeight',window='tsc')


	# compute 2pcf
	#---------------
	#auto 2pcf
	r = ConvolvedFFTPower(mesh1, poles=[0,2,4], dk=0.005, kmin=0.)

	k = r.poles['k']
	sn[:,i] = r.attrs['shotnoise']

	mltpls = [0, 2, 4]
	for count,ell in enumerate(mltpls):
		if ell == 0:
			ps[:,count,i] = r.poles['power_%d' %ell].real - r.attrs['shotnoise']
		else:
			ps[:,count,i] = r.poles['power_%d' %ell].real

	
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
	for j in range(Ncat):
		if j == 0:
			plt.plot(k, k*ps[:,0,j], c='b', label=r'$\ell=0$', alpha=0.5)
			plt.plot(k, k*ps[:,1,j], c='r', label=r'$\ell=2$', alpha=0.5)
			plt.plot(k, k*ps[:,2,j], c='g', label=r'$\ell=4$', alpha=0.5)
		else:
			plt.plot(k, k*ps[:,0,j], c='b', alpha=0.5)
			plt.plot(k, k*ps[:,1,j], c='r', alpha=0.5)
			plt.plot(k, k*ps[:,2,j], c='g', alpha=0.5)

	# format the axes
	plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]", fontsize=16)
	plt.ylabel(r"$k \ P_\ell$ [$h^{-2} \ \mathrm{Mpc}^2$]", fontsize=16)
	plt.xlim(0.01, 0.25)
	# ~plt.plot(rval, np.mean(corrval,axis=1)*rval**2, c='b', label=gal_type[1])
	plt.title(galtype+', '+'Redshift '+str(ZMIN)+' < z < '+str(ZMAX), fontsize=16)
	plt.legend(loc='best', fontsize=16)
	plt.tick_params(labelsize=16)
	plt.show()
	plt.close()

	

	




