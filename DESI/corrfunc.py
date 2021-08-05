from nbodykit.lab import *
import numpy as np
import matplotlib.pyplot as plt

########################################################################
# read catalogs
#---------------

# initialize the catalog
fits_loc = '/home/david/codes/Analysis/DESI/data/'
fits_name = fits_loc + 'LRGAlltiles_N_clustering.dat.fits'
# ~fits_name = fits_loc + 'ELGnotqsoAlltiles_0_clustering.ran.fits'

f = FITSCatalog(fits_name)
ra = f['RA']
print(f)
print("columns = ", f.columns) # default Weight,Selection also present
print("total size = ", f.csize)


########################################################################
# plot pair counts
#---------------

cosmo = cosmology.Cosmology()
edges = np.logspace(-1, np.log10(150), 1000)


# ~result = SurveyDataPairCount('1d',f,edges, cosmo=cosmo, redshift='Z')
# ~print(result)
# ~plt.plot(edges, )


########################################################################
# plot PS and CF theorique
#---------------

Plin = cosmology.LinearPower(cosmo, redshift=0, transfer='CLASS')
Pnl = cosmology.HalofitPower(cosmo, redshift=0)
Pzel = cosmology.ZeldovichPower(cosmo, redshift=0)

# initialize the correlation objects
cf_lin = cosmology.CorrelationFunction(Plin)
cf_nl = cosmology.CorrelationFunction(Pnl)
cf_zel = cosmology.CorrelationFunction(Pzel)

# plot each kind
r = np.logspace(-1, np.log10(150), 1000)
plt.plot(r, r**2 * cf_lin(r), label='linear')
plt.plot(r, r**2 * cf_nl(r), label='nonlinear')
plt.plot(r, r**2 * cf_zel(r), label="Zel'dovich")

# format the axes
plt.legend()
plt.xlabel(r"$r$ $[h^{-1} \mathrm{Mpc}]$")
plt.ylabel(r"$r^2 \xi \ [h^{-2} \mathrm{Mpc}^2]$")
plt.show()
plt.close()
