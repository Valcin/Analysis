import numpy as np
import matplotlib.pyplot as plt
omalley = np.loadtxt('catalogs/omalley.txt')
# ~ax2.errorbar(omalley[:,4], metal_fin_dar[(omalley[:,5]).astype(int)] , yerr =[metal_fin_dar[(omalley[:,5]).astype(int)] - metal_low_dar[(omalley[:,5]).astype(int)],

meta = np.loadtxt('metal_comp.txt')
harris = meta[:,0]
moi = meta[:,1]

metarr = np.linspace(-2.5,0,50)

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

sigma_harris = np.sqrt(np.sum((harris - moi)**2)/len(moi))
sigma_omalley = np.sqrt(np.sum((omalley[:,4] - moi[(omalley[:,5]).astype(int)] )**2)/len(moi[(omalley[:,5]).astype(int)]))

print(sigma_harris, len(moi))
print(sigma_omalley, len(moi[(omalley[:,5]).astype(int)]))
