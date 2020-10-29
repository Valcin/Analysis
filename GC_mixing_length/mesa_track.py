import numpy as np
import mesa_reader as mr
import matplotlib.pyplot as plt

Zsun = 0.0134
Fe_h = -2.0

# ~Zstar = Zsun * 10**(Fe_h)
# ~print(Zstar)

# ~# load and plot data
h = mr.MesaData('/home/david/codes/data/GC_mixing_length/he_core_flash/trimmed_history.data')
plt.scatter(h.log_Teff, h.log_L)

# set axis labels
plt.xlabel('log Effective Temperature')
plt.ylabel('log Luminosity')

# invert the x-axis
plt.gca().invert_xaxis()

plt.show()
plt.close()
