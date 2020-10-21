import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math



h = 0.67
rho_c = 2.775e11
rho_m = rho_c * 0.3 / (h*h)



print rho_m

M = [20,100]

L_20 = ((3 * M[0])/(4 * rho_m * math.pi))**(0.333333333333)
L_100 = ((3 * M[1])/(4 * rho_m * math.pi))**(0.333333333333)
print L_20, L_100

print 1/L_20, 1/L_100
