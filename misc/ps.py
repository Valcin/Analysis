import numpy as np
import matplotlib.pyplot as plt
import matplotlib



r = np.loadtxt('/home/david/codes/class/output/test_pk.dat', skiprows=4)
k = r[:,0]
pk= r[:,1]
real = np.loadtxt('/home/david/codes/python scripts/realspace_halo.dat')

a = np.loadtxt('/home/david/codes/python scripts/mono0.dat')
a2 = np.loadtxt('/home/david/codes/python scripts/dmono0.dat')

b = np.loadtxt('/home/david/codes/python scripts/mono05.dat')
b2 = np.loadtxt('/home/david/codes/python scripts/dmono05.dat')

c = np.loadtxt('/home/david/codes/python scripts/mono1.dat')
c2 = np.loadtxt('/home/david/codes/python scripts/dmono1.dat')

d = np.loadtxt('/home/david/codes/python scripts/mono2.dat')
d2 = np.loadtxt('/home/david/codes/python scripts/dmono2.dat')

#--------------------------------------------------------------------


e = np.loadtxt('/home/david/codes/python scripts/quadru0.dat')
e2 = np.loadtxt('/home/david/codes/python scripts/dquadru0.dat')

f = np.loadtxt('/home/david/codes/python scripts/quadru05.dat')
f2 = np.loadtxt('/home/david/codes/python scripts/dquadru05.dat')

g = np.loadtxt('/home/david/codes/python scripts/quadru1.dat')
g2 = np.loadtxt('/home/david/codes/python scripts/dquadru1.dat')

h = np.loadtxt('/home/david/codes/python scripts/quadru2.dat')
h2 = np.loadtxt('/home/david/codes/python scripts/dquadru2.dat')



#--------------------------------------------------------------------




#### get the linear scale and power spectrum
km0 = a[:,0]
pkm0 = a[:,1]
pk2m0 = a2[:,1]
k2m0 = a2[:,0]
pkbism0 = np.interp(k2m0, km0, pkm0)

kq0 = e[:,0]
pkq0 = e[:,1]
pk2q0 = e2[:,1]
k2q0 = e2[:,0]
pkbisq0 = np.interp(k2q0, kq0, pkq0)

#---------------------
km05 = b[:,0]
pkm05 = b[:,1]
pk2m05 = b2[:,1]
k2m05 = b2[:,0]
pkbism05 = np.interp(k2m05, km05, pkm05)

kq05 = f[:,0]
pkq05 = f[:,1]
pk2q05 = f2[:,1]
k2q05 = f2[:,0]
pkbisq05 = np.interp(k2q05, kq05, pkq05)

#----------------------

km1 = c[:,0]
pkm1 = c[:,1]
pk2m1 = c2[:,1]
k2m1 = c2[:,0]
pkbism1 = np.interp(k2m1, km1, pkm1)

kq1 = g[:,0]
pkq1 = g[:,1]
pk2q1 = g2[:,1]
k2q1 = g2[:,0]
pkbisq1 = np.interp(k2q1, kq1, pkq1)

#---------------------

km2 = d[:,0]
pkm2 = d[:,1]
pk2m2 = d2[:,1]
k2m2 = d2[:,0]
pkbism2 = np.interp(k2m2, km2, pkm2)

kq2 = h[:,0]
pkq2 = h[:,1]
pk2q2 = h2[:,1]
k2q2 = h2[:,0]
pkbisq2 = np.interp(k2q2, kq2, pkq2)



khh = real[:,0]
phh = real[:,1]
phhbis = np.interp(k2m0, khh, phh)

#----------------------------------------------------------------------------

plt.figure()
plt.plot(k2m0,pk2m0,color='k', linestyle='--', label='David z = 0')
plt.plot(k2m0,pkbism0,color='k', label='Paco z = 0')
plt.plot(k2m05,pk2m05,color='b', linestyle='--', label='David z = 0.5')
plt.plot(k2m05,pkbism05,color='b',label='Paco z = 0.5')
plt.plot(k2m1,pk2m1,color='g', linestyle='--', label='David z = 1')
plt.plot(k2m1,pkbism1,color='g', label='Paco z = 0')
plt.plot(k2m2,pk2m2,color='r', linestyle='--', label='David z = 2')
plt.plot(k2m2,pkbism2,color='r', label='Paco  z = 0')	
plt.title('halo monopole comparison' )
plt.legend(loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.xlim(8e-3,0.5)
plt.ylim(1e3)
plt.ylabel('P(k)')
plt.show()  
plt.figure()

print pkbism05-pk2m05
plt.plot(k2m0,pkbism0-pk2m0,color='k', linestyle='-.', label=' z = 0')	
plt.plot(k2m05,pkbism05-pk2m05,color='b', linestyle='-.', label=' z = 0.5')
plt.plot(k2m1,pkbism1-pk2m1,color='g', linestyle='-.', label=' z = 1')
plt.plot(k2m2,pkbism2-pk2m2,color='r', linestyle='-.', label=' z = 2')
plt.title('halo monopole difference' )
plt.legend(loc='lower center')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.xlim(8e-3,0.5)
#~ plt.ylim(1e3)
plt.ylabel('P(k)')
plt.show()  


plt.figure()
#plt.plot(k2m0,pk2m0,color='k', linestyle='--')
#plt.plot(k2m0,pkbism0,color='k')
plt.plot(k2m05,pk2m05,color='b', linestyle='--', label='David z = 0.5')
plt.plot(k2m05,pkbism05,color='r', label=' Paco z = 0.5')

plt.plot(k2m05,pkbism05-pk2m05,color='g', linestyle='-.', label='diff z = 0.5')
#plt.plot(k2m1,pkbism1-pk2m1,color='g', linestyle='-.')
#plt.plot(k2m2,pkbism2-pk2m2,color='r', linestyle='-.')	
plt.title('halo monopole contribution' )
plt.legend(loc='lower center')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.xlim(8e-3,0.5)
#~ plt.ylim(1e3)
plt.ylabel('P(k)')
plt.show()  


#------------------------------------------------------------------

plt.figure()
plt.plot(k2q0,pk2q0,color='k', linestyle='--', label='David z = 0')
plt.plot(k2q0,pkbisq0,color='k', label='Paco z = 0')
plt.plot(k2q05,pk2q05,color='b', linestyle='--', label='David z = 0.5')
plt.plot(k2q05,pkbisq05,color='b', label='Paco z = 0.5')
plt.plot(k2q1,pk2q1,color='g', linestyle='--', label='David z = 1')
plt.plot(k2q1,pkbisq1,color='g', label='Paco z = 1')
plt.plot(k2q2,pk2q2,color='r', linestyle='--', label='David z = 2')
plt.plot(k2q2,pkbisq2,color='r', label='Paco z = 2')	
plt.title('halo quadrupole comparison' )
plt.legend(loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.xlim(8e-3,0.5)
plt.ylim(1e3)
plt.ylabel('P(k)')
plt.show() 

 
plt.figure()
plt.plot(k2q0,pkbisq0-pk2q0,color='k', linestyle='-.', label='z = 0')	
plt.plot(k2q05,pkbisq05-pk2q05,color='b', linestyle='-.', label='z = 0.5')
plt.plot(k2q1,pkbisq1-pk2q1,color='g', linestyle='-.', label='z = 1')
plt.plot(k2q2,pkbisq2-pk2q2,color='r', linestyle='-.', label='z = 2')
plt.title('halo quadrupole difference' )
plt.legend(loc='lower center')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.xlim(8e-3,0.5)
#~ plt.ylim(1e3)
plt.ylabel('P(k)')
plt.show()  


plt.figure()
plt.plot(k2q05,k2q05 * (pk2q05),color='b', linestyle='--', label='David z = 0.5')
plt.plot(k2q05,k2q05 * (pkbisq05),color='r', label='Paco z = 0.5')
plt.plot(k2q05,k2q05 * (pkbisq05-pk2q05),color='g', linestyle='-.', label='diff z = 0.5')
plt.title('halo quadrupole contribution' )
plt.legend(loc='center right')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('k')
#plt.xlim(8e-3,0.5)
#plt.ylim(1e3)
plt.ylabel('k*P(k)')
plt.show()  


plt.figure()
plt.plot(k2q05, (pk2q05),color='b', linestyle='--', label='David z = 0.5')
plt.plot(k2q05, (pkbisq05),color='r', label='Paco z = 0.5')
plt.plot(k2q05, (pkbisq05-pk2q05),color='g', linestyle='-.', label='diff z = 0.5')
plt.title('halo quadrupole contribution' )
plt.legend(loc='upper right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.xlim(8e-3,0.5)
#~ plt.ylim(1e3)
plt.ylabel('P(k)')
plt.show()  
plt.figure()


	

