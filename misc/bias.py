import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
#from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.stats import norm
from sklearn import mixture 
from pylab import *
from scipy.optimize import curve_fit
from matplotlib import ticker

#______________________________________________________________________
# DEFINE QUANTITIES
#______________________________________________________________________

nx, nsteps = 20,5
cmap = mpl.cm.autumn


#----------------------------------------------------------------------
#raccanelli
#----------------------------------------------------------------------

#for redshift 0,1,2,3 

b1 = [1.14,2.13,3.85,6.46]
b2 = [-1.29,3.37,22.06,94.02]
b3 = [-2.19,-12.94,-60.5,-262.05]
b4 = [4.77,11.6,49.2,227.27]

k  = np.linspace(0.006,0.5,100)
z=[0,1,2,3]



plt.figure()
for i in z:
	b = b1[i] + b2[i]*(k**2) + b3[i]*(k**3) + b4[i]*(k**4)
	
	plt.scatter(k,b, marker='.',label= 'bias at z = '+ str(i), color=cmap(i / float(nsteps)))
	plt.xlabel('k')
	plt.ylabel('bias bcc')
	plt.title('bias vs scale Raccanelli et al.')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(0.004, 0.8)
	plt.ylim(0.5,15)
	plt.legend(loc='upper left')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
#Amendola
#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------


redshift = [0.8,1.0,1.2,1.4,1.6,1.8]
n = [1,1.28,2]
kA  = np.linspace(0.001,0.5,100)
k1 = 1

#FM1-PL and FM1-Q
bno = 1
#------------------------------------------------
#-------------------------------------------------
#FM2-PL 
#n= 1
boa = [1.04,1.13,1.22,1.36,1.49,1.61]
b1a = [0.67,0.74,0.99,1.09,1.22,1.40]

#plt.figure()
#for i in xrange(0,len(redshift)):
#	bAa = boa[i] + b1a [i]*(kA/k1)**n[0]
#	
#	plt.scatter(kA,bAa, marker='.',label= 'bias at z = '+ str(redshift[i]), color=cmap(i / float(nsteps)))
##	plt.scatter(kA,bno, marker='.',label= 'constant bias')
#	plt.xlabel('kA')
#	plt.ylabel('bias')
#	plt.xscale('log')
#	plt.title('Power law bias vs scale with n = '+str(n[0])+' Amendola et al.')
##	plt.yscale('log')
#	plt.xlim(0.0008, 0.8)
#	plt.ylim(0.8,2.7)
#	plt.legend(loc='upper left')
#	mng = plt.get_current_fig_manager()
#	mng.resize(*mng.window.maxsize())
#plt.show()
#-------------------------------------------------
#---------------------------------------------------

#n= 1.28
bob = [1.09,1.19,1.30,1.44,1.58,1.72]
b1b = [0.66,0.75,0.97,1.06,1.19,1.40]

#plt.figure()
#for i in xrange(0,len(redshift)):
#	bAb = bob[i] + b1b [i]*(kA/k1)**n[1]
#	
#	plt.scatter(kA,bAb, marker='.',label= 'bias at z = '+ str(redshift[i]), color=cmap(i / float(nsteps)))
##	plt.scatter(kA,bno, marker='.',label= 'constant bias')
#	plt.xlabel('kA')
#	plt.ylabel('bias')
#	plt.xscale('log')
#	plt.title('Power law bias vs scale with n = '+str(n[1])+' Amendola et al.')
##	plt.yscale('log')
#	plt.xlim(0.0008, 0.8)
##	plt.ylim(0.5,15)
#	plt.legend(loc='upper left')
#	mng = plt.get_current_fig_manager()
#	mng.resize(*mng.window.maxsize())
#plt.show()

#----------------------------------------------------------
#------------------------------------------------------------
#FM2-PL 
#n= 2
boc = [1.17,1.28,1.41,1.55,1.71,1.88]
b1c = [0.68,0.79,1.02,1.12,1.25,1.44]

#plt.figure()
#for i in xrange(0,len(redshift)):
#	bAc = boc[i] + b1c [i]*(kA/k1)**n[2]
#	
#	plt.scatter(kA,bAc, marker='.',label= 'bias at z = '+ str(redshift[i]), color=cmap(i / float(nsteps)))
##	plt.scatter(kA,bno, marker='.',label= 'constant bias')
#	plt.xlabel('kA')
#	plt.ylabel('bias')
#	plt.xscale('log')
#	plt.title('Power law bias vs scale with n = '+str(n[2])+' Amendola et al.')
##	plt.yscale('log')
#	plt.xlim(0.0008, 0.8)
##	plt.ylim(0.5,15)
#	plt.legend(loc='upper left')
#	mng = plt.get_current_fig_manager()
#	mng.resize(*mng.window.maxsize())
#plt.show()

#---------------------------------------------------------
#---------------------------------------------------------

#FM2-Q
boq = [1.26,1.36,1.49,1.63,1.75,1.92]
A = [1.7,1.7,1.7,1.7,1.7,1.7]
Q = [4.54,4.92,5.50,5.70,6.62,6.99]

#plt.figure()
#for i in xrange(0,len(redshift)):
#	bAq = boq[i]*((1+Q[i]*(kA/k1)**2)/(1+A[i]*(kA/k1))**0.5)
#	
#	plt.scatter(kA,bAq, marker='.',label= 'bias at z = '+ str(redshift[i]), color=cmap(i / float(nsteps)))
##	plt.scatter(kA,bno, marker='.',label= 'constant bias')
#	plt.xlabel('kA')
#	plt.ylabel('bias')
#	plt.xscale('log')
#	plt.title('Q model vs scale Amendola et al.')
##	plt.yscale('log')
#	plt.xlim(0.0008, 0.8)
##	plt.ylim(0.5,15)
#	plt.legend(loc='upper left')
#	mng = plt.get_current_fig_manager()
#	mng.resize(*mng.window.maxsize())
#plt.show()


#--------------------------------------------------------
#---------------------------------------------------------
#comparison

#plt.figure()
#for i in xrange(0,len(redshift)):
#	bAa = boa[i] + b1a [i]*(kA/k1)**n[0]
#	bAb = bob[i] + b1b [i]*(kA/k1)**n[1]
#	bAc = boc[i] + b1c [i]*(kA/k1)**n[2]
#	bAq = boq[i]*((1+Q[i]*(kA/k1)**2)/(1+A[i]*(kA/k1))**0.5)
#	
#	plt.scatter(kA,bAa, marker='.',label= 'Power law with n=1 ', color='b')
#	plt.scatter(kA,bAb, marker='.',label= 'Power law with n=1.28 ', color='r')
#	plt.scatter(kA,bAc, marker='.',label= 'Power law with n=2 ', color='g')
#	plt.scatter(kA,bAq, marker='.',label= 'Q model', color='y')
#	plt.xlabel('kA')
#	plt.ylabel('bias')
#	plt.xscale('log')
#	plt.title('Comparison of the different model at redshift z = ' + str(redshift[i]) + ' Amendola et al.')
##	plt.yscale('log')
#	plt.xlim(0.0008, 0.8)
##	plt.ylim(0.5,15)
#	plt.legend(loc='upper left')
#	mng = plt.get_current_fig_manager()
#	mng.resize(*mng.window.maxsize())
#	plt.show()



plt.figure()
i = 1
bAa = boa[i] + b1a [i]*(kA/k1)**n[0]
bAb = bob[i] + b1b [i]*(kA/k1)**n[1]
bAc = boc[i] + b1c [i]*(kA/k1)**n[2]
bAq = boq[i]*((1+Q[i]*(kA/k1)**2)/(1+A[i]*(kA/k1))**0.5)
b = b1[i] + b2[i]*(k**2) + b3[i]*(k**3) + b4[i]*(k**4)
	
plt.scatter(kA,bAa, marker='.',label= 'Power law with n=1 ', color='b')
plt.scatter(kA,bAb, marker='.',label= 'Power law with n=1.28 ', color='r')
plt.scatter(kA,bAc, marker='.',label= 'Power law with n=2 ', color='g')
plt.scatter(kA,bAq, marker='.',label= 'Q model', color='y')
plt.scatter(kA,b, marker='.',label= 'massive neutrino', color='k')
plt.xlabel('kA')
plt.ylabel('bias')
plt.xscale('log')
plt.title('Comparison of the different model at redshift z = ' + str(redshift[i]) + ' + massive neutrinos')
#	plt.yscale('log')
plt.xlim(0.0008, 0.8)
#	plt.ylim(0.5,15)
plt.legend(loc='upper left')
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()


