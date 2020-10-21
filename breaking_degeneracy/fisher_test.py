#~ from classy import Class
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse
import sys,os
from scipy import integrate
from scipy import interpolate
from pylab import *
from itertools import product

def main():

	########################################################################
	########################################################################
	### COMPUTE THE FIDUCIAL MODEL
	
	

	# define the parameters and their fiducial value
	prm = ['Om','Ob','h', 'ns', 's8', 'Mnu']
	parameter_label = [r'$\Omega_{\rm m}$', r'$\Omega_{\rm b}$', r'$h$', r'$n_{\rm s}$', r'$\sigma_8$', r'$M_\nu\,({\rm eV})$']
	fiducial = [0.3175, 0.049, 0.6711, 0.9624, 0.834, 0.0]
	
	V = 1.0e9 # Mpc**3/h**3
	z = 0.0
	kmax = 0.2
			
	F1 = Fisher_lss(prm, z, V, kmax, 0) # matter
	#~ F2 = Fisher_lss(prm, z, V, kmax, 1) # cdm
	
	#--------------------------------------------
	lred = [0.0, 0.4, 0.55, 0.70, 0.85, 1.0, 1.15]
	z = lred[0]
	lmax = 2000
			
	#~ F1 += Fisher_lens(prm, z, lmax) 
	#~ F2 += Fisher_lens(prm, z, lmax) 
	F2 = Fisher_lens(prm, z, lmax) 

	#~ print(F2)
	#~ print(np.shape(F1))
	#~ kill

	fig = figure(figsize=(14,14))
	############################################################################################

	# read the inverse Fisher matrix and find the number of parameters
	Cov1 = np.linalg.inv(F1)
	Cov2 = np.linalg.inv(F2)
	

	# find the number of parameters
	parameters = Cov1.shape[0]

	colors = ['r', 'b']
	Cov    = [Cov1, Cov2]

	# do a loop over the different subpanels and plot contours
	for i in xrange(parameters-1):
		for j in xrange(i+1, parameters):

			number = (parameters-1)*(j-1) + i + 1
			
			ax1 = fig.add_subplot(parameters-1, parameters-1, number)
			ax1.patch.set_alpha(0.5)

			# set the x- and y- limits of the subplot
			x2_max = np.max([Cov1[i,i], Cov2[i,i]])
			y2_max = np.max([Cov1[j,j], Cov2[j,j]])
			x_range = np.sqrt(x2_max)*1.5
			y_range = np.sqrt(y2_max)*1.5
			ax1.set_xlim([fiducial[i]-x_range, fiducial[i]+x_range])
			ax1.set_ylim([fiducial[j]-y_range, fiducial[j]+y_range])

			# compute the ellipses area: to improve visualization we plot first largest ellipses
			areas = np.array([Cov1[i,i]*Cov1[j,j], Cov2[i,i]*Cov2[j,j]])
			indexes = np.argsort(areas)[::-1]

			# plot the ellipses
			for k in xrange(len(indexes)):
				Cov_aux = Cov[indexes[k]]
				subCov = np.array([[Cov_aux[i,i], Cov_aux[i,j]], [Cov_aux[j,i], Cov_aux[j,j]]])
				a,b,theta = ellipse_params(subCov)
				plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c=colors[indexes[k]])

			"""
			# take the i,j subFisher, compute the ellipse parameters and plot it
			subCov = np.array([[Cov1[i,i], Cov1[i,j]], [Cov1[j,i], Cov1[j,j]]])
			a,b,theta = ellipse_params(subCov)
			plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c='r')

			subCov = np.array([[Cov2[i,i], Cov2[i,j]], [Cov2[j,i], Cov2[j,j]]])
			a,b,theta = ellipse_params(subCov)
			plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c='b')

			subCov = np.array([[Cov3[i,i], Cov3[i,j]], [Cov3[j,i], Cov3[j,j]]])
			a,b,theta = ellipse_params(subCov)
			plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c='g')

			subCov = np.array([[Cov4[i,i], Cov4[i,j]], [Cov4[j,i], Cov4[j,j]]])
			a,b,theta = ellipse_params(subCov)
			plot_ellipses(ax1, fiducial[i], fiducial[j], a, b, theta, c='k')
			"""

			# clean the x- and y- axes to make the figure nice
			if i>0 and j<(parameters-1):
				ax1.xaxis.set_major_formatter( NullFormatter() ) #unset x label
				ax1.yaxis.set_major_formatter( NullFormatter() ) #unset y label 

			if j==parameters-1:
				ax1.set_xlabel(parameter_label[i], fontsize=18)
				if i>0:
					ax1.yaxis.set_major_formatter( NullFormatter() ) #unset y label 

			if i==0:
				ax1.set_ylabel(parameter_label[j], fontsize=18)
				if j<parameters-1:
					ax1.xaxis.set_major_formatter( NullFormatter() ) #unset x label

			p1,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[0],alpha=0.7,lw=7)
			p2,=ax1.plot([-10,-9],[-10,-9],linestyle='-',marker='None',c=colors[1],alpha=0.7,lw=7)

	# legend
	leg = ax1.legend([p1,p2],
					 [r"$P_{\rm m}(k)\,\,:\,\,k_{\rm max}=0.2\,h{\rm Mpc}^{-1}$",
					 #~ [r"$P_{\rm mm}(k)\ + P_{\rm lensing}(l)$",
					 #~ r"$P_{\rm cdm}(k)\ + P_{\rm lensing}(l)$"],
					  #~ r"$P_{\rm cb}(k)\,\,:\,\,k_{\rm max}=0.2\,h{\rm Mpc}^{-1}$"],
					  r"$P_{\rm lensing}(l)\,\,:\,\,z_{\rm lens}=0.0\,z_{\rm source}=1.0$"],
					 loc=(-1.5,3.5),prop={'size':20},ncol=1,frameon=True)
	leg.get_frame().set_edgecolor('k')

			
	subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0.0)
	plt.show()

########################################################################
########################################################################

#~ def cosmo_change(fid, param, sign):

	#~ fid_new = np.zeros(len(fid))
	#~ for l in range(len(fid)):
		#~ if l == param and sign == 'plus':
			#~ fid_new[l] = fid[l] + fid[l]*0.01
		#~ elif l == param and sign == 'minus':
			#~ fid_new[l] = fid[l] - fid[l]*0.01
		#~ else:
			#~ fid_new[l] = fid[l]
	#~ return fid_new

#~ def derivative(kmiddle, fid, param1, param2, cdm) :
	
	#~ #compute delta parameter
	
	#~ fid_p1 = cosmo_change(fid, param1, 'plus')
	#~ fid_m1 = cosmo_change(fid, param1, 'minus')
	#~ fid_p2 = cosmo_change(fid, param2, 'plus')
	#~ fid_m2 = cosmo_change(fid, param2, 'minus')
	
		
	#~ delta1 = fid_p1[param1] - fid_m1[param1] 
	#~ delta2 = fid_p2[param2] - fid_m2[param2] 
	
	#~ pk_lin_p1, lpk_p1 = signal(kmiddle*fid[2], fid_p1, cdm)# here k multiplied by h because classy in 1/mpc
	#~ pk_lin_m1, lpk_m1 = signal(kmiddle*fid[2], fid_m1, cdm)# here k multiplied by h because classy in 1/mpc
	
	#~ pk_lin_p2, lpk_p2 = signal(kmiddle*fid[2], fid_p2, cdm)
	#~ pk_lin_m2, lpk_m2 = signal(kmiddle*fid[2], fid_m2, cdm)
		
	#~ der1 = (pk_lin_p1 -  pk_lin_m1)/ (delta1)
	#~ der2 = (pk_lin_p2 -  pk_lin_m2)/ (delta2)

	#~ return der1, der2
	

def Fisher_lss(parameter, z, V, kmax, cdm):
	num_params = len(parameter)
	
	root_derv = 'mm'

	# define the Fisher matrix
	Fisher = np.zeros((num_params,num_params), dtype=np.float64)

	# compute the value of kmin
	kmin = 2.0*np.pi/V**(1.0/3.0)

	# check that all ks are equally spaced in log
	for i in xrange(num_params):
		k, deriv = np.loadtxt('%s/derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), 
							  unpack=True)
		dk = np.log10(k[1:]) - np.log10(k[:-1])
		if not(np.allclose(dk, dk[0])):
			raise Exception('k-values not log distributed')

	# compute sub-Fisher matrix
	for i in xrange(num_params):
		for j in xrange(i,num_params):
			if cdm == 0:
				#~ k1, deriv1 = np.loadtxt('log_derivative_%s_z=%.1f.txt'%(parameter[i],z), unpack=True)
				#~ k2, deriv2 = np.loadtxt('log_derivative_%s_z=%.1f.txt'%(parameter[j],z), unpack=True)
				root_derv = 'mm'
				k1, deriv1 = np.loadtxt('%s/log_derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), unpack=True)
				k2, deriv2 = np.loadtxt('%s/log_derivative_%s_z=%.1f.txt'%(root_derv,parameter[j],z), unpack=True)

			elif cdm == 1:
				root_derv = 'cb'
				k1, deriv1 = np.loadtxt('%s/log_derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), unpack=True)
				k2, deriv2 = np.loadtxt('%s/log_derivative_%s_z=%.1f.txt'%(root_derv,parameter[j],z), unpack=True)
				
			if np.any(k1!=k2):  raise Exception('k-values are different!')
		
			I = integ(deriv1, deriv2 ,k1, kmin, kmax)

			Fisher[i,j] = I
			if i!=j:  Fisher[j,i] = Fisher[i,j]

	# add prefactors to subFisher matrix
	Fisher = Fisher*V/(2.0*np.pi)**2

	return Fisher
	
def Fisher_lens(parameter, z, lmax):
	num_params = len(parameter)
	
	root_derv = 'lens'


	# define the Fisher matrix
	Fisher = np.zeros((num_params,num_params), dtype=np.float64)

	# check that all ks are equally spaced in log
	for i in xrange(num_params):
		k, deriva, derivb, derivc = np.loadtxt('%s/derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), unpack=True)
		dk = np.log10(k[1:]) - np.log10(k[:-1])
		#~ if not(np.allclose(dk, dk[0])):
			#~ raise Exception('k-values not log distributed')
			

	# compute sub-Fisher matrix
	for i in xrange(num_params):
		for j in xrange(i,num_params):
			
			#in files the column are [k, 'W1xW1','W1xW2', 'W2xW2' ]
			deriv1 = np.loadtxt('%s/derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), unpack=True)
			
			#-------------------------------------------------------------
			#in files the column are [k, 'W1xW1','W1xW2', 'W2xW2' ]
			deriv2 = np.loadtxt('%s/derivative_%s_z=%.1f.txt'%(root_derv,parameter[j],z), unpack=True)

			#------------------------------------------------------------
			#in files the column are [k, 'W1xW1','W1xW2', 'W2xW2' ]
			cls = np.loadtxt('%s/pk_z=%s.txt'%(root_derv,z), unpack=True)
			#~ print(cls.shape)
			k1 = deriv1[0]; k2 = deriv2[0]; k3 = cls[0]
			

			if np.any(k1!=k2):  raise Exception('k-values are different!')
			if np.any(k1!=k3):  raise Exception('k-values are different!')
		
			#~ total = np.zeros(len(cls[0]))
			#~ covariance = np.zeros(len(cls[0]))
			#~ for l in xrange(len(cls[0])):
			total = np.zeros(800)
			covariance = np.zeros(800)
			for l in xrange((800)):
				#~ print l
				for i1, i2, i3, i4 in product(range(2), range(2), range(2), range(2)):
					#~ print i1, i2, i3, i4
					covariance[l] = cls[i1+i3+1, l]*cls[i1+i4+1, l] + cls[i1+i4+1,l]*cls[i2+i3+1,l]
					total[l] = deriv1[i1+i2+1,l] * 1/covariance[l] *  deriv2[i3+i4+1,l]
					
			I = integ_lens(total, k1[2:802])

			Fisher[i,j] = I
			if i!=j:  Fisher[j,i] = Fisher[i,j]
			
			#~ print(parameter[i], total)

	# add prefactors to subFisher matrix
	fsky = 0.5
	Fisher = Fisher * fsky

	return Fisher

# This function takes a subFisher and computes the ellipse parameters
def ellipse_params(subCov):
	a2 = 0.5*(subCov[0,0]+subCov[1,1]) + np.sqrt(0.25*(subCov[0,0]-subCov[1,1])**2 + subCov[0,1]**2)
	a  = np.sqrt(a2)
	b2 = 0.5*(subCov[0,0]+subCov[1,1]) - np.sqrt(0.25*(subCov[0,0]-subCov[1,1])**2 + subCov[0,1]**2)
	b  = np.sqrt(b2)
	theta = 0.5*np.arctan2(2.0*subCov[0,1],(subCov[0,0]-subCov[1,1]))
	return a,b,theta
	
# This function plots the ellipses
def plot_ellipses(ax, fid_x, fid_y, a, b, theta, c='r'):
	e1 = Ellipse(xy=(fid_x,fid_y), width=1.52*a, height=1.52*b,
				 angle=theta*360.0/(2.0*np.pi))
	e2 = Ellipse(xy=(fid_x,fid_y), width=2.48*a, height=2.48*b,
				 angle=theta*360.0/(2.0*np.pi))

	for e in [e1,e2]:
		ax.add_artist(e)
		if e==e1:  alpha = 0.7
		if e==e2:  alpha = 0.4
		e.set_alpha(alpha)
		e.set_facecolor(c)
		


def integ(der1, der2 ,k, kmin, kmax):
	
	f1 = interpolate.interp1d(k, der1)
	f2 = interpolate.interp1d(k, der2)
		
	def func(x):
		return f1(x) *  (x**2)* f2(x)
	

	intf,_ = integrate.fixed_quad(func, kmin, kmax, n = 200)

	return intf
	
def integ_lens(total ,k1):
	
	f1 = interpolate.interp1d(k1, total)
		
	def func(x):
		return f1(x) *  (2*x +1)
	
	kmin = np.min(k1)
	kmax = np.max(k1)
	intf,_ = integrate.fixed_quad(func, kmin, kmax, n = 200)

	return intf
	

########################################################################
########################################################################


if __name__ == '__main__':
	main()



















