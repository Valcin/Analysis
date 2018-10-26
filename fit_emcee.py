import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import emcee
import sys
sys.path.append('/home/david/codes/FAST-PT')
import myFASTPT as FPT
import math
import time
from scipy.optimize import curve_fit
from scipy.special import erf



########################################################################
########### POWER LAW even
########################################################################

def coeffit_pl2 (kstop,lb1, errlb1, pop, k ,b ,errb):
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	def lnlike(theta, x, y, yerr):
		b1, b2, b4 = theta
		model = b1 + b2*x[lim]**2 + b4*x[lim]**4 
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		b1, b2, b4 = theta
		if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1  and b1 > 0:
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
		
		
	

	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [pop],  method='Nelder-Mead', args=(k, b ,errb ),  options={'maxfev': 2000} )
	#~ result = op.minimize(nll, [pop], args=(k, b ,errb ))
	b1_ml, b2_ml, b4_ml = result["x"]
	print pop
	print result
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = len(pop), 300
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	



	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$b3$", "$b4$" ], truths=[b1_ml, b2_ml, b3_ml, b4_ml])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc, b2_mcmc, b4_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc, b2_mcmc, b4_mcmc
	

	#~ plt.figure()
	#~ ax1 = plt.subplot(221)
	#~ ax1.set_title('b1')
	#~ for i in xrange(0,nwalkers):
		#~ ax1.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ ax2 = plt.subplot(222)
	#~ ax2.set_title('b2')
	#~ for i in xrange(0,nwalkers):
		#~ ax2.plot(np.arange(1000), sampler.chain[i,:,1])
	#~ ax3 = plt.subplot(223)
	#~ ax3.set_title('b3')
	#~ for i in xrange(0,nwalkers):
		#~ ax3.plot(np.arange(1000), sampler.chain[i,:,2])
	#~ plt.show()
	
	return b1_mcmc, b2_mcmc, b4_mcmc
########################################################################
########### POWER LAW odd
########################################################################

def coeffit_pl (kstop,lb1, errlb1, pop, k ,b ,errb):
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	def lnlike(theta, x, y, yerr):
		b1, b2, b3, b4 = theta
		model = b1 + b2*x[lim]**2 + b3*x[lim]**3 + b4*x[lim]**4 
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		b1, b2, b3, b4 = theta
		if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1  and b1 > 0:
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
		
		
	

	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [pop],  method='Nelder-Mead', args=(k, b ,errb ),  options={'maxfev': 2000} )
	#~ result = op.minimize(nll, [pop], args=(k, b ,errb ))
	b1_ml, b2_ml, b3_ml, b4_ml = result["x"]
	print pop
	print result
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = len(pop), 300
	pos = [result["x"] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
	



	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$b3$", "$b4$" ], truths=[b1_ml, b2_ml, b3_ml, b4_ml])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc
	

	#~ plt.figure()
	#~ ax1 = plt.subplot(221)
	#~ ax1.set_title('b1')
	#~ for i in xrange(0,nwalkers):
		#~ ax1.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ ax2 = plt.subplot(222)
	#~ ax2.set_title('b2')
	#~ for i in xrange(0,nwalkers):
		#~ ax2.plot(np.arange(1000), sampler.chain[i,:,1])
	#~ ax3 = plt.subplot(223)
	#~ ax3.set_title('b3')
	#~ for i in xrange(0,nwalkers):
		#~ ax3.plot(np.arange(1000), sampler.chain[i,:,2])
	#~ ax4 = plt.subplot(224)
	#~ ax4.set_title('b4')
	#~ for i in xrange(0,nwalkers):
		#~ ax4.plot(np.arange(1000), sampler.chain[i,:,3])
	#~ plt.show()
	
	return b1_mcmc, b2_mcmc, b3_mcmc, b4_mcmc
	#~ return b1_ml, b2_ml, b3_ml, b4_ml
########################################################################
######### bias expansion 2nd order
########################################################################
def coeffit_exp1(kstop, Pmm, A, B, C, D, E, lb1, errlb1, pop, k ,b ,errb):
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	
	def lnlike(theta, x, y, yerr):
		b1, b2, bs = theta
		model = np.sqrt((b1**2 * Pmm[lim]+ b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + \
		1/4.*bs**2*E[lim])/Pmm[lim])
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		b1, b2, bs = theta
		#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1  and b1 > 0:
		if lb1 - 5*errlb1 < b1 < lb1 + 5*errlb1  and b1 > 0 and -4/7.*((lb1 - 20*errlb1)-1) > bs > -4/7.*((lb1 + 20*errlb1)-1):
			return 0.0
		#~ return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)


	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [pop],  method='Nelder-Mead', args=(k, b ,errb ),  options={'maxfev': 2000} )
	#~ result = op.minimize(nll, [pop], args=(k, b ,errb ),  options={'maxiter': 2000} )
	b1_ml, b2_ml, bs_ml = result["x"]
	print pop
	print(result)
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*3. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = len(pop), 300
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$" ], truths=[b1_ml, b2_ml, bs_ml])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc, b2_mcmc, bs_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc, b2_mcmc, bs_mcmc
	

	#~ plt.figure()
	#~ ax1 = plt.subplot(221)
	#~ ax1.set_title('b1')
	#~ for i in xrange(0,nwalkers):
		#~ ax1.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ ax2 = plt.subplot(222)
	#~ ax2.set_title('b2')
	#~ for i in xrange(0,nwalkers):
		#~ ax2.plot(np.arange(1000), sampler.chain[i,:,1])
	#~ ax3 = plt.subplot(223)
	#~ ax3.set_title('bs')
	#~ for i in xrange(0,nwalkers):
		#~ ax3.plot(np.arange(1000), sampler.chain[i,:,2])

	#~ plt.show()

	return b1_mcmc, b2_mcmc, bs_mcmc
	
########################################################################
######### bias expansion 3rd order free
########################################################################
def coeffit_exp2(kstop, Pmm, A, B, C, D, E, F, lb1, errlb1, pop, k ,b ,errb):
	
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	#~ print k[lim]
	#~ kill
	#~ print lim
	def lnlike(theta, x, y, yerr):
		b1, b2, bs, b3nl = theta
		model = np.sqrt((b1**2 * Pmm[lim]+ b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim]\
		+ 2*b1*b3nl*F[lim])/Pmm[lim])
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		b1, b2, bs, b3nl = theta
		#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1  and b1 > 0:
		if lb1 - 5*errlb1 < b1 < lb1 + 5*errlb1  and b1 > 0 and -4/7.*((lb1 - 70*errlb1)-1) > bs > -4/7.*((lb1 + 70*errlb1)-1)\
		and 32/315.*((lb1 - 30*errlb1)-1) < b3nl < 32/315.*((lb1 + 30*errlb1)-1):
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
		
		
	

	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [pop],  method='Nelder-Mead', args=(k, b ,errb ),  options={'maxfev': 2000} )
	#~ result = op.minimize(nll, [pop],  method='BFGS', args=(k, b ,errb ))
	b1_ml, b2_ml, bs_ml, b3nl = result["x"]
	print pop
	print(result)
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = len(pop), 300
	pos = [result["x"] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
	

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$", "$b3nl$" ], truths=[b1_ml, b2_ml, bs_ml, b3nl])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc, b2_mcmc, bs_mcmc, b3_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc, b2_mcmc, bs_mcmc, b3_mcmc
	

	#~ plt.figure()
	#~ ax1 = plt.subplot(221)
	#~ ax1.set_title('b1')
	#~ for i in xrange(0,nwalkers):
		#~ ax1.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ ax2 = plt.subplot(222)
	#~ ax2.set_title('b2')
	#~ for i in xrange(0,nwalkers):
		#~ ax2.plot(np.arange(1000), sampler.chain[i,:,1])
	#~ ax3 = plt.subplot(223)
	#~ ax3.set_title('bs')
	#~ for i in xrange(0,nwalkers):
		#~ ax3.plot(np.arange(1000), sampler.chain[i,:,2])
	#~ ax3 = plt.subplot(224)
	#~ ax3.set_title('b3nl')
	#~ for i in xrange(0,nwalkers):
		#~ ax3.plot(np.arange(1000), sampler.chain[i,:,3])

	#~ plt.show()

	return b1_mcmc, b2_mcmc, bs_mcmc, b3_mcmc
	#~ return b1_ml, b2_ml, bs_ml, b3nl
########################################################################
######### bias expansion 3rd order fixed
########################################################################
def coeffit_exp3(kstop, Pmm, A, B, C, D, E, F, lb1, errlb1, pop, k ,b ,errb):
	
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	def lnlike(theta, x, y, yerr):
		b1, b2 = theta
		bs = -4/7.*(b1-1)
		b3nl = 32/315.*(b1-1)
		model = np.sqrt((b1**2 * Pmm[lim]+ b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim]\
		+ 2*b1*b3nl*F[lim])/Pmm[lim])
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		b1, b2 = theta
		if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1  and b1 > 0:
			return 0.0
		#~ return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
		
		
	

	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [pop],  method='Nelder-Mead', args=(k, b ,errb ),  options={'maxfev': 2000} )
	#~ result = op.minimize(nll, [pop], args=(k, b ,errb ))
	b1_ml, b2_ml = result["x"]
	print pop
	print result
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = len(pop), 300
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$", "$b3nl$" ], truths=[b1_ml, b2_ml, bs_ml, b3nl])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc, b2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc, b2_mcmc
	

	#~ plt.figure()
	#~ ax1 = plt.subplot(221)
	#~ ax1.set_title('b1')
	#~ for i in xrange(0,nwalkers):
		#~ ax1.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ ax2 = plt.subplot(222)
	#~ ax2.set_title('b2')
	#~ for i in xrange(0,nwalkers):
		#~ ax2.plot(np.arange(1000), sampler.chain[i,:,1])
	#~ ax3 = plt.subplot(223)
	#~ ax3.set_title('bs')
	#~ for i in xrange(0,nwalkers):
		#~ ax3.plot(np.arange(1000), sampler.chain[i,:,2])
	

	#~ plt.show()

	return b1_mcmc, b2_mcmc

########################################################################
#### Linear Kaiser
########################################################################
def coeffit_Kaiser(j, fcc, kstop, Pmm, lb1, k ,b ,errb):
	

	start = time.time()
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]

	def lnlike(theta, x, y, yerr):
		sigma = theta
		#~ kappa = x[lim]*sigma
		#~ coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		#~ coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		#~ coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		kappa = x[lim]*sigma*fcc[lim]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		model = Pmm[lim]*(lb1[lim]**2*coeffA +  2/3.*lb1[lim]*fcc[lim]*coeffB + 1/5.*fcc[lim]**2*coeffC)
		#~ model = Pmm[lim]*(lb1**2*coeffA +  2/3.*lb1*fcc[lim]*coeffB + 1/5.*fcc[lim]**2)
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		sigma = theta
		#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1 :
		if 0 < sigma < 100:
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
	z = [0.0,0.5,1.0,2.0]
	red = ['0.0','0.5','1.0','2.0','3.0']
	ind = red.index(str(z[j]))
	f = [0.518,0.754,0.872,0.956,0.98]
	Dz = [ 1.,0.77,0.61,0.42]

	nll = lambda *args: -lnlike(*args)
	#~ result = op.minimize(nll, [pop], bounds= [(1,1000)], args=(k, b ,errb ))
	result = op.minimize_scalar(nll, bounds=(1,1000), method='bounded',args=(k, b ,errb ))
	b1_ml = result["x"]
	print(result)
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = 1, 200
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]



	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
     

	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$", "$b3nl$" ], truths=[b1_ml, b2_ml, bs_ml, b3nl])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc
	

	#~ plt.figure()
	#~ plt.title('sigma kaiser')
	#~ for i in xrange(0,nwalkers):
		#~ plt.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ plt.show()
	
	end = time.time()
	print 'time is '+str((end - start))
	return b1_mcmc
	
########################################################################
#### Scoccimaro
########################################################################
def coeffit_Scocci(j, fcc, kstop,Pmod_dd, Pmod_dt, Pmod_tt, lb1, k ,b ,errb):
	start = time.time()
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	
	def lnlike(theta, x, y, yerr):
		sigma = theta
		#~ kappa = x[lim]*sigma
		#~ coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		#~ coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		#~ coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		kappa = x[lim]*sigma*fcc[lim][ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		model = lb1[lim]**2*Pmod_dd[lim]*coeffA + 2/3.*lb1[lim]*fcc[lim]*Pmod_dt[lim]*coeffB + 1/5.*fcc[lim]**2*Pmod_tt[lim]*coeffC
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	
	def lnprior(theta):
		sigma = theta
		#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1 :
		if 0 < sigma < 100:
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
	z = [0.0,0.5,1.0,2.0]
	red = ['0.0','0.5','1.0','2.0','3.0']
	ind = red.index(str(z[j]))
	f = [0.518,0.754,0.872,0.956,0.98]
	Dz = [ 1.,0.77,0.61,0.42]

	nll = lambda *args: -lnlike(*args)
	#~ result = op.minimize(nll, [pop], bounds= [(1,1000)], args=(k, b ,errb ))
	result = op.minimize_scalar(nll, bounds=(1,1000), method='bounded',args=(k, b ,errb ))
	b1_ml = result["x"]
	print(result)
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = 1, 200
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]



	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
     

	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$", "$b3nl$" ], truths=[b1_ml, b2_ml, bs_ml, b3nl])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc
	

	#~ plt.figure()
	#~ plt.title('sigma scocci')
	#~ for i in xrange(0,nwalkers):
		#~ plt.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ plt.show()
	
	end = time.time()
	print 'time is '+str((end - start))
	return b1_mcmc

########################################################################
#### TNS
########################################################################
def coeffit_TNS(j, fcc, kstop,Pmod_dd, Pmod_dt, Pmod_tt, lb1, k ,b ,errb,AB2,AB4,AB6,AB8 ):
		
	start = time.time()
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	
	def lnlike(theta, x, y, yerr):
		sigma = theta
		#~ kappa = x[lim]*sigma
		#~ coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		#~ coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		#~ coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		#~ coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		#~ coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		kappa = x[lim]*sigma*fcc[ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		coeffD = -2/3./kappa**2*(coeffC - 2/(2+kappa**2))
		coeffE = -4/10./kappa**2*(7.*coeffD - 2/(2+kappa**2))
		model = lb1[lim]**2*Pmod_dd[lim]*coeffA + 2/3.*lb1[lim]*fcc[lim]*Pmod_dt[lim]*coeffB + 1/5.*fcc[lim]**2*Pmod_tt[lim]*coeffC \
		+ (1/3.*AB2[lim]*coeffB+ 1/5.*AB4[lim]*coeffC+ 1/7.*AB6[lim]*coeffD+ 1/9.*AB8[lim]*coeffE)
		inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
	def lnprior(theta):
		sigma = theta
		#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1 :
		if 0 < sigma < 100:
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
	z = [0.0,0.5,1.0,2.0]
	red = ['0.0','0.5','1.0','2.0','3.0']
	ind = red.index(str(z[j]))
	f = [0.518,0.754,0.872,0.956,0.98]
	Dz = [ 1.,0.77,0.61,0.42]

	nll = lambda *args: -lnlike(*args)
	#~ result = op.minimize(nll, [pop], bounds= [(1,1000)], args=(k, b ,errb ))
	result = op.minimize_scalar(nll, bounds=(1,1000), method='bounded',args=(k, b ,errb ))
	b1_ml = result["x"]
	print(result)
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = 1, 200
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]



	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
     

	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$", "$b3nl$" ], truths=[b1_ml, b2_ml, bs_ml, b3nl])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc
	

	#~ plt.figure()
	#~ plt.title('sigma tns')
	#~ for i in xrange(0,nwalkers):
		#~ plt.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ plt.show()
	
	end = time.time()
	print 'time is '+str((end - start))
	return b1_mcmc
	
########################################################################
#### TNS + saito or eTns
########################################################################
def coeffit_eTNS(j, fcc, kstop, b1, b2, bs, b3nl, Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H, k ,b ,errb,AB2,AB4,AB6,AB8, sca=None ):
	
	start = time.time()
	#~ lim = np.where(k < kstop)[0]
	lim = np.where((k < kstop)&(k > 1e-2))[0]
	
	def lnlike(theta, x, y, yerr):
		#~ b1, b2, bs, b3nl, sigma = theta
		sigma = theta
		PsptD1z = b1**2*Pmod_dd[lim] + b1*b2*A[lim] + 1/4.*b2**2*B[lim] + b1*bs*C[lim] + 1/2.*b2*bs*D[lim] + 1/4.*bs**2*E[lim] \
		+ 2*b1*b3nl*F[lim]
		PsptT = b1* Pmod_dt[lim] + b2*G[lim] + bs*H[lim] + b3nl*F[lim]
		kappa = x[lim]*sigma
		#~ coeffA = math.sqrt(math.pi)/2. * erf(kappa)/kappa
		#~ coeffB = 3./2./kappa**2*(coeffA - np.exp(-kappa**2))
		#~ coeffC = 5./2./kappa**2*(coeffB - np.exp(-kappa**2))
		#~ coeffD = 7./2./kappa**2*(coeffC - np.exp(-kappa**2))
		#~ coeffE = 9./2./kappa**2*(coeffD - np.exp(-kappa**2))
		kappa = x[lim]*sigma*fcc[lim][ind]*Dz[ind]
		coeffA = np.arctan(kappa/math.sqrt(2))/(math.sqrt(2)*kappa) + 1/(2+kappa**2)
		coeffB = 6/kappa**2*(coeffA - 2/(2+kappa**2))
		coeffC = -10/kappa**2*(coeffB - 2/(2+kappa**2))
		coeffD = -2/3./kappa**2*(coeffC - 2/(2+kappa**2))
		coeffE = -4/10./kappa**2*(7.*coeffD - 2/(2+kappa**2))
		if sca:
			model = PsptD1z*coeffA*sca**2 + 2/3.*fcc[lim]*PsptT*coeffB*sca + 1/5.*fcc[lim]**2*Pmod_tt[lim]*coeffC \
			+ (1/3.*AB2[lim]*coeffB+ 1/5.*AB4[lim]*coeffC+ 1/7.*AB6[lim]*coeffD+ 1/9.*AB8[lim]*coeffE)
			inv_sigma2 = 1.0/(yerr[lim]**2)
		else:
			model = PsptD1z*coeffA + 2/3.*fcc[lim]*PsptT*coeffB + 1/5.*fcc[lim]**2*Pmod_tt[lim]*coeffC \
			+ (1/3.*AB2[lim]*coeffB+ 1/5.*AB4[lim]*coeffC+ 1/7.*AB6[lim]*coeffD+ 1/9.*AB8[lim]*coeffE)
			inv_sigma2 = 1.0/(yerr[lim]**2)
		return -0.5*(np.sum((y[lim]-model)**2*inv_sigma2 - np.log(inv_sigma2)))
		
	def lnprior(theta):
		sigma = theta
		#~ if lb1 - 3*errlb1 < b1 < lb1 + 3*errlb1 :
		if 0 < sigma < 100:
			return 0.0
		return -np.inf
	
	def lnprob(theta, x, y, yerr):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(theta, x, y, yerr)
		
	z = [0.0,0.5,1.0,2.0]
	red = ['0.0','0.5','1.0','2.0','3.0']
	ind = red.index(str(z[j]))
	f = [0.518,0.754,0.872,0.956,0.98]
	Dz = [ 1.,0.77,0.61,0.42]

	nll = lambda *args: -lnlike(*args)
	#~ result = op.minimize(nll, [pop], bounds= [(1,1000)], args=(k, b ,errb ))
	result = op.minimize_scalar(nll, bounds=(1,1000), method='bounded',args=(k, b ,errb ))
	b1_ml = result["x"]
	print(result)
	
	#~ max_l = lnlike(result["x"], k, b, errb )
	#~ AIC = 2*4. - 2 * max_l
	#~ print 'maximum likelihood is '+str(max_l)
	#~ print 'AIC = '+str(AIC)
	
	ndim, nwalkers = 1, 200
	pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]



	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, b, errb))
	sampler.run_mcmc(pos, 1000)
	
     

	samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

	
	#~ import corner
	#~ fig = corner.corner(samples, labels=["$b1$", "$b2$", "$bs$", "$b3nl$" ], truths=[b1_ml, b2_ml, bs_ml, b3nl])
	#~ fig.savefig("/home/david/triangle.png")
	

	b1_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
	
	print b1_mcmc
	

	#~ plt.figure()
	#~ plt.title('sigma tns')
	#~ for i in xrange(0,nwalkers):
		#~ plt.plot(np.arange(1000), sampler.chain[i,:,0])
	#~ plt.show()
	end = time.time()
	print 'time is '+str((end - start))
	return b1_mcmc
