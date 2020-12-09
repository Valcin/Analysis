import emcee
import math
import numpy as np
from multiprocessing import Pool
########################################################################
########################################################################

def lnlike(theta):
	age, FeH, dist, A1, afe = theta # varied parameters
	### compute the likelihood
	lnl = -0.5*np.sum((gauss_mean - Color_new)**2 / (gauss_disp)**2 )
	return lnl
		

def lnprior(theta):
	age, FeH, dist, A1, afe = theta # varied parameters
	###gaussian prior on metallicity
	fe_mu = metal
	me_mu = FeH
	me_sigma = 0.2
	lnl_me = (math.log(1.0/(math.sqrt(2*math.pi)*me_sigma))-0.5*(me_mu-fe_mu)**2/me_sigma**2)
	#~ #flat priors on some parameters
	if 9 < age < 10.175 and -2.5 < FeH < 0  and 0.0 < dist and 0 < A1 < 3.0 and -0.2 <= afe <= 0.8:
		# ~return 0.0
		return lnl_me
	return -np.inf


#~ @profile
def lnprob(theta):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta)
	

########################################################################
########################################################################
# WHERE EMCEE IS CALLED

ite = 10000 # number of steps
nwalkers = 100 # number of chains
ndim = 5  # number of parameters to vary

# chains need to be initialized, can be whatever
pos = np.random.uniform(low=[Age -0.1, metal-0.1, distance-2000, Abs-0.1, afe_init-0.1], high=[Age +0.1, metal+0.1, distance+2000, Abs+0.1, afe_init+0.1],
size=(nwalkers, ndim))

# ~sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #single thread
with Pool() as pool:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

	for i, (results) in enumerate(zip(sampler.sample(pos, iterations=ite))):
		print(i)
		if (i+1) % 200 == 0:
			ind = int((i+1)/1)
	# 		with open('test2_'+str(clus_nb)+'_'+str(model)+'.txt', 'a+') as fid_file:
			print("first phase is at {0:.1f}%\n".format(100 * float(i) /ite))
	# 		fid_file.close()

			
			print(sampler.acceptance_fraction)
			print(np.mean(sampler.acceptance_fraction))
			
				
		
			samples = sampler.chain[:,:i, :].reshape((-1, ndim))
			#~ samples2 = sampler2.chain[:,:, :].reshape((-1, ndim))
			#~ samples3 = sampler3.chain[:,:, :].reshape((-1, ndim))

			# if you want 16th, 50th and 84th percentile for each parameters (Gaussian)
			p1, p2, p3, p4, p5 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84], axis=0)))


		pass








########################################################################
########################################################################


# if __name__ == '__main__':
# 	main()

#~ samp

