from montepython.likelihood_class import Likelihood
from classy import Class
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import csv

class NCDM_bias(Likelihood):

     def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

	# require the power spectra and transfer functions from class
        self.need_cosmo_arguments(data, {'output': 'mPk mTk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})

        
	#### check if the non linear power spectrum is needed
	try:
            self.use_halofit
        except:
            self.use_halofit = False

        if self.use_halofit:
            self.need_cosmo_arguments(data, {'non linear': 'halofit'})
    
	#### Store the selected redshifts in a array and deduce its length for the loops
        red = np.array(self.redshift)
	self.znumber = red.size 
	self.z = np.zeros(self.znumber,'float64') #because len() and size only work for znumber >1
	self.z[:] = red


	#### check if the number of nuisance parameters of the bcc_fit formulae in the param file 
	#### corresponds to the number of redshifts declared in the data file
	count = 0
	keys = data.mcmc_parameters.keys()
	
	
	for zp in xrange(self.znumber): 
		for i in range(1,5):
			if 'b'+str(i)+'_z'+str(zp) in keys:
				count += 1		
		
	if count != 4*self.znumber:
		raise ValueError('You forgot to declare '+ str(4*self.znumber - count)+' coefficients') 
	

	#### If the file exists, initialize the fiducial values, the spectrum will
        #### be read first, with k size equal to 119 (FIND A WAY TO AUTOMIZE THE K VALUES) and znumber the number of redshifts. 
        self.fid_values_exist = False
        self.pk_nl_fid = np.zeros((119, self.znumber), 'float64')
	self.pk_lin_fid = np.zeros((119, self.znumber), 'float64')
        
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
        if os.path.exists(fid_file_path):
            self.fid_values_exist = True
            with open(fid_file_path, 'r') as fid_file:
                line = fid_file.readline()
                while line.find('#') != -1:
                    line = fid_file.readline()
                while (line.find('\n') != -1 and len(line) == 1):
                    line = fid_file.readline()
                for index_k in xrange(119):
                    for index_z in xrange(self.znumber):
                        self.pk_nl_fid[index_k,index_z] = float(line)
                        line = fid_file.readline()
		for index_k in xrange(119):
                    for index_z in xrange(self.znumber):
                        self.pk_lin_fid[index_k,index_z] = float(line)
                        line = fid_file.readline()
               
		
     	
        #### Else the file will be created in the loglkl() function.
        return

####################################################################################################
####################################################################################################

     def loglkl(self, cosmo, data):
	
	#### Since get_transfer only gives transfer functions for specific k
	#### and the cosmo.pk computes pk on given k, we must extract the k values from the get_transfer list 
	#### so they coincide. 
	#### For this "cosmo" they are 121 k values but we need to remove some terms of the arrays 
	#### because they are out of bounds in in classy.Class.pk. CHECK WHY ???? (cf. euclid_lensing __init__ file)
	self.kget = cosmo.get_transfer(self.z[0])
	self.k_dot = self.kget.get('k (h/Mpc)')[0:119]
	self.d_b = np.zeros((len(self.k_dot), self.znumber), 'float64')
	self.d_cdm = np.zeros((len(self.k_dot), self.znumber), 'float64')
	self.d_tot = np.zeros((len(self.k_dot), self.znumber), 'float64')
	for i in xrange(self.znumber):
		self.transfer = cosmo.get_transfer(self.z[i])
		self.d_b[:,i] = self.transfer.get('d_b')[0:119]
		self.d_cdm[:,i] = self.transfer.get('d_cdm')[0:119]
		self.d_tot[:,i] = self.transfer.get('d_tot')[0:119]
	
	

	#### If the fiducial model does not exists, recover the power spectrum and
        #### store it, then exit. The pk function in classy will give the pk for a given k and z  		
	#### and will be non linear if requested to Class, linear otherwise.
        if self.fid_values_exist is False:
            pk = np.zeros((len(self.k_dot), self.znumber), 'float64')
	    pk_lin = np.zeros((len(self.k_dot), self.znumber), 'float64')
            
            fid_file_path = os.path.join(
                self.data_directory, self.fiducial_file)
            with open(fid_file_path, 'w') as fid_file:
                fid_file.write('# Fiducial parameters')
                for key, value in data.mcmc_parameters.iteritems():
                    fid_file.write(', %s = %.5g' % (
                        key, value['current']*value['scale']))
                fid_file.write('\n')
                for index_k in xrange(len(self.k_dot)):
                    for index_z in xrange(self.znumber):
                        pk[index_k,index_z] = cosmo.pk(self.k_dot[index_k], self.z[index_z])
                        fid_file.write('%.8g\n' % pk[index_k,index_z])
		for index_k in xrange(len(self.k_dot)):
                    for index_z in xrange(self.znumber):
                        pk_lin[index_k,index_z] = cosmo.pk_lin(self.k_dot[index_k], self.z[index_z])
                        fid_file.write('%.8g\n' % pk_lin[index_k,index_z])

			
#		for index_k in xrange(len(k_dot)):
#                for index_z in xrange(self.znumber):
#                    transfer = cosmo.get_transfer(self.redshift)
#		    keys = sorted(transfer.keys())
#		    writer = csv.writer(fid_file, delimiter = "\t")
#   		    writer.writerow(keys)
#   		    writer.writerows(zip(*[transfer[key] for key in keys]))
##		    fid_file.write("{}\n".format(transfer))
            print '\n'
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

	
	#### import Omega_b and Omega_cdm from class. Remember to add Omega_cdm in classy and recompile after
        Omega_b = cosmo.Omega_b()
	Omega_cdm = cosmo.Omega_cdm()

	#### define the CDM + baryons transfer function from Raccanelli et al 2017.
	T_cb = np.zeros((len(self.k_dot), self.znumber), 'float64')
	T_cb = (Omega_cdm * self.d_cdm + Omega_b * self.d_b)/(Omega_cdm + Omega_b)
	
	
	#### compute Pcc the baryons + CDM power spectrum from Raccanelli et al 2017.
	self.Pcc = np.zeros((len(self.k_dot), self.znumber), 'float64')
	for i in xrange(self.znumber):
		self.Pcc[:,i] = self.pk_lin_fid[:,i] * ((T_cb[:,i]/self.d_tot[:,i])**2)
	
	#### compute the total matter bias from the ratio of halo power spectrum and matter power spectrum and the bias bcc from 
	#### the ratio of halo power spectrum and the CDM + baryons power
	bias = np.sqrt(self.pk_nl_fid/self.pk_lin_fid)
	bcc = np.sqrt(self.pk_nl_fid/self.Pcc)	

####---------------------------------------------------------------------------------------
####---------------------------------------------------------------------------------------

	#### compute the bias using the fit formula from Raccanelli et al 2017.
	bcc_fit = np.zeros((len(self.k_dot), self.znumber), 'float64')
	for i in xrange(self.znumber):
        	bcc_fit[:,i] = (data.mcmc_parameters['b1_z'+str(i)]['current'] * data.mcmc_parameters['b1_z'+str(i)]['scale']) + (data.mcmc_parameters['b2_z'+str(i)]['current'] * data.mcmc_parameters['b2_z'+str(i)]['scale']) * self.k_dot**2 + (data.mcmc_parameters['b3_z'+str(i)]['current'] + data.mcmc_parameters['b3_z'+str(i)]['scale']) * self.k_dot**3 + (data.mcmc_parameters['b4_z'+str(i)]['current'] + data.mcmc_parameters['b4_z'+str(i)]['scale']) * self.k_dot**4

	
	#### Since Raccanelli et al. 2017 are using a purely phenomenological fit over a reduced range of scales
	#### the user is asked to define a kmax value in accordance with the 3 k-scale models in the article. 
	#### This part of the code creates a k matrix with the values of kmax for each possible 
	#### redshift 0,1,2 or 3.(Appox. values from Fig.4) and then select the k model chosen in the data file
	k = [[0.15,0.15,0.15,0.15],[0.12,0.2,0.2,0.2],[0.16,0.25,0.35,0.4]]
	if self.kmode == 1:
		kmax = k[0]
	elif self.kmode == 2:
		kmax = k[1]
	elif self.kmode == 3:
		kmax = k[2]
	
	#### get the index of the kmax value in the k.dot array
	#### divide by mean of first element to rescale
	klim = np.zeros(self.znumber, 'int')
	for i in xrange(self.znumber):
		lim = np.where(self.k_dot <= kmax[i])[0]
		klim[i] = np.amax(lim)
		bcc_fit_mean = np.mean(bcc_fit[0:20,i])
		bcc_fit[:,i] /= bcc_fit_mean

	
	

	
####------------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------------
	
	
	#### Plot different parameters Pk, bias, bias ratio to check settings.
	#### The first shows linear and non linear power spectrum 
	#### The second shows the scale dependence of the bias induce by massive neutrinos. 
	#### The third shows the fit of bcc compared to the value computed from class
	
#	plt.figure()
#	plt.plot(self.k_dot,self.pk_nl_fid[:,0])
#	plt.plot(self.k_dot,self.pk_lin_fid[:,0])
#	plt.plot(self.k_dot,self.pk_nl_fid[:,1])
#	plt.plot(self.k_dot,self.pk_lin_fid[:,1])
#	plt.plot(self.k_dot,self.pk_nl_fid[:,2])
#	plt.plot(self.k_dot,self.pk_lin_fid[:,2])
##	plt.plot(self.k_dot,Pcc)
#	plt.xscale('log')
#	plt.xlabel('k')
#	plt.yscale('log')
#	plt.ylabel('P(k)')
#	plt.show()  

#	plt.figure()
#	plt.plot(self.k_dot,bias[:,0]/bcc[:,0])
#	plt.plot(self.k_dot,bias[:,1]/bcc[:,1])
##	plt.plot(self.k_dot,bias[:,2]/bcc[:,2])
##	plt.plot(self.k_dot,bias[:,3]/bcc[:,3])
#	plt.xscale('log')
#	plt.xlim(1e-2,1)
#	plt.xlabel('k')
#	#plt.yscale('log')
#	#plt.ylim(0,10)
#	plt.ylabel('bias')
#	plt.show()

	
	plt.figure()
#	plt.plot(self.k_dot[0:klim[0]+1],bias[0:klim[0]+1,0], color='b')
#	plt.plot(self.k_dot[0:klim[0]+1],bcc[0:klim[0]+1,0], color='r')
#	plt.scatter(self.k_dot[0:klim[0]+1],bcc_fit[0:klim[0]+1,0], color='g', marker='.')
##	plt.plot(self.k_dot[0:klim[1]+1],bias[0:klim[1]+1,1], color='b')
#	plt.plot(self.k_dot[0:klim[1]+1],bcc[0:klim[1]+1,1], color='r')
#	plt.scatter(self.k_dot[0:klim[1]+1],bcc_fit[0:klim[1]+1,1], color='g',  marker='.')
##	plt.plot(self.k_dot[0:klim[2]+1],bias[0:klim[2]+1,2], color='b')
#	plt.plot(self.k_dot[0:klim[2]+1],bcc[0:klim[2]+1,2], color='r')
#	plt.scatter(self.k_dot[0:klim[2]+1],bcc_fit[0:klim[2]+1,2], color='g', marker='.')
##	plt.plot(self.k_dot[0:klim[3]+1],bias[0:klim[3]+1,3], color='b')
	plt.plot(self.k_dot[0:klim[3]+1],bcc[0:klim[3]+1,3], color='r')
	plt.scatter(self.k_dot[0:klim[3]+1],bcc_fit[0:klim[3]+1,3], color='g', marker='.')
	plt.xscale('log')
	plt.xlim(1e-3,0.5)
	plt.xlabel('k')
#	plt.yscale('log')
#	plt.ylim(0.9,1.4)
	plt.ylabel('bias')
	plt.show()  

	

####------------------------------------------------------------------------------------------------
####------------------------------------------------------------------------------------------------
	#### Compute the likelihood

	#### Loop over the redshifts and k. The expression "[0:a[i]+1]" means that chi2 is computed for k < kmax
	#### sigma is arbitrrily defined for test, must be defined in init part later
	chi2 = 0
	self.sigma = 0.01
	for index_z in xrange(self.znumber): 
        	for index_k in xrange(len(self.k_dot[0:klim[i]+1])): 
			chi2 += (bcc[index_k,index_z] - bcc_fit[index_k,index_z] ) ** 2 / (self.sigma**2)
	loglkl = -0.5 * chi2
        return loglkl



