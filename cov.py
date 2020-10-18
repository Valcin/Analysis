import numpy as np
import matplotlib 
import matplotlib.pyplot as plt




dir_name = '/home/DATA/hector/boss_reanalysis/patchy_mocks/lin/box/0001/Power_Spectrum_cmass_ngc_v5_Patchy_Om0.450_0001.txt'
with open(dir_name, 'r') as f:
	lines=f.readlines()[25:]
	lfile = len(lines)

mock_num = ["%04d" % x for x in xrange(1, 2049)]


exp_name =['cmass_ngc_v5_Patchy', 'cmass_sgc_v5_Patchy', 'lowz_ngc_v5_Patchy', 'lowz_sgc_v5_Patchy']

### compute the mean of the power spectrum for a given Om
#~ for index, ename in enumerate(exp_name):
	#~ if index == 0 or index == 1:
		#~ lfile = 40
	#~ else:
		#~ lfile = 50
	#~ mono_all_ps = np.zeros((lfile,len(mock_num)))
	#~ quadru_all_ps = np.zeros((lfile,len(mock_num)))
	#~ hexa_all_ps = np.zeros((lfile,len(mock_num)))
	
	#~ for y in xrange(150, 460,5):
		#~ om = y/1000.
		#~ print "%.3f" % om

		#~ for count, j in enumerate(mock_num):
			#~ dir_name = '/home/DATA/hector/boss_reanalysis/patchy_mocks/lin/box/'+str(j)+'/Power_Spectrum_'+ename+'_Om'+("%.3f" % om)+ '_'+str(j)+'.txt'
			#~ with open(dir_name, 'r') as f:
				#~ lines=f.readlines()[25:]
				
				#~ for count2,x in enumerate(lines):
					#~ mono_all_ps[count2,count] = (x.split(' ')[2])
					#~ quadru_all_ps[count2,count] = (x.split(' ')[3])
					#~ hexa_all_ps[count2,count] = (x.split(' ')[4])
			#~ f.close()
		#~ mono_mu_ps = np.mean(mono_all_ps,axis=1)
		#~ quadru_mu_ps = np.mean(quadru_all_ps,axis=1)
		#~ hexa_mu_ps = np.mean(hexa_all_ps,axis=1)

		#~ with open('mean/mean_mono_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(mono_mu_ps)):
				#~ fid_file.write('%.8g\n' % ( mono_mu_ps[index_k]))
		#~ fid_file.close()
		#~ with open('mean/mean_quadru_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(quadru_mu_ps)):
				#~ fid_file.write('%.8g\n' % ( quadru_mu_ps[index_k]))
		#~ fid_file.close()
		#~ with open('mean/mean_hexa_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(hexa_mu_ps)):
				#~ fid_file.write('%.8g\n' % ( hexa_mu_ps[index_k]))
		#~ fid_file.close()
		
#----------------------------------------------------------------------------	
#----------------------------------------------------------------------------	
#~ ### compute the std of the power spectrum for a given Om
#~ for index, ename in enumerate(exp_name):
	#~ if index == 0 or index == 1:
		#~ lfile = 40
	#~ else:
		#~ lfile = 50
	#~ mono_all_ps = np.zeros((lfile,len(mock_num)))
	#~ quadru_all_ps = np.zeros((lfile,len(mock_num)))
	#~ hexa_all_ps = np.zeros((lfile,len(mock_num)))
	
	#~ for y in xrange(150, 460,5):
		#~ om = y/1000.
		#~ print "%.3f" % om

		#~ for count, j in enumerate(mock_num):
			#~ dir_name = '/home/DATA/hector/boss_reanalysis/patchy_mocks/lin/box/'+str(j)+'/Power_Spectrum_'+ename+'_Om'+("%.3f" % om)+ '_'+str(j)+'.txt'
			#~ with open(dir_name, 'r') as f:
				#~ lines=f.readlines()[25:]
				
				#~ for count2,x in enumerate(lines):
					#~ mono_all_ps[count2,count] = (x.split(' ')[2])
					#~ quadru_all_ps[count2,count] = (x.split(' ')[3])
					#~ hexa_all_ps[count2,count] = (x.split(' ')[4])
			#~ f.close()
		#~ mono_mu_ps = np.std(mono_all_ps,axis=1, ddof = 1)
		#~ quadru_mu_ps = np.std(quadru_all_ps,axis=1, ddof = 1)
		#~ hexa_mu_ps = np.std(hexa_all_ps,axis=1, ddof = 1)

		#~ with open('std/std_mono_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(mono_mu_ps)):
				#~ fid_file.write('%.8g\n' % ( mono_mu_ps[index_k]))
		#~ fid_file.close()
		#~ with open('std/std_quadru_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(quadru_mu_ps)):
				#~ fid_file.write('%.8g\n' % ( quadru_mu_ps[index_k]))
		#~ fid_file.close()
		#~ with open('std/std_hexa_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			#~ for index_k in xrange(len(hexa_mu_ps)):
				#~ fid_file.write('%.8g\n' % ( hexa_mu_ps[index_k]))
		#~ fid_file.close()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
### Computation of the covariance matrix


#~ for index, ename in enumerate(exp_name):
	#~ if index == 0 or index == 1:
		#~ lfile = 40
	#~ else:
		#~ lfile = 50
		
	#~ mono_mock_ps = np.zeros((lfile,len(mock_num)))
	#~ quadru_mock_ps = np.zeros((lfile,len(mock_num)))
	#~ hexa_mock_ps = np.zeros((lfile,len(mock_num)))
	
	#~ print ename
	#~ for y in xrange(150, 460,5):
		#~ om = y/1000.
		#~ print "%.3f" % om
		
		#~ # read mean value
		#~ mono_mu_om = []
		#~ with open('/home/dvalcin/codes/data_analysis/mean/mean_mono_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as ffile:
			#~ flines=ffile.readlines()
			#~ for x in flines:
				#~ mono_mu_om.append(x)
		#~ ffile.close()
		#~ quadru_mu_om = []
		#~ with open('/home/dvalcin/codes/data_analysis/mean/mean_quadru_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as ffile:
			#~ flines=ffile.readlines()
			#~ for x in flines:
				#~ quadru_mu_om.append(x)
		#~ ffile.close()
		#~ hexa_mu_om = []
		#~ with open('/home/dvalcin/codes/data_analysis/mean/mean_hexa_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as ffile:
			#~ flines=ffile.readlines()
			#~ for x in flines:
				#~ hexa_mu_om.append(x)
		#~ ffile.close()
	
		#~ # compute the covariance
		#~ C = np.zeros((3*lfile, 3*lfile))
		#~ var1 = np.zeros((3*lfile, 3*lfile))
		#~ var2 = np.zeros((3*lfile, 3*lfile))
		#~ for count, j in enumerate(mock_num):
		
			#~ dir_name = '/home/DATA/hector/boss_reanalysis/patchy_mocks/lin/box/'+str(j)+'/Power_Spectrum_'+ename+'_Om'+("%.3f" % om)+ '_'+str(j)+'.txt'
			#~ with open(dir_name, 'r') as f:
				#~ lines=f.readlines()[25:]
				
				#~ for count2,x in enumerate(lines):
					#~ mono_mock_ps[count2, count] = (x.split(' ')[2])
					#~ quadru_mock_ps[count2, count] = (x.split(' ')[3])
					#~ hexa_mock_ps[count2, count] = (x.split(' ')[4])
			
				#~ for i1 in xrange(3*lfile):
					#~ for i2 in xrange(3*lfile):
						#~ print count, i1, i2
						#~ ### diagonal terms
						#~ if 0 <= i1 <= lfile -1 and 0 <= i2 <= lfile -1:
							#~ C[i1,i2] += (mono_mock_ps[i1,count] - float(mono_mu_om[i1])) * (mono_mock_ps[i2,count] - float(mono_mu_om[i2]))
						
						#~ elif lfile <= i1 <= 2*lfile -1 and lfile <= i2 <= 2*lfile -1:
							
							#~ C[i1,i2] += (quadru_mock_ps[i1-lfile,count] - float(quadru_mu_om[i1-lfile])) * (quadru_mock_ps[i2-lfile,count] - float(quadru_mu_om[i2-lfile]))
						#~ elif 2*lfile <= i1 <= 3*lfile -1 and 2*lfile <= i2 <= 3*lfile -1:
							
							#~ C[i1,i2] += (hexa_mock_ps[i1-2*lfile,count] - float(hexa_mu_om[i1-2*lfile])) * (hexa_mock_ps[i2-2*lfile,count] - float(hexa_mu_om[i2-2*lfile]))
						
						#~ ### cross terms
						#~ elif (0 <= i1 <= lfile -1 and lfile <= i2 <= 2*lfile -1):
							
							#~ C[i1,i2] += (mono_mock_ps[i1,count] - float(mono_mu_om[i1])) * (quadru_mock_ps[i2-lfile,count] - float(quadru_mu_om[i2-lfile]))
						#~ elif (0 <= i1 <= lfile -1 and 2*lfile <= i2 <= 3*lfile -1):
							
							#~ C[i1,i2] += (mono_mock_ps[i1,count] - float(mono_mu_om[i1])) * (hexa_mock_ps[i2-2*lfile,count] - float(hexa_mu_om[i2-2*lfile]))
						#~ #-------------------
						#~ elif (lfile <= i1 <= 2*lfile -1 and 0 <= i2 <= lfile -1):
							
							#~ C[i1,i2] += (quadru_mock_ps[i1-lfile,count] - float(quadru_mu_om[i1-lfile])) * (mono_mock_ps[i2,count] - float(mono_mu_om[i2]))
						#~ if (lfile <= i1 <= 2*lfile -1 and 2*lfile <= i2 <= 3*lfile -1):
							
							#~ C[i1,i2] += (quadru_mock_ps[i1-lfile,count] - float(quadru_mu_om[i1-lfile])) * (hexa_mock_ps[i2-2*lfile,count] - float(hexa_mu_om[i2-2*lfile]))
						#~ #-------------------
						#~ elif (2*lfile <= i1 <= 3*lfile -1 and 0 <= i2 <= lfile -1):
							
							#~ C[i1,i2] += (hexa_mock_ps[i1-2*lfile,count] - float(hexa_mu_om[i1-2*lfile])) * (mono_mock_ps[i2,count] - float(mono_mu_om[i2]))
						#~ elif (2*lfile <= i1 <= 3*lfile -1 and lfile <= i2 <= 2*lfile -1):
							
							#~ C[i1,i2] += (hexa_mock_ps[i1-2*lfile,count] - float(hexa_mu_om[i1-2*lfile])) * (quadru_mock_ps[i2-lfile,count] - float(quadru_mu_om[i2-lfile]))
				
			
			#~ f.close()
		#~ C /= float(len(mock_num) - 1 )
		#~ print C, np.shape(C)

			

		#~ with open('/home/dvalcin/codes/data_analysis/covariance/covariance_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'w+') as fid_file:
			 #~ for line in C:
				#~ fid_file.write(" ".join(str(elem) for elem in line) + "\n")
		#~ fid_file.close()
		
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
### plot the different covariant matrix

ename = exp_name[0]
om = 0.150
	
Cov = np.zeros((3*lfile, 3*lfile))

with open('/home/dvalcin/codes/data_analysis/covariance/covariance_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as fid_file:
	flines=fid_file.readlines()
	for ind, x in enumerate(flines):
		for ii in xrange(3*lfile):
			Cov[ind,ii] = x.split(' ')[ii]
fid_file.close()

	
# read std value
mono_std_om = []
with open('/home/dvalcin/codes/data_analysis/std/std_mono_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as ffile:
	flines=ffile.readlines()
	for x in flines:
		mono_std_om.append(x)
ffile.close()
quadru_std_om = []
with open('/home/dvalcin/codes/data_analysis/std/std_quadru_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as ffile:
	flines=ffile.readlines()
	for x in flines:
		quadru_std_om.append(x)
ffile.close()
hexa_std_om = []
with open('/home/dvalcin/codes/data_analysis/std/std_hexa_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as ffile:
	flines=ffile.readlines()
	for x in flines:
		hexa_std_om.append(x)
ffile.close()


with open('/home/dvalcin/codes/data_analysis/covariance/covariance_ps_'+ename+'_'+("%.3f" % om)+'.txt', 'r') as fid_file:
	flines=fid_file.readlines()
	for ind, x in enumerate(flines):
		for ii in xrange(3*lfile):
			if 0 <= ind <= lfile -1 and 0 <= ii <= lfile -1:
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(mono_std_om[ind]) * float(mono_std_om[ii]))
				
			elif lfile <= ind <= 2*lfile -1 and lfile <= ii <= 2*lfile -1:
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(quadru_std_om[ind - lfile]) * float(quadru_std_om[ii - lfile]))
				
			elif 2*lfile <= ind <= 3*lfile -1 and 2*lfile <= ii <= 3*lfile -1:
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(hexa_std_om[ind - 2*lfile]) * float(hexa_std_om[ii - 2*lfile]))
				
			#~ ### cross terms
			elif (0 <= ind <= lfile -1 and lfile <= ii <= 2*lfile -1):
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(mono_std_om[ind]) * float(quadru_std_om[ii - lfile]))
				
			elif (0 <= ind <= lfile -1 and 2*lfile <= ii <= 3*lfile -1):
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(mono_std_om[ind]) * float(hexa_std_om[ii - 2*lfile]))
				
			#~ #-------------------
			elif (lfile <= ind <= 2*lfile -1 and 0 <= ii <= lfile -1):
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(quadru_std_om[ind - lfile]) * float(mono_std_om[ii]))
				
			if (lfile <= ind <= 2*lfile -1 and 2*lfile <= ii <= 3*lfile -1):
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(quadru_std_om[ind - lfile]) * float(hexa_std_om[ii -2*lfile]))
	
			#~ #-------------------
			elif (2*lfile <= ind <= 3*lfile -1 and 0 <= ii <= lfile -1):
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(hexa_std_om[ind - 2*lfile]) * float(mono_std_om[ii]))

			elif (2*lfile <= ind <= 3*lfile -1 and lfile <= ii <= 2*lfile -1):
				Cov[ind,ii] = float(x.split(' ')[ii]) / (float(hexa_std_om[ind - 2*lfile]) * float(quadru_std_om[ii - lfile]))
				
fid_file.close()

plt.imshow(Cov, origin='lower',norm=matplotlib.colors.Normalize(vmin=0, vmax=0.5));
#~ plt.imshow(Cov, origin='lower');
plt.colorbar()
plt.show()
