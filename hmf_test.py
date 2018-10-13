#~ #----------------------------------------------------------------
#~ #----------Tinker, Crocce param bias ----------------------------
#~ #----------------------------------------------------------------
if mv == 0:
	#~ #compute tinker stuff
	#limM = [5e11,1e12,3e12,1e13, 3.2e15]
	limM = [4.2e11,1e12,3e12,1e13, 5e15]
	loglim = [ 11.623, 12., 12.477, 13., 15.698]
	camb2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cc_z='+str(z[j])+'00.txt')
	kcamb2 = camb2[:,0]
	Pcamb2 = camb2[:,1]

	#~ #### get the mass function from simulation
	massf = np. loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf_z='+str(z[j])+'.txt')
	m_middle = massf[:,10]
	dm = massf[:,11]
	hmf_temp = np.zeros((len(m_middle),10))
	for i in xrange(0,10):
		hmf_temp[:,i]= massf[:,i]
	hmf = np.mean(hmf_temp[:,0:11], axis=1)
	
	#~ dndM=MFL.Tinker_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
	
	#~ dndMbis=MFL.Crocce_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
	
	
	#~ bt=np.empty(len(m_middle),dtype=np.float64)
	#~ bst=np.empty(len(m_middle),dtype=np.float64)
	#~ bsmt=np.empty(len(m_middle),dtype=np.float64)
	#~ for i in range(len(m_middle)):
		#~ bt[i]=bias(kcamb2,Pcamb2,Omega_m,m_middle[i],'Tinker')
		#~ bst[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Crocce')
		#~ bsmt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'SMT01')
	
	
	#~ dndM = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt')
	#~ diff = hmf/dndM
	#~ bin1 = np.where(m_middle > 5e11 )[0]
	#~ bin2 = np.where(m_middle > 1e12 )[0]
	#~ bin3 = np.where(m_middle > 3e12 )[0]
	#~ bin4 = np.where(m_middle > 1e13 )[0]


	#### get tinker and crocce hmf
	#Tb1 = halo_bias('Tinker', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#Tb2 = halo_bias('Tinker', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#Tb3 = halo_bias('Tinker', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#Tb4 = halo_bias('Tinker', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )

	#~ Cb1 = halo_bias('Crocce', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ Cb2 = halo_bias('Crocce', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ Cb3 = halo_bias('Crocce', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ Cb4 = halo_bias('Crocce', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	
	#~ with open('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (dndM[m]))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (dndMbis[m]))
	#~ fid_file.close()

	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (bt[m]))
	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (bst[m]))
	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (bsmt[m]))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/bcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k[index_k], bias1[index_k], bias2[index_k], bias3[index_k], bias4[index_k]))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/LS2_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Cb1,Cb2, Cb3, Cb4))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Tb1,Tb2, Tb3, Tb4))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.0eV/large_scale/ccl_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (lb1,lb2, lb3, lb4))
	#~ fid_file.close()

	Tb0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt')
	Tb1 = Tb0ev[0]
	Tb2 = Tb0ev[1]
	Tb3 = Tb0ev[2]
	Tb4 = Tb0ev[3]

	ccl00 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/ccl_z='+str(z[j])+'_.txt')
	lb1 = ccl00[0]
	lb2 = ccl00[1]
	lb3 = ccl00[2]
	lb4 = ccl00[3]


	dndM = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt')
	dndMbis = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt')
	
	
	### Simu hmf #############################################
	bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
	bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
	bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
	bin1 = np.where(m_middle > 5e11 )[0]
	bin2 = np.where(m_middle > 1e12 )[0]
	bin3 = np.where(m_middle > 3e12 )[0]
	bin4 = np.where(m_middle > 1e13 )[0]
	#------------------------------
	bias_eff0_t1=np.sum(hmf[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*hmf[bin1])
	bias_eff0_t2=np.sum(hmf[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*hmf[bin2])
	bias_eff0_t3=np.sum(hmf[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*hmf[bin3])
	bias_eff0_t4=np.sum(hmf[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*hmf[bin4])
	#------------------------------
	bias_eff0_st1=np.sum(hmf[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*hmf[bin1])
	bias_eff0_st2=np.sum(hmf[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*hmf[bin2])
	bias_eff0_st3=np.sum(hmf[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*hmf[bin3])
	bias_eff0_st4=np.sum(hmf[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*hmf[bin4])
	#------------------------------
	bias_eff0_smt1=np.sum(hmf[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*hmf[bin1])
	bias_eff0_smt2=np.sum(hmf[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*hmf[bin2])
	bias_eff0_smt3=np.sum(hmf[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*hmf[bin3])
	bias_eff0_smt4=np.sum(hmf[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*hmf[bin4])
	
	### Crocce hmf ############################################
	#~ bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
	#~ bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
	#~ bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
	#~ #------------------------------
	#~ Bias_eff0_t1=np.sum(dndM0bis[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
	#~ Bias_eff0_t2=np.sum(dndM0bis[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
	#~ Bias_eff0_t3=np.sum(dndM0bis[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
	#~ Bias_eff0_t4=np.sum(dndM0bis[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
	#------------------------------
	#~ Bias_eff0_st1=np.sum(dndM0bis[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
	#~ Bias_eff0_st2=np.sum(dndM0bis[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
	#~ Bias_eff0_st3=np.sum(dndM0bis[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
	#~ Bias_eff0_st4=np.sum(dndM0bis[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
	#~ #------------------------------
	#~ Bias_eff0_smt1=np.sum(dndM0bis[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
	#~ Bias_eff0_smt2=np.sum(dndM0bis[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
	#~ Bias_eff0_smt3=np.sum(dndM0bis[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
	#~ Bias_eff0_smt4=np.sum(dndM0bis[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
	
	#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff0_t1,Bias_eff0_t2, Bias_eff0_t3, Bias_eff0_t4))
	#~ fid_file.close()
	#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff_t1,Bias_eff_t2, Bias_eff_t3, Bias_eff_t4))
	#~ fid_file.close()
	
#######################################################################################################################
########################################################################################################################
########################################################################################################################

if mv == 0.15:

	#----------------------------------------------------------------
	#----------Tinker, Crocce param bias ----------------------------
	#----------------------------------------------------------------

	#~ #compute tinker stuff
	##~ limM = [5e11,1e12,3e12,1e13, 3.2e15]
	limM = [4.2e11,1e12,3e12,1e13, 5e15]
	loglim = [ 11.623, 12., 12.477, 13., 15.698]
	camb2 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/CAMB/Pk_cb_z='+str(z[j])+'00.txt')
	kcamb2 = camb2[:,0]
	Pcamb2 = camb2[:,1]

	#~ #### get the mass function from simulation
	massf = np. loadtxt('/home/david/codes/Paco/data2/0.15eV/hmf_z='+str(z[j])+'.txt')
	m_middle = massf[:,10]
	dm = massf[:,11]
	hmf_temp = np.zeros((len(m_middle),10))
	for i in xrange(0,10):
		hmf_temp[:,i]= massf[:,i]
	hmf = np.mean(hmf_temp[:,0:11], axis=1)
	################################################################
	
	#~ dndM=MFL.Tinker_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
	
	#~ dndMbis=MFL.Crocce_mass_function(kcamb2,Pcamb2,Omega_m,z[j],limM[0],limM[4],len(m_middle),Masses=m_middle)[1]
	
	
	#~ bt=np.empty(len(m_middle),dtype=np.float64)
	#~ bst=np.empty(len(m_middle),dtype=np.float64)
	#~ bsmt=np.empty(len(m_middle),dtype=np.float64)
	#~ for i in range(len(m_middle)):
		#~ bt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Tinker')
		#~ bst[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'Crocce')
		#~ bsmt[i]=bias(kcamb,Pcamb,Omega_m,m_middle[i],'SMT01')
	
	
	
	#~ #### get tinker and crocce hmf
	#~ def linb(hmf,lim1, lim2):
		#~ halo_mass = np.logspace(lim1, lim2,100) #a vector of halo masses between 1e10 and 1e15
		#~ M_middle=10**(0.5*(np.log10(halo_mass[1:])+np.log10(halo_mass[:-1]))) #center of the bin
		#~ deltaM=halo_mass[1:]-halo_mass[:-1]
		#~ a = 1./(1+z[j])
		#~ hmf = np.interp(M_middle, m_middle,hmf)
		#~ #---------------------------------------
		#~ b=np.empty(99,dtype=np.float64)
		#~ for i in range(99):
			#~ b[i] = bias(klin,Plin,Omega_m,M_middle[i],'Tinker')
		#~ bias_eff=np.sum(hmf*deltaM*b)/np.sum(hmf*deltaM)
		#~ return bias_eff

	#~ lb1 = linb(hmf,loglim[0],loglim[4])
	#~ lb2 = linb(hmf,loglim[1],loglim[4])
	#~ lb3 = linb(hmf,loglim[2],loglim[4])
	#~ lb4 = linb(hmf,loglim[3],loglim[4])
	#~ print lb1, lb2, lb3, lb4

	#### get tinker and crocce hmf
	#Tb1 = halo_bias('Tinker', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#Tb2 = halo_bias('Tinker', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#Tb3 = halo_bias('Tinker', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#Tb4 = halo_bias('Tinker', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	
	#~ #Tb1 = halo_bias('Tinker', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ #Tb2 = halo_bias('Tinker', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ #Tb3 = halo_bias('Tinker', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ #Tb4 = halo_bias('Tinker', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	
	#~ Cb1 = halo_bias('Crocce', z[j], limM[0],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ Cb2 = halo_bias('Crocce', z[j], limM[1],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ Cb3 = halo_bias('Crocce', z[j], limM[2],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	#~ Cb4 = halo_bias('Crocce', z[j], limM[3],limM[4], cname,Omega_c, Omega_b, do_DM=True )
	
	#~ with open('/home/david/codes/Paco/data2/0.15eV/chmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (dndMbis[m]))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.15eV/thmf_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (dndM[m]))
	#~ fid_file.close()

	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/tlb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (bt[m]))
			
	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/clb1_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (bst[m]))
			
	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/clb2_z='+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for m in xrange(0, len(m_middle)):
			#~ fid_file.write('%.8g\n' % (bsmt[m]))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/LS2_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Cb1,Cb2, Cb3, Cb4))
	#~ fid_file.close()
	
	
	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Tb1,Tb2, Tb3, Tb4))
	#~ fid_file.close()

	
	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/ccl_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (lb1,lb2, lb3, lb4))
	#~ fid_file.close()
	
	#~ with open('/home/david/codes/Paco/data2/0.15eV/large_scale/bcc_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(k)):
			#~ fid_file.write('%.8g %.8g %.8g %.8g %.8g\n' % ( k[index_k], bias1[index_k], bias2[index_k], bias3[index_k], bias4[index_k]))
	#~ fid_file.close()
	################################################################
	
	dndM = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/thmf_z='+str(z[j])+'.txt')
	dndMbis = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/chmf_z='+str(z[j])+'.txt')
	diff = hmf/dndM
	bin1 = np.where(m_middle > 5e11 )[0]
	bin2 = np.where(m_middle > 1e12 )[0]
	bin3 = np.where(m_middle > 3e12 )[0]
	bin4 = np.where(m_middle > 1e13 )[0]
	
	Tb0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt')
	tb1 = Tb0ev[0]
	tb2 = Tb0ev[1]
	tb3 = Tb0ev[2]
	tb4 = Tb0ev[3]
	
	Tb15ev = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/LS_z='+str(z[j])+'_.txt')
	Tb1 = Tb15ev[0]
	Tb2 = Tb15ev[1]
	Tb3 = Tb15ev[2]
	Tb4 = Tb15ev[3]
	
	Cb0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/LS2_z='+str(z[j])+'_.txt')
	cb1 = Cb0ev[0]
	cb2 = Cb0ev[1]
	cb3 = Cb0ev[2]
	cb4 = Cb0ev[3]
	
	Cb15ev = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/LS2_z='+str(z[j])+'_.txt')
	Cb1 = Cb15ev[0]
	Cb2 = Cb15ev[1]
	Cb3 = Cb15ev[2]
	Cb4 = Cb15ev[3]
	
	ccl00 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/ccl_z='+str(z[j])+'_.txt')
	Lb1 = ccl00[0]
	Lb2 = ccl00[1]
	Lb3 = ccl00[2]
	Lb4 = ccl00[3]
	
	ccl15 = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/ccl_z='+str(z[j])+'_.txt')
	lb1 = ccl15[0]
	lb2 = ccl15[1]
	lb3 = ccl15[2]
	lb4 = ccl15[3]
	#~ print lb1, lb2, lb3, lb4
	
	
	
	bias_0ev = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/bcc_z='+str(z[j])+'_.txt')
	bias1_0ev = bias_0ev[:,1]
	bias2_0ev = bias_0ev[:,2]
	bias3_0ev = bias_0ev[:,3]
	bias4_0ev = bias_0ev[:,4]

	als = np.where(k < 0.5)[0]


	dndM0bis = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/chmf_z='+str(z[j])+'.txt')
	dndM0 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/thmf_z='+str(z[j])+'.txt')
	massf0 = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/hmf/hmf_z='+str(z[j])+'.txt')
	hmf_temp0 = np.zeros((len(m_middle),10))
	for i in xrange(0,10):
		hmf_temp0[:,i]= massf0[:,i]
	hmf0 = np.mean(hmf_temp0[:,0:11], axis=1)
	
	############################################### WITH HMF SIMU
	#### 0.0 eV ################
	bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
	bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
	bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
	#------------------------------
	bias_eff0_t1=np.sum(hmf0[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*hmf0[bin1])
	bias_eff0_t2=np.sum(hmf0[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*hmf0[bin2])
	bias_eff0_t3=np.sum(hmf0[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*hmf0[bin3])
	bias_eff0_t4=np.sum(hmf0[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*hmf0[bin4])
	#------------------------------
	bias_eff0_st1=np.sum(hmf0[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*hmf0[bin1])
	bias_eff0_st2=np.sum(hmf0[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*hmf0[bin2])
	bias_eff0_st3=np.sum(hmf0[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*hmf0[bin3])
	bias_eff0_st4=np.sum(hmf0[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*hmf0[bin4])
	#------------------------------
	bias_eff0_smt1=np.sum(hmf0[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*hmf0[bin1])
	bias_eff0_smt2=np.sum(hmf0[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*hmf0[bin2])
	bias_eff0_smt3=np.sum(hmf0[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*hmf0[bin3])
	bias_eff0_smt4=np.sum(hmf0[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*hmf0[bin4])
	#### 0.15 eV ################
	bt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/tlb1_z='+str(z[j])+'.txt')
	bst = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb1_z='+str(z[j])+'.txt')
	bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb2_z='+str(z[j])+'.txt')
	#------------------------------
	bias_eff_t1=np.sum(hmf[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*hmf[bin1])
	bias_eff_t2=np.sum(hmf[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*hmf[bin2])
	bias_eff_t3=np.sum(hmf[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*hmf[bin3])
	bias_eff_t4=np.sum(hmf[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*hmf[bin4])
	#------------------------------
	bias_eff_st1=np.sum(hmf[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*hmf[bin1])
	bias_eff_st2=np.sum(hmf[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*hmf[bin2])
	bias_eff_st3=np.sum(hmf[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*hmf[bin3])
	bias_eff_st4=np.sum(hmf[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*hmf[bin4])
	#------------------------------
	bias_eff_smt1=np.sum(hmf[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*hmf[bin1])
	bias_eff_smt2=np.sum(hmf[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*hmf[bin2])
	bias_eff_smt3=np.sum(hmf[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*hmf[bin3])
	bias_eff_smt4=np.sum(hmf[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*hmf[bin4])
	#~ ############################################### WITH HMF CROCCE
	#~ #### 0.0 eV ################
	bt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/tlb1_z='+str(z[j])+'.txt')
	bst = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb1_z='+str(z[j])+'.txt')
	bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.0eV/large_scale/clb2_z='+str(z[j])+'.txt')
	#------------------------------
	Bias_eff0_t1=np.sum(dndM0bis[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
	Bias_eff0_t2=np.sum(dndM0bis[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
	Bias_eff0_t3=np.sum(dndM0bis[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
	Bias_eff0_t4=np.sum(dndM0bis[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
	#------------------------------
	Bias_eff0_st1=np.sum(dndM0bis[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
	Bias_eff0_st2=np.sum(dndM0bis[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
	Bias_eff0_st3=np.sum(dndM0bis[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
	Bias_eff0_st4=np.sum(dndM0bis[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
	#------------------------------
	Bias_eff0_smt1=np.sum(dndM0bis[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*dndM0bis[bin1])
	Bias_eff0_smt2=np.sum(dndM0bis[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*dndM0bis[bin2])
	Bias_eff0_smt3=np.sum(dndM0bis[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*dndM0bis[bin3])
	Bias_eff0_smt4=np.sum(dndM0bis[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*dndM0bis[bin4])
	
	#### 0.15 eV ################
	bt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/tlb1_z='+str(z[j])+'.txt')
	bst = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb1_z='+str(z[j])+'.txt')
	bsmt = np.loadtxt('/home/david/codes/Paco/data2/0.15eV/large_scale/clb2_z='+str(z[j])+'.txt')
	#------------------------------
	Bias_eff_t1=np.sum(dndMbis[bin1]*dm[bin1]*bt[bin1])/np.sum(dm[bin1]*dndMbis[bin1])
	Bias_eff_t2=np.sum(dndMbis[bin2]*dm[bin2]*bt[bin2])/np.sum(dm[bin2]*dndMbis[bin2])
	Bias_eff_t3=np.sum(dndMbis[bin3]*dm[bin3]*bt[bin3])/np.sum(dm[bin3]*dndMbis[bin3])
	Bias_eff_t4=np.sum(dndMbis[bin4]*dm[bin4]*bt[bin4])/np.sum(dm[bin4]*dndMbis[bin4])
	#------------------------------
	Bias_eff_st1=np.sum(dndMbis[bin1]*dm[bin1]*bst[bin1])/np.sum(dm[bin1]*dndMbis[bin1])
	Bias_eff_st2=np.sum(dndMbis[bin2]*dm[bin2]*bst[bin2])/np.sum(dm[bin2]*dndMbis[bin2])
	Bias_eff_st3=np.sum(dndMbis[bin3]*dm[bin3]*bst[bin3])/np.sum(dm[bin3]*dndMbis[bin3])
	Bias_eff_st4=np.sum(dndMbis[bin4]*dm[bin4]*bst[bin4])/np.sum(dm[bin4]*dndMbis[bin4])
	#------------------------------
	Bias_eff_smt1=np.sum(dndMbis[bin1]*dm[bin1]*bsmt[bin1])/np.sum(dm[bin1]*dndMbis[bin1])
	Bias_eff_smt2=np.sum(dndMbis[bin2]*dm[bin2]*bsmt[bin2])/np.sum(dm[bin2]*dndMbis[bin2])
	Bias_eff_smt3=np.sum(dndMbis[bin3]*dm[bin3]*bsmt[bin3])/np.sum(dm[bin3]*dndMbis[bin3])
	Bias_eff_smt4=np.sum(dndMbis[bin4]*dm[bin4]*bsmt[bin4])/np.sum(dm[bin4]*dndMbis[bin4])
	
	
	#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/0.0eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff0_t1,Bias_eff0_t2, Bias_eff0_t3, Bias_eff0_t4))
	#~ fid_file.close()
	#~ with open('/home/david/codes/montepython_public/BE_HaPPy/coefficients/'+str(Mnu)+'eV/large_scale/LS_z='+str(z[j])+'_.txt', 'w+') as fid_file:
		#~ fid_file.write('%.8g %.8g %.8g %.8g\n' % (Bias_eff_t1,Bias_eff_t2, Bias_eff_t3, Bias_eff_t4))
	#~ fid_file.close()
