
import numpy as np
import h5py
import math
import readsnap
import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


Mnu = 0.15

z = [0.0,0.5,1.0,2.0]


#~ bfile = np.loadtxt('3rdorder_0.0.txt')

#~ for i in range(0,30,15):
	#~ print i
	#~ ktest = bfile[i:i+15,0]
	#~ b1a = bfile[i:i+15,1]
	#~ b1b = bfile[i:i+15,2]
	#~ b1c = bfile[i:i+15,3]
	#~ b1d = bfile[i:i+15,4]
	#~ b3a = bfile[i:i+15,5]
	#~ b3b = bfile[i:i+15,6]
	#~ b3c = bfile[i:i+15,7]
	#~ b3d = bfile[i:i+15,8]


	#~ M1, =plt.plot(ktest, b3a/(b1a-1))
	#~ M2, =plt.plot(ktest, b3b/(b1b-1))
	#~ M3, =plt.plot(ktest, b3c/(b1c-1))
	#~ M4, =plt.plot(ktest, b3d/(b1d-1))
	#~ M1 =plt.scatter(ktest, b3a/(b1a-1))
	#~ M2 =plt.scatter(ktest, b3b/(b1b-1))
	#~ M3 =plt.scatter(ktest, b3c/(b1c-1))
	#~ M4 =plt.scatter(ktest, b3d/(b1d-1))
	#~ plt.plot(ktest, b1a)
	#~ plt.plot(ktest, b1b)
	#~ plt.plot(ktest, b1c)
	#~ plt.plot(ktest, b1d)
	#~ plt.plot(ktest, b3a)
	#~ plt.plot(ktest, b3b)
	#~ plt.plot(ktest, b3c)
	#~ plt.plot(ktest, b3d)
	#~ plt.figlegend( (M1,M2,M3,M4), ('$M_{1}$','$M_{2}$','$M_{3}$','$M_{4}$'),\
	#~ loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu), fontsize=14)
#~ plt.xscale('log')
#~ plt.ylabel(r'$b_{3nl}$ / $(b_{1} - 1)$ ', fontsize = 14)
#~ plt.xlabel(r'$k$ [h/Mpc] ', fontsize = 14)
#~ plt.axhline(32/315., color='k')
#~ plt.xlim(0.04,0.2)
#~ plt.ylim(0.8,0.9)
#~ plt.ylim(-2,2)
#~ plt.show()

#~ j=0
#~ cname = 'chi2a_z='+str(z[j])+'.txt'
#~ goodfit = np.loadtxt(cname) 
#~ kmax = goodfit[:,0]
#~ chipbis1 = goodfit[:,9]
#~ chipbis2 = goodfit[:,10]
#~ chipbis3 = goodfit[:,11]
#~ chipbis4 = goodfit[:,12]
#~ Chipbis = np.array([chipbis1,chipbis2,chipbis3,chipbis4])
#~ chipbis = np.mean(Chipbis, axis=0)
#~ echipbis = np.std(Chipbis, axis=0)
#~ plt.figure()
#~ plt.plot(kmax, chipbis1)
#~ plt.plot(kmax, chipbis2)
#~ plt.plot(kmax, chipbis3)
#~ plt.plot(kmax, chipbis4)
#~ plt.xscale('log')
#~ plt.xlim(0.04,0.2)
#~ plt.ylim(0,10)
#~ plt.show()
#~ kill
########################################################################
############# 	0.0 eV Masseless neutrino 
########################################################################
for j in xrange(0,len(z)):
########################################################################
########################################################################
	####################################################################
	##### scale factor 

	red = ['0.0','0.5','1.0','2.0']
	ind = red.index(str(z[j]))
	#~ fz = [0.524,0.759,0.875,0.958]
	Dz = [ 1.,0.77,0.61,0.42]
	print 'For redshift z = ' + str(z[j])

	if Mnu == 0.0:
		cname1 = 'chi2a_z='+str(z[j])+'.txt' # for massless neutrinos
	elif Mnu == 0.15:
		cname1 = 'chi2b_z='+str(z[j])+'.txt' # for massive neutrinos
	cname2 = 'chi2rsd_z='+str(z[j])+'.txt'

	goodfit1 = np.loadtxt(cname1) 
	kmax = goodfit1[:,0]
	nbpoints = goodfit1[:,17]
	#~ print nbpoints
	#--------------------
	chipl1 = goodfit1[:,1]/(nbpoints - 4.)
	chipl2 = goodfit1[:,2]/(nbpoints - 4.)
	chipl3 = goodfit1[:,3]/(nbpoints - 4.)
	chipl4 = goodfit1[:,4]/(nbpoints - 4.)
	chipl1_ = goodfit1[:,1]/( 4.)
	chipl2_ = goodfit1[:,2]/( 4.)
	chipl3_ = goodfit1[:,3]/( 4.)
	chipl4_ = goodfit1[:,4]/( 4.)
	Chipl = np.array([chipl1,chipl2,chipl3,chipl4])
	chipl = np.mean(Chipl, axis=0)
	echipl = np.std(Chipl, axis=0)
	

	#--------------------
	chipt1 = goodfit1[:,5]/(nbpoints - 3.)
	chipt2 = goodfit1[:,6]/(nbpoints - 3.)
	chipt3 = goodfit1[:,7]/(nbpoints - 3.)
	chipt4 = goodfit1[:,8]/(nbpoints - 3.)
	chipt1_ = goodfit1[:,5]/( 3.)
	chipt2_ = goodfit1[:,6]/( 3.)
	chipt3_ = goodfit1[:,7]/( 3.)
	chipt4_ = goodfit1[:,8]/( 3.)
	Chipt = np.array([chipt1,chipt2,chipt3,chipt4])
	chipt = np.mean(Chipt, axis=0)
	echipt = np.std(Chipt, axis=0)
	
	
	#--------------------
	chipbis1 = goodfit1[:,9]/(nbpoints - 4.)
	chipbis2 = goodfit1[:,10]/(nbpoints - 4.)
	chipbis3 = goodfit1[:,11]/(nbpoints - 4.)
	chipbis4 = goodfit1[:,12]/(nbpoints - 4.)
	chipbis1_ = goodfit1[:,9]/( 4.)
	chipbis2_ = goodfit1[:,10]/( 4.)
	chipbis3_ = goodfit1[:,11]/( 4.)
	chipbis4_ = goodfit1[:,12]/( 4.)
	Chipbis = np.array([chipbis1,chipbis2,chipbis3,chipbis4])
	chipbis = np.mean(Chipbis, axis=0)
	echipbis = np.std(Chipbis, axis=0)
	
	#--------------------
	chipter1 = goodfit1[:,13]/(nbpoints - 2.)
	chipter2 = goodfit1[:,14]/(nbpoints - 2.)
	chipter3 = goodfit1[:,15]/(nbpoints - 2.)
	chipter4 = goodfit1[:,16]/(nbpoints - 2.)
	chipter1_ = goodfit1[:,13]/( 2.)
	chipter2_ = goodfit1[:,14]/( 2.)
	chipter3_ = goodfit1[:,15]/( 2.)
	chipter4_ = goodfit1[:,16]/( 2.)
	Chipter = np.array([chipter1,chipter2,chipter3,chipter4])
	chipter = np.mean(Chipter, axis=0)
	echipter = np.std(Chipter, axis=0)
	
	
	value = 1.5
	idxpl = (np.abs(chipl4-value)).argmin()

	####### RSD
	goodfit2 = np.loadtxt(cname2) 
	kmax2 = goodfit2[:,0]
	nbpoints2 = goodfit2[:,5]
	#~ print nbpoints
	#--------------------
	chir1 = goodfit2[:,1]/(nbpoints2 - 1.)
	chir2 = goodfit2[:,2]/(nbpoints2 - 1.)
	chir3 = goodfit2[:,3]/(nbpoints2 - 1.)
	chir4 = goodfit2[:,4]/(nbpoints2 - 1.)
	
	#~ plt.figure()
	#~ plt.scatter(kmax,goodfit1[:,8]/(nbpoints - 3.), c='r', marker='.')
	#~ plt.scatter(kmax,goodfit1[:,12]/(nbpoints - 4.), c='b', marker='.')
	#~ plt.scatter(kmax,goodfit1[:,16]/(nbpoints - 2.), c='g', marker='.')
	#~ plt.plot(kmax,goodfit1[:,8]/(nbpoints - 3.), c='r')
	#~ plt.plot(kmax,goodfit1[:,12]/(nbpoints - 4.), c='b')
	#~ plt.plot(kmax,goodfit1[:,16]/(nbpoints - 2.), c='g')
	#~ plt.xscale('log')
	#~ plt.ylim(0,0.5)
	#~ plt.show()
	#~ kill
####################################################################
	#######--------- mean and std of bias and ps ratio ------------#####
	if j == z[0]:
		fig2 = plt.figure()
	J = j + 1
	
	if len(z) == 1:
		ax2 = fig2.add_subplot(1, len(z), J)
	elif len(z) == 2:
		ax2 = fig2.add_subplot(1, 2, J)
	elif len(z) > 2:
		ax2 = fig2.add_subplot(2, 2, J)
	#~ ######### pl residuals comparison #################
	#~ ax2.plot(kmax, chipl1, color='C0')
	#~ ax2.plot(kmax, chipl2, color='C0')
	#~ ax2.plot(kmax, chipl3, color='C0')
	P1, =ax2.plot(kmax, chipl4, color='C0', label='z = '+str(z[j]))
	#~ P1, =ax2.plot(kmax2, chir1, color='c', label='etns model')
	#---------------------------------
	#~ ax2.plot(kmax, chipt1, color='C1')
	#~ ax2.plot(kmax, chipt2, color='C1')
	#~ ax2.plot(kmax, chipt3, color='C1')
	P2, =ax2.plot(kmax, chipt4, color='C1')

	#---------------------------------
	#~ ax2.plot(kmax, chipbis1, color='C2')
	#~ ax2.plot(kmax, chipbis2, color='C2')
	#~ ax2.plot(kmax, chipbis3, color='C2')
	P3, =ax2.plot(kmax, chipbis4, color='C2')

	#---------------------------------
	#~ ax2.plot(kmax, chipter1, color='C3')
	#~ ax2.plot(kmax, chipter2, color='C3')
	#~ ax2.plot(kmax, chipter3, color='C3')
	P4, = ax2.plot(kmax, chipter4, color='C3')

	#---------------------------------
	plt.figlegend( (P1,P2, P3, P4), ('Polynomial','2nd order PT with free $b_{s}$',r'3nd order PT with free $b_{s}$,$b_{3nl}$',\
	r'3nd order PT with fixed $b_{s}$,$b_{3nl}$'), \
	######################################
	#~ loc = 'upper center', ncol=1, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+ ', for mass range M4', fontsize=14)
	loc = 'upper center', ncol=3, labelspacing=0., title =r' M$\nu$ = '+str(Mnu)+ ', for mass range M4', fontsize=12)
	ax2.legend(loc = 'upper left', fancybox=True, fontsize=14, handlelength=0, handletextpad=0)
	ax2.legend(loc = 'upper left', title = 'z = '+str(z[j]), fancybox=True, fontsize=14)
	plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	#~ plt.suptitle(r' M$\nu$ = '+str(Mnu)+ ', for mass range M1', fontsize = 14)
	ax2.set_xscale('log')
	#~ ax2.axhline(1.5, c='k')
	#~ ax2.set_yscale('log')
	ax2. set_xlim(0.04,0.5)
	ax2. set_ylim(0,3)
	if j == 0 :
		ax2.tick_params(bottom='off', labelbottom='off',labelleft=True)
		ax2.set_ylabel(r'$\chi^2/dof$', fontsize=16)
	if j == 1 :
		ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		ax2.set_ylabel(r'$\chi^2$/dof', fontsize=16)
		ax2.yaxis.set_label_position("right")
	if j == 2 :
		#ax.tick_params(labelleft=True)
		ax2.set_ylabel(r'$\chi^2/dof$', fontsize=16)
		ax2.set_xlabel('kmax [h/Mpc]', fontsize=16)
	if j == 3 :
		ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		ax2.set_xlabel('kmax [h/Mpc]', fontsize=16)
		ax2.set_ylabel(r'$\chi^2/dof$', fontsize=16)
		ax2.yaxis.set_label_position("right")
	if j == len(z) -1:
		plt.show()
	

#~ plt.figlegend( (B1,B2,B3), ('Power law','FAST-PT 2nd order','FAST-PT 3rd order'), \
#~ plt.figlegend( (B1,B3), ('Power law','FAST-PT 3rd order'), \
#~ loc = 'upper center', ncol=3, labelspacing=0. , title='z = 0.0')


kill



#########################################################################
#########################################################################
#~ def find_nearest(array,value):
    #~ idx = (np.abs(array-value)).argmin()
    #~ return idx, array[idx]
    

#~ kf = np.zeros((4,4))
#~ kpt = np.zeros((4,4))
#~ kptbis = np.zeros((4,4))



#~ chi2F1a = chi2F1a[~np.isnan(chi2F1a)]
#~ chi2F2a = chi2F2a[~np.isnan(chi2F2a)]
#~ chi2F3a = chi2F3a[~np.isnan(chi2F3a)]
#~ chi2F4a = chi2F4a[~np.isnan(chi2F4a)]
#~ chi2PT1a = chi2PT1a[~np.isnan(chi2PT1a)]
#~ chi2PT2a = chi2PT2a[~np.isnan(chi2PT2a)]
#~ chi2PT3a = chi2PT3a[~np.isnan(chi2PT3a)]
#~ chi2PT4a = chi2PT4a[~np.isnan(chi2PT4a)]
#~ chi2PTbis1a = chi2PTbis1a[~np.isnan(chi2PTbis1a)]
#~ chi2PTbis2a = chi2PTbis2a[~np.isnan(chi2PTbis2a)]
#~ chi2PTbis3a = chi2PTbis3a[~np.isnan(chi2PTbis3a)]
#~ chi2PTbis4a = chi2PTbis4a[~np.isnan(chi2PTbis4a)]

#~ chi2F1b = chi2F1b[~np.isnan(chi2F1b)]
#~ chi2F2b = chi2F2b[~np.isnan(chi2F2b)]
#~ chi2F3b = chi2F3b[~np.isnan(chi2F3b)]
#~ chi2F4b = chi2F4b[~np.isnan(chi2F4b)]
#~ chi2PT1b = chi2PT1b[~np.isnan(chi2PT1b)]
#~ chi2PT2b = chi2PT2b[~np.isnan(chi2PT2b)]
#~ chi2PT3b = chi2PT3b[~np.isnan(chi2PT3b)]
#~ chi2PT4b = chi2PT4b[~np.isnan(chi2PT4b)]
#~ chi2PTbis1b = chi2PTbis1b[~np.isnan(chi2PTbis1b)]
#~ chi2PTbis2b = chi2PTbis2b[~np.isnan(chi2PTbis2b)]
#~ chi2PTbis3b = chi2PTbis3b[~np.isnan(chi2PTbis3b)]
#~ chi2PTbis4b = chi2PTbis4b[~np.isnan(chi2PTbis4b)]

#~ chi2F1c = chi2F1c[~np.isnan(chi2F1c)]
#~ chi2F2c = chi2F2c[~np.isnan(chi2F2c)]
#~ chi2F3c = chi2F3c[~np.isnan(chi2F3c)]
#~ chi2F4c = chi2F4c[~np.isnan(chi2F4c)]
#~ chi2PT1c = chi2PT1c[~np.isnan(chi2PT1c)]
#~ chi2PT2c = chi2PT2c[~np.isnan(chi2PT2c)]
#~ chi2PT3c = chi2PT3c[~np.isnan(chi2PT3c)]
#~ chi2PT4c = chi2PT4c[~np.isnan(chi2PT4c)]
#~ chi2PTbis1c = chi2PTbis1c[~np.isnan(chi2PTbis1c)]
#~ chi2PTbis2c = chi2PTbis2c[~np.isnan(chi2PTbis2c)]
#~ chi2PTbis3c = chi2PTbis3c[~np.isnan(chi2PTbis3c)]
#~ chi2PTbis4c = chi2PTbis4c[~np.isnan(chi2PTbis4c)]

#~ chi2F1d = chi2F1d[~np.isnan(chi2F1d)]
#~ chi2F2d = chi2F2d[~np.isnan(chi2F2d)]
#~ chi2F3d = chi2F3d[~np.isnan(chi2F3d)]
#~ chi2F4d = chi2F4d[~np.isnan(chi2F4d)]
#~ chi2PT1d = chi2PT1d[~np.isnan(chi2PT1d)]
#~ chi2PT2d = chi2PT2d[~np.isnan(chi2PT2d)]
#~ chi2PT3d = chi2PT3d[~np.isnan(chi2PT3d)]
#~ chi2PT4d = chi2PT4d[~np.isnan(chi2PT4d)]
#~ chi2PTbis1d = chi2PTbis1d[~np.isnan(chi2PTbis1d)]
#~ chi2PTbis2d = chi2PTbis2d[~np.isnan(chi2PTbis2d)]
#~ chi2PTbis3d = chi2PTbis3d[~np.isnan(chi2PTbis3d)]
#~ chi2PTbis4d = chi2PTbis4d[~np.isnan(chi2PTbis4d)]




#~ kf[0,0]=kmax1a[np.argmin(chi2F1a)]
#~ kf[1,0]=kmax1b[np.argmin(chi2F1b)]
#~ kf[2,0]=kmax1c[np.argmin(chi2F1c)]
#~ kf[3,0]=kmax1d[np.argmin(chi2F1d)]
#~ kf[0,1]=kmax1a[np.argmin(chi2F2a)]
#~ kf[1,1]=kmax1b[np.argmin(chi2F2b)]
#~ kf[2,1]=kmax1c[np.argmin(chi2F2c)]
#~ kf[3,1]=kmax1d[np.argmin(chi2F2d)]
#~ kf[0,2]=kmax1a[np.argmin(chi2F3a)]
#~ kf[1,2]=kmax1b[np.argmin(chi2F3b)]
#~ kf[2,2]=kmax1c[np.argmin(chi2F3c)]
#~ kf[3,2]=kmax1d[np.argmin(chi2F3d)]
#~ kf[0,3]=kmax1a[np.argmin(chi2F4a)]
#~ kf[1,3]=kmax1b[np.argmin(chi2F4b)]
#~ kf[2,3]=kmax1c[np.argmin(chi2F4c)]
#~ kf[3,3]=kmax1d[np.argmin(chi2F4d)]

#~ kpt[0,0]=kmax1a[np.argmin(chi2PT1a)]
#~ kpt[1,0]=kmax1b[np.argmin(chi2PT1b)]
#~ kpt[2,0]=kmax1c[np.argmin(chi2PT1c)]
#~ kpt[3,0]=kmax1d[np.argmin(chi2PT1d)]
#~ kpt[0,1]=kmax1a[np.argmin(chi2PT2a)]
#~ kpt[1,1]=kmax1b[np.argmin(chi2PT2b)]
#~ kpt[2,1]=kmax1c[np.argmin(chi2PT2c)]
#~ kpt[3,1]=kmax1d[np.argmin(chi2PT2d)]
#~ kpt[0,2]=kmax1a[np.argmin(chi2PT3a)]
#~ kpt[1,2]=kmax1b[np.argmin(chi2PT3b)]
#~ kpt[2,2]=kmax1c[np.argmin(chi2PT3c)]
#~ kpt[3,2]=kmax1d[np.argmin(chi2PT3d)]
#~ kpt[0,3]=kmax1a[np.argmin(chi2PT4a)]
#~ kpt[1,3]=kmax1b[np.argmin(chi2PT4b)]
#~ kpt[2,3]=kmax1c[np.argmin(chi2PT4c)]
#~ kpt[3,3]=kmax1d[np.argmin(chi2PT4d)]


#~ kptbis[0,0]=kmax1a[np.argmin(chi2PTbis1a)]
#~ kptbis[1,0]=kmax1b[np.argmin(chi2PTbis1b)]
#~ kptbis[2,0]=kmax1c[np.argmin(chi2PTbis1c)]
#~ kptbis[3,0]=kmax1d[np.argmin(chi2PTbis1d)]
#~ kptbis[0,1]=kmax1a[np.argmin(chi2PTbis2a)]
#~ kptbis[1,1]=kmax1b[np.argmin(chi2PTbis2b)]
#~ kptbis[2,1]=kmax1c[np.argmin(chi2PTbis2c)]
#~ kptbis[3,1]=kmax1d[np.argmin(chi2PTbis2d)]
#~ kptbis[0,2]=kmax1a[np.argmin(chi2PTbis3a)]
#~ kptbis[1,2]=kmax1b[np.argmin(chi2PTbis3b)]
#~ kptbis[2,2]=kmax1c[np.argmin(chi2PTbis3c)]
#~ kptbis[3,2]=kmax1d[np.argmin(chi2PTbis3d)]
#~ kptbis[0,3]=kmax1a[np.argmin(chi2PTbis4a)]
#~ kptbis[1,3]=kmax1b[np.argmin(chi2PTbis4b)]
#~ kptbis[2,3]=kmax1c[np.argmin(chi2PTbis4c)]
#~ kptbis[3,3]=kmax1d[np.argmin(chi2PTbis4d)]


