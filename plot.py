
import numpy as np
import h5py
import math
import readsnap
import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


Mnu = 0.0

z = [0.0,0.5,1.0,2.0]


bfile = np.loadtxt('3rdorder_0.0.txt')
ktest = bfile[:,0]
b1a = bfile[:,1]
b1b = bfile[:,2]
b1c = bfile[:,3]
b1d = bfile[:,4]
b3a = bfile[:,5]
b3b = bfile[:,6]
b3c = bfile[:,7]
b3d = bfile[:,8]

plt.figure()
plt.plot(ktest, b3a/(b1a-1))
plt.plot(ktest, b3b/(b1b-1))
plt.plot(ktest, b3c/(b1c-1))
plt.plot(ktest, b3d/(b1d-1))
#~ plt.plot(ktest, b1a)
#~ plt.plot(ktest, b1b)
#~ plt.plot(ktest, b1c)
#~ plt.plot(ktest, b1d)
#~ plt.plot(ktest, b3a)
#~ plt.plot(ktest, b3b)
#~ plt.plot(ktest, b3c)
#~ plt.plot(ktest, b3d)
plt.xscale('log')
plt.axhline(32/315., color='k')
plt.xlim(0.03,0.2)
plt.ylim(-0.5,5)
plt.show()

j=0
cname = 'chi2_z='+str(z[j])+'.txt'
goodfit = np.loadtxt(cname) 
kmax = goodfit[:,0]
chipbis1 = goodfit[:,9]
chipbis2 = goodfit[:,10]
chipbis3 = goodfit[:,11]
chipbis4 = goodfit[:,12]
Chipbis = np.array([chipbis1,chipbis2,chipbis3,chipbis4])
chipbis = np.mean(Chipbis, axis=0)
echipbis = np.std(Chipbis, axis=0)
plt.figure()
plt.plot(kmax, chipbis1)
plt.plot(kmax, chipbis2)
plt.plot(kmax, chipbis3)
plt.plot(kmax, chipbis4)
plt.xscale('log')
plt.xlim(0.04,0.2)
plt.ylim(0,10)
plt.show()
kill
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

	cname = 'chi2_z='+str(z[j])+'.txt'

	goodfit = np.loadtxt(cname) 
	kmax = goodfit[:,0]
	#--------------------
	chipl1 = goodfit[:,1]
	chipl2 = goodfit[:,2]
	chipl3 = goodfit[:,3]
	chipl4 = goodfit[:,4]
	Chipl = np.array([chipl1,chipl2,chipl3,chipl4])
	chipl = np.mean(Chipl, axis=0)
	echipl = np.std(Chipl, axis=0)

	#--------------------
	chipt1 = goodfit[:,5]
	chipt2 = goodfit[:,6]
	chipt3 = goodfit[:,7]
	chipt4 = goodfit[:,8]
	Chipt = np.array([chipt1,chipt2,chipt3,chipt4])
	chipt = np.mean(Chipt, axis=0)
	echipt = np.std(Chipt, axis=0)
	
	#--------------------
	chipbis1 = goodfit[:,9]
	chipbis2 = goodfit[:,10]
	chipbis3 = goodfit[:,11]
	chipbis4 = goodfit[:,12]
	Chipbis = np.array([chipbis1,chipbis2,chipbis3,chipbis4])
	chipbis = np.mean(Chipbis, axis=0)
	echipbis = np.std(Chipbis, axis=0)
	#--------------------
	chipter1 = goodfit[:,13]
	chipter2 = goodfit[:,14]
	chipter3 = goodfit[:,15]
	chipter4 = goodfit[:,16]
	Chipter = np.array([chipter1,chipter2,chipter3,chipter4])
	chipter = np.mean(Chipter, axis=0)
	echipter = np.std(Chipter, axis=0)



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
	#~ ax2.plot(kmax, chipl4, color='C0')
	P1, =ax2.plot(kmax, chipl, color='C0', label='z = '+str(z[j]))
	ax2.errorbar(kmax, chipl, yerr= echipl,fmt='.')
	#---------------------------------
	#~ ax2.plot(kmax, chipt1, color='C1')
	#~ ax2.plot(kmax, chipt2, color='C1')
	#~ ax2.plot(kmax, chipt3, color='C1')
	#~ ax2.plot(kmax, chipt4, color='C1')
	P2, =ax2.plot(kmax, chipt, color='C1')
	ax2.errorbar(kmax, chipt, yerr= echipt,fmt='.')
	#---------------------------------
	#~ ax2.plot(kmax, chipbis1, color='C2')
	#~ ax2.plot(kmax, chipbis2, color='C2')
	#~ ax2.plot(kmax, chipbis3, color='C2')
	#~ ax2.plot(kmax, chipbis4, color='C2')
	P3, =ax2.plot(kmax, chipbis, color='C2')
	ax2.errorbar(kmax, chipbis, yerr= echipbis,fmt='.')
	#---------------------------------
	#~ ax2.plot(kmax, chipter1, color='C3')
	#~ ax2.plot(kmax, chipter2, color='C3')
	#~ ax2.plot(kmax, chipter3, color='C3')
	#~ ax2.plot(kmax, chipter4, color='C3')
	P4, =ax2.plot(kmax, chipter, color='C3')
	ax2.errorbar(kmax, chipter, yerr= echipter,fmt='.')
	#---------------------------------
	plt.figlegend( (P1,P2, P3, P4), ('Polynomial','2nd order PT',r'3nd order PT with free $b_{s}$,$b_{3nl}$',\
	r'3nd order PT with fixed $b_{s}$,$b_{3nl}$'), \
	######################################
	loc = 'upper center', ncol=5, labelspacing=0., title =r' M$\nu$ = '+str(Mnu), fontsize=14)
	ax2.legend(loc = 'upper left', fancybox=True, fontsize=14)
	plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
	ax2.set_xscale('log')
	#~ ax2.set_yscale('log')
	ax2. set_xlim(0.05,0.55)
	ax2. set_ylim(0,30)
	if j == 0 :
		ax2.tick_params(bottom='off', labelbottom='off',labelleft=True)
		ax2.set_ylabel(r'$\chi^2$', fontsize=16)
	if j == 1 :
		ax2.tick_params(bottom='off', labelbottom='off', labelright=True, right= True, labelleft='off', left='off')
		ax2.set_ylabel(r'$\chi^2$', fontsize=16)
		ax2.yaxis.set_label_position("right")
	if j == 2 :
		#ax.tick_params(labelleft=True)
		ax2.set_ylabel(r'$\chi^2$', fontsize=16)
		ax2.set_xlabel('k [h/Mpc]', fontsize=16)
	if j == 3 :
		ax2.tick_params(labelright=True, right= True, labelleft='off', left='off')
		ax2.set_xlabel('k [h/Mpc]', fontsize=16)
		ax2.set_ylabel(r'$\chi^2$', fontsize=16)
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


