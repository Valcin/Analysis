import numpy as np
import h5py
import math
import readsnap
import matplotlib
#~ matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
sys.path.append('/home/david/codes/FAST-PT')
import myFASTPT as FPT
import scipy.interpolate as sp
import pyximport
pyximport.install()
import redshift_space_library as RSL
from readfof import FoF_catalog
import MAS_library as MASL
import Pk_library as PKL
import mass_function_library as MFL
import bias_library as BL
import tempfile
import expected_CF
import exp2
from load_data import ld_data
from time import time
from bias_library import halo_bias, bias
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.special import gamma

####################################################################
#### compute the one loop correction 

def pt_terms(kbis, Plinbis):
	# set the parameters for the power spectrum window and
	# Fourier coefficient window 
	#P_window=np.array([.2,.2])  
	C_window=0.95

	# padding length 
	nu=-2; n_pad=len(kbis)
	n_pad=int(0.5*len(kbis))
	to_do=['all']
					
	# initialize the FASTPT class 
	# including extrapolation to higher and lower k  
	# time the operation
	t1=time()
	fastpt=FPT.FASTPT(kbis,to_do=to_do,n_pad=n_pad, verbose=True) 
	t2=time()
		
	# calculate 1loop SPT (and time the operation) for density
	P_spt_dd=fastpt.one_loop_dd(Plinbis,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	# calculate 1loop SPT (and time the operation) for velocity
	P_spt_tt=fastpt.one_loop_tt(Plinbis,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	# calculate 1loop SPT (and time the operation) for velocity - density
	P_spt_dt=fastpt.one_loop_dt(Plinbis,C_window=C_window)
		
	t3=time()
	print('initialization time for', to_do, "%10.3f" %(t2-t1), 's')
	print('one_loop_dd recurring time', "%10.3f" %(t3-t2), 's')
		
	#calculate tidal torque EE and BB P(k)
	#~ P_RSD=fastpt.RSD_components(P,1.0,C_window=C_window)	

	# update the power spectrum
	Pmod_dd=Plinbis+P_spt_dd[0]
	Pmod_dt=Plinbis+P_spt_dt[0]
	Pmod_tt=Plinbis+P_spt_tt[0]	
	A = P_spt_dd[2]
	B = P_spt_dd[3]
	C = P_spt_dd[4]
	D = P_spt_dd[5]
	E = P_spt_dd[6]
	F = P_spt_dd[7]
	G = P_spt_dt[2]
	H = P_spt_dt[3]
	
	return Pmod_dd, Pmod_dt, Pmod_tt, A, B, C, D, E, F, G, H 
	### for scoccimaro comparison
	#~ plt.figure()
	#~ plt.suptitle('z = '+str(z[j])+' ,expansion at 11th order, class h = 0.7, omega_b =0.05, omega_cdm = 0.25')
	#~ ax1=plt.subplot(311)
	#~ ax1.plot(kbis,Pmod_dd/Plinbis,label=r'$ \delta \delta FAST PT $', color='r')
	#~ ax1.plot(ksdd,psdd, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ ax2=plt.subplot(312)
	#~ ax2.plot(kbis,Pmod_dt/Plinbis,label=r'$ \delta \theta FAST PT $',color='r')
	#~ ax2.plot(ksdt,psdt, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ ax3=plt.subplot(313)
	#~ ax3.plot(kbis,Pmod_tt/Plinbis,label=r'$ \theta \theta FAST PT$', color='r')
	#~ ax3.plot(kstt,pstt, color='b',label='scoccimaro')
	#~ plt.axhline(1, linestyle='--', color='k')
	#~ plt.xscale('log')
	#~ plt.legend(loc='lower left')
	#~ plt.xlim(0.02,0.205)
	#~ plt.ylim(0.5,1.5)
	#~ plt.tick_params(labelleft=True, labelright=True)
	#~ plt.show()

	#~ plt.figure()
	#~ plt.plot(kbis,Pmod_dd)
	#~ plt.plot(kbis,Pmod_dt)
	#~ plt.plot(kbis,Pmod_tt)
	#~ plt.plot(kbis,Plinbis)
	#~ plt.xscale('log')
	#~ plt.yscale('log')
	#~ plt.xlim(0.0008,10)
	#~ plt.ylim(3e3,4e4)
	#~ plt.ylim(1e-1,4e4)
	#~ plt.show()
	#~ kill
	
	
	
	####################################################################
	#### save PT coeff if needed
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/Pmod_dd_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], Pmod_dd[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/Pmod_dt_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], Pmod_dt[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/Pmod_tt_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], Pmod_tt[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/A_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], A[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/B_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], B[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/C_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], C[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/D_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], D[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/E_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], E[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/F_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], F[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/G_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], G[index_k]))
	#~ fid_file.close()
	#~ with open('/home/david/codes/Paco/data2/0.0eV/exp/H_'+str(z[j])+'.txt', 'w+') as fid_file:
		#~ for index_k in xrange(len(kbis)):
			#~ fid_file.write('%.8g %.8g\n' % ( kbis[index_k], H[index_k]))
	#~ fid_file.close()
