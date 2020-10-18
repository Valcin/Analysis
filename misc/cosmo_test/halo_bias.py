import numpy as np
import sys,os
import readfof
import Pk_library as PKL
import MAS_library as MASL

root = '/home/fvillaescusa/data/pdf_information/'
################################### INPUT ##############################################
cosmo = 's8_p/'
grid  = 512

snapnum = 4

ptypes = [1]
MAS    = 'CIC'
do_RSD = False
axis   = 0
threads = 16

BoxSize = 1000.0 #Mpc/h
Mmin    = 5e13   #Msun/h

realizations = 10
bins         = 443
########################################################################################

if not(os.path.exists(cosmo)):  os.system('mkdir %s'%cosmo)

b = np.zeros((2*realizations, bins), dtype=np.float64)

for i in xrange(10):
    for pair in [0,1]:

        f1 = '%s/%s/NCV_%d_%d/snapdir_%03d/snap_%03d'%(root,cosmo,pair,i,snapnum,snapnum)
        f2 = '%s/%s/NCV_%d_%d/'%(root,cosmo,pair,i)

        # compute delta_c
        delta_c = MASL.density_field_gadget(f1, ptypes, grid, MAS, do_RSD, axis)
        delta_c /= np.mean(delta_c, dtype=np.float64);  delta_c -= 1.0

        # compute delta_h
        delta_h = np.zeros((grid,grid,grid), dtype=np.float32)
        FoF   = readfof.FoF_catalog(f2,snapnum,long_ids=False,
                                    swap=False,SFR=False,read_IDs=False)
        pos_h = FoF.GroupPos/1e3            #Mpc/h                       
        mass  = FoF.GroupMass*1e10          #Msun/h                      
        indexes = np.where(mass>Mmin)[0]
        pos_h = pos_h[indexes];  del indexes
        
        MASL.MA(pos_h, delta_h, BoxSize, MAS)
        delta_h /= np.mean(delta_h, dtype=np.float64);  delta_h -= 1.0

        # compute power spectra
        Pk = PKL.XPk([delta_c,delta_h], BoxSize, axis, [MAS,MAS], threads)

        b[2*i+pair] = (Pk.Pk[:,0,1]-BoxSize**3/len(pos_h))/Pk.Pk[:,0,0]


fout = '%s/b_z=0.txt'%(cosmo)
np.savetxt(fout, np.transpose([Pk.k3D, np.mean(b, axis=0), np.std(b, axis=0)]))
