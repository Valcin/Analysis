import numpy as np
import sys,os
sys.path.append('/simons/scratch/fvillaescusa/pdf_information/library')
import BCP_library as BCPL

root_derv = '/simons/scratch/fvillaescusa/pdf_information/linear_derivatives'
###################################### INPUT ############################################
BoxSize = 1000.0 #Mpc/h
grid    = 1024
z       = 0

#parameters = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
parameters = ['Mnu']
#########################################################################################

num_params = len(parameters)

#fin  = '../fiducial/CAMB_TABLES/CAMB_matterpow_0.dat'
#fout = '../fiducial/CAMB_TABLES/expected_CAMB_matterpow_0.dat'
#k, Pk, Nmodes = BCPL.Pk_binning(fin, BoxSize, grid)
#np.savetxt(fout, np.transpose([k, Pk, Nmodes]))
#sys.exit()


# rebin the derivatives to use them in the Fisher matrix
for i in xrange(num_params):
    fin  = '%s/derivative_%s_z=%.1f.txt'%(root_derv, parameters[i], z)
    fout = '%s/derivative_rebin_%s_z=%.1f.txt'%(root_derv, parameters[i], z)
    
    k, Pk, Nmodes = BCPL.Pk_binning(fin, BoxSize, grid)
    np.savetxt(fout, np.transpose([k, Pk, Nmodes]))


"""
# compute the covariance matrix
fin = '/simons/scratch/fvillaescusa/pdf_information/fiducial/CAMB_TABLES/CAMB_matterpow_%s.dat'%z
k, Pk, Nmodes = BCPL.Pk_binning(fin, BoxSize, grid)

f = open('%s/linear_covariance_z=%.1f.txt'%(root_derv, z), 'w')
for i in xrange(len(k)):
    for j in xrange(len(k)):
        if i!=j:  f.write(str(k[i])+' '+str(k[j])+' '+str(0.0)+'\n')
        else:     f.write(str(k[i])+' '+str(k[j])+' '+str(Pk[i]**2/Nmodes[i])+'\n')
f.close()
"""
