#########################################################################################
# This routine computes the analytic Fisher matrix of a set of parameters
# parameter ------> set of paramters used ['Om', 'Ob', 'h', 'ns', 's8']
# z --------------> considered redshift
# V --------------> considered volume in (Mpc/h)^3
# kmax -----------> maximum considered k in h/Mpc
# root_derv ------> folder containing the analytic derivatives
def analytic_Fisher(parameter, z, V, kmax, root_derv):

    num_params = len(parameter)

    # define the Fisher matrix
    Fisher = np.zeros((num_params,num_params), dtype=np.float64)

    # compute the value of kmin
    kmin = 2.0*np.pi/V**(1.0/3.0)

    # check that all ks are equally spaced in log
    for i in xrange(num_params):
        k, deriv = np.loadtxt('%s/derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), 
                              unpack=True)
        dk = np.log10(k[1:]) - np.log10(k[:-1])
        if not(np.allclose(dk, dk[0])):
            raise Exception('k-values not log distributed')

    # compute sub-Fisher matrix
    for i in xrange(num_params):
        for j in xrange(i,num_params):
            #if i==5:
            #    k1, deriv1 = np.loadtxt('%s/log_derivative_%s_cb_z=%.1f.txt'%(root_derv,parameter[i],z), unpack=True)
            #else:
            k1, deriv1 = np.loadtxt('%s/log_derivative_%s_z=%.1f.txt'%(root_derv,parameter[i],z), unpack=True)
            #if j==5:
            #    k2, deriv2 = np.loadtxt('%s/log_derivative_%s_cb_z=%.1f.txt'%(root_derv,parameter[j],z), unpack=True)
            #else:
            k2, deriv2 = np.loadtxt('%s/log_derivative_%s_z=%.1f.txt'%(root_derv,parameter[j],z), unpack=True)

            if np.any(k1!=k2):  raise Exception('k-values are different!')
        
            yinit = np.zeros(1, dtype=np.float64) 
            eps   = 1e-16
            h1    = 1e-18 
            hmin  = 0.0   
            function = 'log'

            I = IL.odeint(yinit, kmin, kmax, eps, h1, hmin,
                          np.log10(k1), k**2*deriv1*deriv2,
                          function, verbose=False)[0]

            Fisher[i,j] = I
            if i!=j:  Fisher[j,i] = Fisher[i,j]

    # add prefactors to subFisher matrix
    Fisher = Fisher*V/(2.0*np.pi)**2

    return Fisher
#########################################################################################
