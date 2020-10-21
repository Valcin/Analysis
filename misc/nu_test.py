import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import ConfigParser
from classy import Class



## Define a cosmological scenario ( CLASS default otherwise )
#params = { ' omega_b ': 0.02 , 'h ': 0.7 , ' output ': ' mPk '}

## Create a Class instance
#cosmo = Class ()

## Set the instance to the cosmology
#cosmo . set ( params )

## Run the _init methods
#cosmo . compute ()

## Do something with the pk
#pk = cosmo . pk (0 , 0.1)

## Clean
#cosmo . struct_cleanup () ; cosmo . empty ()

cosmo = Class ()
cosmo . set ({ ' output ': 'tCl , pCl , lCl ' , ' lensing ': ' yes ' })
cosmo . compute ()
l = np . array ( range (2 ,2501) )
factor = l *( l +1) /(2* np . pi )
lensed_cl = cosmo . lensed_cl (2500)

lensed_cl . viewkeys ()

#plt . loglog (l , factor * lensed_cl [ ' tt ' ][2:] , l , factor * lensed_cl [ 'ee ' ][2:])
#plt . xlabel ( r " $ \ ell$ " )
#plt . ylabel ( r " $ \ ell (\ ell +1) /(2\ pi ) C_ \ ell$ "
#plt . tight_layout ()
#plt . savefig ( " output / T T_ EE _ La mb da C DM . pdf " )


