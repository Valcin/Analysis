import multiprocessing as mp
p = mp.Pool()

def poly(args):
	a,b,c = args
	y = a**2 + b**2 + c**2
    
	return y



dim1, dim2, dim3 = 10, 20, 30

import itertools

args = [(i,j,k) for i in xrange(dim1) for j in xrange(dim2) for k in xrange(dim3)]
print args
results = p.map(poly, args)
p.close()
p.join()

print results
