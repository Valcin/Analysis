import numpy as np
import matplotlib.pyplot as plt
version1 = '9'
version2 = '15'
model2 = 'dar'

Age_mean_dar1 = np.loadtxt('/home/david/codes/GC/plots/data_'+ version1 +'_'+str(model2)+'.txt', usecols=(2,))
Age_high_dar1 = np.loadtxt('/home/david/codes/GC/plots/data_'+ version1 +'_'+str(model2)+'.txt', usecols=(3,))
Age_low_dar1 = np.loadtxt('/home/david/codes/GC/plots/data_'+ version1 +'_'+str(model2)+'.txt', usecols=(1,))

Age_mean_dar2 = np.loadtxt('/home/david/codes/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(2,))
Age_high_dar2 = np.loadtxt('/home/david/codes/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(3,))
Age_low_dar2 = np.loadtxt('/home/david/codes/GC/plots/data_'+ version2 +'_'+str(model2)+'.txt', usecols=(1,))


x = np.linspace(7,15.5, 100)
plt.scatter(Age_mean_dar1, Age_mean_dar2)
plt.plot(x,x)
plt.xlim(7,15.5)
plt.ylim(7,15.5)
plt.xlabel('paper ages')
plt.ylabel('referee report ages')
plt.show()
plt.close()
