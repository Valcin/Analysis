import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects

x1 = 600
x2 = 380
x3 = 100
fts=7
fts1=7
fts2=10

def a(n):
	a = n*x1
	return a
def b(n):
	b = n*x1-x2
	return b
def c(n):
	c = n*x1+x3
	return c

ecb = np.sqrt(0.15**2 + 0.5**2)
ecb_o1 = np.sqrt(0.15**2 + 0.23**2)
ecb_o2 = np.sqrt(0.15**2 + 0.33**2)
print('option 0',ecb, ecb_o2, ecb_o1)
# ~cb2 = np.sum([13.2/0.4**2, 13.4/0.5**2, 13.5/ecb_o2**2])/np.sum([1./0.4**2, 1./0.5**2, 1./ecb_o2**2])
# ~ecb2 = np.sqrt(1./np.sum([1./0.4**2, 1./0.5**2, 1./ecb_o2**2]))
# ~print('option 2',cb2, ecb2)
# ~cb1 = np.sum([13.2/0.4**2, 13.4/0.5**2, 13.5/ecb_o1**2])/np.sum([1./0.4**2, 1./0.5**2, 1./ecb_o1**2])
# ~ecb1 = np.sqrt(1./np.sum([1./0.4**2, 1./0.5**2, 1./ecb_o1**2]))
# ~print('option 1',cb1, ecb1)


plt.figure()

#shoes
n=1
plt.errorbar(12.93, a(n), xerr =np.array([[0.29,0.29]]).T, color='g', fmt='o', markersize=5)
plt.text(12.93, b(n), s='-   SHOES', color='k', horizontalalignment='left', fontsize=fts, weight='bold')
plt.text(12.93, b(n), s=r'$\mathbf{12.93^{+0.29}_{-0.29}}$   ', color='g', horizontalalignment='right', fontsize=fts1, weight='bold')
# ~#-------------------------------------------------------------------------
#### SEPARATION
n=2
plt.text(12.93,a(n),'Late time LCDM',ha='center',fontsize=fts2,alpha=1.,
			bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
# ~n=3
# ~plt.axhline(a(n), c='grey', linestyle='--')
# ~#-------------------------------------------------------------------------

#Raul1
n=4
plt.errorbar(13.2, a(n), xerr =0.4, color='orange', fmt='o', markersize=5)
plt.text(13.2, b(n), s='-   O\'Malley et al 2017. (GC)', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.2, b(n), s=r'$\mathbf{13.2^{+0.4}_{-0.4}}$   ', color='orange', horizontalalignment='right', fontsize=fts1, weight='bold')
sig = (13.2 - 12.93)/np.sqrt(0.4**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='orange', verticalalignment='center', fontsize=fts1, weight='bold')
# ~#Raul2
n=5
plt.errorbar(13.4, a(n), xerr =0.5, color='violet', fmt='o', markersize=5)
plt.text(13.4, b(n), s='-   Jimenez et al 2019. (Old objects)', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.4, b(n), s=r'$\mathbf{13.4^{+0.5}_{-0.5}}$   ', color='violet', horizontalalignment='right', fontsize=fts1, weight='bold')
sig = (13.4 - 12.93)/np.sqrt(0.5**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='violet', verticalalignment='center', fontsize=fts1, weight='bold')
#moi1
n=6
plt.axvline(13.52-ecb, alpha=0.4, color='b', linestyle='--')
plt.axvline(13.52+ecb, alpha=0.4, color='b', linestyle='--')
plt.axvline(13.52-ecb_o2, alpha=0.4, color='b', linestyle='--')
plt.axvline(13.52+ecb_o2, alpha=0.4, color='b', linestyle='--')
plt.axvline(13.52-ecb_o1, alpha=0.4, color='b', linestyle='--')
plt.axvline(13.52+ecb_o1, alpha=0.4, color='b', linestyle='--')
# ~plt.errorbar(13.52, a(n), xerr =np.array([[0.14+0.5,0.16+0.5]]).T, color='b', fmt='o', markersize=5)
plt.errorbar(13.52, a(n), xerr =np.array([[ecb,ecb]]).T, color='b', fmt='o', markersize=5)
plt.errorbar(13.52, a(n), xerr =np.array([[0.14,0.16]]).T, color='c', fmt='o', markersize=5)
plt.text(13.52, b(n), s='-   Valcin et al 2020.', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.52, b(n), s=r'$\mathbf{\pm 0.52}$ (w/Sys.)   ', color='b', horizontalalignment='right', fontsize=fts1, weight='bold')
plt.text(13.52, b(n), s=r'$\mathbf{13.52^{+0.16}_{-0.14}}$                           ', color='c', horizontalalignment='right', fontsize=fts1, weight='bold')
sig = (13.52 - 12.93)/np.sqrt(ecb**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='b', verticalalignment='center', fontsize=fts1, weight='bold')
#moi2a
n=7

# ~plt.errorbar(13.52, a(n), xerr =np.array([[0.14+0.33,0.16+0.33]]).T, color='b', fmt='o', markersize=5)
plt.errorbar(13.52, a(n), xerr =np.array([[ecb_o2,ecb_o2]]).T, color='b', fmt='o', markersize=5)
plt.errorbar(13.52, a(n), xerr =np.array([[0.14,0.16]]).T, color='c', fmt='o', markersize=5)
plt.text(13.52, b(n), s='-   Valcin et al 2021. (option 2).', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.52, b(n), s=r'$\mathbf{\pm 0.36}$ (w/Sys.)   ', color='b', horizontalalignment='right', fontsize=fts1, weight='bold')
plt.text(13.52, b(n), s=r'$\mathbf{13.52^{+0.16}_{-0.14}}$                           ', color='c', horizontalalignment='right', fontsize=fts1, weight='bold')
sig = (13.52 - 12.93)/np.sqrt(ecb_o2**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='b', verticalalignment='center', fontsize=fts1, weight='bold')
#moi2b
n=8

# ~plt.errorbar(13.52, a(n), xerr =np.array([[0.14+0.23,0.16+0.23]]).T, color='b', fmt='o', markersize=5)
plt.errorbar(13.52, a(n), xerr =np.array([[ecb_o1,ecb_o1]]).T, color='b', fmt='o', markersize=5)
plt.errorbar(13.52, a(n), xerr =np.array([[0.14,0.16]]).T, color='c', fmt='o', markersize=5)
plt.text(13.52, b(n), s='-   Valcin et al 2021. (option 1).', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.52, b(n), s=r'$\mathbf{\pm 0.27}$ (w/Sys.)   ', color='b', horizontalalignment='right', fontsize=fts1, weight='bold')
plt.text(13.52, b(n), s=r'$\mathbf{13.52^{+0.16}_{-0.14}}$                           ', color='c', horizontalalignment='right', fontsize=fts1, weight='bold')
sig = (13.52 - 12.93)/np.sqrt(ecb_o1**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='b', verticalalignment='center', fontsize=fts1, weight='bold')
#direct combined option 2
# ~n=10
# ~plt.errorbar(cb2, a(n), xerr =np.array([[0.24,0.24]]).T, color='k', fmt='x', markersize=5)
# ~plt.text(cb2, b(n), s='-   Combined GC + Old objects + option 2.', color='k', horizontalalignment='left', fontsize=fts)
# ~plt.text(cb2, b(n), s=r'$13.37^{+0.24}_{-0.24}$   ', color='k', horizontalalignment='right', fontsize=fts1)
# ~sig = (cb2 - 12.93)/0.24
# ~format_sig = "{:.1f}".format(sig)
# ~plt.text(14.2, a(n), format_sig + r'$\sigma$', color='k', verticalalignment='center', fontsize=fts1)
# ~#direct combined option 1
# ~n=11
# ~plt.errorbar(cb1, a(n), xerr =np.array([[0.21,0.21]]).T, color='k', fmt='^', markersize=5)
# ~plt.text(cb1, b(n), s='-   Combined GC + Old objects + option 1.', color='k', horizontalalignment='left', fontsize=fts)
# ~plt.text(cb1, b(n), s=r'$13.40^{+0.21}_{-0.21}$   ', color='k', horizontalalignment='right', fontsize=fts1)
# ~sig = (cb1 - 12.93)/0.21
# ~format_sig = "{:.1f}".format(sig)
# ~plt.text(14.2, a(n), format_sig + r'$\sigma$', color='k', verticalalignment='center', fontsize=fts1)

### plot vertical band for stats. error
plt.axvspan(13.52-0.14, 13.52+0.16, alpha=0.2, color='c')

# ~#-------------------------------------------------------------------------
#### SEPARATION
n=9
plt.text(13.52,a(n),'Direct measurement',ha='center',fontsize=fts2,alpha=1.,
			bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
# ~n=11
# ~plt.axhline(a(n), c='grey', linestyle='--')
# ~#-------------------------------------------------------------------------
#trgb
# ~n=9
# ~plt.errorbar(13.62, a(n), xerr =np.array([[0.42,0.42]]).T, color='lightgreen', fmt='o', markersize=5)
# ~plt.text(13.62, b(n), s='-   TRGB', color='k', horizontalalignment='left', fontsize=fts)
# ~plt.text(13.62, b(n), s=r'$13.62^{+0.42}_{-0.42}$   ', color='lightgreen', horizontalalignment='right', fontsize=fts1)

# ~#-------------------------------------------------------------------------
#PLANCK
n=12
plt.errorbar(13.8, a(n), xerr =np.array([[0.02,0.02]]).T, color='r', fmt='o', markersize=5)
plt.text(13.8, b(n), s='-   Planck', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.8, b(n), s=r'$\mathbf{13.8^{+0.02}_{-0.02}}$   ', color='r', horizontalalignment='right', fontsize=fts1, weight='bold')
# ~txt12a.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
# ~txt12b.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

sig = (13.8 - 12.93)/np.sqrt(0.02**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='r', verticalalignment='center', fontsize=fts1, weight='bold')

#PLANCK ede
n=13
plt.errorbar(13.76, a(n), xerr =np.array([[0.16,0.06]]).T, color='m', fmt='o', markersize=5)
plt.text(13.76, b(n), s='-   Planck EDE', color='k', horizontalalignment='left', fontsize=fts1, weight='bold')
plt.text(13.76, b(n), s=r'$\mathbf{13.76^{+0.06}_{-0.16}}$   ', color='m', horizontalalignment='right', fontsize=fts1, weight='bold')
# ~txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

sig = (13.76 - 12.93)/np.sqrt(0.11**2 + 0.29**2)
format_sig = "{:.1f}".format(sig)
plt.text(14.2, a(n), format_sig + r'$\mathbf{\sigma}$', color='m', verticalalignment='center', fontsize=fts1, weight='bold')
# ~#-------------------------------------------------------------------------
#### SEPARATION
n=14
plt.text(13.8,a(n),'LCDM early+late',ha='center',fontsize=fts2,alpha=1.,
			bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))
# ~plt.text(12.7,a(n),'LCDM early+late',va='center',fontsize=fts2,alpha=1.,
			# ~bbox=dict(boxstyle="round",fc=(1., 1.,1.,1.),ec='k'))

# ~#-----------------------------------------------------------------------


plt.xlim(12.6,14.3)
plt.xlabel('Age of the Universe [Gyr]', fontsize=14)
plt.ylim(0,a(15))
plt.subplots_adjust(bottom=0.13, top=0.94)
frame1 = plt.gca()
frame1.axes.get_yaxis().set_visible(False)

plt.show()

# ~a = np.sqrt(0.27**2 + 0.29**2)
# ~print(a)
# ~b = 13.52 -12.93
# ~c = b/a
# ~print(b,c)

