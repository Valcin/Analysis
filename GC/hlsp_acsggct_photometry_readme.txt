------------------------------------------------------------------------+
The ACS Globular Cluster Survey
PI: Ata Sarajedini  - University of Florida ata@astro.ufl.edu
HST Program ID: 10775

Introduction and observations

The ACS Survey of Globular Clusters was designed to provide a
nearly complete catalog of all the stars present in the central two 
arcminutes of 66 targeted clusters. Such a uniform data set has many 
scientific applications and the catalog may be used for broad studies of: 
binary-star distributions, absolute and relative ages, horizontal-branch 
morphology, blue stragglers, isochrone fitting, mass functions, and 
dynamical models.

For each of the observed clusters the team archived the reference image and a
catalog.  The methodology for creating the catalog is described in 
Paper V: Anderson et al. 2008, "The ACS Survey of Galactic Globular Clusters.
V. Generating a Comprehensive Star Catalog for each Cluster," AJ, 135, 2055

The following describes the final pass photometry. 

A baseline catalog is created for each cluster that includes the x, y 
positions and the converted RA and Dec using the F606W files.  This file 
is then input into fortran code that applies the CTE correction to the long 
and short exposure magnitudes and refines the magnitude offsets between
the long and short magnitudes. The code also calculates the photometric error
for stars with only 1 measurement using the root(n) noise in the star. 
Note that any star with wV or wI equal to 3 comes from the satphot algorithm 
applied to saturated stars.

These files are then adjusted to the astrometric zeropoint derived from
the 2MASS catalog and named *.RDVIQ.cal.adj.  Note that the Palomar 1 file 
does not need the 2MASS astrometric adjustment so that its final 
photometry/astrometry file is called *.RDVIQ.cal.zpt.

The *.RDVIQ.cal.*.zpt files include the new photometric zeropoints for ACS/WFC
from ACS-ISR-2007-02 (Mack et al. 2007).

The columns in the *.RDVIQ.cal.* files are:

id      = star number from the original files
x       = star x position on the master frame
y       = star y position on the master frame
Vvega   = F606W VEGA mag from Sirianni et al. (2005) (Table 10)
err     = error in F606W mag (see below for an explanation)
VIvega  = V-I VEGAmag
err     = error in V-I VEGAmag (see below for an explanation)
Ivega   = F814W VEGA mag from Sirianni (2005) (Table 10)
err     = error in F814W mag (see below for an explanation)
Vground = ground-based V mag from Sirianni et al. (2005) (Table 18)
Iground = ground-based I mag from Sirianni et al. (2005) (Table 18)
Nv      = number of measurements in the V mag
Ni      = number of measurements in the I mag
wV      = where did the V mag come from (1=deep unsat, 2=short unsat)
wI      = where did the I mag come from (3=short sat, 4=deep sat)
xsig    = RMS of x positions (see below for an explanation)
ysig    = RMS of y positions (see below for an explanation)
othv    = fraction of F606W light from other stars
othi    = fraction of F814W light from other stars
qfitV   = quality of the V fit (smaller is better)
qfitI   = quality of the I fit
RA      = RA from the COMB.fits file header
Dec     = Dec from the COMB.fits file header

A value of 9.900 for any quantity means that it has no measurement.


Note on the errors given in the data files
------------------------------------------
The columns entitled "err", in cols. 5, 7, and 9, are not the errors
themselves; their meaning is as follows:

When n (the number of observations in col. 12 or 13) is >2, "err" gives the rms residual of the
individual measurements.  When n=2, they are the difference between the two measurements, which
is twice the rms residual. When n=1, they are error estimates calculated from the number of
counts in the star image; these tend to be underestimates of the real error, by a factor of
~3-4.

The user should also note that xsig and ysig are not astrometric error
estimates; instead xsig is the difference between the x positions in the
V and I images, and correspondingly for ysig.  The positions are not of
high astrometric accuracy, and xsig and ysig should be used only as an
additional flag of the photometric accuracy (since stars with poor
photometry tend also to have poor astrometry).


