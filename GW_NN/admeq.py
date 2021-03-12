import torch
from torch import nn, optim
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from ncf import generator_3dspatial_cube
from ncf import _solve_3dspatial_temporal
from ncf import SingleNetworkApproximator3DSpatialTemporal
from neurodiffeq.temporal import FirstOrderInitialCondition, BoundaryCondition, generator_temporal
from neurodiffeq.temporal import MonitorMinimal, generator_1dspatial


########################################################################
########################################################################
#-----------------------------------------------------------------------
### define the function ####
#-----------------------------------------------------------------------


# specify the ODE system and its parameters
def ADM(K11,K12,K13,K21,K22,K23,K31,K32,K33,K,a,b1,b2,b3,G11,G12,G13,G21,G22,G23,G31,G32,G33, r, y, z, t):

	# Underscore in name for contravariant

	# define the contravariant metric
	G_11 = 1./G11
	G_12 = 1./G12
	G_13 = 1./G13
	G_21 = 1./G21
	G_22 = 1./G22
	G_23 = 1./G23
	G_31 = 1./G31
	G_32 = 1./G32
	G_33 = 1./G33

	# define the christoffel coefficients only accounting for G11, G22, G33 (TO MODIFY)
	C111 = 1/2.*G_11*(diff(G11, x) + diff(G11, x) - diff(G11, x)) + 1/2.*G_12*(diff(G21, x) + diff(G21, x) - diff(G11, y)) + 1/2.*G_13*(diff(G31, x) + diff(G31, x) - diff(G11, z))
	C112 = 1/2.*G_11*(diff(G11, y) + diff(G12, x) - diff(G12, x)) + 1/2.*G_12*(diff(G21, y) + diff(G22, x) - diff(G12, y)) + 1/2.*G_13*(diff(G31, y) + diff(G32, x) - diff(G12, z))
	C113 = 1/2.*G_11*(diff(G11, z) + diff(G13, x) - diff(G13, x))	+ 1/2.*G_12*(diff(G21, z) + diff(G23, x) - diff(G13, y))	+ 1/2.*G_13*(diff(G31, z) + diff(G33, x) - diff(G13, z))	
	C121 = 1/2.*G_11*(diff(G12, x) + diff(G11, y) - diff(G21, x)) + 1/2.*G_12*(diff(G22, x) + diff(G21, y) - diff(G21, y)) + 1/2.*G_13*(diff(G32, x) + diff(G31, y) - diff(G21, z))
	C122 = 1/2.*G_11*(diff(G12, y) + diff(G12, y) - diff(G22, x)) + 1/2.*G_12*(diff(G22, y) + diff(G22, y) - diff(G22, y)) + 1/2.*G_13*(diff(G32, y) + diff(G32, y) - diff(G22, z))
	C123 = 1/2.*G_11*(diff(G12, z) + diff(G13, y) - diff(G23, x)) + 1/2.*G_12*(diff(G22, z) + diff(G23, y) - diff(G23, y)) + 1/2.*G_13*(diff(G32, z) + diff(G33, y) - diff(G23, z))
	C131 = 1/2.*G_11*(diff(G13, x) + diff(G11, z) - diff(G31, x)) + 1/2.*G_12*(diff(G23, x) + diff(G21, z) - diff(G31, y)) + 1/2.*G_13*(diff(G33, x) + diff(G31, z) - diff(G31, z))
	C132 = 1/2.*G_11*(diff(G13, y) + diff(G12, z) - diff(G32, x)) + 1/2.*G_12*(diff(G23, y) + diff(G22, z) - diff(G32, y)) + 1/2.*G_13*(diff(G33, y) + diff(G32, z) - diff(G32, z))
	C133 = 1/2.*G_11*(diff(G13, z) + diff(G13, z) - diff(G33, x)) + 1/2.*G_12*(diff(G23, z) + diff(G23, z) - diff(G33, y)) + 1/2.*G_13*(diff(G33, z) + diff(G33, z) - diff(G33, z))
	C211 = 1/2.*G_21*(diff(G11, x) + diff(G11, x) - diff(G11, x)) + 1/2.*G_22*(diff(G21, x) + diff(G21, x) - diff(G11, y)) + 1/2.*G_23*(diff(G31, x) + diff(G31, x) - diff(G11, z))
	C212 = 1/2.*G_21*(diff(G11, y) + diff(G12, x) - diff(G12, x)) + 1/2.*G_22*(diff(G21, y) + diff(G22, x) - diff(G12, y)) + 1/2.*G_23*(diff(G31, y) + diff(G32, x) - diff(G12, z))
	C213 = 1/2.*G_21*(diff(G11, z) + diff(G13, x) - diff(G13, x)) + 1/2.*G_22*(diff(G21, z) + diff(G23, x) - diff(G13, y)) + 1/2.*G_23*(diff(G31, z) + diff(G33, x) - diff(G13, z))
	C221 = 1/2.*G_21*(diff(G12, x) + diff(G11, y) - diff(G21, x)) + 1/2.*G_22*(diff(G22, x) + diff(G21, y) - diff(G21, y)) + 1/2.*G_23*(diff(G32, x) + diff(G31, y) - diff(G21, z))
	C222 = 1/2.*G_21*(diff(G12, y) + diff(G12, y) - diff(G22, x)) + 1/2.*G_22*(diff(G22, y) + diff(G22, y) - diff(G22, y)) + 1/2.*G_23*(diff(G32, y) + diff(G32, y) - diff(G22, z))
	C223 = 1/2.*G_21*(diff(G12, z) + diff(G13, y) - diff(G23, x)) + 1/2.*G_22*(diff(G22, z) + diff(G23, y) - diff(G23, y)) + 1/2.*G_23*(diff(G32, z) + diff(G33, y) - diff(G23, z))
	C231 = 1/2.*G_21*(diff(G13, x) + diff(G11, z) - diff(G31, x)) + 1/2.*G_22*(diff(G23, x) + diff(G21, z) - diff(G31, y)) + 1/2.*G_23*(diff(G33, x) + diff(G31, z) - diff(G31, z))
	C232 = 1/2.*G_21*(diff(G13, y) + diff(G12, z) - diff(G32, x)) + 1/2.*G_22*(diff(G23, y) + diff(G22, z) - diff(G32, y)) + 1/2.*G_32*(diff(G33, y) + diff(G32, z) - diff(G32, z))
	C233 = 1/2.*G_21*(diff(G13, z) + diff(G13, z) - diff(G33, x)) + 1/2.*G_22*(diff(G23, z) + diff(G23, z) - diff(G33, y)) + 1/2.*G_23*(diff(G33, z) + diff(G33, z) - diff(G33, z))
	C311 = 1/2.*G_31*(diff(G11, x) + diff(G11, x) - diff(G11, x)) + 1/2.*G_32*(diff(G21, x) + diff(G21, x) - diff(G11, y)) + 1/2.*G_33*(diff(G31, x) + diff(G31, x) - diff(G11, z))
	C312 = 1/2.*G_31*(diff(G11, y) + diff(G12, x) - diff(G12, x)) + 1/2.*G_32*(diff(G21, y) + diff(G22, x) - diff(G12, y)) + 1/2.*G_33*(diff(G31, y) + diff(G32, x) - diff(G12, z))
	C313 = 1/2.*G_31*(diff(G11, z) + diff(G13, x) - diff(G13, x)) + 1/2.*G_32*(diff(G21, z) + diff(G23, x) - diff(G13, y)) + 1/2.*G_33*(diff(G31, z) + diff(G33, x) - diff(G13, z))
	C321 = 1/2.*G_31*(diff(G12, x) + diff(G11, y) - diff(G21, x)) + 1/2.*G_32*(diff(G22, x) + diff(G21, y) - diff(G21, y)) + 1/2.*G_33*(diff(G32, x) + diff(G31, y) - diff(G21, z))
	C322 = 1/2.*G_31*(diff(G12, y) + diff(G12, y) - diff(G22, x)) + 1/2.*G_32*(diff(G22, y) + diff(G22, y) - diff(G22, y)) + 1/2.*G_33*(diff(G32, y) + diff(G32, y) - diff(G22, z))
	C323 = 1/2.*G_31*(diff(G12, z) + diff(G13, y) - diff(G23, x)) + 1/2.*G_32*(diff(G22, z) + diff(G23, y) - diff(G23, y)) + 1/2.*G_33*(diff(G32, z) + diff(G33, y) - diff(G23, z))
	C331 = 1/2.*G_31*(diff(G13, x) + diff(G11, z) - diff(G31, x)) + 1/2.*G_32*(diff(G23, x) + diff(G21, z) - diff(G31, y)) + 1/2.*G_33*(diff(G33, x) + diff(G31, z) - diff(G31, z))
	C332 = 1/2.*G_31*(diff(G13, y) + diff(G12, z) - diff(G32, x)) + 1/2.*G_32*(diff(G23, y) + diff(G22, z) - diff(G32, y)) + 1/2.*G_33*(diff(G33, y) + diff(G32, z) - diff(G32, z))
	C333 = 1/2.*G_31*(diff(G13, z) + diff(G13, z) - diff(G33, x)) + 1/2.*G_32*(diff(G23, z) + diff(G23, z) - diff(G33, y)) + 1/2.*G_33*(diff(G33, z) + diff(G33, z) - diff(G33, z))
 

	# define Ricci iteration
	R11_1 = diff(C111, x) - diff(C111, x) + C111*C111 + C111*C212 + C111*C313 - C111*C111 - C112*C211 - C113*C311
	R11_2 = diff(C211, y) - diff(C212, x) + C211*C121 + C211*C222 + C211*C323 - C211*C112 - C212*C212 - C213*C312
	R11_3 = diff(C311, z) - diff(C313, x) + C311*C131 + C311*C232 + C311*C333 - C311*C113 - C312*C213 - C313*C313
	R12_1 = diff(C112, x) - diff(C111, y) + C112*C111 + C112*C212 + C112*C313 - C111*C121 - C112*C221 - C113*C321
	R12_2 = diff(C212, y) - diff(C212, y) + C212*C121 + C212*C222 + C212*C323 - C211*C122 - C212*C222 - C213*C322
	R12_3 = diff(C312, z) - diff(C313, y) + C312*C131 + C312*C232 + C312*C333 - C311*C123 - C312*C223 - C313*C323
	R13_1 = diff(C113, x) - diff(C111, z) + C113*C111 + C113*C212 + C113*C313 - C111*C131 - C112*C231 - C113*C331
	R13_2 = diff(C213, y) - diff(C212, z) + C213*C121 + C213*C222 + C213*C323 - C211*C132 - C212*C232 - C213*C332
	R13_3 = diff(C313, z) - diff(C313, z) + C313*C131 + C313*C232 + C313*C333 - C311*C133 - C312*C233 - C313*C333
	
	R21_1 = diff(C121, x) - diff(C121, x) + C121*C111 + C121*C212 + C121*C313 - C121*C111 - C122*C211 - C123*C311
	R21_2 = diff(C221, y) - diff(C222, x) + C221*C121 + C221*C222 + C221*C323 - C221*C112 - C222*C212 - C223*C312
	R21_3 = diff(C321, z) - diff(C323, x) + C311*C131 + C321*C232 + C321*C333 - C321*C113 - C322*C213 - C323*C313
	R22_1 = diff(C122, x) - diff(C121, y) + C122*C111 + C122*C212 + C122*C313 - C121*C121 - C122*C221 - C123*C321
	R22_2 = diff(C222, y) - diff(C222, y) + C222*C121 + C222*C222 + C222*C323 - C221*C122 - C222*C222 - C223*C322
	R22_3 = diff(C322, z) - diff(C323, y) + C322*C131 + C322*C232 + C322*C333 - C321*C123 - C322*C223 - C323*C323
	R23_1 = diff(C123, x) - diff(C121, z) + C123*C111 + C123*C212 + C123*C313 - C121*C131 - C122*C231 - C123*C331
	R23_2 = diff(C223, y) - diff(C222, z) + C223*C121 + C223*C222 + C223*C323 - C221*C132 - C222*C232 - C223*C332
	R23_3 = diff(C323, z) - diff(C323, z) + C323*C131 + C323*C232 + C323*C333 - C321*C133 - C322*C233 - C323*C333

	R31_1 = diff(C131, x) - diff(C131, x) + C131*C111 + C131*C212 + C131*C313 - C131*C111 - C132*C211 - C133*C311
	R31_2 = diff(C231, y) - diff(C232, x) + C231*C121 + C231*C222 + C231*C323 - C231*C112 - C232*C212 - C233*C312
	R31_3 = diff(C331, z) - diff(C333, x) + C331*C131 + C331*C232 + C331*C333 - C331*C113 - C332*C213 - C333*C313
	R32_1 = diff(C132, x) - diff(C131, y) + C132*C111 + C132*C212 + C132*C313 - C131*C121 - C132*C221 - C133*C321
	R32_2 = diff(C232, y) - diff(C232, y) + C232*C121 + C232*C222 + C232*C323 - C231*C122 - C232*C222 - C233*C322
	R32_3 = diff(C332, z) - diff(C333, y) + C332*C131 + C332*C232 + C332*C333 - C331*C123 - C332*C223 - C333*C323
	R33_1 = diff(C133, x) - diff(C131, z) + C133*C111 + C133*C212 + C133*C313 - C131*C131 - C132*C231 - C133*C331
	R33_2 = diff(C233, y) - diff(C232, z) + C233*C121 + C233*C222 + C233*C323 - C231*C132 - C232*C232 - C233*C332
	R33_3 = diff(C333, z) - diff(C333, z) + C333*C131 + C333*C232 + C333*C333 - C331*C133 - C332*C233 - C333*C333

	
	# define the Ricci tensor for all configurations
	R11 = R11_1 + R11_2 + R11_3
	R12 = R12_1 + R12_2 + R12_3
	R13 = R13_1 + R13_2 + R13_3
	R21 = R21_1 + R21_2 + R21_3
	R22 = R22_1 + R22_2 + R22_3
	R23 = R23_1 + R23_2 + R23_3
	R31 = R31_1 + R31_2 + R31_3
	R32 = R32_1 + R32_2 + R32_3
	R33 = R33_1 + R33_2 + R33_3


	# Compute total Ricci scalar
	R = G_11*R11 + G_22*R22 + G_33*R33

	#define source terms (here vacuum)
	rho = 0
	S1 = 0
	S2 = 0
	S3 = 0
	S = 0
	S11 = 0
	S12 = 0
	S13 = 0
	S21 = 0
	S22 = 0
	S23 = 0
	S31 = 0
	S32 = 0
	S33 = 0

	#define contravariant Ricci tensor
	K_11 = G_11*(G_11*K11 + G_12*K12 + G_13*K13) + G_12*(G_11*K21 + G_12*K22 + G_13*K23) + G_13*(G_11*K31 + G_12*K32 + G_13*K33)
	K_12 = G_11*(G_21*K11 + G_22*K12 + G_23*K13) + G_12*(G_21*K21 + G_22*K22 + G_23*K23) + G_13*(G_21*K31 + G_22*K32 + G_23*K33)
	K_13 = G_11*(G_31*K11 + G_32*K12 + G_33*K13) + G_12*(G_31*K21 + G_32*K22 + G_33*K23) + G_13*(G_31*K31 + G_32*K32 + G_33*K33)
	K_21 = G_21*(G_11*K11 + G_12*K12 + G_13*K13) + G_22*(G_11*K21 + G_12*K22 + G_13*K23) + G_23*(G_11*K31 + G_12*K32 + G_13*K33)
	K_22 = G_21*(G_21*K11 + G_22*K12 + G_23*K13) + G_22*(G_21*K21 + G_22*K22 + G_23*K23) + G_23*(G_21*K31 + G_22*K32 + G_23*K33)
	K_23 = G_21*(G_31*K11 + G_32*K12 + G_33*K13) + G_22*(G_31*K21 + G_32*K22 + G_33*K23) + G_23*(G_31*K31 + G_32*K32 + G_33*K33)
	K_31 = G_31*(G_11*K11 + G_12*K12 + G_13*K13) + G_32*(G_11*K21 + G_12*K22 + G_13*K23) + G_33*(G_11*K31 + G_12*K32 + G_13*K33)
	K_32 = G_31*(G_21*K11 + G_22*K12 + G_23*K13) + G_32*(G_21*K21 + G_22*K22 + G_23*K23) + G_33*(G_21*K31 + G_22*K32 + G_23*K33)
	K_33 = G_31*(G_31*K11 + G_32*K12 + G_33*K13) + G_32*(G_31*K21 + G_32*K22 + G_33*K23) + G_33*(G_31*K31 + G_32*K32 + G_33*K33)

	
	#define mixed Ricci tensor (first index up second down)
	K1_1 = G_11*K11 + G_12*K21 + G_13*K31
	K1_2 = G_11*K12 + G_12*K22 + G_13*K32
	K1_3 = G_11*K13 + G_12*K23 + G_13*K33
	K2_1 = G_21*K11 + G_22*K21 + G_23*K31
	K2_2 = G_21*K12 + G_22*K22 + G_23*K32
	K2_3 = G_21*K13 + G_22*K23 + G_23*K33
	K3_1 = G_31*K11 + G_32*K21 + G_33*K31
	K3_2 = G_31*K12 + G_32*K22 + G_33*K32
	K3_3 = G_31*K13 + G_32*K23 + G_33*K33

	#define k group to derive for the momentum constraint
	kder_11 = K_11 - G_11*K
	kder_12 = K_12 - G_12*K
	kder_13 = K_13 - G_13*K
	kder_21 = K_21 - G_21*K
	kder_22 = K_22 - G_22*K
	kder_23 = K_23 - G_23*K
	kder_31 = K_31 - G_31*K
	kder_32 = K_32 - G_32*K
	kder_33 = K_33 - G_33*K
	
	#define covariant derivative of k group
	D1kder_11 = diff(kder_11, x) + kder_11*C111 + kder_21*C121 + kder_31*C131 + kder_11*C111 + Kder_12*C121 + Kder_13*C131
	D2kder_12 = diff(kder_12, y) + kder_12*C112 + kder_22*C122 + kder_32*C132 + kder_11*C212 + Kder_12*C222 + Kder_13*C232
	D3kder_13 = diff(kder_13, z) + kder_13*C113 + kder_23*C123 + kder_33*C133 + kder_11*C313 + Kder_12*C323 + Kder_13*C333
	D1kder_21 = diff(kder_21, x) + kder_11*C211 + kder_21*C221 + kder_31*C231 + kder_21*C111 + Kder_22*C121 + Kder_23*C131
	D2kder_22 = diff(kder_22, y) + kder_12*C212 + kder_22*C222 + kder_32*C232 + kder_21*C212 + Kder_22*C222 + Kder_23*C232
	D3kder_23 = diff(kder_23, z) + kder_13*C213 + kder_23*C223 + kder_33*C233 + kder_21*C313 + Kder_22*C323 + Kder_23*C333
	D1kder_31 = diff(kder_31, x) + kder_11*C311 + kder_21*C321 + kder_31*C331 + kder_31*C111 + Kder_32*C121 + Kder_33*C131
	D2kder_32 = diff(kder_32, y) + kder_12*C312 + kder_22*C322 + kder_32*C332 + kder_31*C212 + Kder_32*C222 + Kder_33*C232
	D3kder_33 = diff(kder_33, z) + kder_13*C313 + kder_23*C323 + kder_33*C333 + kder_31*C313 + Kder_32*C323 + Kder_33*C333
	

	# define lapse second covariant derivative
	DDa_11 = diff(diff(a,x),x) - diff(a,x)*C111 - diff(a,y)*C211 - diff(a,z)*C311
	DDa_12 = diff(diff(a,y),x) - diff(a,x)*C112 - diff(a,y)*C212 - diff(a,z)*C312
	DDa_13 = diff(diff(a,z),x) - diff(a,x)*C113 - diff(a,y)*C213 - diff(a,z)*C313
	DDa_21 = diff(diff(a,x),y) - diff(a,x)*C121 - diff(a,y)*C221 - diff(a,z)*C321
	DDa_22 = diff(diff(a,y),y) - diff(a,x)*C122 - diff(a,y)*C222 - diff(a,z)*C322
	DDa_23 = diff(diff(a,z),y) - diff(a,x)*C123 - diff(a,y)*C223 - diff(a,z)*C323
	DDa_31 = diff(diff(a,x),z) - diff(a,x)*C131 - diff(a,y)*C231 - diff(a,z)*C331
	DDa_32 = diff(diff(a,y),z) - diff(a,x)*C132 - diff(a,y)*C232 - diff(a,z)*C332
	DDa_33 = diff(diff(a,z),z) - diff(a,x)*C133 - diff(a,y)*C233 - diff(a,z)*C333
	
	# define source groupe term in curvature evolution equation
	Stot11 = S11 - 1/2.*G11*(S-rho)
	Stot12 = S12 - 1/2.*G12*(S-rho)
	Stot13 = S13 - 1/2.*G13*(S-rho)
	Stot21 = S21 - 1/2.*G21*(S-rho)
	Stot22 = S22 - 1/2.*G22*(S-rho)
	Stot23 = S23 - 1/2.*G23*(S-rho)
	Stot31 = S31 - 1/2.*G31*(S-rho)
	Stot32 = S32 - 1/2.*G32*(S-rho)
	Stot33 = S33 - 1/2.*G33*(S-rho)


	# written only for the 6 independent terms
	return [R + K**2 - K11*k_11 - K12*k_12 - K13*k_13 - K21*k_21 - K22*k_22 - K23*k_23 - K31*k_31 - K32*k_32 - K33*k_33 - 16*np.pi*rho,
	D1kder_11 + D2kder_12 + D3kder_13 - 8*np.pi*S1,
	D1kder_21 + D2kder_22 + D3kder_23 - 8*np.pi*S2,
	D1kder_31 + D2kder_32 + D3kder_33 - 8*np.pi*S3,
	-2*a*K11 + (diff(b1,x) - b1*C111 - b2*C211 - b3*C311) + (diff(b1,x) - b1*C111 - b2*C211 - b3*C311) - diff(G11,t),
	-2*a*K12 + (diff(b2,x) - b1*C112 - b2*C212 - b3*C312) + (diff(b1,y) - b1*C121 - b2*C221 - b3*C321) - diff(G12,t),
	-2*a*K13 + (diff(b3,x) - b1*C113 - b2*C213 - b3*C313) + (diff(b1,z) - b1*C131 - b2*C231 - b3*C331) - diff(G13,t),
	-2*a*K21 + (diff(b1,y) - b1*C121 - b2*C221 - b3*C321) + (diff(b2,x) - b1*C112 - b2*C212 - b3*C312) - diff(G21,t),
	-2*a*K22 + (diff(b2,y) - b1*C122 - b2*C222 - b3*C322) + (diff(b2,y) - b1*C122 - b2*C222 - b3*C322) - diff(G22,t),
	-2*a*K23 + (diff(b3,y) - b1*C123 - b2*C223 - b3*C323) + (diff(b2,z) - b1*C132 - b2*C232 - b3*C332) - diff(G23,t),
	-2*a*K31 + (diff(b1,z) - b1*C131 - b2*C231 - b3*C331) + (diff(b3,x) - b1*C113 - b2*C213 - b3*C313) - diff(G31,t),
	-2*a*K32 + (diff(b2,z) - b1*C132 - b2*C232 - b3*C332) + (diff(b3,y) - b1*C123 - b2*C223 - b3*C323) - diff(G32,t),
	-2*a*K33 + (diff(b3,z) - b1*C133 - b2*C233 - b3*C333) + (diff(b3,z) - b1*C133 - b2*C233 - b3*C333) - diff(G33,t),
	a*(R11 - 2*K11*K1_1 - 2*K12*K2_1 - 2*K13*K3_1 + K*K11) - DDa_11 - 8*np.pi*a*Stot11 + b1*diff(K11,x) + b2*diff(K11,y) + b3*diff(K11,z) + K11*diff(b1,x) + K12*diff(b2,x) + K13*diff(b3,x) + K11*diff(b1,x) + K21*diff(b2,x) + K31*diff(b3,x) - diff(K11,t),
	a*(R12 - 2*K11*K1_2 - 2*K12*K2_2 - 2*K13*K3_2 + K*K12) - DDa_12 - 8*np.pi*a*Stot12 + b1*diff(K12,x) + b2*diff(K12,y) + b3*diff(K12,z) + K11*diff(b1,y) + K12*diff(b2,y) + K13*diff(b3,y) + K12*diff(b1,x) + K22*diff(b2,x) + K32*diff(b3,x) - diff(K12,t),
	a*(R13 - 2*K11*K1_3 - 2*K12*K2_3 - 2*K13*K3_3 + K*K13) - DDa_13 - 8*np.pi*a*Stot13 + b1*diff(K13,x) + b2*diff(K13,y) + b3*diff(K13,z) + K11*diff(b1,z) + K12*diff(b2,z) + K13*diff(b3,z) + K13*diff(b1,x) + K23*diff(b2,x) + K33*diff(b3,x) - diff(K13,t),
	a*(R21 - 2*K21*K1_1 - 2*K22*K2_1 - 2*K23*K3_1 + K*K11) - DDa_21 - 8*np.pi*a*Stot21 + b1*diff(K21,x) + b2*diff(K21,y) + b3*diff(K21,z) + K21*diff(b1,x) + K22*diff(b2,x) + K23*diff(b3,x) + K11*diff(b1,y) + K21*diff(b2,y) + K31*diff(b3,y) - diff(K21,t),
	a*(R22 - 2*K21*K1_2 - 2*K22*K2_2 - 2*K23*K3_2 + K*K22) - DDa_22 - 8*np.pi*a*Stot22 + b1*diff(K22,x) + b2*diff(K22,y) + b3*diff(K22,z) + K21*diff(b1,y) + K22*diff(b2,y) + K23*diff(b3,y) + K12*diff(b1,y) + K22*diff(b2,y) + K32*diff(b3,y) - diff(K22,t),
	a*(R23 - 2*K21*K1_3 - 2*K22*K2_3 - 2*K23*K3_3 + K*K23) - DDa_23 - 8*np.pi*a*Stot23 + b1*diff(K23,x) + b2*diff(K23,y) + b3*diff(K23,z) + K21*diff(b1,z) + K22*diff(b2,z) + K23*diff(b3,z) + K13*diff(b1,y) + K23*diff(b2,y) + K33*diff(b3,y) - diff(K23,t),
	a*(R31 - 2*K31*K1_1 - 2*K32*K2_1 - 2*K33*K3_1 + K*K31) - DDa_31 - 8*np.pi*a*Stot31 + b1*diff(K31,x) + b2*diff(K31,y) + b3*diff(K31,z) + K31*diff(b1,x) + K32*diff(b2,x) + K33*diff(b3,x) + K11*diff(b1,z) + K21*diff(b2,z) + K31*diff(b3,z) - diff(K31,t),
	a*(R32 - 2*K31*K1_2 - 2*K32*K2_2 - 2*K33*K3_2 + K*K32) - DDa_32 - 8*np.pi*a*Stot32 + b1*diff(K32,x) + b2*diff(K32,y) + b3*diff(K32,z) + K31*diff(b1,y) + K32*diff(b2,y) + K33*diff(b3,y) + K12*diff(b1,z) + K22*diff(b2,z) + K32*diff(b3,z) - diff(K32,t),
	a*(R33 - 2*K31*K1_3 - 2*K32*K2_3 - 2*K33*K3_3 + K*K33) - DDa_33 - 8*np.pi*a*Stot33 + b1*diff(K33,x) + b2*diff(K33,y) + b3*diff(K33,z) + K31*diff(b1,z) + K32*diff(b2,z) + K33*diff(b3,z) + K13*diff(b1,z) + K23*diff(b2,z) + K33*diff(b3,z) - diff(K33,t),
	G_11*K11 + G_22*K22 + G_33*K33 - K
	]

########################################################################
########################################################################
#-----------------------------------------------------------------------
### configure the solver ####
#-----------------------------------------------------------------------
# define cst and boundary values
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
Z_MIN, Z_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 12.0


# specify the initial conditions
# ~init_vals = [IVP(t_0=0.0, x_0=0.0),#K11 
# ~IVP(t_0=0.0, x_0=0.0),  #K12
# ~IVP(t_0=0.0, x_0=0.0), #K13
# ~IVP(t_0=0.0, x_0=0.0), #K21
# ~IVP(t_0=0.0, x_0=0.0), #K22
# ~IVP(t_0=0.0, x_0=0.0), #K23
# ~IVP(t_0=0.0, x_0=0.0), #K31
# ~IVP(t_0=0.0, x_0=0.0), #K32
# ~IVP(t_0=0.0, x_0=0.0), #K33
# ~IVP(t_0=0.0, x_0=0.0), #K
# ~IVP(t_0=0.0, x_0=lambda x: torch.sqrt(1 - 2*M/x)), #a
# ~IVP(t_0=0.0, x_0=0.0), #b1
# ~IVP(t_0=0.0, x_0=0.0), #b2
# ~IVP(t_0=0.0, x_0=0.0), #b3
# ~IVP(t_0=0.0, x_0=lambda x: 1./(1 - (2*M)/x)), #G11
# ~IVP(t_0=0.0, x_0=0.0), #G12
# ~IVP(t_0=0.0, x_0=0.0), #G13
# ~IVP(t_0=0.0, x_0=0.0), #G21
# ~IVP(t_0=0.0, x_0=lambda x: 1./(1 - (2*M)/x) * x**2), #G22
# ~IVP(t_0=0.0, x_0=0.0), #G23
# ~IVP(t_0=0.0, x_0=0.0), #G31
# ~IVP(t_0=0.0, x_0=0.0), #G32
# ~IVP(t_0=0.0, x_0=lambda x: 1./(1 - (2*M)/x) * x**2 * torch.sin(y)**2), #G33
# ~]

init_vals = [FirstOrderInitialCondition(u0=0.0),#K11 
FirstOrderInitialCondition(u0=0.0),  #K12
FirstOrderInitialCondition(u0=0.0), #K13
FirstOrderInitialCondition(u0=0.0), #K21
FirstOrderInitialCondition(u0=0.0), #K22
FirstOrderInitialCondition(u0=0.0), #K23
FirstOrderInitialCondition(u0=0.0), #K31
FirstOrderInitialCondition(u0=0.0), #K32
FirstOrderInitialCondition(u0=0.0), #K33
FirstOrderInitialCondition(u0=0.0), #K
FirstOrderInitialCondition(u0=lambda x: torch.sqrt(1 - 2*M/x)), #a
FirstOrderInitialCondition(u0=0.0), #b1
FirstOrderInitialCondition(u0=0.0), #b2
FirstOrderInitialCondition(u0=0.0), #b3
FirstOrderInitialCondition(u0=lambda x: 1./(1 - (2*M)/x)), #G11
FirstOrderInitialCondition(u0=0.0), #G12
FirstOrderInitialCondition(u0=0.0), #G13
FirstOrderInitialCondition(u0=0.0), #G21
FirstOrderInitialCondition(u0=lambda x: 1./(1 - (2*M)/x) * x**2), #G22
FirstOrderInitialCondition(u0=0.0), #G23
FirstOrderInitialCondition(u0=0.0), #G31
FirstOrderInitialCondition(u0=0.0), #G32
FirstOrderInitialCondition(u0=lambda x: 1./(1 - (2*M)/x) * x**2 * torch.sin(y)**2), #G33
]

# specify the network to be used to approximate each dependent variable
# ~fcnn = [FCNN(n_hidden_units=32, n_hidden_layers=2, actv=SinActv),
# ~FCNN(n_hidden_units=32, n_hidden_layers=2, actv=SinActv)]


#n_input is the number of independent variables
#n_output is the number of dependent variables
fcnn = FCNN(
    n_input_units=4,
    n_output_units=23,
    n_hidden_units=64,
    n_hidden_layers=2,
    actv=nn.Tanh
)

fcnn_approximator = SingleNetworkApproximator3DSpatialTemporal(
    single_network=fcnn,
    pde=ADM,
    initial_condition=init_vals
)

adam = optim.Adam(fcnn_approximator.parameters(), lr=0.001)

train_gen_spatial = generator_3dspatial_cube(size=(16, 16), x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, z_min=Z_MIN, z_max=Z_MAX)
train_gen_temporal = generator_temporal(size=32, t_min=T_MIN, t_max=T_MAX)
train_gen_spatial = generator_3dspatial_cube(size=(8, 8), x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, z_min=Z_MIN, z_max=Z_MAX)
valid_gen_temporal = generator_temporal(size=16, t_min=T_MIN, t_max=T_MAX, random=False)

some_3d_time_dependent_pde_solution, _ = _solve_3dspatial_temporal(
    train_generator_spatial=train_gen_spatial,
    train_generator_temporal=train_gen_temporal,
    valid_generator_spatial=valid_gen_spatial,
    valid_generator_temporal=valid_gen_temporal,
    approximator=fcnn_approximator,
    optimizer=adam,
    batch_size=512,
    max_epochs=5000,
    shuffle=True,
    metrics={},
    monitor=MonitorMinimal(check_every=10)
)

