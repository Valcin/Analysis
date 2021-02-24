import torch
from torch import nn, optim
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.temporal import generator_3dspatial_body, generator_3dspatial_surface, generator_temporal
from neurodiffeq.temporal import FirstOrderInitialCondition, BoundaryCondition
from neurodiffeq.temporal import SingleNetworkApproximator3DSpatialTemporal
from neurodiffeq.temporal import MonitorMinimal
from neurodiffeq.temporal import _solve_3dspatial_temporal

def some_3d_time_dependent_pde(u, x, y, z, t):
    return diff(u, x) + diff(u, y) + diff(u, z) + diff(u, t) ...

# e.g. make u(x, y, z, t) = x^2 +y^2 + z^2 at the boundary
boundary_surface_1 = BoundaryCondition(
    form=lambda u, x, y, z: u - (x**2 + y**2 + z**2),
    points_generator=generator_3dspatial_surface( ... )
)
boundary_surface_2 = BoundaryCondition(
    form=lambda u, x, y, z: u - (x**2 + y**2 + z**2),
    points_generator=generator_3dspatial_surface( ... )
)
boundary_surface_3 = BoundaryCondition(
    form=lambda u, x, y, z: u - (x**2 + y**2 + z**2),
    points_generator=generator_3dspatial_surface( ... )
)

fcnn = FCNN(
    n_input_units=4,
    n_output_units=1,
    n_hidden_units=32,
    n_hidden_layers=1,
    actv=nn.Tanh
)
fcnn_approximator = SingleNetworkApproximator3DSpatialTemporal(
    single_network=fcnn,
    pde=some_3d_time_dependent_pde,
    boundary_conditions=[
        boundary_surface_1,
        boundary_surface_2,
        boundary_surface_3,
    ]
)
adam = optim.Adam(fcnn_approximator.parameters(), lr=0.001)

train_gen_spatial = generator_3dspatial_body(...)
train_gen_temporal = generator_temporal(...)
valid_gen_spatial = generator_3dspatial_body(...)
valid_gen_temporal = generator_temporal(...)

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

########################################################################
########################################################################
#-----------------------------------------------------------------------
### configure the solver ####
#-----------------------------------------------------------------------


# specify the ODE system and its parameters
def ADM(K11,K12,K13,K22,K23,K33,K,a,b1,b2,b3,G11,G12,G13,G22,G23,G33, r, theta, phi, t):

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
	C111 = 1/2.*G_11*(diff(G11, r) + diff(G11, r) - diff(G11, r))
	C112 = 1/2.*G_11*(diff(G11, theta) + diff(G12, r) - diff(G12, r))
	C113 = 1/2.*G_11*(diff(G13, phi) + diff(G13, r) - diff(G13, r))
	C121 = 1/2.*G_11*(diff(G12, r) + diff(G11, theta) - diff(G21, r))
	C122 = 1/2.*G_11*(diff(G12, theta) + diff(G12, theta) - diff(G22, r))
	C123 = 1/2.*G_11*(diff(G12, phi) + diff(G13, theta) - diff(G23, r))
	C131 = 1/2.*G_11*(diff(G13, r) + diff(G11, phi) - diff(G31, r))
	C132 = 1/2.*G_11*(diff(G13, theta) + diff(G12, phi) - diff(G32, r))
	C133 = 1/2.*G_11*(diff(G13, phi) + diff(G13, phi) - diff(G33, r))
	C211 = 1/2.*G_22*(diff(G21, r) + diff(G21, r) - diff(G11, theta))
	C212 = 1/2.*G_22*(diff(G21, theta) + diff(G22, r) - diff(G12, theta))
	C213 = 1/2.*G_22*(diff(G21, phi) + diff(G23, r) - diff(G13, theta))
	C221 = 1/2.*G_22*(diff(G22, r) + diff(G21, theta) - diff(G21, theta))
	C222 = 1/2.*G_22*(diff(G22, theta) + diff(G22, theta) - diff(G22, theta))
	C223 = 1/2.*G_22*(diff(G22, phi) + diff(G23, theta) - diff(G23, theta))
	C231 = 1/2.*G_22*(diff(G23, r) + diff(G21, phi) - diff(G31, theta))
	C232 = 1/2.*G_22*(diff(G23, theta) + diff(G22, phi) - diff(G32, theta))
	C233 = 1/2.*G_22*(diff(G23, phi) + diff(G23, phi) - diff(G33, theta))
	C311 = 1/2.*G_33*(diff(G31, r) + diff(G31, r) - diff(G11, phi))
	C312 = 1/2.*G_33*(diff(G31, theta) + diff(G32, r) - diff(G12, phi))
	C313 = 1/2.*G_33*(diff(G31, phi) + diff(G33, r) - diff(G13, phi))
	C321 = 1/2.*G_33*(diff(G32, r) + diff(G31, theta) - diff(G21, phi))
	C322 = 1/2.*G_33*(diff(G32, theta) + diff(G32, theta) - diff(G22, phi))
	C323 = 1/2.*G_33*(diff(G32, phi) + diff(G33, theta) - diff(G23, phi))
	C331 = 1/2.*G_33*(diff(G33, r) + diff(G31, phi) - diff(G31, phi))
	C332 = 1/2.*G_33*(diff(G33, theta) + diff(G32, phi) - diff(G32, phi))
	C333 = 1/2.*G_33*(diff(G33, phi) + diff(G33, phi) - diff(G33, phi))
 

	# define Ricci iteration
	R11_1 = diff(C111, r) - diff(C111, r) + C111*C111 + C111*C212 + C111*C313 - C111*C111 - C112*C211 - C113*C311
	R11_2 = diff(C211, theta) - diff(C212, r) + C211*C121 + C211*C222 + C211*C323 - C211*C112 - C212*C212 - C213*C312
	R11_3 = diff(C311, phi) - diff(C313, r) + C311*C131 + C311*C232 + C311*C333 - C311*C113 - C312*C213 - C313*C313
	R12_1 = diff(C112, r) - diff(C111, theta) + C112*C111 + C112*C212 + C112*C313 - C111*C121 - C112*C221 - C113*C321
	R12_2 = diff(C212, theta) - diff(C212, theta) + C212*C121 + C212*C222 + C212*C323 - C211*C122 - C212*C222 - C213*C322
	R12_3 = diff(C312, phi) - diff(C313, theta) + C312*C131 + C312*C232 + C312*C333 - C311*C123 - C312*C223 - C313*C323
	R13_1 = diff(C113, r) - diff(C111, phi) + C113*C111 + C113*C212 + C113*C313 - C111*C131 - C112*C231 - C113*C331
	R13_2 = diff(C213, theta) - diff(C212, phi) + C213*C121 + C213*C222 + C213*C323 - C211*C132 - C212*C232 - C213*C332
	R13_3 = diff(C313, phi) - diff(C313, phi) + C313*C131 + C313*C232 + C313*C333 - C311*C133 - C312*C233 - C313*C333
	
	R21_1 = diff(C121, r) - diff(C121, r) + C121*C111 + C121*C212 + C121*C313 - C121*C111 - C122*C211 - C123*C311
	R21_2 = diff(C221, theta) - diff(C222, r) + C221*C121 + C221*C222 + C221*C323 - C221*C112 - C222*C212 - C223*C312
	R21_3 = diff(C321, phi) - diff(C323, r) + C311*C131 + C321*C232 + C321*C333 - C321*C113 - C322*C213 - C323*C313
	R22_1 = diff(C122, r) - diff(C121, theta) + C122*C111 + C122*C212 + C122*C313 - C121*C121 - C122*C221 - C123*C321
	R22_2 = diff(C222, theta) - diff(C222, theta) + C222*C121 + C222*C222 + C222*C323 - C221*C122 - C222*C222 - C223*C322
	R22_3 = diff(C322, phi) - diff(C323, theta) + C322*C131 + C322*C232 + C322*C333 - C321*C123 - C322*C223 - C323*C323
	R23_1 = diff(C123, r) - diff(C121, phi) + C123*C111 + C123*C212 + C123*C313 - C121*C131 - C122*C231 - C123*C331
	R23_2 = diff(C223, theta) - diff(C222, phi) + C223*C121 + C223*C222 + C223*C323 - C221*C132 - C222*C232 - C223*C332
	R23_3 = diff(C323, phi) - diff(C323, phi) + C323*C131 + C323*C232 + C323*C333 - C321*C133 - C322*C233 - C323*C333

	R31_1 = diff(C131, r) - diff(C131, r) + C131*C111 + C131*C212 + C131*C313 - C131*C111 - C132*C211 - C133*C311
	R31_2 = diff(C231, theta) - diff(C232, r) + C231*C121 + C231*C222 + C231*C323 - C231*C112 - C232*C212 - C233*C312
	R31_3 = diff(C331, phi) - diff(C333, r) + C331*C131 + C331*C232 + C331*C333 - C331*C113 - C332*C213 - C333*C313
	R32_1 = diff(C132, r) - diff(C131, theta) + C132*C111 + C132*C212 + C132*C313 - C131*C121 - C132*C221 - C133*C321
	R32_2 = diff(C232, theta) - diff(C232, theta) + C232*C121 + C232*C222 + C232*C323 - C231*C122 - C232*C222 - C233*C322
	R32_3 = diff(C332, phi) - diff(C333, theta) + C332*C131 + C332*C232 + C332*C333 - C331*C123 - C332*C223 - C333*C323
	R33_1 = diff(C133, r) - diff(C131, phi) + C133*C111 + C133*C212 + C133*C313 - C131*C131 - C132*C231 - C133*C331
	R33_2 = diff(C233, theta) - diff(C232, phi) + C233*C121 + C233*C222 + C233*C323 - C231*C132 - C232*C232 - C233*C332
	R33_3 = diff(C333, phi) - diff(C333, phi) + C333*C131 + C333*C232 + C333*C333 - C331*C133 - C332*C233 - C333*C333

	
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

	#define contravariant Ricci tensor (assuming only G11, G22 and G33 to modified if different)
	K_11 = G_11*G_11*K11
	K_12 = G_11*G_22*K12
	K_13 = G_11*G_33*K13
	K_21 = G_22*G_11*K21
	K_22 = G_22*G_22*K22
	K_23 = G_22*G_33*K23
	K_31 = G_33*G_11*K31
	K_32 = G_33*G_22*K32
	K_33 = G_33*G_33*K33

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
	D1kder_11 = diff(kder_11, r) + kder_11*C111 + kder_21*C121 + kder_31*C131 + kder_11*C111 + Kder_12*C121 + Kder_13*C131
	D2kder_12 = diff(kder_12, theta) + kder_12*C112 + kder_22*C122 + kder_32*C132 + kder_11*C212 + Kder_12*C222 + Kder_13*C232
	D3kder_13 = diff(kder_13, phi) + kder_13*C113 + kder_23*C123 + kder_33*C133 + kder_11*C313 + Kder_12*C323 + Kder_13*C333
	D1kder_21 = diff(kder_21, r) + kder_11*C211 + kder_21*C221 + kder_31*C231 + kder_21*C111 + Kder_22*C121 + Kder_23*C131
	D2kder_22 = diff(kder_22, theta) + kder_12*C212 + kder_22*C222 + kder_32*C232 + kder_21*C212 + Kder_22*C222 + Kder_23*C232
	D3kder_23 = diff(kder_23, phi) + kder_13*C213 + kder_23*C223 + kder_33*C233 + kder_21*C313 + Kder_22*C323 + Kder_23*C333
	D1kder_31 = diff(kder_31, r) + kder_11*C311 + kder_21*C321 + kder_31*C331 + kder_31*C111 + Kder_32*C121 + Kder_33*C131
	D2kder_32 = diff(kder_32, theta) + kder_12*C312 + kder_22*C322 + kder_32*C332 + kder_31*C212 + Kder_32*C222 + Kder_33*C232
	D3kder_33 = diff(kder_33, phi) + kder_13*C313 + kder_23*C323 + kder_33*C333 + kder_31*C313 + Kder_32*C323 + Kder_33*C333
	


	return [R + K**2 - K11*k_11 - K12*k_12 - K13*k_13 - K22*k_22 - K23*k_23 - K33*k_33 - 16*np.pi*rho,
	D1kder_11 + D2kder_12 + D3kder_13 - 8*np.pi*S1,
	D1kder_21 + D2kder_22 + D3kder_23 - 8*np.pi*S2,
	D1kder_31 + D2kder_32 + D3kder_33 - 8*np.pi*S3,
	-2*a*K11 + (diff(b1,r) - b1*C111 - b2*C211 - b3*C311) + (diff(b1,r) - b1*C111 - b2*C211 - b3*C311) - diff(G11,t),
	-2*a*K12 + (diff(b2,r) - b1*C112 - b2*C212 - b3*C312) + (diff(b1,theta) - b1*C121 - b2*C221 - b3*C321) - diff(G12,t),
	-2*a*K13 + (diff(b3,r) - b1*C113 - b2*C213 - b3*C313) + (diff(b1,phi) - b1*C131 - b2*C231 - b3*C331) - diff(G13,t),
	-2*a*K21 + (diff(b1,theta) - b1*C121 - b2*C221 - b3*C321) + (diff(b2,r) - b1*C112 - b2*C212 - b3*C312) - diff(G21,t),
	-2*a*K22 + (diff(b2,theta) - b1*C122 - b2*C222 - b3*C322) + (diff(b2,theta) - b1*C122 - b2*C222 - b3*C322) - diff(G22,t),
	-2*a*K23 + (diff(b3,theta) - b1*C123 - b2*C223 - b3*C323) + (diff(b2,phi) - b1*C132 - b2*C232 - b3*C332) - diff(G23,t),
	-2*a*K31 + (diff(b1,phi) - b1*C131 - b2*C231 - b3*C331) + (diff(b3,r) - b1*C113 - b2*C213 - b3*C313) - diff(G31,t),
	-2*a*K32 + (diff(b2,phi) - b1*C132 - b2*C232 - b3*C332) + (diff(b3,theta) - b1*C123 - b2*C223 - b3*C323) - diff(G32,t),
	-2*a*K33 + (diff(b3,phi) - b1*C133 - b2*C233 - b3*C333) + (diff(b3,phi) - b1*C133 - b2*C233 - b3*C333) - diff(G33,t),
	]

# specify the initial conditions
init_vals_lv = [IVP(t_0=0.0, x_0=0.0),#K11 
IVP(t_0=0.0, x_0=0.0),  #K12
IVP(t_0=0.0, x_0=0.0), #K13
IVP(t_0=0.0, x_0=0.0), #K22
IVP(t_0=0.0, x_0=0.0), #K23
IVP(t_0=0.0, x_0=0.0), #K33
IVP(t_0=0.0, x_0=0.0), #K
IVP(t_0=0.0, x_0=lambda r: torch.sqrt(1 - 2*M/r)), #a
IVP(t_0=0.0, x_0=0.0), #b1
IVP(t_0=0.0, x_0=0.0), #b2
IVP(t_0=0.0, x_0=0.0), #b3
IVP(t_0=0.0, x_0=lambda r: 1./(1 - (2*M)/r)), #G11
IVP(t_0=0.0, x_0=0.0), #G12
IVP(t_0=0.0, x_0=0.0), #G13
IVP(t_0=0.0, x_0=lambda r: 1./(1 - (2*M)/r) * r**2), #G22
IVP(t_0=0.0, x_0=0.0), #G23
IVP(t_0=0.0, x_0=lambda r: 1./(1 - (2*M)/r) * r**2 * torch.sin(theta)**2), #G33
]

# specify the network to be used to approximate each dependent variable
nets_lv = [FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),
FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv)]

# solve the ODE system
#~ solution_lv, _ = solve_system(ode_system=cov, conditions=init_vals_lv, t_min=0.0, t_max=12,
#~ nets=nets_lv, max_epochs=48000, monitor=Monitor(t_min=0.0, t_max=12, check_every=100))
solution_lv, _ = solve2D_system(pde_system=adm, conditions=init_vals_lv, xy_min=(0, 0), xy_max=(12, 12), max_epochs=2000)


