import numpy as np
import math
import matplotlib.pyplot as plt
from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve # the ANN-based solver
from neurodiffeq.ode import IVP   # the initial condition
from neurodiffeq.ode import solve_system
from neurodiffeq.pde import solve2D_system
from neurodiffeq.networks import FCNN    # fully-connect neural network
from neurodiffeq.networks import SinActv # sin activation
from neurodiffeq.ode import Monitor


#-------------------------------------------------------------------------
# define the Ricci tensor for all configurations
# ~R11 = (8*r*M)/(2*r**2 + M**2 * r)**2
R11 = 0
R12 = 0
R13 = 0
# ~R22 = (4*r**3 * M)/(2*r**2 + M**2 * r)**2
R22 = 0
R23 = 0
# ~R33 = np.sin(theta)**2 * R22
R33 = 0
R = 0
rho = 0
S1 = 0
S2 = 0
S3 = 0
S = 0
S11 = 0
S12 = 0
S13 = 0
S22 = 0
S23 = 0
S33 = 0

# specify the ODE system and its parameters
adm = lambda K11,K12,K13,K22,K23,K33,K,a,b1,b2,b3,G11,G12,G13,G22,G23,G33,t,r,theta,phi: [R + K**2 + (K11**2 + K12**2 + K13**2 + K22**2 + K23**2 + K33**2) - 16*math.pi*rho,
diff((K11-G11*K), r) + diff((K12-G12*K), theta) + diff((K13-G13*K), phi) - 8*math.pi*S1,
diff((K22-G22*K), theta) + diff((K23-G23*K), phi) - 8*math.pi*S2,
diff((K33-G33*K), phi) - 8*math.pi*S3,
-diff(G11,t) -2*a*K11 + diff(b1,r) + diff(b1,r),
-diff(G12,t) -2*a*K12 + diff(b2,r) + diff(b1,theta),
-diff(G13,t) -2*a*K13 + diff(b3,r) + diff(b1,phi),
-diff(G22,t) -2*a*K22 + diff(b2,theta) + diff(b2,theta),
-diff(G23,t) -2*a*K23 + diff(b3,theta) + diff(b2,phi),
-diff(G33,t) -2*a*K33 + diff(b3,phi) + diff(b3,phi),
-diff(K11,t) + a(R11 -2*(K11*K11) + K*K11) - diff(diff(a,r),r) - 8*math.pi*a*(S11 - 1/2.*G11*(S-rho)) + b1*diff(K11,r) + b2*diff(K11,theta) + b3*diff(K11,phi) + K11*diff(b1,r) + K12*diff(b2,r) + K13*diff(b3,r) + K11*diff(b1,r),
-diff(K12,t) + a(R12 -2*(K11*K12 + K12*K22) + K*K12) - diff(diff(a,theta),r) - 8*math.pi*a*(S12 - 1/2.*G12*(S-rho)) + b1*diff(K12,r) + b2*diff(K12,theta) + b3*diff(K12,phi) + K11*diff(b1,theta) + K12*diff(b2,theta) + K13*diff(b3,theta) + K12*diff(b1,r) + K22*diff(b2,r),
-diff(K13,t) + a(R13 -2*(K11*K13 + K12*K23 + K13*K33) + K*K13) - diff(diff(a,phi),r) - 8*math.pi*a*(S13 - 1/2.*G13*(S-rho)) + b1*diff(K13,r) + b2*diff(K13,theta) + b3*diff(K13,phi) + K11*diff(b1,phi) + K12*diff(b2,phi) + K13*diff(b3,phi) + K13*diff(b1,r) + K23*diff(b2,r) + K33*diff(b3,theta),
-diff(K22,t) + a(R22 -2*(K22*K22) + K*K22) - diff(diff(a,theta),theta) - 8*math.pi*a*(S22 - 1/2.*G22*(S-rho)) + b1*diff(K22,r) + b2*diff(K22,theta) + b3*diff(K22,phi) + K22*diff(b2,theta) + K23*diff(b3,theta) + K12*diff(b1,theta) + K22*diff(b2,theta),
-diff(K23,t) + a(R23 -2*(K22*K23 + K23*K33) + K*K23) - diff(diff(a,phi),theta) - 8*math.pi*a*(S23 - 1/2.*G23*(S-rho)) + b1*diff(K23,r) + b2*diff(K23,theta) + b3*diff(K23,phi) + K22*diff(b2,phi) + K23*diff(b3,phi) + K13*diff(b1,theta) + K23*diff(b2,theta) + K33*diff(b3,theta),
-diff(K33,t) + a(R33 -2*(K33*K33) + K*K33) - diff(diff(a,phi),phi) - 8*math.pi*a*(S33 - 1/2.*G33*(S-rho)) + b1*diff(K33,r) + b2*diff(K33,theta) + b3*diff(K33,phi) + K33*diff(b3,phi) + K13*diff(b1,phi) + K23*diff(b2,phi) + K33*diff(b3,phi),
-K + K11 + K22 + K33]

# specify the initial conditions
init_vals_lv = [IVP(t_0=0.0, x_0=0.0), 
IVP(t_0=0.0, x_0=0.0),  #K12
IVP(t_0=0.0, x_0=0.0), #K13
IVP(t_0=0.0, x_0=0.0), #K22
IVP(t_0=0.0, x_0=0.0), #K23
IVP(t_0=0.0, x_0=0.0), #K33
IVP(t_0=0.0, x_0=0.0), #K
IVP(t_0=0.0, x_0=lambda r: torch.sqrt(1 - 2*M/r)), #a
# ~IVP(t_0=0.0, x_0=0.0), #a
IVP(t_0=0.0, x_0=0.0), #b1
IVP(t_0=0.0, x_0=0.0), #b2
IVP(t_0=0.0, x_0=0.0), #b3
IVP(t_0=0.0, x_0=lambda r: 1./(1 - 2*M/r)), #G11
# ~IVP(t_0=0.0, x_0=0.0), #G11
IVP(t_0=0.0, x_0=0.0), #G12
IVP(t_0=0.0, x_0=0.0), #G13
IVP(t_0=0.0, x_0=lambda r: r**2), #G22
# ~IVP(t_0=0.0, x_0=0.0), #G22
IVP(t_0=0.0, x_0=0.0), #G23
IVP(t_0=0.0, x_0=lambda r: r**2 * torch.sin(theta)**2), #G33
# ~IVP(t_0=0.0, x_0=0.0), #G33
]

# specify the network to be used to approximate each dependent variable
nets_lv = [FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),
FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv)]

# solve the ODE system
#~ solution_lv, _ = solve_system(ode_system=cov, conditions=init_vals_lv, t_min=0.0, t_max=12,
#~ nets=nets_lv, max_epochs=48000, monitor=Monitor(t_min=0.0, t_max=12, check_every=100))
solution_lv, _ = solve2D_system(pde_system=adm, conditions=init_vals_lv, xy_min=(0, 0), xy_max=(12, 12), max_epochs=2000)

# ~ts = np.linspace(0, 12, 100)

# ANN-based solution
# ~s_net, i_net, r_net = solution_lv(ts, as_type='np')

# ~plt.figure()
# ~plt.plot(ts, s_net, label='suceptible')
# ~plt.plot(ts, i_net, label='infected')
# ~plt.plot(ts, r_net, label='recoverd')

# ~plt.ylabel('population')
# ~plt.xlabel('t')
# ~plt.title('beta = '+str(beta)+', gamma = '+str(gamma))
# ~plt.legend()
# ~plt.show()
