import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq import diff      # the differentiation operation
from neurodiffeq.ode import solve # the ANN-based solver
from neurodiffeq.ode import IVP   # the initial condition
from neurodiffeq.ode import solve_system
# ~from neurodiffeq.ode import solve3D_system
from neurodiffeq.networks import FCNN    # fully-connect neural network
from neurodiffeq.networks import SinActv # sin activation
from neurodiffeq.monitors import Monitor1D


#-------------------------------------------------------------------------

# specify the ODE system and its parameters
N = 100
beta, gamma = 0.8,0.3
cov = lambda s, i, r, t : [diff(s, t) + (beta*s*i/N),
diff(i, t) - (beta*s*i/N) + gamma*i,
diff(r, t) - gamma*i, s+i+r-N]

# specify the initial conditions
init_vals_lv = [IVP(t_0=0.0, x_0=N-5.0), IVP(t_0=0.0, x_0=5.0), IVP(t_0=0.0, x_0=0.0)]

# specify the network to be used to approximate each dependent variable
nets_lv = [FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),
FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv)]

# solve the ODE system
#~ solution_lv, _ = solve_system(ode_system=cov, conditions=init_vals_lv, t_min=0.0, t_max=12,
#~ nets=nets_lv, max_epochs=48000, monitor=Monitor(t_min=0.0, t_max=12, check_every=100))
solution_lv, loss_ex = solve_system(ode_system=cov, conditions=init_vals_lv, t_min=0.0, t_max=12,
nets=nets_lv, max_epochs=2000, monitor=Monitor1D(t_min=0.0, t_max=12, check_every=100))


# ~plt.figure()
# ~plt.plot(loss_ex['train_loss'], label='training loss')
# ~plt.plot(loss_ex['valid_loss'], label='validation loss')
# ~plt.yscale('log')
# ~plt.title('loss during training')
# ~plt.legend()
# ~plt.show()

ts = np.linspace(0, 12, 100)

# ANN-based solution
s_net, i_net, r_net = solution_lv(ts, as_type='np')

plt.figure()
plt.plot(ts, s_net, label='suceptible')
plt.plot(ts, i_net, label='infected')
plt.plot(ts, r_net, label='recoverd')

plt.ylabel('population')
plt.xlabel('t')
plt.title('beta = '+str(beta)+', gamma = '+str(gamma))
plt.legend()
plt.show()
