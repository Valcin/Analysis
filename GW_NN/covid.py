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
from neurodiffeq.callbacks import MonitorCallback
from neurodiffeq.solvers import Solver1D
from neurodiffeq.conditions import DirichletBVP2D
from neurodiffeq.solvers import Solver2D
from neurodiffeq.monitors import Monitor2D
from neurodiffeq.generators import Generator2D
import torch
from mpl_toolkits.mplot3d  import Axes3D


def plt_surf(xx, yy, zz, z_label='u', x_label='x', y_label='y', title=''):
    fig  = plt.figure(figsize=(16, 8))
    ax   = Axes3D(fig)
    surf = ax.plot_surface(xx, yy, zz, rstride=2, cstride=1, alpha=0.8, cmap='hot')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    fig.suptitle(title)
    ax.set_proj_type('ortho')
    plt.show()
#-------------------------------------------------------------------------

# specify the ODE system and its parameters
# ~N = 100
# ~beta, gamma = 0.8,0.3
# ~cov = lambda s, i, r, t : [diff(s, t) + (beta*s*i/N),
# ~diff(i, t) - (beta*s*i/N) + gamma*i,
# ~diff(r, t) - gamma*i, s+i+r-N]

# ~# specify the initial conditions
# ~init_vals_lv = [IVP(t_0=0.0, x_0=N-5.0), IVP(t_0=0.0, x_0=5.0), IVP(t_0=0.0, x_0=0.0)]

# ~# specify the network to be used to approximate each dependent variable
# ~nets_lv = [FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),
# ~FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv),FCNN(n_hidden_units=32, n_hidden_layers=1, actv=SinActv)]

# ~# solve the ODE system
# ~#~ solution_lv, _ = solve_system(ode_system=cov, conditions=init_vals_lv, t_min=0.0, t_max=12,
# ~#~ nets=nets_lv, max_epochs=48000, monitor=Monitor(t_min=0.0, t_max=12, check_every=100))
# ~solution_lv, loss_ex = solve_system(ode_system=cov, conditions=init_vals_lv, t_min=0.0, t_max=22,
# ~nets=nets_lv, max_epochs=2000)


# ~plt.figure()
# ~plt.plot(loss_ex['train_loss'], label='training loss')
# ~plt.plot(loss_ex['valid_loss'], label='validation loss')
# ~plt.yscale('log')
# ~plt.title('loss during training')
# ~plt.legend()
# ~plt.show()

# ~ts = np.linspace(0, 22, 200)

# ~# ANN-based solution
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

########################################################################
########################################################################
from neurodiffeq.conditions import IBVP1D
from neurodiffeq.pde import make_animation

k, L, T = 0.3, 2, 3
# Define the PDE system
# There's only one (heat) equation in the system, so the function maps (u, x, y) to a single entry
heat = lambda u, x, t: [
    diff(u, t) - k * diff(u, x, order=2)
]

# Define the initial and boundary conditions
# There's only one function to be solved for, so we only have a single condition object
conditions = [
    IBVP1D(
        t_min=0, t_min_val=lambda x: torch.sin(np.pi * x / L),
        x_min=0, x_min_prime=lambda t:  np.pi/L * torch.exp(-k*np.pi**2*t/L**2),
        x_max=L, x_max_prime=lambda t: -np.pi/L * torch.exp(-k*np.pi**2*t/L**2)
    )
]

# Define the neural network to be used
# Again, there's only one function to be solved for, so we only have a single network
nets = [
    FCNN(n_input_units=2, hidden_units=(32, 32))
]


# Define the monitor callback
monitor=Monitor2D(check_every=10, xy_min=(0, 0), xy_max=(L, T))
monitor_callback = MonitorCallback(monitor)

# Instantiate the solver
solver = Solver2D(
    pde_system=heat,
    conditions=conditions,
    xy_min=(0, 0),  # We can omit xy_min when both train_generator and valid_generator are specified
    xy_max=(L, T),  # We can omit xy_max when both train_generator and valid_generator are specified
    nets=nets,
    train_generator=Generator2D((32, 32), (0, 0), (L, T), method='equally-spaced-noisy'),
    valid_generator=Generator2D((32, 32), (0, 0), (L, T), method='equally-spaced'),
)

# Fit the neural network
solver.fit(max_epochs=200, callbacks=[monitor_callback])

# Obtain the solution
solution_neural_net_heat = solver.get_solution()

xs = np.linspace(0, L, 101)
ts = np.linspace(0, T, 101)
xx, tt = np.meshgrid(xs, ts)
make_animation(solution_neural_net_heat, xs, ts)
solution_analytical_heat = lambda x, t: np.sin(np.pi*x/L) * np.exp(-k * np.pi**2 * t / L**2)
sol_ana = solution_analytical_heat(xx, tt)
sol_net = solution_neural_net_heat(xx, tt, to_numpy=True)
plt_surf(xx, tt, sol_net-sol_ana, y_label='t', z_label='residual of the neural network solution')
