import torch
import numpy as np
from neurodiffeq.temporal import Approximator
from neurodiffeq.temporal import generator_1dspatial, _solve_spatial_temporal, _cartesian_prod_dims

# the model and the loss
# should be similar to `SingleNetworkApproximator2DSpatialTemporal`
class SingleNetworkApproximator3DSpatialTemporal(Approximator):
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        self.u0 = initial_condition.u0
        self.u0dot = initial_condition.u0dot if hasattr(initial_condition, 'u0dot') else None
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy, zz, tt):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        zz = torch.unsqueeze(zz, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xyzt = torch.cat((xx, yy, zz, tt), dim=1)
        # ~uu_raw = self.single_network(xyzt)
        if self.u0dot is None:
            uu = torch.exp(-tt) * self.u0(xx, yy, zz) + (1 - torch.exp(-tt)) * self.single_network(xyzt)
        else:
            # not sure about this line
            uu = (1 - (1 - torch.exp(-tt))**2) * self.u0(xx, yy,zz) \
                 + (1 - torch.exp(-tt)) * self.u0dot(xx, yy,zz) \
                 + (1 - torch.exp(-tt))**2 * self.single_network(xyzt)
        return torch.squeeze(uu)

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy, zz, tt, x, y, z, t):
        uu = self.__call__(xx, yy, zz, tt)
        equation_mse = torch.mean(self.pde(uu, xx, yy, zz, tt)**2)
        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)
        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x, y, z = next(bc.points_generator)

        xx, tt = _cartesian_prod_dims(x, t, x_grad=True, t_grad=False)
        yy, tt = _cartesian_prod_dims(y, t, x_grad=True, t_grad=False)
        zz, tt = _cartesian_prod_dims(z, t, x_grad=True, t_grad=False)
        uu = self.__call__(xx, yy, zz, tt)
        return torch.mean(bc.form(uu, xx, yy, zz, tt) ** 2)


    def calculate_metrics(self, xx, yy, zz, tt, x, y, z, t, metrics):
        uu = self.__call__(xx, yy, zz, tt)

        return {
            metric_name: metric_func(uu, xx, yy, zz, tt)
            for metric_name, metric_func in metrics.items()
        }

class SingleNetworkApproximator3DSpatialTemporalSystem(Approximator):
    def __init__(self, single_network, pde, initial_condition, boundary_conditions, boundary_strictness=1.):
        self.single_network = single_network
        self.pde = pde
        # ~self.u0 = initial_condition.u0
        # ~self.u0dot = initial_condition.u0dot if hasattr(initial_condition, 'u0dot') else None
        self.initial_condition = (ic.u0 for ic in initial_condition)
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness

    def __call__(self, xx, yy, zz, tt):
        xx = torch.unsqueeze(xx, dim=1)
        yy = torch.unsqueeze(yy, dim=1)
        zz = torch.unsqueeze(zz, dim=1)
        tt = torch.unsqueeze(tt, dim=1)
        xyzt = torch.cat((xx, yy, zz, tt), dim=1)
        uu_raw = self.single_network(xyzt)
        print(uu_raw.size())

        # ~if self.u0dot is None:
        if not hasattr(self.initial_condition, 'u0dot'): #check if udot is defined 
            uu = torch.exp(-tt) * next(self.initial_condition)(xx, yy, zz) + (1 - torch.exp(-tt)) * self.single_network(xyzt)

        else:
            # not sure about this line
            uu = (1 - (1 - torch.exp(-tt))**2) * self.u0(xx, yy,zz) \
                 + (1 - torch.exp(-tt)) * self.u0dot(xx, yy,zz) \
                 + (1 - torch.exp(-tt))**2 * self.single_network(xyzt)
        return tuple(torch.squeeze(uu[:, i]) for i in range(uu.shape[1]))

    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy, zz, tt, x, y, z, t):
        uu = self.__call__(xx, yy, zz, tt)

        equation_mse = sum(
            torch.mean(eq**2)
            for eq in self.pde(*uu, xx, yy, zz, tt)
        )

        boundary_mse = self.boundary_strictness * sum(self._boundary_mse(t, bc) for bc in self.boundary_conditions)
        return equation_mse + boundary_mse

    def _boundary_mse(self, t, bc):
        x, y, z = next(bc.points_generator)

        xx, tt = _cartesian_prod_dims(x, t, x_grad=True, t_grad=False)
        yy, tt = _cartesian_prod_dims(y, t, x_grad=True, t_grad=False)
        zz, tt = _cartesian_prod_dims(z, t, x_grad=True, t_grad=False)
        uu = self.__call__(xx, yy, zz, tt)
        return torch.mean(bc.form(uu, xx, yy, zz, tt) ** 2)


    def calculate_metrics(self, xx, yy, zz, tt, x, y, z, t, metrics):
        uu = self.__call__(xx, yy, zz, tt)

        return {
            metric_name: metric_func(uu, xx, yy, zz, tt)
            for metric_name, metric_func in metrics.items()
        }


def _solve_3dspatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor
):
    return _solve_spatial_temporal(
        train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,
        train_routine=_train_3dspatial_temporal, valid_routine=_valid_3dspatial_temporal
    )

def _train_3dspatial_temporal(train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size):
    x, y, z = next(train_generator_spatial)
    t = next(train_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)
    yy, tt = _cartesian_prod_dims(y, t)
    zz, tt = _cartesian_prod_dims(z, t)
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)

    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx]
        batch_yy = yy[batch_idx]
        batch_zz = zz[batch_idx]
        batch_tt = tt[batch_idx]

        batch_loss = approximator.calculate_loss(batch_xx, batch_yy, batch_zz, batch_tt, x, y, z, t)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_start += batch_size
        batch_end += batch_size

    # TODO: this can give us the real loss after an epoch, but can be very memory intensive
    epoch_loss = approximator.calculate_loss(xx, yy, zz, tt, x, y, z, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, zz, tt, x, y, z, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


def _valid_3dspatial_temporal(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    x, y, z = next(valid_generator_spatial)
    t = next(valid_generator_temporal)
    xx, tt = _cartesian_prod_dims(x, t)
    yy, tt = _cartesian_prod_dims(y, t)
    zz, tt = _cartesian_prod_dims(z, t)

    epoch_loss = approximator.calculate_loss(xx, yy, zz, tt, x, y, z, t).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, zz, tt, x, y, z, t, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics


def generator_3dspatial_segment(size, start, end, random=True):
    """Return a generator that generates 3D points in a line segment.

    :param size:
        Number of points to generated when `__next__` is invoked.
    :type size: int
    :param x_min:
        Lower bound of x.
    :type x_min: float
    :param x_max:
        Upper bound of x.
    :type x_max: float
    :param y_min:
        Lower bound of y.
    :type y_min: float
    :param y_max:
        Upper bound of y.
    :param z_min:
        Lower bound of z.
    :type z_min: float
    :param z_max:
        Upper bound of z.
    :type z_max: float
    :param random:

        - If set to False, then return a grid where the points are equally spaced in the x,y and z dimension.
        - If set to True then generate points randomly.

        Defaults to True.
    :type random: bool
    """
    x1, y1, z1 = start
    x2, y2, z2 = end
    step = 1./size
    center = torch.linspace(0. + 0.5*step, 1. - 0.5*step, size)
    noise_lo = -step*0.5
    while True:
        if random:
            noise = step*torch.rand(size) + noise_lo
            center = center + noise
        yield x1 + (x2-x1)*center, y1 + (y2-y1)*center, z1 + (z2-z1)*center


def generator_3dspatial_cube(size, x_min, x_max, y_min, y_max, z_min, z_max, random=True):
    """Return a generator that generates 3D points in a cube.

    :param size:
        Number of points to generated when `__next__` is invoked.
    :type size: int
    :param start:
        The starting point of the line segment.
    :type start: tuple[float, float]
    :param end:
        The ending point of the line segment.
    :type end: tuple[float, float]
    :param random:
        - If set to False, then return eqally spaced points range from `start` to `end`.
        - If set to Rrue then generate points randomly.

        Defaults to True.
    :type random: bool
    """
    x_size, y_size, z_size = size
    x_generator = generator_1dspatial(x_size, x_min, x_max, random)
    y_generator = generator_1dspatial(y_size, y_min, y_max, random)
    z_generator = generator_1dspatial(z_size, z_min, z_max, random)
    while True:
        x = next(x_generator)
        y = next(y_generator)
        z = next(z_generator)
        xyz = torch.cartesian_prod(x, y, z)
        xx = torch.squeeze(xyz[:, 0])
        yy = torch.squeeze(xyz[:, 1])
        zz = torch.squeeze(xyz[:, 2])
        yield xx, yy, zz

########################################################################
# A generator for generating 3D points in the problem domain: yield (xx, yy, zz) where xx, yy, zz are 1-D tensors
# ~def generator_3dspatial_body(...):
    # ~pass

# ~# A generator for generating 3D points on the boundary: yield (xx, yy, zz) where xx, yy, zz are 1-D tensors
# ~def generator_3dspatial_surface(...):
    # ~pass


# ~class GeneratorSpherical(BaseGenerator):
    # ~r"""A generator for generating points in spherical coordinates.

    # ~:param size: Number of points in 3-D sphere.
    # ~:type size: int
    # ~:param r_min: Radius of the interior boundary.
    # ~:type r_min: float, optional
    # ~:param r_max: Radius of the exterior boundary.
    # ~:type r_max: float, optional
    # ~:param method:
        # ~The distribution of the 3-D points generated.

        # ~- If set to 'equally-radius-noisy', radius of the points will be drawn
          # ~from a uniform distribution :math:`r \sim U[r_{min}, r_{max}]`.
        # ~- If set to 'equally-spaced-noisy', squared radius of the points will be drawn
          # ~from a uniform distribution :math:`r^2 \sim U[r_{min}^2, r_{max}^2]`

        # ~Defaults to 'equally-spaced-noisy'.

    # ~:type method: str, optional

    # ~.. note::
        # ~Not to be confused with ``Generator3D``.
    # ~"""

    # ~# noinspection PyMissingConstructor
    # ~def __init__(self, size, r_min=0., r_max=1., method='equally-spaced-noisy'):
        # ~super(GeneratorSpherical, self).__init__()
        # ~if r_min < 0 or r_max < r_min:
            # ~raise ValueError(f"Illegal range [{r_min}, {r_max}]")

        # ~if method == 'equally-spaced-noisy':
            # ~lower = r_min ** 2
            # ~upper = r_max ** 2
            # ~rng = upper - lower
            # ~self.get_r = lambda: torch.sqrt(rng * torch.rand(self.shape) + lower)
        # ~elif method == "equally-radius-noisy":
            # ~lower = r_min
            # ~upper = r_max
            # ~rng = upper - lower
            # ~self.get_r = lambda: rng * torch.rand(self.shape) + lower
        # ~else:
            # ~raise ValueError(f'Unknown method: {method}')

        # ~self.size = size  # stored for `solve_spherical_system` to access
        # ~self.shape = (size,)  # used for `self.get_example()`

    # ~def get_examples(self):
        # ~a = torch.rand(self.shape)
        # ~b = torch.rand(self.shape)
        # ~c = torch.rand(self.shape)
        # ~denom = a + b + c
        # ~# `x`, `y`, `z` here are just for computation of `theta` and `phi`
        # ~epsilon = 1e-6
        # ~x = torch.sqrt(a / denom) + epsilon
        # ~y = torch.sqrt(b / denom) + epsilon
        # ~z = torch.sqrt(c / denom) + epsilon
        # ~# `sign_x`, `sign_y`, `sign_z` are either -1 or +1
        # ~sign_x = torch.randint(0, 2, self.shape, dtype=x.dtype) * 2 - 1
        # ~sign_y = torch.randint(0, 2, self.shape, dtype=y.dtype) * 2 - 1
        # ~sign_z = torch.randint(0, 2, self.shape, dtype=z.dtype) * 2 - 1

        # ~x = x * sign_x
        # ~y = y * sign_y
        # ~z = z * sign_z

        # ~theta = torch.acos(z).requires_grad_(True)
        # ~phi = -torch.atan2(y, x) + np.pi  # atan2 ranges (-pi, pi] instead of [0, 2pi)
        # ~phi.requires_grad_(True)
        # ~r = self.get_r().requires_grad_(True)

        # ~return r, theta, phi



