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
########################################################################
# A generator for generating 3D points in the problem domain: yield (xx, yy, zz) where xx, yy, zz are 1-D tensors
def generator_3dspatial_body(...):
    pass

# A generator for generating 3D points on the boundary: yield (xx, yy, zz) where xx, yy, zz are 1-D tensors
def generator_3dspatial_surface(...):
    pass





