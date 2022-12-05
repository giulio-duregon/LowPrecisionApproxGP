from email.mime import base
import gpytorch
import torch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from .inducing_point_kernel import VarPrecisionInducingPointKernel


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            # TODO: Should this always be :500?
            inducing_points=train_x[:500, :],
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VarPrecisionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dtype, mean_module, base_kernel):
        self._dtype_to_set = dtype
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = VarPrecisionInducingPointKernel(
            base_kernel,
            inducing_points=torch.empty(1),
            likelihood=likelihood,
            dtype=self._dtype_to_set,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
