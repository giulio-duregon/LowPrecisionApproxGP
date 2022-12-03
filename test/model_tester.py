import torch
import gpytorch
from LowPrecisionApproxGP.util.dtype_converter import convert_tensors_to_dtype


class ModelTester(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dtype, mean_module, covar_module):
        self._dtype_to_set = dtype
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)