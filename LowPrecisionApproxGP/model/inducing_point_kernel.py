import copy
import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from linear_operator import to_dense
from linear_operator.operators import (
    LowRankRootAddedDiagLinearOperator,
    MatmulLinearOperator,
    DiagLinearOperator,
    LowRankRootLinearOperator,
)
from kernel_added_loss_term import (
    VarPrecisionInducingPointKernelAddedLossTerm,
)
from prediction_strategy import VarPrecisionSGPRPredictionStrategy
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.kernels import Kernel


class VarPrecisionInducingPointKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        inducing_points: Tensor,
        likelihood: Likelihood,
        dtype: torch.dtype,
        active_dims: Optional[Tuple[int, ...]] = None,
    ):
        super(VarPrecisionInducingPointKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.likelihood = likelihood
        self._dtype_to_set = dtype

        # Added check for torch.dtypes
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Dtype must be instance of torch.dtype, {dtype} is of instance {type(dtype)}"
            )

        if inducing_points.ndimension() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        inducing_points = inducing_points.type(self._dtype_to_set)
        self.register_parameter(
            name="inducing_points", parameter=torch.nn.Parameter(inducing_points)
        )
        self.register_added_loss_term("inducing_point_loss_term")

    def _clear_cache(self):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        if hasattr(self, "_cached_kernel_inv_root"):
            del self._cached_kernel_inv_root

    @property
    def _inducing_mat(self):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = to_dense(self.base_kernel(self.inducing_points, self.inducing_points))
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_mat, upper=True)
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.linalg.solve_triangular(chol, eye, upper=True)

            # TODO: See if this conversion is necessary
            res = inv_root.type(self._dtype_to_set)
            if not self.training:
                self._cached_kernel_inv_root = res
            return res

    def _get_covariance(self, x1, x2):
        k_ux1 = to_dense(self.base_kernel(x1, self.inducing_points))
        if torch.equal(x1, x2):
            covar = LowRankRootLinearOperator(k_ux1.matmul(self._inducing_inv_root))

            # Diagonal correction for predictive posterior
            if not self.training and settings.sgpr_diagonal_correction.on():
                correction = (
                    self.base_kernel(x1, x2, diag=True)
                    - covar.diagonal(dim1=-1, dim2=-2)
                ).clamp(0, math.inf)
                covar = LowRankRootAddedDiagLinearOperator(
                    covar, DiagLinearOperator(correction)
                )
        else:
            k_ux2 = to_dense(self.base_kernel(x2, self.inducing_points))
            covar = MatmulLinearOperator(
                k_ux1.matmul(self._inducing_inv_root),
                k_ux2.matmul(self._inducing_inv_root).transpose(-1, -2),
            )

        return covar

    def _covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = to_dense(self.base_kernel(inputs, diag=True))
        return DiagLinearOperator(covar_diag)

    def forward(self, x1, x2, diag=False, **kwargs):
        # TODO: See if this conversion is necessary
        x1, x2 = x1.type(self._dtype_to_set), x2.type(self._dtype_to_set)
        covar = self._get_covariance(x1, x2)

        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")

            # TODO: See if this conversion is necessary
            zero_mean = torch.zeros_like(x1.select(-1, 0)).type(self._dtype_to_set)
            new_added_loss_term = VarPrecisionInducingPointKernelAddedLossTerm(
                MultivariateNormal(zero_mean, self._covar_diag(x1)),
                MultivariateNormal(zero_mean, covar),
                self.likelihood,
            )
            self.update_added_loss_term("inducing_point_loss_term", new_added_loss_term)

        if diag:
            return covar.diagonal(dim1=-1, dim2=-2)
        else:
            return covar

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def __deepcopy__(self, memo):
        replace_inv_root = False
        replace_kernel_mat = False

        if hasattr(self, "_cached_kernel_inv_root"):
            replace_inv_root = True
            kernel_inv_root = self._cached_kernel_inv_root
        if hasattr(self, "_cached_kernel_mat"):
            replace_kernel_mat = True
            kernel_mat = self._cached_kernel_mat

        cp = self.__class__(
            base_kernel=copy.deepcopy(self.base_kernel),
            inducing_points=copy.deepcopy(self.inducing_points),
            likelihood=self.likelihood,
            active_dims=self.active_dims,
        )

        if replace_inv_root:
            cp._cached_kernel_inv_root = kernel_inv_root

        if replace_kernel_mat:
            cp._cached_kernel_mat = kernel_mat

        return cp

    def prediction_strategy(
        self, train_inputs, train_prior_dist, train_labels, likelihood
    ):
        # Use our Custom Strategy as built-in expects InducingPointKernel
        return VarPrecisionSGPRPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )
