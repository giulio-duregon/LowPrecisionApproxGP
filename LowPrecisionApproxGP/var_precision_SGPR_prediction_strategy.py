import torch

from linear_operator import to_linear_operator
from linear_operator.operators import (
    AddedDiagLinearOperator,
    LowRankRootAddedDiagLinearOperator,
    MatmulLinearOperator,
)
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.utils.memoize import (
    cached,
)

from var_precision_inducing_point_kernel import VarPrecisionInducingPointKernel


class VarPrecisionSGPRPredictionStrategy(DefaultPredictionStrategy):
    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Here, the covar_cache is going to be K_{UU}^{-1/2} K_{UX}( K_{XX} + \sigma^2 I )^{-1} K_{XU} K_{UU}^{-1/2}
        # This is easily computed using Woodbury
        # K_{XX} + \sigma^2 I = R R^T + \sigma^2 I
        #                     = \sigma^{-2} ( I - \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T  )
        train_train_covar = self.lik_train_train_covar.evaluate_kernel()

        # Get terms needed for woodbury
        root = train_train_covar._linear_op.root_decomposition().root.to_dense()  # R
        inv_diag = train_train_covar._diag_tensor.inverse()  # \sigma^{-2}

        # Form LT using woodbury
        ones = torch.tensor(1.0, dtype=root.dtype, device=root.device)
        chol_factor = to_linear_operator(
            root.transpose(-1, -2) @ (inv_diag @ root)
        ).add_diagonal(
            ones
        )  # (I + \sigma^{-2} R^T R)^{-1}
        woodbury_term = inv_diag @ torch.linalg.solve_triangular(
            chol_factor.cholesky().to_dense(), root.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        # woodbury_term @ woodbury_term^T = \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T \sigma^{-2}

        inverse = AddedDiagLinearOperator(
            inv_diag,
            MatmulLinearOperator(-woodbury_term, woodbury_term.transpose(-1, -2)),
        )
        # \sigma^{-2} ( I - \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T  )

        return root.transpose(-1, -2) @ (inverse @ root)

    def get_fantasy_strategy(
        self, inputs, targets, full_inputs, full_targets, full_output, **kwargs
    ):
        raise NotImplementedError(
            "Fantasy observation updates not yet supported for models using VarPrecisionSGPRPredictionStrategy"
        )

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]

        # If we're in lazy evaluation mode, let's use the base kernel of the SGPR output to compute the prior covar
        test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
        if isinstance(test_test_covar, LazyEvaluatedKernelTensor) and isinstance(
            test_test_covar.kernel, VarPrecisionInducingPointKernel
        ):
            test_test_covar = LazyEvaluatedKernelTensor(
                test_test_covar.x1,
                test_test_covar.x2,
                test_test_covar.kernel.base_kernel,
                test_test_covar.last_dim_is_batch,
                **test_test_covar.params,
            )

        test_train_covar = joint_covar[
            ..., self.num_train :, : self.num_train
        ].evaluate_kernel()

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        covar_cache = self.covar_cache
        # covar_cache = K_{UU}^{-1/2} K_{UX}( K_{XX} + \sigma^2 I )^{-1} K_{XU} K_{UU}^{-1/2}

        # Decompose test_train_covar = l, r
        # Main case: test_x and train_x are different - test_train_covar is a MatmulLinearOperator
        if isinstance(test_train_covar, MatmulLinearOperator):
            L = test_train_covar.left_linear_op.to_dense()
        # Edge case: test_x and train_x are the same - test_train_covar is a LowRankRootAddedDiagLinearOperator
        elif isinstance(test_train_covar, LowRankRootAddedDiagLinearOperator):
            L = test_train_covar._linear_op.root.to_dense()
        else:
            # We should not hit this point of the code - this is to catch potential bugs in GPyTorch
            raise ValueError(
                "Expected SGPR output to be a MatmulLinearOperator or AddedDiagLinearOperator. "
                f"Got {test_train_covar.__class__.__name__} instead. "
                "This is likely a bug in GPyTorch."
            )

        res = test_test_covar - (L @ (covar_cache @ L.transpose(-1, -2)))
        return res
