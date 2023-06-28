from typing import Optional

import gpytorch
import gpytorch.variational
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import pop_from_cache_ignore_args
from linear_operator.operators import DenseLinearOperator, DiagLinearOperator, MatmulLinearOperator, SumLinearOperator
from torch.utils.data import DataLoader, TensorDataset

from custom_ops import *

MODEL_FILE = 'gp_test.onnx'


class PatchedMatmulLinearOperator(MatmulLinearOperator):
    def _diagonal(self) -> torch.Tensor:
        if isinstance(self.left_linear_op, DenseLinearOperator) and isinstance(
                self.right_linear_op, DenseLinearOperator
        ):
            return (self.left_linear_op.tensor * self.right_linear_op.tensor.transpose(-1, -2)).sum(-1)
        elif isinstance(self.left_linear_op, DiagLinearOperator) or isinstance(
                self.right_linear_op, DiagLinearOperator
        ):
            return self.left_linear_op._diagonal() * self.right_linear_op._diagonal()
        else:
            return super()._diagonal()


class VariationalStrategyWithInvCholeskyCaching(gpytorch.variational.VariationalStrategy):
    cached_cholesky_inv: Optional[torch.Tensor] = None

    def precompute_cholesky_inv(self) -> None:
        self.num_induc = self.inducing_points.size(-2)

        induc_output = self.model.forward(self.inducing_points)
        induc_induc_covar = induc_output.lazy_covariance_matrix.add_jitter(self.jitter_val)
        L = self._cholesky_factor(induc_induc_covar).to(_linalg_dtype_cholesky.value())

        I = torch.eye(L.shape[-2], dtype=_linalg_dtype_cholesky.value())
        cached_cholesky_inv = L.solve(I)

        self.cached_cholesky_inv = cached_cholesky_inv.detach()

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):  # type: ignore
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        if self.cached_cholesky_inv is not None:
            interp_term = (self.cached_cholesky_inv @ induc_data_covar.to(_linalg_dtype_cholesky.value())).to(
                full_inputs.dtype
            )

        else:
            L = self._cholesky_factor(induc_induc_covar)
            if L.shape != induc_induc_covar.shape:
                # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
                # TODO: Use a hook fo this
                try:
                    pop_from_cache_ignore_args(self, "cholesky_factor")
                except CachingError:
                    pass
                L = self._cholesky_factor(induc_induc_covar)
            interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                    data_data_covar.add_jitter(self.jitter_val).to_dense()
                    + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                # MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
                # The next line was originally:
                #       MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
                # which fails with onnx. I have no idea why using MatmulLinearOperator fixes it.
                # The ONNX export fails with the message:
                # IndexError: Argument passed to at() was not in the map.
                # Adding .to_dense() on middle_term also fixes it.
                PatchedMatmulLinearOperator(interp_term.transpose(-1, -2), middle_term.to_dense() @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)


class ApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.shape[0])
        variational_strategy = VariationalStrategyWithInvCholeskyCaching(
            self, inducing_points, variational_distribution, learn_inducing_locations=True, jitter_val=1e-12
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ONNXWrapperModel(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.model.variational_strategy.precompute_cholesky_inv()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model.variational_strategy(x)
        return output.mean, output.variance


class ApproximateGPSimple(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.shape[0])
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True, jitter_val=1e-12
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ONNXWrapperModelSimple(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model.variational_strategy(x)
        return output.mean, output.variance


def export_gp(onnx_model):
    register_custom_ops()
    sample_x = torch.zeros((10, 1), dtype=torch.float32)
    inputs = (sample_x)

    with torch.no_grad():
        torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
            onnx_model, inputs
        )
        print("Unconvertible Ops:")
        print(unconvertible_ops)

        torch.onnx.export(onnx_model, inputs, MODEL_FILE, verbose=False,
                          input_names=["input"], output_names=["mean", "variance"],
                          custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION},
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


def test_gp():
    N = 10
    X = torch.linspace(-3, 3, N, dtype=torch.float32)
    Y = torch.sin(X)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(0))
    model = ApproximateGPSimple(X)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X.shape[0])
    optimizer = torch.optim.Adam(model.parameters())

    data = DataLoader(TensorDataset(X, Y), batch_size=X.shape[0])
    for _ in range(1):
        for x, y in data:
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            optimizer.step()

    # Finalize the model
    model.eval()
    likelihood.eval()  # This was missing and simplifies the export.

    # Test predictions
    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in data:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
    means = means[1:]
    print('Train MAE: {}'.format(torch.mean(torch.abs(means - Y.cpu()))))

    # ONNX tests
    dummy_input = torch.zeros((10, 1), dtype=torch.float32)
    onnx_model = ONNXWrapperModelSimple(model)

    with torch.no_grad():
        direct_mean, direct_variance = onnx_model.forward(dummy_input)
        print("Direct pred:")
        print("mean")
        print(direct_mean)
        print("variance")
        print(direct_variance)
        print()

    # Export model
    export_gp(onnx_model)

    # Run model
    session = create_session(MODEL_FILE)
    mean, variance = session.run(["mean", "variance"], {"input": dummy_input.numpy()})

    print("mean")
    print(mean)
    assert torch.allclose(direct_mean, torch.from_numpy(mean))

    print("variance")
    print(variance)
    assert torch.allclose(direct_variance, torch.from_numpy(variance), rtol=0.1, atol=0.1)
