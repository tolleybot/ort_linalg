from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import gpytorch
import gpytorch.variational
import onnxruntime as ort
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import pop_from_cache_ignore_args
from linear_operator.operators import DenseLinearOperator, DiagLinearOperator, MatmulLinearOperator, SumLinearOperator
from torch.utils.data import DataLoader, TensorDataset

from custom_ops import *

MODEL_FILE = 'gp_test.onnx'


class ApproximateGP(gpytorch.models.ApproximateGP):
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


class ONNXWrapperModel(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model.variational_strategy(x)
        return torch.stack((output.mean, output.variance))


def export_gp(model):
    register_custom_ops()
    dtype = torch.float32
    sample_x = torch.zeros((10, 1))
    inputs = (sample_x)
    onnx_model = ONNXWrapperModel(model)


    with torch.no_grad():
        torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
            ONNXWrapperModel(model), sample_x
        )
        print("Unconvertible Ops:")
        print(unconvertible_ops)

        torch.onnx.export(onnx_model, sample_x, MODEL_FILE, verbose=False,
                          input_names=["input"], output_names=["mean", "variance"],
                          custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION},
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


def test_gp():
    N = 10
    X = torch.linspace(-3, 3, N)
    Y = torch.sin(X)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(0))
    model = ApproximateGP(X)
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

    dummy_input = torch.zeros((10, 1))
    onnx_model = ONNXWrapperModel(model)

    with torch.no_grad():
        pred = onnx_model.forward(dummy_input)
        print(pred)


    # Export model
    export_gp(model)



