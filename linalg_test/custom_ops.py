# Custom operator from PyTorch

# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
# Added python custom operator as per: https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/pyop.md
# Original C++ custom operator as described in: https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md


import numpy as np
import torch
import scipy
import onnxruntime
import onnxruntime_extensions as ortx

# Global settings
CUSTOM_OP_DOMAIN = 'ai.onnx.contrib'
CUSTOM_OP_VERSION = 9  # Not sure what opset version to use, or if it matters


# Register custom onnx-runtime implementations in python
# This will be registered to the domain ai.onnx.contrib
@ortx.onnx_op(op_type="linalg_cholesky", inputs=[ortx.PyCustomOpDef.dt_double],
              outputs=[ortx.PyCustomOpDef.dt_double])
def linalg_cholesky(x):
    L = np.linalg.cholesky(x)
    return L


@ortx.onnx_op(op_type="linalg_cholesky_ex", inputs=[ortx.PyCustomOpDef.dt_double],
              outputs=[ortx.PyCustomOpDef.dt_double, ortx.PyCustomOpDef.dt_int32])
def linalg_cholesky_ex(x):
    L = np.linalg.cholesky(x)
    info = np.zeros(1, dtype=torch.int32)
    return L, info


@ortx.onnx_op(op_type="linalg_solve_triangular",
              inputs=[ortx.PyCustomOpDef.dt_double, ortx.PyCustomOpDef.dt_double,
                      ortx.PyCustomOpDef.dt_bool, ortx.PyCustomOpDef.dt_bool, ortx.PyCustomOpDef.dt_bool],
              outputs=[ortx.PyCustomOpDef.dt_double])
def linalg_solve_triangular(a, b, upper, left=True, unitriangular=False):
    if (left != True):
        raise RuntimeError('left = False is not supported for this implementation of solve_triangular')
    x = scipy.linalg.solve_triangular(a, b, lower=not upper, unit_diagonal=unitriangular)
    return x


@ortx.onnx_op(op_type="numpy_transpose", inputs=[ortx.PyCustomOpDef.dt_float], outputs=[ortx.PyCustomOpDef.dt_float])
def numpy_transpose(x):
    return np.transpose(x, axes=(-2, -1))


rem = """
Tensor diag_embed(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
  int64_t nDims = self.dim() + 1;
  int64_t dim1 = maybe_wrap_dim(dim1_, nDims);
  int64_t dim2 = maybe_wrap_dim(dim2_, nDims);
  TORCH_CHECK(dim1 != dim2, "diagonal dimensions cannot be identical ", dim1_, ", ", dim2_);
  int64_t new_dim_len = std::abs(offset) + self.size(-1);
  auto sizes = self.sizes().vec();
  sizes.pop_back();
  sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
  sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
  auto result = at::zeros(sizes, self.options());
  auto diag = result.diagonal(offset, dim1, dim2);
  diag.copy_(self);
  return result;
}
"""

@ortx.onnx_op(op_type="numpy_diag_embed",
              inputs=[ortx.PyCustomOpDef.dt_float,
                      ortx.PyCustomOpDef.dt_int64,
                      ortx.PyCustomOpDef.dt_int64,
                      ortx.PyCustomOpDef.dt_int64],
              outputs=[ortx.PyCustomOpDef.dt_float])
def numpy_diag_embed(x, offset=0, dim1=- 2, dim2=- 1):
    if x.ndim > 1:
        new = np.zeros_like(x)
        np.fill_diagonal(new, x.diagonal())
    else:
        new = np.zeros((x.shape[0], x.shape[0]), dtype=x.dtype)
        np.fill_diagonal(new, x)

    return new


# Register the bindings from pytorch aten functions to implementations in onnx-runtime
def register_custom_ops():
    def bind_custom_op_cholesky(g, x, upper):
        return g.op("ai.onnx.contrib::linalg_cholesky", x)

    def bind_custom_op_cholesky_ex(g, x, upper, check_errors):
        return g.op("ai.onnx.contrib::linalg_cholesky_ex", x)

    def bind_custom_op_solve_triangular(g, a, b, upper, left, unittriangular):
        return g.op("ai.onnx.contrib::linalg_solve_triangular", a, b, upper, left, unittriangular)

    def bind_custom_op_mT(g, x):
        # x.mT is equivalent to x.transpose(-2, -1).
        return g.op("ai.onnx.contrib::numpy_transpose", x)

    def bind_custom_op_diag_embed(g, x, offset, dim1, dim2):
        return g.op("ai.onnx.contrib::numpy_diag_embed", x, offset, dim1, dim2)

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic(symbolic_name='aten::linalg_cholesky',
                                symbolic_fn=bind_custom_op_cholesky,
                                opset_version=CUSTOM_OP_VERSION)

    register_custom_op_symbolic(symbolic_name='aten::linalg_cholesky_ex',
                                symbolic_fn=bind_custom_op_cholesky_ex,
                                opset_version=CUSTOM_OP_VERSION)

    register_custom_op_symbolic(symbolic_name='aten::linalg_solve_triangular',
                                symbolic_fn=bind_custom_op_solve_triangular,
                                opset_version=CUSTOM_OP_VERSION)

    register_custom_op_symbolic(symbolic_name='aten::mT',
                                symbolic_fn=bind_custom_op_mT,
                                opset_version=1)

    register_custom_op_symbolic(symbolic_name='aten::diag_embed',
                                symbolic_fn=bind_custom_op_diag_embed,
                                opset_version=1)


# Create an ONNX Runtime session with the provided model and custom ops library
def create_session(model: str) -> onnxruntime.InferenceSession:
    so1 = onnxruntime.SessionOptions()
    so1.register_custom_ops_library(ortx.get_library_path())

    # Model loading successfully indicates that the custom op node could be resolved successfully
    providers = ['CPUExecutionProvider']
    sess1 = onnxruntime.InferenceSession(model, so1, providers=providers)

    return sess1
