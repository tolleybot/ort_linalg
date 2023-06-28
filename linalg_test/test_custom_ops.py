# Custom operator from PyTorch

# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
# Added python custom operator as per: https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/pyop.md
# Original C++ custom operator as described in: https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md


import torch

from custom_ops import *

# Global settings
torch.manual_seed(0)
MODEL_FILE = 'custom_op_test.onnx'
DEVICE_NAME = 'cpu'
DEVICE_INDEX = 0  # Replace this with the index of the device you want to run on
DEVICE = f'{DEVICE_NAME}:{DEVICE_INDEX}'

verbose = False
verboseprint = print if verbose else lambda *a, **k: None

class CustomModelCholesky(torch.nn.Module):
    def forward(self, x):
        L = torch.linalg.cholesky(x)
        return L


def create_custom_model_cholesky():
    dtype = torch.double
    a = torch.randn(2, 2, dtype=dtype)
    sample_x = a @ a.mT + 1e-3  # make symmetric positive-definite
    inputs = (sample_x)

    torch.onnx.export(CustomModelCholesky(), inputs, MODEL_FILE,
                      opset_version=9,
                      input_names=["x"], output_names=["z"],
                      dynamic_axes={"x": {0: "rows_x", 1: "cols_x"}},
                      custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})


class CustomModelSolveTrianguler(torch.nn.Module):
    def forward(self, a, b):
        x = torch.linalg.solve_triangular(a, b, upper=False)
        return x


def create_custom_model_solve_triangular():
    dtype = np.double

    a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=dtype)
    b = np.array([4, 2, 4, 2], dtype=dtype)
    b = np.reshape(b, (4, 1))
    # x = solve_triangular(a, b, lower=True)
    # x = array([1.33333333, -0.66666667, 2.66666667, -1.33333333])

    sample_a = torch.from_numpy(a)
    sample_b = torch.from_numpy(b)
    inputs = (sample_a, sample_b)

    torch.onnx.export(CustomModelSolveTrianguler(), inputs, MODEL_FILE,
                      opset_version=9,
                      input_names=["a", "b"], output_names=["x"],
                      dynamic_axes={"a": {0: "rows_a", 1: "cols_a"}, "b": {0: "rows_b", 1: "cols_b"}},
                      custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})


# Run the model from torch
def run_pytorch_chol(x: np.array) -> np.array:
    model = CustomModelCholesky()
    model.eval()
    with torch.no_grad():
        z = model(x)
    return z


def run_pytorch_solve(a: np.array, b: np.array) -> np.array:
    model = CustomModelSolveTrianguler()
    model.eval()
    with torch.no_grad():
        x = model(a, b)
    return x


# Run the model on CPU consuming and producing numpy arrays
def run_chol(x: np.array) -> np.array:
    session = create_session(MODEL_FILE)
    z = session.run(["z"], {"x": x})
    return z[0]


def run_solve(a: np.array, b: np.array) -> np.array:
    session = create_session(MODEL_FILE)
    x = session.run(["x"], {"a": a, "b": b})
    return x[0]


# Run the model on device consuming and producing native PyTorch tensors
def run_torch_device_chol(x: torch.Tensor, np_type: np.dtype = np.double,
                          torch_type: torch.dtype = torch.double) -> torch.Tensor:
    session = create_session(MODEL_FILE)

    binding = session.io_binding()

    x_tensor = x.contiguous()

    binding.bind_input(
        name='x',
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=np_type,
        shape=tuple(x_tensor.shape),
        buffer_ptr=x_tensor.data_ptr(),
    )

    # Allocate the PyTorch tensor for the model output
    z_tensor = torch.empty(x_tensor.shape, dtype=torch_type, device=DEVICE).contiguous()
    binding.bind_output(
        name='z',
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=np_type,
        shape=tuple(z_tensor.shape),
        buffer_ptr=z_tensor.data_ptr(),
    )

    session.run_with_iobinding(binding)
    return z_tensor


def print_sep():
    verboseprint("################################################################################")


def test_cholesky():
    print_sep()
    register_custom_ops()
    create_custom_model_cholesky()

    A = np.array([[25, 15, -5],
                  [15, 18, 0],
                  [-5, 0, 11]]
                 , dtype=np.double)

    # Expected output
    L = np.array([[5., 0., 0.],
                  [3., 3., 0.],
                  [-1., 1., 3.]]
                 , dtype=np.double)

    torch_L = torch.from_numpy(L)

    verboseprint("Expected Cholesky output:")
    verboseprint(L)

    verboseprint("\nDirect pytorch run (copy matrix):")
    direct = run_pytorch_chol(x=torch.from_numpy(A))
    verboseprint(direct)
    assert torch.allclose(direct, torch_L)

    verboseprint("\nRuntime invocation with numpy data:")
    numpy_onnx = run_chol(x=A)
    verboseprint(numpy_onnx)
    assert np.allclose(numpy_onnx, L)

    verboseprint("\nRuntime invocation with torch data:")
    torch_onnx = run_torch_device_chol(torch.from_numpy(A))
    verboseprint(torch_onnx)
    assert torch.allclose(torch_onnx, torch_L)
    print_sep()


def test_triangular_solve():
    print_sep()
    register_custom_ops()
    create_custom_model_solve_triangular()

    a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.double)
    b = np.array([4, 2, 4, 2], dtype=np.double)
    b = np.reshape(b, (4, 1))
    x = scipy.linalg.solve_triangular(a, b, lower=True)

    torch_a = torch.from_numpy(a)
    torch_b = torch.from_numpy(b)
    torch_x = torch.from_numpy(x)

    # x = array([1.33333333, -0.66666667, 2.66666667, -1.33333333])
    # verboseprint(a.dot(x))  # Check the result
    # array([4., 2., 4., 2.])

    verboseprint("Expected solve_triangular output:")
    verboseprint(x)

    verboseprint("\nDirect pytorch run (copy matrix):")
    direct = run_pytorch_solve(torch.from_numpy(a), torch.from_numpy(b))
    verboseprint(direct)
    assert torch.allclose(direct, torch_x)

    verboseprint("\nRuntime invocation with numpy data:")
    numpy_onnx = run_solve(a, b)
    verboseprint(numpy_onnx)
    assert np.allclose(numpy_onnx, x)

    print_sep()

