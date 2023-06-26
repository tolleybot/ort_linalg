# Custom operator from PyTorch

# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
# Added python custom operator as per: https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/pyop.md
# Original C++ custom operator as described in: https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md


import numpy as np
import torch
import onnxruntime
import onnxruntime_extensions as ortx


torch.manual_seed(0)
CUSTOM_OP_DOMAIN = 'ai.onnx.contrib'
CUSTOM_OP_VERSION = 9  # Not sure what opset version to use, or if it matters
MODEL_FILE = 'custom_op_test.onnx'
DEVICE_NAME = 'cpu'
DEVICE_INDEX = 0  # Replace this with the index of the device you want to run on
DEVICE = f'{DEVICE_NAME}:{DEVICE_INDEX}'


# Register custom operator implementation in python
@ortx.onnx_op(op_type="linalg_cholesky", inputs=[ortx.PyCustomOpDef.dt_float])
def linalg_cholesky(x):
    return np.linalg.cholesky(x)


def register_custom_op():
    def bind_custom_op_cholesky(g, x, upper):
        return g.op("ai.onnx.contrib::linalg_cholesky", x)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic(symbolic_name='aten::linalg_cholesky', symbolic_fn=bind_custom_op_cholesky,
                                opset_version=CUSTOM_OP_VERSION)


class CustomModel(torch.nn.Module):
    def forward(self, x):
        L = torch.linalg.cholesky(x)
        return L


def create_custom_model():
    dtype = torch.float32
    a = torch.randn(2, 2, dtype=dtype)
    sample_x = a @ a.mT + 1e-3  # make symmetric positive-definite
    inputs = (sample_x)

    torch.onnx.export(CustomModel(), inputs, MODEL_FILE,
                      opset_version=9,
                      input_names=["x"], output_names=["z"],
                      dynamic_axes={"x": {0: "rows_x", 1: "cols_x"}},
                      custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})


# Create an ONNX Runtime session with the provided model and custom ops library
def create_session(model: str) -> onnxruntime.InferenceSession:
    so1 = onnxruntime.SessionOptions()
    so1.register_custom_ops_library(ortx.get_library_path())

    # Model loading successfully indicates that the custom op node could be resolved successfully
    providers = ['CPUExecutionProvider']
    sess1 = onnxruntime.InferenceSession(model, so1, providers=providers)

    return sess1


# Run the model from torch
def run_pytorch(x: np.array) -> np.array:
    model = CustomModel()
    model.eval()
    with torch.no_grad():
        z = model(x)
    return z


# Run the model on CPU consuming and producing numpy arrays
def run(x: np.array) -> np.array:
    session = create_session(MODEL_FILE)
    z = session.run(["z"], {"x": x})
    return z[0]


# Run the model on device consuming and producing native PyTorch tensors
def run_with_torch_tensors_on_device(x: torch.Tensor, np_type: np.dtype = np.float32,
                                     torch_type: torch.dtype = torch.float32) -> torch.Tensor:
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


def main():
    register_custom_op()
    create_custom_model()

    A = np.array([[25, 15, -5],
                  [15, 18, 0],
                  [-5, 0, 11]]
                 , dtype=np.float32)

    # Expected output
    L = np.array([[5., 0., 0.],
                  [3., 3., 0.],
                  [-1., 1., 3.]]
                 , dtype=np.float32)

    print("Expected Cholesky output:")
    print(L)

    print("\nDirect pytorch run (copy matrix):")
    print(run_pytorch(x=torch.from_numpy(A)))

    print("\nRuntime invocation with numpy data:")
    print(run(x=A))

    print("\nRuntime invocation with torch data:")
    print(run_with_torch_tensors_on_device(torch.from_numpy(A)))


if __name__ == "__main__":
    main()
