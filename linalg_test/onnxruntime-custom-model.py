# A set of code samples showing different usage of the ONNX Runtime Python API
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
# Added custom operator as described in: https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md

import os
import numpy as np
import torch
import onnxruntime

torch.manual_seed(0)
CUSTOM_OP_DOMAIN = 'test.customop'
CUSTOM_OP_VERSION = 1
MODEL_FILE = 'custom_op_test.onnx'
DEVICE_NAME = 'cpu'
DEVICE_INDEX = 0  # Replace this with the index of the device you want to run on
DEVICE = f'{DEVICE_NAME}:{DEVICE_INDEX}'


def register_custom_op():
    def custom_op_one(g, x, y):
        return g.op(CUSTOM_OP_DOMAIN + "::CustomOpOne", x, y)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic(symbolic_name='mynamespace::custom_op_one', symbolic_fn=custom_op_one,
                                opset_version=CUSTOM_OP_VERSION)


class CustomModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.mynamespace.custom_op_one(x, y)

def create_custom_model():
    type = torch.float32
    sample_x = torch.ones(3, dtype=type)
    sample_y = torch.zeros(3, dtype=type)
    inputs = (sample_x, sample_y)

    torch.onnx.export(CustomModel(), inputs, MODEL_FILE,
                      opset_version=9,
                      input_names=["x", "y"], output_names=["z"],
                      dynamic_axes={"x": {0: "array_length_x"}, "y": {0: "array_length_y"}, },
                      custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})


# Create an ONNX Runtime session with the provided model and custom ops library
def create_session(model: str) -> onnxruntime.InferenceSession:
    lib_dir = "../cmake-build-debug/linalg_op"
    shared_library = lib_dir + "/libcustom_op_library.so"
    if not os.path.exists(shared_library):
        raise FileNotFoundError(f"Unable to find '{shared_library}'")

    so1 = onnxruntime.SessionOptions()
    so1.register_custom_ops_library(shared_library)

    # Model loading successfully indicates that the custom op node could be resolved successfully
    providers = ['CPUExecutionProvider']
    sess1 = onnxruntime.InferenceSession(model, so1, providers=providers)

    return sess1


# Run the model from torch
def run_pytorch(x: np.array, y: np.array) -> np.array:
    model = CustomModel()
    model.eval()
    with torch.no_grad():
        z = model(x, y)
    return z


# Run the model on CPU consuming and producing numpy arrays
def run(x: np.array, y: np.array) -> np.array:
    session = create_session(MODEL_FILE)

    z = session.run(["z"], {"x": x, "y": y})

    return z[0]


# Run the model on device consuming and producing ORTValues
def run_with_data_on_device(x: np.array, y: np.array) -> onnxruntime.OrtValue:
    session = create_session(MODEL_FILE)

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, DEVICE_NAME, DEVICE_INDEX)
    y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(y, DEVICE_NAME, DEVICE_INDEX)

    io_binding = session.io_binding()
    io_binding.bind_input(name='x', device_type=x_ortvalue.device_name(), device_id=0, element_type=x.dtype,
                          shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
    io_binding.bind_input(name='y', device_type=y_ortvalue.device_name(), device_id=0, element_type=y.dtype,
                          shape=y_ortvalue.shape(), buffer_ptr=y_ortvalue.data_ptr())
    io_binding.bind_output(name='z', device_type=DEVICE_NAME, device_id=DEVICE_INDEX, element_type=x.dtype,
                           shape=x_ortvalue.shape())
    session.run_with_iobinding(io_binding)

    z = io_binding.get_outputs()

    return z[0]


# Run the model on device consuming and producing native PyTorch tensors
def run_with_torch_tensors_on_device(x: torch.Tensor, y: torch.Tensor, np_type: np.dtype = np.float32,
                                     torch_type: torch.dtype = torch.float32) -> torch.Tensor:
    session = create_session(MODEL_FILE)

    binding = session.io_binding()

    x_tensor = x.contiguous()
    y_tensor = y.contiguous()

    binding.bind_input(
        name='x',
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=np_type,
        shape=tuple(x_tensor.shape),
        buffer_ptr=x_tensor.data_ptr(),
    )

    binding.bind_input(
        name='y',
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=np_type,
        shape=tuple(y_tensor.shape),
        buffer_ptr=y_tensor.data_ptr(),
    )

    ## Allocate the PyTorch tensor for the model output
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
    torch.ops.load_library(
        "build/lib.linux-x86_64-cpython-310/custom_op_one.cpython-310-x86_64-linux-gnu.so")
    register_custom_op()
    create_custom_model()

    print(run_pytorch(x=torch.tensor([1.0, 2.0, 3.0]), y=torch.tensor([4.0, 5.0, 6.0])))
    # tensor([1., 2., 3.])  # For now, the pytorch impl is deliberately different from the onnxruntime impl

    print(run(x=np.float32([1.0, 2.0, 3.0]), y=np.float32([4.0, 5.0, 6.0])))
    # [array([5., 7., 9.], dtype=float32)]

    print(run_with_data_on_device(x=np.float32([1.0, 2.0, 3.0, 4.0, 5.0]),
                                  y=np.float32([1.0, 2.0, 3.0, 4.0, 5.0])).numpy())
    # [ 2.  4.  6.  8. 10.]

    print(run_with_torch_tensors_on_device(torch.rand(5).to(DEVICE), torch.rand(5).to(DEVICE)))
    # tensor([1.1303, 1.2583, 0.9849, 0.5877, 0.9397])


if __name__ == "__main__":
    main()
