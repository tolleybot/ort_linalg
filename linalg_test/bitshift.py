import numpy as np
import torch
import onnxruntime
import onnxruntime_extensions as ortx
from torch.onnx import register_custom_op_symbolic
import tempfile
import os

CUSTOM_OP_VERSION = 9
CUSTOM_OP_DOMAIN = 'ai.onnx.contrib'
MODEL_FILE = 'custom_op_shift.onnx'


@ortx.onnx_op(op_type="bitwise_left_shift", inputs=[ortx.PyCustomOpDef.dt_int32, ortx.PyCustomOpDef.dt_int32],
              outputs=[ortx.PyCustomOpDef.dt_int32])
def bitwise_left_shift(a, b):
    """ custom operator for bitwise left shift"""
    return a << b


@ortx.onnx_op(op_type="bitwise_right_shift", inputs=[ortx.PyCustomOpDef.dt_int32, ortx.PyCustomOpDef.dt_int32],
              outputs=[ortx.PyCustomOpDef.dt_int32])
def bitwise_right_shift(a, b):
    """ custom operator for bitwise right shift"""
    return a >> b

# Register the bindings from pytorch aten functions to implementations in
# onnx-runtime


def register_custom_ops():
    def bind_custom_op_bitwise_left_shift(g, a, b):
        return g.op("ai.onnx.contrib::bitwise_left_shift", a, b)

    def bind_custom_op_bitwise_right_shift(g, a, b):
        return g.op("ai.onnx.contrib::bitwise_right_shift", a, b)

    register_custom_op_symbolic(symbolic_name='aten::bitwise_left_shift',
                                symbolic_fn=bind_custom_op_bitwise_left_shift,
                                opset_version=CUSTOM_OP_VERSION)

    register_custom_op_symbolic(symbolic_name='aten::bitwise_right_shift',
                                symbolic_fn=bind_custom_op_bitwise_right_shift,
                                opset_version=CUSTOM_OP_VERSION)


# Create an ONNX Runtime session with the provided model and custom ops library
def create_session(model: str) -> onnxruntime.InferenceSession:
    so1 = onnxruntime.SessionOptions()
    so1.register_custom_ops_library(ortx.get_library_path())

    # Model loading successfully indicates that the custom op node could be
    # resolved successfully
    providers = ['CPUExecutionProvider']
    sess1 = onnxruntime.InferenceSession(model, so1, providers=providers)

    return sess1


class CustomModelBitwiseShiftLeft(torch.nn.Module):
    def forward(self, a, b):
        x = torch.bitwise_left_shift(a, b)
        return x


class CustomModelBitwiseShiftRight(torch.nn.Module):
    def forward(self, a, b):
        x = torch.bitwise_right_shift(a, b)
        return x


def create_custom_model_bitwise_shift(
        left_shift: bool = True, model_file: str = MODEL_FILE):
    """ Create our simple model for testing bitwise left shift or right shift """

    dtype = torch.int32

    inputs = (torch.tensor([16, 4, 1], dtype=dtype),
              torch.tensor(1, dtype=dtype))
    model = None

    if left_shift:
        model = CustomModelBitwiseShiftLeft()
        torch.onnx.export(model, inputs, model_file,
                          opset_version=9,
                          input_names=["a", "b"], output_names=["x"],
                          dynamic_axes={"a": {0: "rows_a"}},
                          custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})
    else:
        model = CustomModelBitwiseShiftRight()
        torch.onnx.export(model, inputs, model_file,
                          opset_version=9,
                          input_names=["a", "b"], output_names=["x"],
                          dynamic_axes={"a": {0: "rows_a"}},
                          custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})

    return model


def run_bitwise_shift_onnx(model: str, a: np.array, b: np.array) -> np.array:
    """ run the model with onnxruntime """
    session = create_session(model)
    x = session.run(None, {"a": a, "b": b})
    return x[0]


def run_bitwise_shift_pytorch(model, a: np.array, b: np.array) -> np.array:
    model.eval()
    with torch.no_grad():
        z = model(a, b)
    return z


def bitwise_shift_test():
    """ test our operators """
    register_custom_ops()
    with tempfile.TemporaryDirectory() as tmpdirname:
        # combine tmpdirname with filename
        lft_path = os.path.join(tmpdirname, 'lft_shift.onnx')
        rsh_path = os.path.join(tmpdirname, 'rsh_shift.onnx')
        model_lsh = create_custom_model_bitwise_shift(
            left_shift=True, model_file=lft_path)
        model_rsh = create_custom_model_bitwise_shift(
            left_shift=False, model_file=rsh_path)

        a = np.random.randint(0, 10, size=(3,), dtype=np.int32)
        b = np.array(1, dtype=np.int32)
        print("\nDirect ONNX run Left Shift:")
        lsh_onnx = run_bitwise_shift_onnx(lft_path, a, b)
        print(lsh_onnx)

        print("\nRuntime invocation with PyTorch, Left Shift:")
        lsh_torch = run_bitwise_shift_pytorch(
            model_lsh, torch.from_numpy(a), torch.from_numpy(b)).numpy()
        print(lsh_torch)

        print("\nComparing results with numpy allclose, Left Shift")
        print(np.allclose(lsh_onnx, lsh_torch))

        # print("\nComparing results with numpy allclose, Right Shift")
        np.allclose(lsh_onnx, lsh_torch)

        print("\nDirect ONNX run, Right Shift:")
        rsh_onnx = run_bitwise_shift_onnx(rsh_path, a, b)
        print(rsh_onnx)

        print("\nRuntime invocation with PyTorch, Right Shift:")
        rsh_torch = run_bitwise_shift_pytorch(
            model_rsh, torch.from_numpy(a), torch.from_numpy(b)).numpy()
        print(rsh_torch)

        # print("\nComparing results with numpy allclose, Right Shift")
        np.allclose(rsh_onnx, rsh_torch)


if __name__ == "__main__":
    bitwise_shift_test()
