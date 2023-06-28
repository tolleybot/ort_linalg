import torch
import numpy as np
import onnx
import onnxruntime
import onnxruntime_extensions as ortx

# Register custom onnx-runtime implementations in python
# This will be registered to the domain ai.onnx.contrib
@ortx.onnx_op(op_type="bitwise_left_shift", inputs=[ortx.PyCustomOpDef.dt_int32, ortx.PyCustomOpDef.dt_int32])
def bitwise_left_shift(x, y):
    return torch.bitwise_left_shift(x, y)
    
# Define the custom model
class CustomModelBitWiseShiftL(torch.nn.Module):
    def forward(self, x, shift):
        return torch.bitwise_left_shift(x, shift)

# Register the custom operator
def register_custom_ops():
    def symbolic_bitwise_shift_left(g, input, shift):
        return g.op("ai.onnix.contrib::bitwise_left_shift", input, shift)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic(symbolic_name='aten::bitwise_shift_left', 
                                symbolic_fn=symbolic_bitwise_shift_left,
                                opset_version=9)


def torch_dtype_to_numpy_dtype(dtype):
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def test_with_custom_op(dtype):
    # Register the custom ops
    register_custom_ops()
    # Create the model and the inputs
    model = CustomModelBitWiseShiftL()
    x = torch.tensor([16, 4, 1], dtype=dtype)
    shift = torch.tensor(1, dtype=dtype)
    # Export the model
    torch.onnx.export(model,
                      (x, shift),
                     "model_custom.onnx")

    # Load the model with ONNX
    onnx_model = onnx.load("model_custom.onnx")

    # Check the model
    onnx.checker.check_model(onnx_model)

    # Create a session with ONNX Runtime
    sess = onnxruntime.InferenceSession("model_custom.onnx")

    # Convert torch dtype to numpy dtype
    np_dtype = torch_dtype_to_numpy_dtype(dtype)

    # Run the model with ONNX Runtime
    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    result = sess.run(None, {input_name_1: np.array([16, 4, 1], dtype=np_dtype), input_name_2: np.array(1, dtype=np_dtype)})
    print(result)


def test_without_custom_op(dtype):
    # Create the model and the inputs
    model = CustomModelBitWiseShiftL()
    x = torch.tensor([16, 4, 1], dtype=dtype)
    shift = torch.tensor(1, dtype=dtype)

    try:
        # Try to export the model
        torch.onnx.export(model, (x, shift), "model_noop.onnx")
    except Exception as e:
        print(f"Failed to export the model: {e}")

# Run the tests
test_without_custom_op(torch.int32)
test_with_custom_op(torch.int32)


