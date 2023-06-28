import torch
import numpy as np


# Define the custom model
class CustomModelBitShift(torch.nn.Module):
    def forward(self, x, shift):
        return x << shift

def torch_dtype_to_numpy_dtype(dtype):
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def test_without_custom_op(dtype):
    # Create the model and the inputs
    model = CustomModelBitShift()
    x = torch.tensor([16, 4, 1], dtype=dtype)
    shift = torch.tensor(1, dtype=dtype)

#    try:
    # Try to export the model
    torch.onnx.export(model, (x, shift), "model_noop.onnx")
    #except Exception as e:
    #    print(f"Failed to export the model: {e}")

        # Identify unconvertible operations
    unconvertible_ops = torch.onnx.utils.unconvertible_ops(model, (x, shift))

    # Print the unconvertible operations
    for op in unconvertible_ops:
        print(op)

# Run the tests
test_without_custom_op(torch.int32)



