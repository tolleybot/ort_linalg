# Custom operator from PyTorch

# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
# Added python custom operator as per: https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/pyop.md
# Original C++ custom operator as described in: https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md


import numpy as np
import scipy
import torch
import torch
import torch.nn.functional as F
import math
import onnxruntime
import onnxruntime_extensions as ortx

# Global settings
torch.manual_seed(0)
CUSTOM_OP_DOMAIN = 'ai.onnx.contrib'
CUSTOM_OP_VERSION = 9  # Not sure what opset version to use, or if it matters
MODEL_FILE = 'custom_op_test.onnx'
DEVICE_NAME = 'cpu'
DEVICE_INDEX = 0  # Replace this with the index of the device you want to run on
DEVICE = f'{DEVICE_NAME}:{DEVICE_INDEX}'

def torch_dtype_to_numpy_dtype(dtype : torch.dtype)->np.dtype:
    """ Convert a torch dtype to a numpy dtype"""
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.int32:
        return np.int32
    elif dtype == torch.int64:
        return np.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# Register custom onnx-runtime implementations in python
# This will be registered to the domain ai.onnx.contrib
@ortx.onnx_op(op_type="linalg_cholesky", inputs=[ortx.PyCustomOpDef.dt_float])
def linalg_cholesky(x):
    return np.linalg.cholesky(x)

@ortx.onnx_op(op_type="linalg_solve_triangular",
              inputs=[ortx.PyCustomOpDef.dt_float, ortx.PyCustomOpDef.dt_float,
                      ortx.PyCustomOpDef.dt_bool, ortx.PyCustomOpDef.dt_bool, ortx.PyCustomOpDef.dt_bool])
def linalg_solve_triangular(a, b, upper, left=True, unitriangular=False):
    if (left != True):
        raise RuntimeError('left = False is not supported for this implementation of solve_triangular')
    x = scipy.linalg.solve_triangular(a, b, lower=not upper, unit_diagonal=unitriangular)
    return x

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

@ortx.onnx_op(op_type="flatten", inputs=[ortx.PyCustomOpDef.dt_float],
              outputs=[ortx.PyCustomOpDef.dt_float])
def flatten(a):
    """ custom operator for flatten"""
    return a.flatten()  

@ortx.onnx_op(op_type="scaled_dot_product_attention", inputs=[ortx.PyCustomOpDef.dt_float, 
                                                              ortx.PyCustomOpDef.dt_float, 
                                                              ortx.PyCustomOpDef.dt_float, 
                                                              ortx.PyCustomOpDef.dt_bool],)
def scaled_dot_product_attention(q, k, v, is_causal):
    """ custom operator for scaled dot product attention"""
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q)
    if isinstance(k, np.ndarray):
        k = torch.from_numpy(k)
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)

    # TODO: DON
    attn_mask = None

    attn_weight = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
    if attn_mask is not None:
        attn_mask = attn_mask.to(device=q.device)
        attn_weight = attn_weight.masked_fill(~attn_mask, -float('inf'))
    elif is_causal:
        attn_mask = torch.zeros(q.size(0), k.size(0), dtype=torch.bool).to(device=q.device)
        attn_mask.tril_(diagonal=0).fill_(True)
        attn_weight = attn_weight.masked_fill(~attn_mask, -float('inf'))
        
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    return attn_weight @ v

# Register the bindings from pytorch aten functions to implementations in onnx-runtime
def register_custom_ops():
    def bind_custom_op_cholesky(g, x, upper):
        return g.op("ai.onnx.contrib::linalg_cholesky", x)

    def bind_custom_op_solve_triangular(g, a, b, upper, left, unittriangular):
        return g.op("ai.onnx.contrib::linalg_solve_triangular", a, b, upper, left, unittriangular)
    
    def bind_custom_op_bitwise_left_shift(g, a, b):
        return g.op("ai.onnx.contrib::bitwise_left_shift", a, b)
    
    def bind_custom_op_bitwise_right_shift(g, a, b):
        return g.op("ai.onnx.contrib::bitwise_right_shift", a, b)
    
    def bind_custom_op_flatten(g, a):
        return g.op("ai.onnx.contrib::flatten", a)    
      
    def bind_custom_op_scaled_dot_product_attention(g, q, k, v, attn_mask=None, dropout_p=None, is_causal=False):
        return g.op("ai.onnx.contrib::scaled_dot_product_attention", q, k, v, is_causal)

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic(symbolic_name='aten::linalg_cholesky',
                                symbolic_fn=bind_custom_op_cholesky,
                                opset_version=CUSTOM_OP_VERSION)

    register_custom_op_symbolic(symbolic_name='aten::linalg_solve_triangular',
                                symbolic_fn=bind_custom_op_solve_triangular,
                                opset_version=CUSTOM_OP_VERSION)
    
    register_custom_op_symbolic(symbolic_name='aten::bitwise_left_shift',
                                symbolic_fn=bind_custom_op_bitwise_left_shift,
                                opset_version=CUSTOM_OP_VERSION)
    
    register_custom_op_symbolic(symbolic_name='aten::bitwise_right_shift',
                                symbolic_fn=bind_custom_op_bitwise_right_shift,
                                opset_version=CUSTOM_OP_VERSION)
    
    register_custom_op_symbolic(symbolic_name='aten::flatten',
                                symbolic_fn=bind_custom_op_flatten,
                                opset_version=CUSTOM_OP_VERSION)

    register_custom_op_symbolic(symbolic_name='aten::scaled_dot_product_attention',
                                symbolic_fn=bind_custom_op_scaled_dot_product_attention,
                                opset_version=CUSTOM_OP_VERSION)
    


class CustomModelCholesky(torch.nn.Module):
    def forward(self, x):
        L = torch.linalg.cholesky(x)
        return L


def create_custom_model_cholesky():
    dtype = torch.float32
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
    dtype = np.float32

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
    

class CustomModelBitwiseShiftLeft(torch.nn.Module):
    def forward(self, a, b):
        x = torch.bitwise_left_shift(a, b)
        return x


def create_custom_model_bitwise_left_shift():
    dtype = torch.int32

    inputs = (torch.tensor([16, 4, 1], dtype=dtype), torch.tensor(1, dtype=dtype))

    torch.onnx.export(CustomModelBitwiseShiftLeft(), inputs, MODEL_FILE,
                      opset_version=9,
                      input_names=["a", "b"], output_names=["x"],
                      dynamic_axes={"a": {0: "rows_a"}},
                      custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})
    

class CustomModelTransformer(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.0, is_causal=False):
        super(CustomModelTransformer, self).__init__()
        self.dim = dim
        self.is_causal = is_causal
        self.q_linear = torch.nn.Linear(dim, dim)
        self.k_linear = torch.nn.Linear(dim, dim)
        self.v_linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        return F.scaled_dot_product_attention(q, k, v,is_causal=self.is_causal)
    
def create_custom_scaled_dot_product_attention(dim=512):
    # Create some random input data
    inputs = torch.randn(1, 10, dim)
    model = CustomModelTransformer(dim=dim, is_causal=False)
    # Export the model to an ONNX file
    torch.onnx.export(
        model,
        inputs,
        MODEL_FILE,
        input_names=["x"],
        output_names=["output"],
        dynamic_axes={"x": {0: "batch_size", 1: "sequence_length"}},
        custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION}
    )

    return model



# Create an ONNX Runtime session with the provided model and custom ops library
def create_session(model: str) -> onnxruntime.InferenceSession:
    so1 = onnxruntime.SessionOptions()
    so1.register_custom_ops_library(ortx.get_library_path())

    # Model loading successfully indicates that the custom op node could be resolved successfully
    providers = ['CPUExecutionProvider']
    sess1 = onnxruntime.InferenceSession(model, so1, providers=providers)

    return sess1


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

def run_bitwise_shift_left(a: np.array, b: np.array) -> np.array:
    session = create_session(MODEL_FILE)
    a = a.numpy()  # Convert the PyTorch tensor to a numpy array
    b = b.numpy()  # Convert the PyTorch tensor to a numpy array
    x = session.run(None, {"a": a, "b": b}) 
    return x[0]

def run_scaled_dot_product_attention(x: np.array) -> np.array:
    session = create_session(MODEL_FILE)  
  #  x = np.transpose(x, (1, 0, 2))  # Transpose the input to match the ONNX model
    x = session.run(None, {session.get_inputs()[0].name: x})  # Run the model
   # x = np.transpose(x[0], (1, 0, 2))  # Transpose the output to match the PyTorch model
    return x[0]

# Run the model on device consuming and producing native PyTorch tensors
def run_torch_device_chol(x: torch.Tensor, np_type: np.dtype = np.float32,
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


def print_sep():
    print("################################################################################")


def cholesky_test():
    print_sep()
    register_custom_ops()
    create_custom_model_cholesky()

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
    print(run_pytorch_chol(x=torch.from_numpy(A)))

    print("\nRuntime invocation with numpy data:")
    print(run_chol(x=A))

    print("\nRuntime invocation with torch data:")
    print(run_torch_device_chol(torch.from_numpy(A)))
    print_sep()


def triangular_solve_test():
    print_sep()
    register_custom_ops()
    create_custom_model_solve_triangular()

    a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.float32)
    b = np.array([4, 2, 4, 2], dtype=np.float32)
    b = np.reshape(b, (4, 1))
    x = scipy.linalg.solve_triangular(a, b, lower=True)

    print("Expected solve_triangular output:")
    print(x)

    print("\nDirect pytorch run (copy matrix):")
    print(run_pytorch_solve(torch.from_numpy(a), torch.from_numpy(b)))

    print("\nRuntime invocation with numpy data:")
    print(run_solve(a, b))

    print_sep()

def bitwise_left_shift_test():
    print_sep()
    register_custom_ops()
    create_custom_model_bitwise_left_shift()

    a = np.random.randint(0, 10, size=(3,), dtype=np.int32)   
    b = np.array(1, dtype=np.int32)
    x = np.left_shift(a, b)

    print("Expected bitwise_left_shift output:")
    print(x)

    print("\nDirect pytorch run (copy matrix):")
    print(run_bitwise_shift_left(torch.from_numpy(a), torch.from_numpy(b)))

    print("\nRuntime invocation with numpy data:")
    print(run_solve(a, b))

    print_sep()

def scaled_dot_product_attention_test():
    print_sep()
    register_custom_ops()
    model = create_custom_scaled_dot_product_attention()
    dim = 512    
    # Create some random input data
    x = torch.randn(1, 10, dim)

    # Run the input data through the PyTorch model
    expected_output = model(x)
    actual_output = run_scaled_dot_product_attention(x.numpy())

    # Print the expected and actual outputs
    print(f"Expected output: {expected_output.detach().numpy()}")
    print(f"Actual output: {actual_output}")

    # Compare the expected and actual outputs
    print(np.allclose(actual_output, expected_output.detach().numpy(), atol=1e-5))


if __name__ == "__main__":
   # triangular_solve_test()
   # print()
   # cholesky_test()
    #bitwise_left_shift_test()
    scaled_dot_product_attention_test()
    print()