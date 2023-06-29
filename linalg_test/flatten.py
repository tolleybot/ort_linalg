import numpy as np
import torch
import onnxruntime
import onnxruntime_extensions as ortx
from torch.onnx import register_custom_op_symbolic
import tempfile
import os

CUSTOM_OP_VERSION = 9
CUSTOM_OP_DOMAIN = 'ai.onnx.contrib'
MODEL_FILE = 'custom_op_flatten.onnx'


@ortx.onnx_op(op_type="flatten", inputs=[ortx.PyCustomOpDef.dt_float])
def flatten(a):
    """ custom operator for flatten"""
    return a.flatten()  

# Register the bindings from pytorch aten functions to implementations in onnx-runtime
def register_custom_ops():    
    def bind_custom_op_flatten(g, a, start_dim, end_dim):
        return g.op("ai.onnx.contrib::flatten", a)              
    
    register_custom_op_symbolic(symbolic_name='aten::flatten',
                                symbolic_fn=bind_custom_op_flatten,
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

class CustomModelFlatten(torch.nn.Module):
    def forward(self, a):        
        x = torch.flatten(a)        
        return x

def create_custom_model(model_file: str = MODEL_FILE, dtype=torch.float32):
    """ Create our simple model """    
    inputs = (torch.randn(16, 4, 1, dtype=dtype))
    
    model = CustomModelFlatten()
    torch.onnx.export(model, inputs, model_file,
                        opset_version=9,
                        input_names=["a"], output_names=["x"],
                        dynamic_axes={"a": {0: "rows_a"}},
                        custom_opsets={CUSTOM_OP_DOMAIN: CUSTOM_OP_VERSION})
    
    return model

def run_flatten_onnx(model: str, a: np.array) -> np.array:
    """ run the model with onnxruntime """
    session = create_session(model)
    x = session.run(None, {"a": a})
    return x[0]

def flatten_test():
    with tempfile.TemporaryDirectory() as tmpdirname:    
        # combine tmpdirname with filename
        model_path = os.path.join(tmpdirname, MODEL_FILE)
        # register custom op
        register_custom_ops()
        # create model
        model = create_custom_model(model_file=model_path, dtype=torch.float32)
        # run model
        a = np.random.rand(16, 4, 1).astype(np.float32) 
        print(f"\nInput shape: {a.shape}")
        x = run_flatten_onnx(model_path, a)
        print(f"\nONNX runtime output shape: {x.shape}")
        # compare results
        x_expected = model(torch.tensor(a)).detach().numpy()
        print(f"\nPyTorch runtime output shape: {x_expected.shape}")
        np.testing.assert_allclose(x, x_expected, atol=1e-05)

if __name__ == "__main__":
    flatten_test()