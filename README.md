# ort_linalg
ONNX Runtime Linear Algebra Operations

* linalg_op: 
    - Custom operators coded in C++ for ONNX runtime


* linalg_test: 
    - Create a pytorch model with a custom operator that calls linalg_op.
    - Export the model to ONNX.
    - Run the model in ONNX runtime, calling the new operator in linalg_op.