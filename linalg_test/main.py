import os
import sys

import numpy as np
import onnxruntime as onnxrt


def testRegisterCustomOpsLibrary(self):  # noqa: N802
    lib_dir = "../cmake-build-debug/linalg_op"
    shared_library = lib_dir + "/libcustom_op_library.so"
    if not os.path.exists(shared_library):
        raise FileNotFoundError(f"Unable to find '{shared_library}'")

    this = os.path.dirname(__file__)
    custom_op_model = os.path.join(this, "testdata", "custom_op_library", "custom_op_test.onnx")
    if not os.path.exists(custom_op_model):
        raise FileNotFoundError(f"Unable to find '{custom_op_model}'")

    so1 = onnxrt.SessionOptions()
    so1.register_custom_ops_library(shared_library)

    # Model loading successfully indicates that the custom op node could be resolved successfully
    sess1 = onnxrt.InferenceSession(custom_op_model, so1)

    # Run with input data
    input_name_0 = sess1.get_inputs()[0].name
    input_name_1 = sess1.get_inputs()[1].name
    output_name = sess1.get_outputs()[0].name
    input_0 = np.ones((3, 5)).astype(np.float32)
    input_1 = np.zeros((3, 5)).astype(np.float32)
    res = sess1.run([output_name], {input_name_0: input_0, input_name_1: input_1})
    output_expected = np.ones((3, 5)).astype(np.float32)
    np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testRegisterCustomOpsLibrary()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
