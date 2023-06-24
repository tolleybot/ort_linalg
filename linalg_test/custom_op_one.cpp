/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/script.h>

torch::Tensor custom_op_one(torch::Tensor X) {

  float* X_data = X.data<float>();
  size_t elems = X.numel();
  torch::Tensor output = torch::zeros(X.sizes());
  float* out = output.data<float>();

  // Make this different than the ONNX runtime operator so we can see which one is used
  for (size_t i = 0; i < elems; i++) {
    out[i] = X_data[i];
  }

  return output.clone();
}

static auto registry =
  torch::RegisterOperators("mynamespace::custom_op_one", &custom_op_one);
