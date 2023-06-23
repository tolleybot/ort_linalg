/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/script.h>

torch::Tensor custom_op_one(torch::Tensor X, torch::Tensor Y) {

  float* X_data = X.data<float>();
  float* Y_data = Y.data<float>();
  torch::Tensor output = torch::zeros(X.sizes());
  float* out = output.data<float>();

  // Make this different than the ONNX runtime operator so we can see which one is used
  for (size_t i = 0; i < X.size(0); i++) {
    out[i] = X_data[i] + 0.0*Y_data[i];
  }

  return output.clone();
}

static auto registry =
  torch::RegisterOperators("mynamespace::custom_op_one", &custom_op_one);
