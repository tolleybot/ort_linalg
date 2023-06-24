#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>

#include <armadillo>
// using namespace arma;

#include "core/common/common.h"


static const char* c_OpDomain = "test";

struct KernelChol {
  void Compute(OrtKernelContext* context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    auto input_a = ctx.GetInput(0);
    const float* A = input_a.GetTensorData<float>();

    // A - MxK matrix
    auto dimensions = input_a.GetTensorTypeAndShapeInfo().GetShape();
    int64_t M = dimensions[0];
    int64_t K = dimensions[1];
    if(M != K) {
      ORT_CXX_API_THROW("Input matrix must be square.", ORT_FAIL);
    }

    // Convert from row-major to column-major format
    arma::fmat A_mat(const_cast<float *>(A), M, K, false, true);
    auto At = A_mat.t();

    // Setup output
    auto output = ctx.GetOutput(0, dimensions);
    float* out = output.GetTensorMutableData<float>();

    // Compute Cholesky
    // Return LOWER triangular matrix
    // https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
    // U = Upper-triangular Cholesky factor of A.
    arma::fmat U = arma::chol(At, "upper");
    float *U_mem = U.memptr();
    
    // Copy to output & transpose
    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();
    for (size_t i = 0; i < size; i++) {
      out[i] = U_mem[i] ;
    }
    
  }
};

struct CholOpOne : Ort::CustomOpBase<CholOpOne, KernelChol> {
  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* /* info */) const {
    return std::make_unique<KernelChol>().release();
  };

  const char* GetName() const { return "CholOp"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

////////////////////////////////////////////////

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const CholOpOne c_CholOpOne;

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CholOpOne);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
