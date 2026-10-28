#include "ffi_device_memory.h"
#include "block_tridi.cuh"
#include "cudss_blktridi.cuh"

#include <cudss.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace {

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void CheckCudss(cudssStatus_t status, const char* operation) {
  if (status != CUDSS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with status " +
                             std::to_string(static_cast<int>(status)));
  }
}

size_t ComputeNnz(int32_t T, int32_t n) {
  if (T == 1) {
    return static_cast<size_t>(n) * n;
  }
  return static_cast<size_t>(3 * T - 2) * n * n;
}

enum class ScalarType { kFloat32, kFloat64 };

class WorkspaceBase {
public:
  virtual ~WorkspaceBase() = default;
  virtual void Release(turbompc::FfiDeviceAllocator& allocator) = 0;
  virtual ScalarType scalar_type() const noexcept = 0;
  virtual bool matches(int32_t batch_size, int32_t T,
                       int32_t n) const noexcept = 0;
};

struct WorkspaceEntry {
  std::mutex execution_mutex;
  std::unique_ptr<WorkspaceBase> capacity;
};

template <typename Scalar>
class CudssBlockTridiWorkspace final : public WorkspaceBase {
public:
  CudssBlockTridiWorkspace(turbompc::FfiDeviceAllocator& allocator, int32_t batch_size, int32_t T, int32_t n)
      : batch_size_(batch_size), T_(T), n_(n), total_(batch_size * T * n),
        nnz_(static_cast<size_t>(batch_size) * ComputeNnz(T, n)) {
    if (batch_size_ < 1 || T_ < 1 || n_ < 1) {
      throw std::invalid_argument("cuDSS block-tridiagonal workspace dimensions must be positive");
    }
    try {
      turbompc::FfiAllocate(allocator, &indptr_, (total_ + 1) * sizeof(int32_t));
      turbompc::FfiAllocate(allocator, &indices_, nnz_ * sizeof(int32_t));
      turbompc::FfiAllocate(allocator, &csr_data_, nnz_ * sizeof(Scalar));
      turbompc::FfiAllocate(allocator, &x_buf_, total_ * sizeof(Scalar));
      CheckCudss(cudssCreate(&handle_), "cudssCreate");
      cudss_allocator_ = allocator;
      CheckCudss(turbompc::SetCudssDeviceMemory(handle_, cudss_allocator_), "cudssSetDeviceMemHandler");
      CheckCudss(cudssConfigCreate(&config_), "cudssConfigCreate");
      // cuDSS >= 1.0 enables matching by default; the Schur system does not need it.
      cudssMatchingAlg_t matching_alg = CUDSS_MATCHING_ALG_NONE;
      CheckCudss(cudssConfigSet(config_, CUDSS_CONFIG_MATCHING_ALG,
                                &matching_alg, sizeof(matching_alg)),
                 "cudssConfigSet(MATCHING_ALG)");
      CheckCudss(cudssDataCreate(handle_, &data_), "cudssDataCreate");
      CheckCuda(cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    } catch (...) {
      Release(allocator);
      throw;
    }
  }

  CudssBlockTridiWorkspace(const CudssBlockTridiWorkspace&) = delete;
  CudssBlockTridiWorkspace& operator=(const CudssBlockTridiWorkspace&) = delete;

  ~CudssBlockTridiWorkspace() noexcept override = default;

  ScalarType scalar_type() const noexcept override {
    if constexpr (std::is_same_v<Scalar, float>) {
      return ScalarType::kFloat32;
    }
    return ScalarType::kFloat64;
  }

  bool matches(int32_t batch_size, int32_t T,
               int32_t n) const noexcept override {
    return batch_size == batch_size_ && T == T_ && n == n_;
  }

  void Solve(turbompc::FfiDeviceAllocator& allocator, cudaStream_t stream, Scalar* output, const Scalar* blocks,
             const Scalar* rhs, int32_t batch_size, int32_t T, int32_t n,
             bool factorize) {
    if (batch_size != batch_size_ || T != T_ || n != n_) {
      throw std::invalid_argument("cuDSS block-tridiagonal workspace dimensions do not match buffers");
    }
    if (completion_pending_) {
      CheckCuda(cudaEventSynchronize(completion_event_), "cudaEventSynchronize");
      completion_pending_ = false;
    }
    cudss_allocator_ = allocator;
    CheckCudss(cudssSetStream(handle_, stream), "cudssSetStream");

    if (!factorize && !factorized_) {
      throw std::invalid_argument("cuDSS block-tridiagonal workspace has not been factorized");
    }

    if (factorize && !structure_written_) {
      if (batch_size_ == 1) {
        if constexpr (std::is_same_v<Scalar, float>) {
          blkTridiToCSR(blocks, indptr_, indices_, csr_data_, T_, n_, stream);
        } else {
          blkTridiToCSR_f64(blocks, indptr_, indices_, csr_data_, T_, n_, stream);
        }
      } else if constexpr (std::is_same_v<Scalar, float>) {
        batchedBlkTridiToCSR(blocks, indptr_, indices_, csr_data_, batch_size_, T_, n_, stream);
      } else {
        batchedBlkTridiToCSR_f64(blocks, indptr_, indices_, csr_data_, batch_size_, T_, n_, stream);
      }
      structure_written_ = true;
    } else if (factorize && batch_size_ == 1) {
      if constexpr (std::is_same_v<Scalar, float>) {
        blkTridiToCSR_data_only(blocks, csr_data_, T_, n_, stream);
      } else {
        blkTridiToCSR_data_only_f64(blocks, csr_data_, T_, n_, stream);
      }
    } else if (factorize) {
      if constexpr (std::is_same_v<Scalar, float>) {
        batchedBlkTridiToCSR_data_only(blocks, csr_data_, batch_size_, T_, n_, stream);
      } else {
        batchedBlkTridiToCSR_data_only_f64(blocks, csr_data_, batch_size_, T_, n_, stream);
      }
    }

    const cudaDataType_t value_type = std::is_same_v<Scalar, float> ? CUDSS_R_32F : CUDSS_R_64F;
    if (matA_ == nullptr) {
      CheckCudss(cudssMatrixCreateCsr(
                     &matA_, total_, total_, nnz_, indptr_, nullptr, indices_,
                     csr_data_, CUDSS_R_32I, CUDSS_R_32I, value_type,
                     CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                 "cudssMatrixCreateCsr");
      CheckCudss(cudssMatrixCreateDn(&matX_, total_, 1, total_, x_buf_,
                                     value_type, CUDSS_LAYOUT_COL_MAJOR),
                 "cudssMatrixCreateDn(X)");
      CheckCudss(cudssMatrixCreateDn(&matB_, total_, 1, total_,
                                     const_cast<Scalar*>(rhs), value_type,
                                     CUDSS_LAYOUT_COL_MAJOR),
                 "cudssMatrixCreateDn(B)");
    } else {
      CheckCudss(cudssMatrixSetValues(matB_, const_cast<Scalar*>(rhs)),
                 "cudssMatrixSetValues(B)");
    }

    if (factorize) {
      cudssReorderingAlg_t reorder_alg = CUDSS_REORDERING_ALG_DEFAULT;
      CheckCudss(cudssConfigSet(config_, CUDSS_CONFIG_REORDERING_ALG,
                                &reorder_alg, sizeof(reorder_alg)),
                 "cudssConfigSet");
      if (!analyzed_) {
        CheckCudss(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_,
                                matA_, matX_, matB_),
                   "cuDSS analysis");
        analyzed_ = true;
      }
      CheckCudss(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_,
                              data_, matA_, matX_, matB_),
                 "cuDSS factorization");
      factorized_ = true;
    }
    CheckCudss(cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, matA_,
                            matX_, matB_),
               "cuDSS solve");
    CheckCuda(cudaMemcpyAsync(output, x_buf_, total_ * sizeof(Scalar),
                              cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync(output)");
    CheckCuda(cudaEventRecord(completion_event_, stream), "cudaEventRecord");
    completion_pending_ = true;
  }

  void Release(turbompc::FfiDeviceAllocator& allocator) override {
    if (completion_pending_ && completion_event_ != nullptr) {
      cudaEventSynchronize(completion_event_);
    }
    cudss_allocator_ = allocator;
    if (matA_ != nullptr) {
      cudssMatrixDestroy(matA_);
    }
    if (matX_ != nullptr) {
      cudssMatrixDestroy(matX_);
    }
    if (matB_ != nullptr) {
      cudssMatrixDestroy(matB_);
    }
    if (data_ != nullptr && handle_ != nullptr) {
      cudssDataDestroy(handle_, data_);
    }
    if (config_ != nullptr) {
      cudssConfigDestroy(config_);
    }
    if (handle_ != nullptr) {
      cudssDestroy(handle_);
    }
    if (indptr_ != nullptr) {
      turbompc::FfiFree(allocator, indptr_, (total_ + 1) * sizeof(int32_t));
    }
    if (indices_ != nullptr) {
      turbompc::FfiFree(allocator, indices_, nnz_ * sizeof(int32_t));
    }
    if (csr_data_ != nullptr) {
      turbompc::FfiFree(allocator, csr_data_, nnz_ * sizeof(Scalar));
    }
    if (x_buf_ != nullptr) {
      turbompc::FfiFree(allocator, x_buf_, total_ * sizeof(Scalar));
    }
    if (completion_event_ != nullptr) {
      cudaEventDestroy(completion_event_);
    }
  }

private:
  int32_t batch_size_;
  int32_t T_;
  int32_t n_;
  int32_t total_;
  size_t nnz_;
  turbompc::FfiDeviceAllocator cudss_allocator_{};
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t matA_ = nullptr;
  cudssMatrix_t matX_ = nullptr;
  cudssMatrix_t matB_ = nullptr;
  int32_t* indptr_ = nullptr;
  int32_t* indices_ = nullptr;
  Scalar* csr_data_ = nullptr;
  Scalar* x_buf_ = nullptr;
  cudaEvent_t completion_event_ = nullptr;
  bool completion_pending_ = false;
  bool analyzed_ = false;
  bool factorized_ = false;
  bool structure_written_ = false;
};

std::mutex registry_mutex;
std::unordered_map<std::uint64_t, std::unique_ptr<WorkspaceEntry>> registry;
std::uint64_t next_handle = 1;
thread_local std::string last_error;

std::uint64_t CreateEmptyWorkspace() {
  auto entry = std::make_unique<WorkspaceEntry>();
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (next_handle == 0) {
    throw std::overflow_error("cuDSS workspace handle space is exhausted");
  }
  const std::uint64_t handle = next_handle++;
  registry.emplace(handle, std::move(entry));
  return handle;
}

WorkspaceEntry& FindEntry(std::uint64_t handle) {
  const auto position = registry.find(handle);
  if (position == registry.end()) {
    throw std::invalid_argument("cuDSS block-tridiagonal workspace handle is not live");
  }
  return *position->second;
}

template <typename Scalar>
CudssBlockTridiWorkspace<Scalar>& EnsureCapacity(
    turbompc::FfiDeviceAllocator& allocator, WorkspaceEntry& entry, int32_t batch_size, int32_t T, int32_t n) {
  const ScalarType expected = std::is_same_v<Scalar, float>
                                  ? ScalarType::kFloat32
                                  : ScalarType::kFloat64;
  if (entry.capacity == nullptr ||
      entry.capacity->scalar_type() != expected ||
      !entry.capacity->matches(batch_size, T, n)) {
    if (entry.capacity) entry.capacity->Release(allocator);
    entry.capacity.reset();
    entry.capacity = std::make_unique<CudssBlockTridiWorkspace<Scalar>>(allocator, batch_size, T, n);
  }
  return static_cast<CudssBlockTridiWorkspace<Scalar>&>(*entry.capacity);
}

template <typename Scalar>
void SolveOwned(turbompc::FfiDeviceAllocator& allocator, std::uint64_t handle, cudaStream_t stream, Scalar* output,
                const Scalar* blocks, const Scalar* rhs, int32_t T, int32_t n,
                int32_t batch_size, bool factorize) {
  std::unique_lock<std::mutex> registry_lock(registry_mutex);
  auto& entry = FindEntry(handle);
  std::unique_lock<std::mutex> execution_lock(entry.execution_mutex);
  registry_lock.unlock();
  auto& workspace = EnsureCapacity<Scalar>(allocator, entry, batch_size, T, n);
  workspace.Solve(allocator, stream, output, blocks, rhs, batch_size, T, n, factorize);
}

void DestroyWorkspace(turbompc::FfiDeviceAllocator& allocator, std::uint64_t handle) {
  std::unique_ptr<WorkspaceEntry> entry;
  std::unique_lock<std::mutex> registry_lock(registry_mutex);
  const auto position = registry.find(handle);
  if (position == registry.end()) {
    throw std::invalid_argument(
        "cuDSS block-tridiagonal workspace handle is not live");
  }
  std::unique_lock<std::mutex> execution_lock(position->second->execution_mutex);
  entry = std::move(position->second);
  registry.erase(position);
  registry_lock.unlock();
  execution_lock.unlock();
  if (entry->capacity) entry->capacity->Release(allocator);
  entry.reset();
}

template <typename Result, typename Function>
Result TranslateErrors(Function&& function, Result failure) noexcept {
  try {
    last_error.clear();
    return function();
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown native error";
  }
  return failure;
}

} // namespace

void CudssBlkTridiSolveF32(turbompc::FfiDeviceAllocator& allocator, std::uint64_t workspace_handle, cudaStream_t stream,
                           float* x_out, const float* S, const float* rhs,
                           int32_t T, int32_t n, bool factorize, int32_t Nb) {
  SolveOwned<float>(allocator, workspace_handle, stream, x_out, S, rhs, T, n, Nb,
                    factorize);
}

void CudssBlkTridiSolveF64(turbompc::FfiDeviceAllocator& allocator, std::uint64_t workspace_handle, cudaStream_t stream,
                           double* x_out, const double* S, const double* rhs,
                           int32_t T, int32_t n, bool factorize, int32_t Nb) {
  SolveOwned<double>(allocator, workspace_handle, stream, x_out, S, rhs, T, n, Nb,
                     factorize);
}

std::uint64_t CreateCudssBlockTridiWorkspace() {
  return TranslateErrors([&] { return CreateEmptyWorkspace(); },
                         std::uint64_t{0});
}

int EnsureCudssBlockTridiWorkspaceCapacity(
    turbompc::FfiDeviceAllocator& allocator, std::uint64_t handle, int32_t batch_size, int32_t T, int32_t n,
    int32_t dtype_bytes) {
  return TranslateErrors(
      [&] {
        std::unique_lock<std::mutex> registry_lock(registry_mutex);
        auto& entry = FindEntry(handle);
        std::unique_lock<std::mutex> execution_lock(entry.execution_mutex);
        registry_lock.unlock();
        if (dtype_bytes == 4) {
          EnsureCapacity<float>(allocator, entry, batch_size, T, n);
        } else if (dtype_bytes == 8) {
          EnsureCapacity<double>(allocator, entry, batch_size, T, n);
        } else {
          throw std::invalid_argument("workspace dtype must be 4 or 8 bytes");
        }
        return 0;
      },
      -1);
}

int DestroyCudssBlockTridiWorkspace(
    turbompc::FfiDeviceAllocator& allocator, std::uint64_t handle) {
  return TranslateErrors(
      [&] {
        DestroyWorkspace(allocator, handle);
        return 0;
      },
      -1);
}

const char* CudssBlockTridiLastError() {
  return last_error.c_str();
}
