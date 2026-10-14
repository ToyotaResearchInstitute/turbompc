#include "ffi_device_memory.h"
#include <vector>
#include "cudss_sparse_kkt.cuh"
#include <cudss.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      char _buf[256];                                                          \
      snprintf(_buf, sizeof(_buf), "CUDA error at %s:%d: %s",                  \
               __FILE__, __LINE__, cudaGetErrorString(err));                   \
      throw std::runtime_error(_buf);                                          \
    }                                                                          \
  } while (0)

#define CUDSS_CHECK(call)                                                      \
  do {                                                                         \
    cudssStatus_t err = call;                                                  \
    if (err != CUDSS_STATUS_SUCCESS) {                                         \
      char _buf[256];                                                          \
      snprintf(_buf, sizeof(_buf), "cuDSS error at %s:%d: code %d",            \
               __FILE__, __LINE__, static_cast<int>(err));                     \
      throw std::runtime_error(_buf);                                          \
    }                                                                          \
  } while (0)

struct SparseKktWorkspace {
    turbompc::FfiDeviceAllocator cudss_allocator{};
    std::vector<int32_t> host_row_ptr, host_col_idx;
    bool initialized = false;
    std::mutex execution_mutex;
    int32_t n;
    int32_t nnz;
    int32_t dtype_bytes;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr;
    cudssMatrix_t matX = nullptr;
    cudssMatrix_t matB = nullptr;
    int32_t* row_ptr = nullptr;
    int32_t* col_idx = nullptr;
    // Valid device storage for descriptor creation; each descriptor is
    // redirected to the actual buffers before analysis and solve.
    void* placeholder_values = nullptr;
    void* placeholder_rhs = nullptr;
    void* placeholder_solution = nullptr;
    cudaEvent_t completion_event = nullptr;
    bool analyzed = false;

    SparseKktWorkspace(int32_t n_in, int32_t nnz_in, int32_t dtype_bytes_in)
        : n(n_in), nnz(nnz_in), dtype_bytes(dtype_bytes_in) {
        if (n < 1 || nnz < 1 || (dtype_bytes != 4 && dtype_bytes != 8))
            throw std::invalid_argument("invalid sparse KKT workspace dimensions");
    }

    void initialize(turbompc::FfiDeviceAllocator& allocator) {
        if (initialized) return;
        const auto* row_ptr_host = host_row_ptr.data();
        const auto* col_idx_host = host_col_idx.data();
        const size_t scalar_size = static_cast<size_t>(dtype_bytes);
        const cudssDataType_t value_type = dtype_bytes == 8 ? CUDSS_R_64F : CUDSS_R_32F;
        turbompc::FfiAllocate(allocator, &row_ptr, static_cast<size_t>(n + 1) * sizeof(int32_t));
        turbompc::FfiAllocate(allocator, &col_idx, static_cast<size_t>(nnz) * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpy(row_ptr, row_ptr_host,
                              static_cast<size_t>(n + 1) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(col_idx, col_idx_host,
                              static_cast<size_t>(nnz) * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        turbompc::FfiAllocate(allocator, &placeholder_values, static_cast<size_t>(nnz) * scalar_size);
        turbompc::FfiAllocate(allocator, &placeholder_rhs, static_cast<size_t>(n) * scalar_size);
        turbompc::FfiAllocate(allocator, &placeholder_solution, static_cast<size_t>(n) * scalar_size);
        CUDA_CHECK(cudaMemset(placeholder_values, 0,
                              static_cast<size_t>(nnz) * scalar_size));
        CUDA_CHECK(cudaMemset(placeholder_rhs, 0, static_cast<size_t>(n) * scalar_size));
        CUDA_CHECK(cudaMemset(placeholder_solution, 0,
                              static_cast<size_t>(n) * scalar_size));
        CUDSS_CHECK(cudssCreate(&handle));
        cudss_allocator = allocator;
        CUDSS_CHECK(turbompc::SetCudssDeviceMemory(handle, cudss_allocator));
        CUDSS_CHECK(cudssConfigCreate(&config));
        CUDSS_CHECK(cudssDataCreate(handle, &data));
        cudssReorderingAlg_t reorder_alg = CUDSS_REORDERING_ALG_DEFAULT;
        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
                                   &reorder_alg, sizeof(reorder_alg)));
        CUDSS_CHECK(cudssMatrixCreateCsr(
            &matA, n, n, nnz, row_ptr, nullptr, col_idx, placeholder_values,
            CUDSS_R_32I, CUDSS_R_32I, value_type, CUDSS_MTYPE_SYMMETRIC,
            CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&matX, n, 1, n, placeholder_solution,
                                        value_type, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&matB, n, 1, n, placeholder_rhs, value_type,
                                        CUDSS_LAYOUT_COL_MAJOR));
        CUDA_CHECK(cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(completion_event, nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
        initialized = true;
    }

    SparseKktWorkspace(const SparseKktWorkspace&) = delete;
    SparseKktWorkspace& operator=(const SparseKktWorkspace&) = delete;

    void Release(turbompc::FfiDeviceAllocator& allocator) {
        if (completion_event)     cudaEventSynchronize(completion_event);
        cudss_allocator = allocator;
        if (matB)                 cudssMatrixDestroy(matB);
        if (matX)                 cudssMatrixDestroy(matX);
        if (matA)                 cudssMatrixDestroy(matA);
        if (data && handle)       cudssDataDestroy(handle, data);
        if (config)               cudssConfigDestroy(config);
        if (handle)               cudssDestroy(handle);
        if (completion_event)     cudaEventDestroy(completion_event);
        const size_t scalar_size = static_cast<size_t>(dtype_bytes);
        if (placeholder_solution) turbompc::FfiFree(allocator, placeholder_solution, static_cast<size_t>(n) * scalar_size);
        if (placeholder_rhs)      turbompc::FfiFree(allocator, placeholder_rhs, static_cast<size_t>(n) * scalar_size);
        if (placeholder_values)   turbompc::FfiFree(allocator, placeholder_values, static_cast<size_t>(nnz) * scalar_size);
        if (col_idx)              turbompc::FfiFree(allocator, col_idx, static_cast<size_t>(nnz) * sizeof(int32_t));
        if (row_ptr)              turbompc::FfiFree(allocator, row_ptr, static_cast<size_t>(n + 1) * sizeof(int32_t));
    }
};

static std::mutex g_workspace_registry_mutex;
static std::unordered_map<uint64_t, std::unique_ptr<SparseKktWorkspace>> g_workspace_registry;
static uint64_t g_next_workspace_handle = 1;
static thread_local std::string g_last_error;

uint64_t CreateCudssSparseKktWorkspace(const int32_t* row_ptr,
                                       const int32_t* col_idx, int32_t n,
                                       int32_t nnz, int32_t dtype_bytes) {
    try {
        auto workspace = std::make_unique<SparseKktWorkspace>(n, nnz, dtype_bytes);
        workspace->host_row_ptr.assign(row_ptr, row_ptr + n + 1);
        workspace->host_col_idx.assign(col_idx, col_idx + nnz);
        std::lock_guard<std::mutex> lock(g_workspace_registry_mutex);
        const uint64_t handle = g_next_workspace_handle++;
        g_workspace_registry.emplace(handle, std::move(workspace));
        return handle;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 0;
    }
}

int DestroyCudssSparseKktWorkspace(turbompc::FfiDeviceAllocator& allocator, uint64_t handle) {
    try {
        std::unique_ptr<SparseKktWorkspace> workspace;
        std::unique_lock<std::mutex> registry_lock(g_workspace_registry_mutex);
        auto position = g_workspace_registry.find(handle);
        if (position == g_workspace_registry.end())
          throw std::invalid_argument("sparse KKT workspace handle is not live");
        std::unique_lock<std::mutex> execution_lock(
            position->second->execution_mutex);
        workspace = std::move(position->second);
        g_workspace_registry.erase(position);
        registry_lock.unlock();
        execution_lock.unlock();
        workspace->Release(allocator);
        workspace.reset();
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    }
}

const char* CudssSparseKktLastError() {
    return g_last_error.c_str();
}

// Both eager initialization and the memory probe run inside a JAX FFI call.
int PrepareCudssSparseKktWorkspace(turbompc::FfiDeviceAllocator& allocator, uint64_t handle, bool probe, int64_t* out) {
    try {
        std::unique_lock<std::mutex> registry_lock(g_workspace_registry_mutex);
        auto position = g_workspace_registry.find(handle);
        if (position == g_workspace_registry.end())
            throw std::invalid_argument("sparse KKT workspace handle is not live");
        auto& workspace = *position->second;
        std::unique_lock<std::mutex> execution_lock(workspace.execution_mutex);
        registry_lock.unlock();
        workspace.initialize(allocator);
        workspace.cudss_allocator = allocator;
        if (probe) {
            CUDSS_CHECK(cudssExecute(workspace.handle, CUDSS_PHASE_ANALYSIS,
                workspace.config, workspace.data, workspace.matA,
                workspace.matX, workspace.matB));
            CUDA_CHECK(cudaDeviceSynchronize());
            int64_t estimates[16] = {};
            size_t written = 0;
            CUDSS_CHECK(cudssDataGet(workspace.handle, workspace.data,
                CUDSS_DATA_MEMORY_ESTIMATES, estimates, sizeof(estimates), &written));
            out[0] = 0;
            for (size_t i = 0; i < written / sizeof(int64_t) && i < 16; ++i)
                if (estimates[i] > out[0]) out[0] = estimates[i];
            size_t free_bytes, total_bytes;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            out[1] = static_cast<int64_t>(free_bytes);
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    }
}

namespace turbompc {
namespace cudss_sparse_kkt {

template <typename T>
void solve_sparse_kkt_impl(turbompc::FfiDeviceAllocator& allocator, uint64_t workspace_handle, cudaStream_t stream,
                           const T* values, const T* rhs, T* solution,
                           int32_t n, int32_t nnz) {
    std::unique_lock<std::mutex> registry_lock(g_workspace_registry_mutex);
    auto position = g_workspace_registry.find(workspace_handle);
    if (position == g_workspace_registry.end())
        throw std::invalid_argument("sparse KKT workspace handle is not live");
    SparseKktWorkspace* workspace = position->second.get();
    std::unique_lock<std::mutex> execution_lock(workspace->execution_mutex);
    registry_lock.unlock();
    if (workspace->n != n || workspace->nnz != nnz || workspace->dtype_bytes != sizeof(T))
        throw std::invalid_argument("sparse KKT workspace dimensions do not match buffers");

    workspace->initialize(allocator);
    workspace->cudss_allocator = allocator;
    CUDSS_CHECK(cudssSetStream(workspace->handle, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, workspace->completion_event, 0));
    CUDSS_CHECK(cudssMatrixSetValues(workspace->matA, const_cast<T*>(values)));
    CUDSS_CHECK(cudssMatrixSetValues(workspace->matX, solution));
    CUDSS_CHECK(cudssMatrixSetValues(workspace->matB, const_cast<T*>(rhs)));
    if (!workspace->analyzed) {
        CUDSS_CHECK(cudssExecute(workspace->handle, CUDSS_PHASE_ANALYSIS,
                                 workspace->config, workspace->data,
                                 workspace->matA, workspace->matX,
                                 workspace->matB));
        workspace->analyzed = true;
    }
    CUDSS_CHECK(cudssExecute(workspace->handle, CUDSS_PHASE_FACTORIZATION,
                             workspace->config, workspace->data, workspace->matA,
                             workspace->matX, workspace->matB));
    CUDSS_CHECK(cudssExecute(workspace->handle, CUDSS_PHASE_SOLVE,
                             workspace->config, workspace->data, workspace->matA,
                             workspace->matX, workspace->matB));
    CUDA_CHECK(cudaEventRecord(workspace->completion_event, stream));
}

// Explicit instantiations
void solve_sparse_kkt_f32(turbompc::FfiDeviceAllocator& allocator, uint64_t workspace_handle, cudaStream_t stream,
                          const float* values, const float* rhs,
                          float* solution, int32_t n, int32_t nnz) {
    solve_sparse_kkt_impl<float>(allocator, workspace_handle, stream, values, rhs, solution, n, nnz);
}

void solve_sparse_kkt_f64(turbompc::FfiDeviceAllocator& allocator, uint64_t workspace_handle, cudaStream_t stream,
                          const double* values, const double* rhs,
                          double* solution, int32_t n, int32_t nnz) {
    solve_sparse_kkt_impl<double>(allocator, workspace_handle, stream, values, rhs, solution, n, nnz);
}

}  // namespace cudss_sparse_kkt
}  // namespace turbompc
