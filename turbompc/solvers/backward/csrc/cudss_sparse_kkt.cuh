#pragma once

#include "ffi_device_memory.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <cudss.h>

namespace turbompc {
namespace cudss_sparse_kkt {

// Solve a general sparse linear system KKT @ x = rhs using cuDSS
// All pointers are device pointers
// Input:
//   stream: CUDA stream
//   rowPtr: CSR row pointers on device (size n+1)
//   colIdx: CSR column indices on device (size nnz)
//   values: CSR values on device (size nnz)
//   rhs: right-hand side vector on device (size n)
//   n: matrix dimension
// Output:
//   solution: solution vector on device (size n)
void solve_sparse_kkt_f32(
    turbompc::FfiDeviceAllocator& allocator,
    uint64_t workspace_handle,
    cudaStream_t stream,
    const float* values,
    const float* rhs,
    float* solution,
    int32_t n,
    int32_t nnz
);

void solve_sparse_kkt_f64(
    turbompc::FfiDeviceAllocator& allocator,
    uint64_t workspace_handle,
    cudaStream_t stream,
    const double* values,
    const double* rhs,
    double* solution,
    int32_t n,
    int32_t nnz
);

}  // namespace cudss_sparse_kkt
}  // namespace turbompc
