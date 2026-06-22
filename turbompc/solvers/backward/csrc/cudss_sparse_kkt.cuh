#pragma once

#include <cuda_runtime.h>
#include <cudss.h>
#include <cstdint>

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
    cudaStream_t stream,
    const int32_t* rowPtr,
    const int32_t* colIdx,
    const float* values,
    const float* rhs,
    float* solution,
    int32_t n,
    int32_t nnz
);

void solve_sparse_kkt_f64(
    cudaStream_t stream,
    const int32_t* rowPtr,
    const int32_t* colIdx,
    const double* values,
    const double* rhs,
    double* solution,
    int32_t n,
    int32_t nnz
);

// Dense-input variants: accept a row-major n×n dense matrix and convert to
// CSR on GPU (via cuSPARSE) before solving with cuDSS.
void solve_sparse_kkt_from_dense_f32(
    cudaStream_t stream,
    const float* dense_matrix,
    const float* rhs,
    float* solution,
    int32_t n
);

void solve_sparse_kkt_from_dense_f64(
    cudaStream_t stream,
    const double* dense_matrix,
    const double* rhs,
    double* solution,
    int32_t n
);

}  // namespace cudss_sparse_kkt
}  // namespace turbompc
