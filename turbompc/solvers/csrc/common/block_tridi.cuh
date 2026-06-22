#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Block-tridiagonal matvec: y[t] = A_sub[t] @ x[t-1] + A_diag[t] @ x[t] + A_sup[t] @ x[t+1]
// S_row: pointer to S[t, :, :] which is (n, 3n) stored row-major
// x_prev, x_curr, x_next: pointers to (n,) vectors
// y_out: pointer to (n,) output
// n: block size
// This is a device function called within a CUDA block (threads cooperate on the matvec).
template<typename T>
__device__ void blkTridiMatvecRow(
    T* y_out,
    const T* S_row,     // (n, 3n)
    const T* x_prev,    // (n,) or nullptr if t==0
    const T* x_curr,    // (n,)
    const T* x_next,    // (n,) or nullptr if t==T-1
    int n);

// Assemble block-tridiagonal blocks into CSR arrays on device.
void blkTridiToCSR(
    const float* S_dev,
    int32_t* indptr_out,
    int32_t* indices_out,
    float* data_out,
    int T_blocks,
    int n,
    cudaStream_t stream);

void blkTridiToCSR_f64(
    const double* S_dev,
    int32_t* indptr_out,
    int32_t* indices_out,
    double* data_out,
    int T_blocks,
    int n,
    cudaStream_t stream);

// Values-only variants: update only the CSR data array.
// Call after the full blkTridiToCSR has been called once (indptr/indices unchanged).
void blkTridiToCSR_data_only(
    const float* S_dev,
    float* data_out,
    int T_blocks,
    int n,
    cudaStream_t stream);

void blkTridiToCSR_data_only_f64(
    const double* S_dev,
    double* data_out,
    int T_blocks,
    int n,
    cudaStream_t stream);

// Batched block-diagonal CSR assembly.
// S_dev: [Nb, T, n, 3n] contiguous.
// Produces a single (Nb*T*n) x (Nb*T*n) block-diagonal CSR matrix.
void batchedBlkTridiToCSR(
    const float* S_dev,
    int32_t* indptr_out,
    int32_t* indices_out,
    float* data_out,
    int Nb, int T_blocks, int n,
    cudaStream_t stream);

void batchedBlkTridiToCSR_f64(
    const double* S_dev,
    int32_t* indptr_out,
    int32_t* indices_out,
    double* data_out,
    int Nb, int T_blocks, int n,
    cudaStream_t stream);

// Batched values-only variants.
void batchedBlkTridiToCSR_data_only(
    const float* S_dev,
    float* data_out,
    int Nb, int T_blocks, int n,
    cudaStream_t stream);

void batchedBlkTridiToCSR_data_only_f64(
    const double* S_dev,
    double* data_out,
    int Nb, int T_blocks, int n,
    cudaStream_t stream);
