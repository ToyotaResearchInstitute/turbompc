#pragma once
#include <cuda_runtime.h>
#include <cstdint>

constexpr int PCG_BLKTRIDI_THREADS = 128;

void LaunchPcgBlkTridiF32(
    cudaStream_t stream,
    float* x_out,
    uint32_t* iters_out,
    const float* S,
    const float* Phiinv,
    const float* rhs,
    const float* x0,
    float eps,
    int32_t max_iters,
    int32_t T_blocks,
    int32_t n,
    int32_t Nb = 1);

void LaunchPcgBlkTridiF64(
    cudaStream_t stream,
    double* x_out,
    uint32_t* iters_out,
    const double* S,
    const double* Phiinv,
    const double* rhs,
    const double* x0,
    double eps,
    int32_t max_iters,
    int32_t T_blocks,
    int32_t n,
    int32_t Nb = 1);
