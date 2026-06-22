#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void CudssBlkTridiSolveF32(
    cudaStream_t stream,
    float* x_out_dev,
    const float* S_dev,
    const float* rhs_dev,
    int32_t T_blocks,
    int32_t n,
    int32_t Nb = 1);

void CudssBlkTridiSolveF64(
    cudaStream_t stream,
    double* x_out_dev,
    const double* S_dev,
    const double* rhs_dev,
    int32_t T_blocks,
    int32_t n,
    int32_t Nb = 1);
