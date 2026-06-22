#pragma once

// Block-wide dot product: returns a . b (valid on thread 0 only).
template<typename T>
__device__ T blockDotShared_admm(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* warp_buf,  // no __restrict__: cross-thread reduction buffer
    int total)
{
    int tid = threadIdx.x;
    T local_sum = T(0);
    for (int i = tid; i < total; i += blockDim.x)
        local_sum += a[i] * b[i];

    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    int lane = tid & 31;
    int wid  = tid >> 5;
    if (lane == 0) warp_buf[wid] = local_sum;
    __syncthreads();

    T result = T(0);
    if (wid == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        result = (lane < num_warps) ? warp_buf[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2)
            result += __shfl_down_sync(0xffffffff, result, offset);
    }
    __syncthreads();
    return result;
}

// Block-wide inf-norm: returns max_i |a[i]| (valid on thread 0 only).
template<typename T>
__device__ T blockInfNormShared(
    const T* __restrict__ a,
    T* warp_buf,  // no __restrict__: cross-thread reduction buffer
    int total)
{
    int tid = threadIdx.x;
    T local_max = T(0);
    for (int i = tid; i < total; i += blockDim.x) {
        T v = fabs(a[i]);
        if (v > local_max) local_max = v;
    }

    // Warp-level reduce with max
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }

    int lane = tid & 31;
    int wid  = tid >> 5;
    if (lane == 0) warp_buf[wid] = local_max;
    __syncthreads();

    T result = T(0);
    if (wid == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        result = (lane < num_warps) ? warp_buf[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, result, offset);
            if (other > result) result = other;
        }
    }
    __syncthreads();
    return result;
}
