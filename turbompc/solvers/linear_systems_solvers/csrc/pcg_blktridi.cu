#include "pcg_blktridi.cuh"
#include "block_tridi.cuh"
#include <cmath>

// Portable atomicAdd wrapper — some CUDA header sets (e.g. conda-bundled
// cuda-toolkit) omit the double overload of atomicAdd.  This CAS-based
// helper works on any SM_60+ GPU regardless of header version.
__device__ __forceinline__ void atomicAddPortable(float* addr, float val) {
    atomicAdd(addr, val);  // always available
}
__device__ __forceinline__ void atomicAddPortable(double* addr, double val) {
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

// ============================================================
// Shared-memory device helpers for the single-block PCG kernel
// ============================================================

// Block-tridiagonal matvec: y = M @ v
// M is in global memory (T_blocks*n rows, 3n cols, row-major).
// v and y are in shared memory, length = total = T_blocks * n.
// All threads must call this (inactive threads skip compute but hit syncthreads).
template<typename T>
__device__ void blkTridiMatvecShared(
    T* __restrict__ y,
    const T* __restrict__ M,
    const T* __restrict__ v,
    int T_blocks, int n, int total)
{
    int tid = threadIdx.x;
    if (tid < total) {
        int t = tid / n;
        int i = tid % n;
        int n3 = 3 * n;
        const T* row = M + (t * n + i) * n3;
        T sum = T(0);
        if (t > 0) {
            const T* vp = v + (t - 1) * n;
            for (int j = 0; j < n; ++j) sum += row[j] * vp[j];
        }
        const T* vc = v + t * n;
        for (int j = 0; j < n; ++j) sum += row[n + j] * vc[j];
        if (t < T_blocks - 1) {
            const T* vn = v + (t + 1) * n;
            for (int j = 0; j < n; ++j) sum += row[2 * n + j] * vn[j];
        }
        y[tid] = sum;
    }
    __syncthreads();
}

// Block-wide dot product: returns a · b (valid on thread 0 only).
// warp_buf must have >= (blockDim.x / 32) entries in shared memory.
template<typename T>
__device__ T blockDotShared(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ warp_buf,
    int total)
{
    int tid = threadIdx.x;
    T local_sum = T(0);
    for (int i = tid; i < total; i += blockDim.x)
        local_sum += a[i] * b[i];

    // Warp-level reduce
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
    __syncthreads();   // all threads must sync before shared mem is reused
    return result;     // correct on thread 0 only
}

// Single-block persistent PCG: all iterations in one launch, zero host syncs.
// Working vectors in smem; S/Phiinv in global (L2-cached).
// Constraints: total = T_blocks*n <= 1024; smem = (5*total + 40)*sizeof(T).
template<typename T>
__global__ void pcgBlkTridiSingleBlock(
    T* __restrict__ x_out,
    uint32_t* __restrict__ iters_out,
    const T* __restrict__ S_global,
    const T* __restrict__ Phiinv_global,
    const T* __restrict__ rhs_global,
    const T* __restrict__ x0_global,
    T eps,
    int max_iters,
    int T_blocks,
    int n)
{
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    const int total = T_blocks * n;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Batch-offset pointers: each block solves one independent problem
    const int S_stride   = T_blocks * n * 3 * n;   // elements per S matrix
    const int vec_stride = total;                    // elements per vector (T*n)
    const T*     S_b      = S_global      + bid * S_stride;
    const T*     Phiinv_b = Phiinv_global + bid * S_stride;
    const T*     rhs_b    = rhs_global    + bid * vec_stride;
    const T*     x0_b     = x0_global     + bid * vec_stride;
    T*           xo_b     = x_out         + bid * vec_stride;
    uint32_t*    it_b     = iters_out     + bid;

    // Shared memory layout: x, r, z, p, Sp, scalars[8], warp_buf[32]
    T* x  = smem;
    T* r  = x  + total;
    T* z  = r  + total;
    T* p  = z  + total;
    T* Sp = p  + total;
    T* sc = Sp + total;        // sc[0]=rho, sc[1]=rho_init, sc[2]=denom,
                                // sc[3]=alpha, sc[4]=rho_new, sc[5]=beta
    T* warp_buf = sc + 8;      // 32 entries for warp reduce

    // ── Init: x = x0 ──
    for (int i = tid; i < total; i += blockDim.x)
        x[i] = x0_b[i];
    __syncthreads();

    // ── Sp = S @ x0 ──
    blkTridiMatvecShared(Sp, S_b, x, T_blocks, n, total);

    // ── r = rhs - Sp ──
    for (int i = tid; i < total; i += blockDim.x)
        r[i] = rhs_b[i] - Sp[i];
    __syncthreads();

    // ── z = Phiinv @ r ──
    blkTridiMatvecShared(z, Phiinv_b, r, T_blocks, n, total);

    // ── p = z ──
    for (int i = tid; i < total; i += blockDim.x)
        p[i] = z[i];
    __syncthreads();

    // ── rho = r · z ──
    {
        T dot = blockDotShared(r, z, warp_buf, total);
        if (tid == 0) {
            sc[0] = dot;           // rho
            sc[1] = fabs(dot);     // rho_init
        }
    }
    __syncthreads();

    const T abs_tol = T(1e-12);
    uint32_t final_iter = (uint32_t)max_iters;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Convergence check — all threads read same shared values
        if (fabs(sc[0]) < abs_tol + eps * sc[1]) {
            final_iter = (uint32_t)iter;
            break;
        }

        // ── Sp = S @ p ──
        blkTridiMatvecShared(Sp, S_b, p, T_blocks, n, total);

        // ── denom = p · Sp ──
        {
            T dot = blockDotShared(p, Sp, warp_buf, total);
            if (tid == 0) {
                T d = dot;
                if (fabs(d) < T(1e-30)) d = T(1e-30);
                sc[3] = sc[0] / d;  // alpha = rho / denom
            }
        }
        __syncthreads();

        // ── x += alpha*p,  r -= alpha*Sp ──
        {
            T alpha = sc[3];
            for (int i = tid; i < total; i += blockDim.x) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Sp[i];
            }
        }
        __syncthreads();

        // ── z = Phiinv @ r ──
        blkTridiMatvecShared(z, Phiinv_b, r, T_blocks, n, total);

        // ── rho_new = r · z ──
        {
            T dot = blockDotShared(r, z, warp_buf, total);
            if (tid == 0) {
                T rho_old = sc[0];
                sc[5] = dot / rho_old;  // beta = rho_new / rho
                sc[0] = dot;            // rho = rho_new
            }
        }
        __syncthreads();

        // ── p = z + beta * p ──
        {
            T beta = sc[5];
            for (int i = tid; i < total; i += blockDim.x)
                p[i] = z[i] + beta * p[i];
        }
        __syncthreads();
    }

    // ── Copy result to global output ──
    for (int i = tid; i < total; i += blockDim.x)
        xo_b[i] = x[i];

    if (tid == 0)
        *it_b = final_iter;
}

// ============================================================
// Legacy multi-kernel PCG (fallback for total > 1024)
// ============================================================

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0);
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

template<typename T>
__global__ void blkTridiMatvecKernel(
    T* __restrict__ y,
    const T* __restrict__ S,
    const T* __restrict__ x,
    int T_blocks, int n)
{
    int t = blockIdx.x;
    if (t >= T_blocks) return;

    const T* S_t = S + t * n * 3 * n;
    const T* x_prev = (t > 0)             ? x + (t - 1) * n : nullptr;
    const T* x_curr =                       x + t * n;
    const T* x_next = (t < T_blocks - 1)  ? x + (t + 1) * n : nullptr;
    T* y_t = y + t * n;

    blkTridiMatvecRow(y_t, S_t, x_prev, x_curr, x_next, n);
}

template<typename T>
__global__ void pcgInitResidual(
    T* __restrict__ r,
    const T* __restrict__ rhs,
    const T* __restrict__ Sx0,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) r[i] = rhs[i] - Sx0[i];
}

template<typename T>
__global__ void pcgUpdateXR(
    T* __restrict__ x,
    T* __restrict__ r,
    T alpha,
    const T* __restrict__ p,
    const T* __restrict__ Sp,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Sp[i];
    }
}

template<typename T>
__global__ void pcgUpdateP(
    T* __restrict__ p,
    const T* __restrict__ z,
    T beta,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        p[i] = z[i] + beta * p[i];
    }
}

template<typename T>
__global__ void pcgCopyKernel(T* __restrict__ dst, const T* __restrict__ src, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = src[i];
}

template<typename T>
__global__ void dotProductKernel(
    T* result,
    const T* __restrict__ a,
    const T* __restrict__ b,
    int total)
{
    T sum = T(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) atomicAddPortable(result, sum);
}

template<typename T>
static void pcgBlkTridiSolveLegacy(
    cudaStream_t stream,
    T* x_out,
    uint32_t* iters_out,
    const T* S,
    const T* Phiinv,
    const T* rhs,
    const T* x0,
    T eps,
    int32_t max_iters,
    int32_t T_blocks,
    int32_t n)
{
    int total = T_blocks * n;
    int elem_threads = 256;
    int elem_blocks = (total + elem_threads - 1) / elem_threads;
    int mv_blocks = T_blocks;
    int mv_threads = PCG_BLKTRIDI_THREADS;
    int dot_blocks = (total + elem_threads - 1) / elem_threads;
    if (dot_blocks > 256) dot_blocks = 256;

    T *r_dev, *z_dev, *p_dev, *Sp_dev;
    T *dot_buf;
    cudaMallocAsync(&r_dev, total * sizeof(T), stream);
    cudaMallocAsync(&z_dev, total * sizeof(T), stream);
    cudaMallocAsync(&p_dev, total * sizeof(T), stream);
    cudaMallocAsync(&Sp_dev, total * sizeof(T), stream);
    cudaMallocAsync(&dot_buf, sizeof(T), stream);

    pcgCopyKernel<<<elem_blocks, elem_threads, 0, stream>>>(x_out, x0, total);

    blkTridiMatvecKernel<<<mv_blocks, mv_threads, 0, stream>>>(Sp_dev, S, x_out, T_blocks, n);

    pcgInitResidual<<<elem_blocks, elem_threads, 0, stream>>>(r_dev, rhs, Sp_dev, total);

    blkTridiMatvecKernel<<<mv_blocks, mv_threads, 0, stream>>>(z_dev, Phiinv, r_dev, T_blocks, n);

    pcgCopyKernel<<<elem_blocks, elem_threads, 0, stream>>>(p_dev, z_dev, total);

    T rho_host;
    cudaMemsetAsync(dot_buf, 0, sizeof(T), stream);
    dotProductKernel<<<dot_blocks, elem_threads, 0, stream>>>(dot_buf, r_dev, z_dev, total);
    cudaMemcpyAsync(&rho_host, dot_buf, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    T rho_init = fabs(rho_host);
    T abs_tol = T(1e-12);
    uint32_t k = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        if (fabs(rho_host) < abs_tol + eps * rho_init) break;

        blkTridiMatvecKernel<<<mv_blocks, mv_threads, 0, stream>>>(Sp_dev, S, p_dev, T_blocks, n);

        T denom_host;
        cudaMemsetAsync(dot_buf, 0, sizeof(T), stream);
        dotProductKernel<<<dot_blocks, elem_threads, 0, stream>>>(dot_buf, p_dev, Sp_dev, total);
        cudaMemcpyAsync(&denom_host, dot_buf, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (fabs(denom_host) < T(1e-30)) denom_host = T(1e-30);
        T alpha = rho_host / denom_host;

        pcgUpdateXR<<<elem_blocks, elem_threads, 0, stream>>>(x_out, r_dev, alpha, p_dev, Sp_dev, total);

        blkTridiMatvecKernel<<<mv_blocks, mv_threads, 0, stream>>>(z_dev, Phiinv, r_dev, T_blocks, n);

        T rho_new_host;
        cudaMemsetAsync(dot_buf, 0, sizeof(T), stream);
        dotProductKernel<<<dot_blocks, elem_threads, 0, stream>>>(dot_buf, r_dev, z_dev, total);
        cudaMemcpyAsync(&rho_new_host, dot_buf, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        T beta = rho_new_host / rho_host;
        rho_host = rho_new_host;

        pcgUpdateP<<<elem_blocks, elem_threads, 0, stream>>>(p_dev, z_dev, beta, total);

        k = iter + 1;
    }

    cudaStreamSynchronize(stream);
    cudaMemcpy(iters_out, &k, sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaFreeAsync(r_dev, stream);
    cudaFreeAsync(z_dev, stream);
    cudaFreeAsync(p_dev, stream);
    cudaFreeAsync(Sp_dev, stream);
    cudaFreeAsync(dot_buf, stream);
}

// ============================================================
// Dispatch: single-block (fast) vs legacy (large problems)
// ============================================================

template<typename T>
static void pcgBlkTridiSolve(
    cudaStream_t stream,
    T* x_out,
    uint32_t* iters_out,
    const T* S,
    const T* Phiinv,
    const T* rhs,
    const T* x0,
    T eps,
    int32_t max_iters,
    int32_t T_blocks,
    int32_t n,
    int32_t Nb)
{
    int total = T_blocks * n;

    if (total <= 1024) {
        // Fast path: Nb persistent blocks, zero host-device syncs.
        // Each CUDA block solves one independent problem from the batch.
        int threads = ((total + 31) / 32) * 32;
        if (threads < 32)   threads = 32;
        if (threads > 1024) threads = 1024;
        // Shared memory: 5 vectors + 8 scalars + 32 warp buffer (per block)
        size_t smem = (5 * total + 8 + 32) * sizeof(T);

        pcgBlkTridiSingleBlock<T><<<Nb, threads, smem, stream>>>(
            x_out, iters_out, S, Phiinv, rhs, x0,
            eps, max_iters, T_blocks, n);
    } else {
        // Fallback: multi-kernel with host-device syncs (sequential over batch)
        int S_stride   = T_blocks * n * 3 * n;
        int vec_stride = total;
        for (int b = 0; b < Nb; ++b) {
            pcgBlkTridiSolveLegacy<T>(
                stream,
                x_out    + b * vec_stride,
                iters_out + b,
                S        + b * S_stride,
                Phiinv   + b * S_stride,
                rhs      + b * vec_stride,
                x0       + b * vec_stride,
                eps, max_iters, T_blocks, n);
        }
    }
}

void LaunchPcgBlkTridiF32(
    cudaStream_t stream, float* x_out, uint32_t* iters_out,
    const float* S, const float* Phiinv, const float* rhs, const float* x0,
    float eps, int32_t max_iters, int32_t T_blocks, int32_t n, int32_t Nb)
{
    pcgBlkTridiSolve<float>(stream, x_out, iters_out, S, Phiinv, rhs, x0,
                            eps, max_iters, T_blocks, n, Nb);
}

void LaunchPcgBlkTridiF64(
    cudaStream_t stream, double* x_out, uint32_t* iters_out,
    const double* S, const double* Phiinv, const double* rhs, const double* x0,
    double eps, int32_t max_iters, int32_t T_blocks, int32_t n, int32_t Nb)
{
    pcgBlkTridiSolve<double>(stream, x_out, iters_out, S, Phiinv, rhs, x0,
                             eps, max_iters, T_blocks, n, Nb);
}
