#include "ffi_device_memory.h"
#include "admm_cudss.cuh"
#include "admm_infeasibility.cuh"
#include "admm_status.cuh"
#include "admm_math.cuh"
#include "admm_reductions.cuh"
#include "block_tridi.cuh"
#include <cudss.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <unordered_map>

// Throw on cuDSS/CUDA errors so the FFI handler can return ffi::Error::Internal
// rather than aborting the process.
#define CUDSS_CHECK(call)                                                      \
  do {                                                                         \
    cudssStatus_t _err = (call);                                               \
    if (_err != CUDSS_STATUS_SUCCESS) {                                        \
      throw std::runtime_error(                                                \
          std::string("cuDSS error at " __FILE__ ":") +                        \
          std::to_string(__LINE__) + ": status code " +                        \
          std::to_string(static_cast<int>(_err)));                             \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_THROW(call)                                                 \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(                                                \
          std::string("CUDA error at " __FILE__ ":") +                         \
          std::to_string(__LINE__) + ": " + cudaGetErrorString(_err));         \
    }                                                                          \
  } while (0)

// ============================================================
// Global-memory ADMM body kernels.
// Each kernel launches Nb blocks (one per batch element).
// ADMM state lives in global memory with batch offsets.
// ============================================================

// ---- Kernel 1: init state — copy warm-start to working buffers ----
template<typename T>
__global__ void initStateKernel(
    T* __restrict__ x,           // [Nb, T*n]
    T* __restrict__ z_g,         // [Nb, T*m]
    T* __restrict__ y_g,         // [Nb, T*m]
    T* __restrict__ y_f_0,       // [Nb, nx]
    T* __restrict__ y_f_dyn,     // [Nb, N*nx]
    T* __restrict__ xi_g,        // [Nb, T*m]
    const T* __restrict__ x0,
    const T* __restrict__ z_g0,
    const T* __restrict__ y_g0,
    const T* __restrict__ y_f_0_init,
    const T* __restrict__ y_f_dyn_init,
    const T* __restrict__ xi_g0,
    int total_tn, int total_Tm, int nx_val, int total_Nnx,
    uint32_t* status, uint32_t* iterations, int max_iter)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) {
        status[bid] = static_cast<uint32_t>(ADMM_STATUS_NOT_RUN);
        iterations[bid] = static_cast<uint32_t>(max_iter);
    }

    for (int i = tid; i < total_tn; i += blockDim.x)
        x[bid * total_tn + i] = x0[bid * total_tn + i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        z_g[bid * total_Tm + i] = z_g0[bid * total_Tm + i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        y_g[bid * total_Tm + i] = y_g0[bid * total_Tm + i];
    for (int i = tid; i < nx_val; i += blockDim.x)
        y_f_0[bid * nx_val + i] = y_f_0_init[bid * nx_val + i];
    for (int i = tid; i < total_Nnx; i += blockDim.x)
        y_f_dyn[bid * total_Nnx + i] = y_f_dyn_init[bid * total_Nnx + i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        xi_g[bid * total_Tm + i] = xi_g0[bid * total_Tm + i];
}

// gamma = sigma*x - q + C^T(rho_f*c - y_f) + G^T(rho_bar*z_g - y_g);
// also saves x_old = x for over-relaxation.
template<typename T>
__global__ void __launch_bounds__(768) computeGammaAndSaveXOldKernel(
    T* __restrict__ gamma,       // [Nb, T*n]
    T* __restrict__ x_old,       // [Nb, T*n]
    const T* __restrict__ x,     // [Nb, T*n]
    const T* __restrict__ z_g,   // [Nb, T*m]
    const T* __restrict__ y_g,   // [Nb, T*m]
    const T* __restrict__ y_f_0, // [Nb, n0]
    const T* __restrict__ y_f_dyn, // [Nb, N*nx]
    const T* __restrict__ q_g,   // [Nb, T*n]
    const T* __restrict__ A0_g,  // [Nb, n0, n]
    const T* __restrict__ Am_g,  // [Nb, N, nx, n]
    const T* __restrict__ Ap_g,  // [Nb, N, nx, n]
    const T* __restrict__ G_g,   // [Nb, T, m, n]
    const T* __restrict__ c0_g,  // [Nb, n0]
    const T* __restrict__ cdyn_g,// [Nb, N, nx]
    T sigma, T rho_f_factor,
    const T* __restrict__ rho_bar_arr,  // [Nb]
    int T_blocks, int n, int nx, int n0, int m,
    const uint32_t* __restrict__ status)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int total_tn = T_blocks * n;
    int total_Tm = T_blocks * m;
    int N = T_blocks - 1;
    int total_Nnx = N * nx;

    T rho_bar = rho_bar_arr[bid];
    T rho_f = rho_bar * rho_f_factor;

    // Batch offsets
    int off_tn = bid * total_tn;
    int off_Tm = bid * total_Tm;
    int off_n0 = bid * n0;
    int off_Nnx = bid * total_Nnx;
    int off_A0 = bid * n0 * n;
    int off_Am = bid * N * nx * n;
    int off_Ap = bid * N * nx * n;
    int off_G  = bid * T_blocks * m * n;
    int off_c0 = bid * n0;
    int off_cd = bid * N * nx;

    // Save x_old
    for (int i = tid; i < total_tn; i += blockDim.x)
        x_old[off_tn + i] = x[off_tn + i];
    // cuDSS still solves every RHS. Keep the terminal primal so post-solve
    // can restore it while the other rows continue iterating.
    if (status[bid] != static_cast<uint32_t>(ADMM_STATUS_NOT_RUN)) return;

    // Compute gamma using shared device function (admm_math.cuh)
    computeGammaDevice(
        gamma + off_tn,
        x + off_tn, z_g + off_Tm, y_g + off_Tm,
        y_f_0 + off_n0, y_f_dyn + off_Nnx,
        q_g + off_tn, A0_g + off_A0, Am_g + off_Am, Ap_g + off_Ap,
        G_g + off_G, c0_g + off_c0, cdyn_g + off_cd,
        sigma, rho_f, rho_bar,
        T_blocks, n, nx, n0, m);
}

template<typename T>
__global__ void snapshotCertificateDualsKernel(
    T* previous_y_g, T* previous_y_f_0, T* previous_y_f_dyn,
    T* previous_xi_g, const T* y_g, const T* y_f_0,
    const T* y_f_dyn, const T* xi_g, int total_Tm, int n0, int total_Nnx)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int off_Tm = bid * total_Tm;
    int off_n0 = bid * n0;
    int off_Nnx = bid * total_Nnx;
    for (int i = tid; i < total_Tm; i += blockDim.x) {
        previous_y_g[off_Tm + i] = y_g[off_Tm + i];
        previous_xi_g[off_Tm + i] = xi_g[off_Tm + i];
    }
    for (int i = tid; i < n0; i += blockDim.x)
        previous_y_f_0[off_n0 + i] = y_f_0[off_n0 + i];
    for (int i = tid; i < total_Nnx; i += blockDim.x)
        previous_y_f_dyn[off_Nnx + i] = y_f_dyn[off_Nnx + i];
}

// ---- Kernel 3: post-solve — constraints + over-relax + inequality + dual ----
// Fuses: applyConstraints, overRelax, save z_g_old, inequality proximal, dualUpdate
//
// rho_bar is read per-block from rho_bar_arr[bid]; rho_f = rho_bar * rho_f_factor.
// rho_bar and slack_weight are read per batch.
//
// `Handling` selects the inequality proximal at compile time, so the projection
// path carries none of the barrier code. The host launcher picks the
// instantiation; everything else is shared.
template<typename T, ConstraintHandling Handling>
__global__ void __launch_bounds__(768) postSolveKernel(
    T* __restrict__ x,           // [Nb, T*n] — in/out (over-relaxed)
    T* __restrict__ z_g,         // [Nb, T*m] — in/out (updated slack)
    T* __restrict__ y_g,         // [Nb, T*m] — in/out (updated dual)
    T* __restrict__ y_f_0,       // [Nb, n0]  — in/out
    T* __restrict__ y_f_dyn,     // [Nb, N*nx] — in/out
    // Scratch buffers
    T* __restrict__ Gx,          // [Nb, T*m]  — scratch for G @ x
    T* __restrict__ Cx0,         // [Nb, n0]   — scratch
    T* __restrict__ Cx_dyn,      // [Nb, N*nx] — scratch
    T* __restrict__ z_g_old,     // [Nb, T*m]  — scratch
    const T* __restrict__ x_old, // [Nb, T*n]
    // QP data (read-only)
    const T* __restrict__ A0_g,
    const T* __restrict__ Am_g,
    const T* __restrict__ Ap_g,
    const T* __restrict__ G_g,
    const T* __restrict__ l_g,
    const T* __restrict__ u_g,
    const T* __restrict__ c0_g,
    const T* __restrict__ cdyn_g,
    T alpha_relax, T rho_f_factor,
    const T* __restrict__ rho_bar_arr,    // [Nb]
    int T_blocks, int n, int nx, int n0, int m,
    const T* __restrict__ slack_weight_arr,  // [Nb]
    bool use_slack, T log_barrier_kappa, T* __restrict__ xi_g,
    const uint32_t* __restrict__ status)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int N = T_blocks - 1;
    int total_tn = T_blocks * n;
    int total_Tm = T_blocks * m;
    int total_Nnx = N * nx;

    if (status[bid] != static_cast<uint32_t>(ADMM_STATUS_NOT_RUN)) {
        for (int i = tid; i < total_tn; i += blockDim.x)
            x[bid * total_tn + i] = x_old[bid * total_tn + i];
        return;
    }

    T rho_bar = rho_bar_arr[bid];
    T rho_f = rho_bar * rho_f_factor;
    T slack_weight = slack_weight_arr[bid];

    // Batch offsets
    int off_tn  = bid * total_tn;
    int off_Tm  = bid * total_Tm;
    int off_n0  = bid * n0;
    int off_Nnx = bid * total_Nnx;
    int off_A0  = bid * n0 * n;
    int off_Am  = bid * N * nx * n;
    int off_Ap  = bid * N * nx * n;
    int off_G   = bid * T_blocks * m * n;
    int off_c0  = bid * n0;
    int off_cd  = bid * N * nx;

    // Step 1: Compute constraints (Cx0, Cx_dyn, Gx) — shared with fused kernel
    applyConstraintsDevice(
        Cx0 + off_n0, Cx_dyn + off_Nnx, Gx + off_Tm,
        x + off_tn,
        A0_g + off_A0, Am_g + off_Am, Ap_g + off_Ap, G_g + off_G,
        T_blocks, n, nx, n0, m);

    // Step 2: Over-relaxation — shared with fused kernel
    overRelaxDevice(x + off_tn, x_old + off_tn, alpha_relax, total_tn);

    // Step 3: Save z_g_old, then the inequality proximal
    if (m > 0) {
        for (int i = tid; i < total_Tm; i += blockDim.x)
            z_g_old[off_Tm + i] = z_g[off_Tm + i];
    }
    __syncthreads();

    if constexpr (Handling == ConstraintHandling::LogBarrier) {
        logBarrierUpdateDevice(
            z_g + off_Tm, Gx + off_Tm, y_g + off_Tm,
            l_g + bid * total_Tm, u_g + bid * total_Tm,
            alpha_relax, rho_bar, T_blocks, m,
            log_barrier_kappa, slack_weight, use_slack, xi_g + off_Tm);
    } else {
        projectionUpdateDevice(
            z_g + off_Tm, Gx + off_Tm, y_g + off_Tm,
            l_g + bid * total_Tm, u_g + bid * total_Tm,
            alpha_relax, rho_bar, T_blocks, m,
            slack_weight, use_slack, xi_g + off_Tm);
    }

    // Step 4: Dual update — shared with fused kernel
    dualUpdateDevice(
        y_f_0 + off_n0, y_f_dyn + off_Nnx, y_g + off_Tm,
        Cx0 + off_n0, Cx_dyn + off_Nnx, Gx + off_Tm,
        z_g + off_Tm, z_g_old + off_Tm,
        c0_g + off_c0, cdyn_g + off_cd,
        alpha_relax, rho_f, rho_bar,
        T_blocks, n, nx, n0, m);
}

// ---- Kernel 4: compute residuals and norm terms ----
// Uses shared memory for block reductions (inf-norm).
// Writes per-batch: residuals_out[bid*4+0] = primal_residual
//                   residuals_out[bid*4+1] = dual_residual
//                   residuals_out[bid*4+2] = primal_norm_term
//                   residuals_out[bid*4+3] = dual_norm_term
// Norm terms match JAX _compute_residuals for eps_rel scaling.
template<typename T>
__global__ void __launch_bounds__(768) computeResidualsKernel(
    T* __restrict__ residuals_out, // [Nb, 4]
    uint32_t* __restrict__ finite_out, // [Nb]
    const T* __restrict__ x,
    const T* __restrict__ z_g,
    const T* __restrict__ y_g,
    const T* __restrict__ y_f_0,
    const T* __restrict__ y_f_dyn,
    const T* __restrict__ D_g,
    const T* __restrict__ E_g,
    const T* __restrict__ q_g,
    const T* __restrict__ A0_g,
    const T* __restrict__ Am_g,
    const T* __restrict__ Ap_g,
    const T* __restrict__ G_g,
    const T* __restrict__ c0_g,
    const T* __restrict__ cdyn_g,
    int T_blocks, int n, int nx, int n0, int m,
    const T* __restrict__ xi_g,
    const T* __restrict__ rho_bar,
    const T* __restrict__ slack_weight_arr,  // [Nb]
    bool use_slack)
{
    extern __shared__ char smem_raw[];
    T* warp_buf = reinterpret_cast<T*>(smem_raw);

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int N = T_blocks - 1;
    int total_tn = T_blocks * n;
    int total_Tm = T_blocks * m;
    int total_Nnx = N * nx;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    T slack_weight = slack_weight_arr[bid];

    // Batch offsets
    int off_tn  = bid * total_tn;
    int off_Tm  = bid * total_Tm;
    int off_n0  = bid * n0;
    int off_Nnx = bid * total_Nnx;
    int off_A0  = bid * n0 * n;
    int off_Am  = bid * N * nx * n;
    int off_Ap  = bid * N * nx * n;
    int off_G   = bid * T_blocks * m * n;
    int off_D   = bid * T_blocks * n * n;
    int off_E   = bid * N * n * n;
    int off_c0  = bid * n0;
    int off_cd  = bid * N * nx;

    // Check state finiteness in parallel with residual evaluation.
    if (tid == 0) finite_out[bid] = 1;
    __syncthreads();
    bool local_finite = true;
    if (tid == 0) local_finite = local_finite && isfinite(rho_bar[bid]);

    // ---- Primal residual + norm term ----
    // primal_res = max(||Cx0-c0||, ||Cx_dyn-c_dyn||, ||Gx-z_g||)
    // primal_norm = max(||Cx0||, ||Cx_dyn||, ||c0||, ||c_dyn||, ||Gx||, ||z_g||)
    T local_prim = T(0);
    T local_prim_norm = T(0);

    // Cx0 - c0
    for (int j = tid; j < n0; j += blockDim.x) {
        local_finite = local_finite && isfinite(y_f_0[off_n0 + j]);
        T sum = T(0);
        for (int k = 0; k < n; ++k)
            sum += A0_g[off_A0 + j * n + k] * x[off_tn + k];
        T c_val = c0_g[off_c0 + j];
        T r = fabs(sum - c_val);
        if (r > local_prim) local_prim = r;
        T a = fabs(sum); if (a > local_prim_norm) local_prim_norm = a;
        a = fabs(c_val);  if (a > local_prim_norm) local_prim_norm = a;
    }

    // Cx_dyn - c_dyn
    for (int idx = tid; idx < total_Nnx; idx += blockDim.x) {
        local_finite = local_finite && isfinite(y_f_dyn[off_Nnx + idx]);
        int t = idx / nx;
        int j = idx % nx;
        T sum = T(0);
        const T* Am_row = Am_g + off_Am + t * nx * n + j * n;
        const T* Ap_row = Ap_g + off_Ap + t * nx * n + j * n;
        const T* x_t  = x + off_tn + t * n;
        const T* x_tp = x + off_tn + (t + 1) * n;
        for (int k = 0; k < n; ++k)
            sum += Am_row[k] * x_t[k] + Ap_row[k] * x_tp[k];
        T c_val = cdyn_g[off_cd + idx];
        T r = fabs(sum - c_val);
        if (r > local_prim) local_prim = r;
        T a = fabs(sum); if (a > local_prim_norm) local_prim_norm = a;
        a = fabs(c_val);  if (a > local_prim_norm) local_prim_norm = a;
    }

    // Gx - z_g
    if (m > 0) {
        for (int idx = tid; idx < total_Tm; idx += blockDim.x) {
            local_finite = local_finite && isfinite(z_g[off_Tm + idx])
                && isfinite(y_g[off_Tm + idx])
                && isfinite(xi_g[off_Tm + idx]);
            int t = idx / m;
            int j = idx % m;
            T sum = T(0);
            const T* G_row = G_g + off_G + t * m * n + j * n;
            const T* x_t = x + off_tn + t * n;
            for (int k = 0; k < n; ++k)
                sum += G_row[k] * x_t[k];
            T zg_val = z_g[off_Tm + idx];
            T r = fabs(sum - zg_val);
            if (r > local_prim) local_prim = r;
            T a = fabs(sum);    if (a > local_prim_norm) local_prim_norm = a;
            a = fabs(zg_val);    if (a > local_prim_norm) local_prim_norm = a;
        }
    }

    // Reduce primal_res
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, local_prim, offset);
        if (other > local_prim) local_prim = other;
    }
    if (lane == 0) warp_buf[wid] = local_prim;
    __syncthreads();
    T primal_res = T(0);
    if (wid == 0) {
        primal_res = (lane < num_warps) ? warp_buf[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, primal_res, offset);
            if (other > primal_res) primal_res = other;
        }
    }
    __syncthreads();

    // Reduce primal_norm_term
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, local_prim_norm, offset);
        if (other > local_prim_norm) local_prim_norm = other;
    }
    if (lane == 0) warp_buf[wid] = local_prim_norm;
    __syncthreads();
    T primal_norm_term = T(0);
    if (wid == 0) {
        primal_norm_term = (lane < num_warps) ? warp_buf[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, primal_norm_term, offset);
            if (other > primal_norm_term) primal_norm_term = other;
        }
    }
    __syncthreads();

    // ---- Dual residual + norm term ----
    // dual_res = ||Px + q + C^T*y_f + G^T*y_g||_inf
    // dual_norm = max(||Px||, ||q||, ||C^T*y_f + G^T*y_g||)
    T local_dual = T(0);
    T local_dual_norm = T(0);

    for (int idx = tid; idx < total_tn; idx += blockDim.x) {
        local_finite = local_finite && isfinite(x[off_tn + idx]);
        int t = idx / n;
        int j = idx % n;

        // Px: D[t] @ x[t]
        T px_val = T(0);
        const T* D_t = D_g + off_D + t * n * n;
        for (int k = 0; k < n; ++k)
            px_val += D_t[j * n + k] * x[off_tn + t * n + k];
        if (t > 0) {
            const T* E_tm1 = E_g + off_E + (t - 1) * n * n;
            for (int k = 0; k < n; ++k)
                px_val += E_tm1[j * n + k] * x[off_tn + (t - 1) * n + k];
        }
        if (t < N) {
            const T* E_t = E_g + off_E + t * n * n;
            for (int k = 0; k < n; ++k)
                px_val += E_t[k * n + j] * x[off_tn + (t + 1) * n + k];
        }

        // C^T @ y_f
        T ct_yf = T(0);
        if (t == 0) {
            for (int k = 0; k < n0; ++k)
                ct_yf += A0_g[off_A0 + k * n + j] * y_f_0[off_n0 + k];
            if (N > 0) {
                for (int k = 0; k < nx; ++k)
                    ct_yf += Am_g[off_Am + k * n + j] * y_f_dyn[off_Nnx + k];
            }
        } else if (t < N) {
            const T* Aplus_tm1 = Ap_g + off_Ap + (t - 1) * nx * n;
            const T* Aminus_t = Am_g + off_Am + t * nx * n;
            for (int k = 0; k < nx; ++k)
                ct_yf += Aplus_tm1[k * n + j] * y_f_dyn[off_Nnx + (t - 1) * nx + k];
            for (int k = 0; k < nx; ++k)
                ct_yf += Aminus_t[k * n + j] * y_f_dyn[off_Nnx + t * nx + k];
        } else {
            if (N > 0) {
                const T* Aplus_Nm1 = Ap_g + off_Ap + (N - 1) * nx * n;
                for (int k = 0; k < nx; ++k)
                    ct_yf += Aplus_Nm1[k * n + j] * y_f_dyn[off_Nnx + (N - 1) * nx + k];
            }
        }

        // G^T @ y_g
        T gt_yg = T(0);
        if (m > 0) {
            const T* G_t = G_g + off_G + t * m * n;
            for (int k = 0; k < m; ++k)
                gt_yg += G_t[k * n + j] * y_g[off_Tm + t * m + k];
        }

        T r = fabs(px_val + q_g[off_tn + idx] + ct_yf + gt_yg);
        if (r > local_dual) local_dual = r;

        // Norm terms: max(||Px||, ||q||, ||Aty||)
        T a = fabs(px_val);           if (a > local_dual_norm) local_dual_norm = a;
        a = fabs(q_g[off_tn + idx]);  if (a > local_dual_norm) local_dual_norm = a;
        a = fabs(ct_yf + gt_yg);     if (a > local_dual_norm) local_dual_norm = a;
    }

    // Reduce dual_res
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, local_dual, offset);
        if (other > local_dual) local_dual = other;
    }
    if (lane == 0) warp_buf[wid] = local_dual;
    __syncthreads();
    T dual_res = T(0);
    if (wid == 0) {
        dual_res = (lane < num_warps) ? warp_buf[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, dual_res, offset);
            if (other > dual_res) dual_res = other;
        }
    }
    __syncthreads();

    // Reduce dual_norm_term
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, local_dual_norm, offset);
        if (other > local_dual_norm) local_dual_norm = other;
    }
    if (lane == 0) warp_buf[wid] = local_dual_norm;
    __syncthreads();
    T dual_norm_term = T(0);
    if (wid == 0) {
        dual_norm_term = (lane < num_warps) ? warp_buf[lane] : T(0);
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, dual_norm_term, offset);
            if (other > dual_norm_term) dual_norm_term = other;
        }
    }

    // Slack dual residual + norm terms: ||slack_weight * xi_g + y_g||_inf
    if (use_slack && m > 0) {
        T local_slack_dual = T(0);
        T local_slack_norm = T(0);
        for (int idx = tid; idx < total_Tm; idx += blockDim.x) {
            T r = fabs(slack_weight * xi_g[off_Tm + idx] + y_g[off_Tm + idx]);
            if (r > local_slack_dual) local_slack_dual = r;
            T a = slack_weight * fabs(xi_g[off_Tm + idx]);
            if (a > local_slack_norm) local_slack_norm = a;
            a = fabs(y_g[off_Tm + idx]);
            if (a > local_slack_norm) local_slack_norm = a;
        }
        // Reduce slack dual residual
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, local_slack_dual, offset);
            if (other > local_slack_dual) local_slack_dual = other;
        }
        if (lane == 0) warp_buf[wid] = local_slack_dual;
        __syncthreads();
        if (wid == 0) {
            T slack_dual_res = (lane < num_warps) ? warp_buf[lane] : T(0);
            for (int offset = 16; offset > 0; offset /= 2) {
                T other = __shfl_down_sync(0xffffffff, slack_dual_res, offset);
                if (other > slack_dual_res) slack_dual_res = other;
            }
            if (lane == 0 && slack_dual_res > dual_res)
                dual_res = slack_dual_res;
        }
        __syncthreads();
        // Reduce slack norm terms and merge into dual_norm_term
        for (int offset = 16; offset > 0; offset /= 2) {
            T other = __shfl_down_sync(0xffffffff, local_slack_norm, offset);
            if (other > local_slack_norm) local_slack_norm = other;
        }
        if (lane == 0) warp_buf[wid] = local_slack_norm;
        __syncthreads();
        if (wid == 0) {
            T slack_norm = (lane < num_warps) ? warp_buf[lane] : T(0);
            for (int offset = 16; offset > 0; offset /= 2) {
                T other = __shfl_down_sync(0xffffffff, slack_norm, offset);
                if (other > slack_norm) slack_norm = other;
            }
            if (lane == 0 && slack_norm > dual_norm_term)
                dual_norm_term = slack_norm;
        }
    }

    if (!local_finite) atomicExch(finite_out + bid, uint32_t(0));
    if (tid == 0) {
        residuals_out[bid * 4 + 0] = primal_res;
        residuals_out[bid * 4 + 1] = dual_res;
        residuals_out[bid * 4 + 2] = primal_norm_term;
        residuals_out[bid * 4 + 3] = dual_norm_term;
    }
}

template<typename T>
__global__ void computeInfeasibilityMetricsKernel(
    T* metrics,
    const T* x, const T* yg, const T* yf0, const T* yfdyn, const T* xi,
    const T* previous_x, const T* previous_yg,
    const T* previous_yf0, const T* previous_yfdyn, const T* previous_xi,
    const T* D, const T* E, const T* q,
    const T* A0, const T* Am, const T* Ap, const T* G,
    const T* lower, const T* upper, const T* c0, const T* cdyn,
    const T* slack_weight, bool use_slack,
    int Tn, int n, int nx, int n0, int m)
{
    int batch = blockIdx.x;
    int N = Tn - 1;
    int tn = Tn * n;
    int tm = Tn * m;
    int ndyn = N * nx;
    extern __shared__ unsigned char shared_bytes[];
    T* warp_buffer = reinterpret_cast<T*>(shared_bytes);
    ComputeADMMInfeasibilityMetrics(
        metrics + batch * 7,
        x + batch * tn, yg + batch * tm, yf0 + batch * n0,
        yfdyn + batch * ndyn, xi + batch * tm,
        previous_x + batch * tn, previous_yg + batch * tm,
        previous_yf0 + batch * n0, previous_yfdyn + batch * ndyn,
        previous_xi + batch * tm,
        D + batch * Tn * n * n, E + batch * N * n * n,
        q + batch * tn, A0 + batch * n0 * n,
        Am + batch * N * nx * n, Ap + batch * N * nx * n,
        G + batch * Tn * m * n, lower + batch * tm, upper + batch * tm,
        c0 + batch * n0, cdyn + batch * ndyn,
        slack_weight[batch], use_slack, Tn, n, nx, n0, m, warp_buffer);
}

// Select statuses and signal the host when the native loop must stop.
template<typename T>
__global__ void checkConvergenceSignalKernel(
    const T* __restrict__ residuals,
    const T* __restrict__ metrics,
    const uint32_t* __restrict__ finite_flags,
    uint32_t* __restrict__ status_out,
    uint32_t* __restrict__ iterations_out,
    T* __restrict__ terminal_residuals,
    volatile int* converged_flag,     // mapped memory (host-visible)
    T eps_abs,
    T eps_rel,
    T eps_prim_inf,
    T eps_dual_inf,
    int Nb,
    int current_iter,
    bool check_termination,
    bool check_infeasibility,
    bool iteration_limit_reached)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    bool all_finished = true;
    for (int b = 0; b < Nb; ++b) {
        if (status_out[b] != static_cast<uint32_t>(ADMM_STATUS_NOT_RUN)) continue;
        T p_res  = residuals[b * 4 + 0];
        T d_res  = residuals[b * 4 + 1];
        T p_norm = residuals[b * 4 + 2];
        T d_norm = residuals[b * 4 + 3];
        T p_tol  = eps_abs + eps_rel * p_norm;
        T d_tol  = eps_abs + eps_rel * d_norm;
        const T* certificate = metrics + b * 7;
        T direction_floor = T(10) * (
            sizeof(T) == sizeof(float) ? T(FLT_EPSILON) : T(DBL_EPSILON));
        bool finite = finite_flags[b] != 0
            && isfinite(p_res) && isfinite(d_res)
            && isfinite(p_norm) && isfinite(d_norm);
        bool solved = check_termination && finite
            && p_res <= p_tol && d_res <= d_tol;
        bool primal_infeasible = check_infeasibility
            && certificate[0] > direction_floor
            && certificate[1] <= eps_prim_inf * certificate[0]
            && certificate[2] <= -eps_prim_inf * certificate[0];
        bool dual_infeasible = check_infeasibility
            && certificate[3] > direction_floor
            && certificate[4] <= eps_dual_inf * certificate[3]
            && certificate[5] <= -eps_dual_inf * certificate[3]
            && certificate[6] <= eps_dual_inf * certificate[3];
        int32_t status = SelectADMMStatus(
            !finite, solved, primal_infeasible, dual_infeasible,
            iteration_limit_reached);
        status_out[b] = static_cast<uint32_t>(status);
        if (status != ADMM_STATUS_NOT_RUN)
            iterations_out[b] = static_cast<uint32_t>(current_iter);
        terminal_residuals[b * 4] = p_res;
        terminal_residuals[b * 4 + 1] = d_res;
        terminal_residuals[b * 4 + 2] = p_tol;
        terminal_residuals[b * 4 + 3] = d_tol;
        all_finished = all_finished && status != ADMM_STATUS_NOT_RUN;
    }
    if (all_finished) {
        *converged_flag = current_iter;
        __threadfence_system();
    }
}

// ---- Kernel 4c: OSQP-style adaptive rho update ----
// One thread per batch element. Reads p_norm/d_norm from d_residuals,
// computes new rho via computeRhoCandidateDevice, writes it to
// d_rho_bar_work[bid] when the gate fires, and bumps a host-visible
// "any-changed" flag so the host knows whether to re-factorise.
template<typename T>
__global__ void rhoUpdateKernel(
    const T* __restrict__ residuals,          // [Nb, 4]
    T* __restrict__ rho_bar_work,             // [Nb] in/out
    volatile int* changed_flag,               // mapped memory: 0 = no change,
                                              //                1 = at least one bid updated rho
    T max_factor, T rho_min, T rho_max, T tolerance,
    int Nb, const uint32_t* __restrict__ status)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= Nb) return;
    if (status[bid] != static_cast<uint32_t>(ADMM_STATUS_NOT_RUN)) return;

    T p_res  = residuals[bid * 4 + 0];
    T d_res  = residuals[bid * 4 + 1];
    T p_norm = residuals[bid * 4 + 2];
    T d_norm = residuals[bid * 4 + 3];

    // Normalised residuals, matching the fused-PCG daemon
    // (admm_fused.cu:513-514): scalars[13/14] use a 1e-10 floor, then
    // computeRhoCandidateDevice computes ratio = sqrt(p_norm / d_norm).
    T p_normalized = p_res / (T(1e-10) + p_norm);
    T d_normalized = d_res / (T(1e-10) + d_norm);

    T rho_old = rho_bar_work[bid];
    T rho_new;
    bool should_update = computeRhoCandidateDevice<T>(
        p_normalized, d_normalized, rho_old,
        max_factor, rho_min, rho_max, tolerance, &rho_new);

    if (should_update) {
        rho_bar_work[bid] = rho_new;
        // Any batch element wanting an update flips the host flag on.
        // Mapped int with __threadfence_system() makes the write visible
        // to the CPU after a stream sync.
        *changed_flag = 1;
    }
}

// Per-batch Schur S rebuild when rho changes. Launched <<<Nb, threads, smem>>>;
// smem layout is dtilde_sh[n*n] then gauss_work[n*2n] = 3*n*n elements of T.
template<typename T>
__global__ void recomputeSKernel(
    T* __restrict__ S_work,                  // [Nb, T, n, 3n] writable
    const T* __restrict__ D_g,               // [Nb, T, n, n]
    const T* __restrict__ E_g,               // [Nb, N, n, n]
    const T* __restrict__ A0_g,              // [Nb, n0, n]
    const T* __restrict__ Am_g,              // [Nb, N, nx, n]
    const T* __restrict__ Ap_g,              // [Nb, N, nx, n]
    const T* __restrict__ G_g,               // [Nb, T, m, n]
    T* __restrict__ theta_inv_work,          // [Nb, T*n*n]
    const T* __restrict__ rho_bar_work,      // [Nb]
    T rho_f_factor, T sigma,
    int T_blocks, int n, int nx, int n0, int m)
{
    extern __shared__ char smem_raw[];
    T* dtilde_sh  = reinterpret_cast<T*>(smem_raw);
    T* gauss_work = dtilde_sh + n * n;

    int bid = blockIdx.x;
    int N = T_blocks - 1;
    int n3 = 3 * n;

    // Per-batch slices.
    T*       S_b         = S_work + (size_t)bid * T_blocks * n * n3;
    const T* D_b         = D_g    + (size_t)bid * T_blocks * n * n;
    const T* E_b         = E_g    + (size_t)bid * N        * n * n;
    const T* A0_b        = A0_g   + (size_t)bid * n0       * n;
    const T* Am_b        = Am_g   + (size_t)bid * N        * nx * n;
    const T* Ap_b        = Ap_g   + (size_t)bid * N        * nx * n;
    const T* G_b         = G_g    + (size_t)bid * T_blocks * m * n;
    T*       theta_inv_b = theta_inv_work + (size_t)bid * T_blocks * n * n;

    T rho_bar = rho_bar_work[bid];
    T rho_f   = rho_bar * rho_f_factor;

    recomputeSDevice<T>(
        S_b, D_b, E_b, A0_b, Am_b, Ap_b, G_b,
        theta_inv_b, dtilde_sh, gauss_work,
        rho_bar, rho_f, sigma,
        T_blocks, n, nx, n0, m,
        /*want_theta_inv=*/false);
}

// ---- Kernel 5: copy outputs ----
template<typename T>
__global__ void copyOutputsKernel(
    uint32_t* __restrict__ iters_out,
    T*        __restrict__ x_blocks_out,
    T*        __restrict__ z_g_out,
    T*        __restrict__ y_g_out,
    T*        __restrict__ y_f_0_out,
    T*        __restrict__ y_f_dyn_out,
    T*        __restrict__ xi_g_out,
    T*        __restrict__ rho_bar_out,
    const T* __restrict__ x,
    const T* __restrict__ z_g,
    const T* __restrict__ y_g,
    const T* __restrict__ y_f_0,
    const T* __restrict__ y_f_dyn,
    const T* __restrict__ xi_g,
    const T* __restrict__ rho_bar_in,
    const uint32_t* __restrict__ iter_counts,
    int total_tn, int total_Tm, int nx_val, int total_Nnx)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < total_tn; i += blockDim.x) {
        x_blocks_out[bid * total_tn + i] = x[bid * total_tn + i];
    }
    for (int i = tid; i < total_Tm; i += blockDim.x)
        z_g_out[bid * total_Tm + i] = z_g[bid * total_Tm + i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        y_g_out[bid * total_Tm + i] = y_g[bid * total_Tm + i];
    for (int i = tid; i < nx_val; i += blockDim.x)
        y_f_0_out[bid * nx_val + i] = y_f_0[bid * nx_val + i];
    for (int i = tid; i < total_Nnx; i += blockDim.x)
        y_f_dyn_out[bid * total_Nnx + i] = y_f_dyn[bid * total_Nnx + i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        xi_g_out[bid * total_Tm + i] = xi_g[bid * total_Tm + i];
    if (tid == 0) {
        rho_bar_out[bid] = rho_bar_in[bid];
        iters_out[bid] = iter_counts[bid];
    }
}


// ============================================================
// Python-owned cuDSS workspace.
// ============================================================

struct AdmmCudssWorkspace {
    turbompc::FfiDeviceAllocator cudss_allocator{};
    int32_t Nb = 0, T = 0, n = 0, nx = 0, n0 = 0, m = 0;
    int32_t dtype_bytes = 0;
    bool initialized = false;
    // cuDSS objects (persistent)
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t matA = nullptr;
    cudssMatrix_t matX = nullptr;
    cudssMatrix_t matB = nullptr;

    // CSR structure buffers (persistent)
    int32_t* csr_indptr = nullptr;
    int32_t* csr_indices = nullptr;
    void* csr_data = nullptr;

    // Scratch buffers (persistent — reused across calls)
    void* d_x = nullptr;
    void* d_z_g = nullptr;
    void* d_y_g = nullptr;
    void* d_y_f_0 = nullptr;
    void* d_y_f_dyn = nullptr;
    void* d_xi_g = nullptr;
    void* d_gamma = nullptr;
    void* d_x_old = nullptr;
    void* d_Gx = nullptr;
    void* d_Cx0 = nullptr;
    void* d_Cx_dyn = nullptr;
    void* d_z_g_old = nullptr;
    void* d_residuals = nullptr;
    uint32_t* d_finite_flags = nullptr;
    void* d_infeasibility_metrics = nullptr;
    void* d_previous_y_g = nullptr;
    void* d_previous_y_f_0 = nullptr;
    void* d_previous_y_f_dyn = nullptr;
    void* d_previous_xi_g = nullptr;
    uint32_t* d_iter_counts = nullptr;

    // Adaptive-rho workspaces (mutated in-loop; initialised from inputs each call).
    void* d_S_work = nullptr;          // [Nb, T, n, 3n]
    void* d_rho_bar_work = nullptr;    // [Nb]
    void* d_theta_inv_work = nullptr;  // [Nb, T*n*n]

    // Mapped memory for zero-sync convergence signaling
    volatile int* h_converged_flag = nullptr;
    int* d_converged_flag = nullptr;

    // Mapped memory for the adaptive-rho "any batch updated rho" flag
    volatile int* h_rho_changed_flag = nullptr;
    int* d_rho_changed_flag = nullptr;

    bool analysis_done = false;
    bool structure_written = false;
    cudaEvent_t completion_event = nullptr;

    std::mutex entry_mutex;  // per-entry lock for exclusive access

    AdmmCudssWorkspace() = default;

    AdmmCudssWorkspace(const AdmmCudssWorkspace&) = delete;
    AdmmCudssWorkspace& operator=(const AdmmCudssWorkspace&) = delete;
    AdmmCudssWorkspace& operator=(AdmmCudssWorkspace&&) = delete;

    void release_capacity(turbompc::FfiDeviceAllocator& allocator) {
        if (completion_event) cudaEventSynchronize(completion_event);
        cudss_allocator = allocator;
        if (matA) cudssMatrixDestroy(matA);
        if (matX) cudssMatrixDestroy(matX);
        if (matB) cudssMatrixDestroy(matB);
        if (data && handle) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        const size_t scalar_size = static_cast<size_t>(dtype_bytes);
        const size_t total_tn = static_cast<size_t>(T) * n;
        const size_t total_Tm = static_cast<size_t>(T) * m;
        const size_t total_Nnx = static_cast<size_t>(T - 1) * nx;
        const size_t total = static_cast<size_t>(Nb) * total_tn;
        const size_t nnz_per_sys = T == 1
            ? static_cast<size_t>(n) * n
            : static_cast<size_t>(3 * T - 2) * n * n;
        const size_t nnz = static_cast<size_t>(Nb) * nnz_per_sys;
        auto free = [&](void* pointer, size_t bytes) {
            if (pointer) turbompc::FfiFree(allocator, pointer, bytes);
        };
        free(csr_indptr, (total + 1) * sizeof(int32_t));
        free(csr_indices, nnz * sizeof(int32_t));
        free(csr_data, nnz * scalar_size);
        free(d_x, static_cast<size_t>(Nb) * total_tn * scalar_size);
        free(d_z_g, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_y_g, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_y_f_0, static_cast<size_t>(Nb) * n0 * scalar_size);
        free(d_y_f_dyn, static_cast<size_t>(Nb) * total_Nnx * scalar_size);
        free(d_xi_g, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_gamma, static_cast<size_t>(Nb) * total_tn * scalar_size);
        free(d_x_old, static_cast<size_t>(Nb) * total_tn * scalar_size);
        free(d_Gx, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_Cx0, static_cast<size_t>(Nb) * n0 * scalar_size);
        free(d_Cx_dyn, static_cast<size_t>(Nb) * total_Nnx * scalar_size);
        free(d_z_g_old, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_residuals, static_cast<size_t>(Nb) * 4 * scalar_size);
        free(d_finite_flags, static_cast<size_t>(Nb) * sizeof(uint32_t));
        free(d_infeasibility_metrics, static_cast<size_t>(Nb) * 7 * scalar_size);
        free(d_previous_y_g, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_previous_y_f_0, static_cast<size_t>(Nb) * n0 * scalar_size);
        free(d_previous_y_f_dyn, static_cast<size_t>(Nb) * total_Nnx * scalar_size);
        free(d_previous_xi_g, static_cast<size_t>(Nb) * total_Tm * scalar_size);
        free(d_iter_counts, static_cast<size_t>(Nb) * sizeof(uint32_t));
        free(d_S_work, static_cast<size_t>(Nb) * T * n * 3 * n * scalar_size);
        free(d_rho_bar_work, static_cast<size_t>(Nb) * scalar_size);
        free(d_theta_inv_work, static_cast<size_t>(Nb) * T * n * n * scalar_size);

        // Mapped host memory (h_* and d_* point to the same underlying region;
        // free once via cudaFreeHost on the host pointer).
        if (h_converged_flag)   cudaFreeHost((void*)h_converged_flag);
        if (h_rho_changed_flag) cudaFreeHost((void*)h_rho_changed_flag);

        handle = nullptr;
        config = nullptr;
        data = nullptr;
        matA = nullptr;
        matX = nullptr;
        matB = nullptr;
        csr_indptr = nullptr;
        csr_indices = nullptr;
        csr_data = nullptr;
        d_x = d_z_g = d_y_g = d_y_f_0 = d_y_f_dyn = nullptr;
        d_xi_g = d_gamma = d_x_old = d_Gx = d_Cx0 = d_Cx_dyn = nullptr;
        d_z_g_old = d_residuals = d_infeasibility_metrics = nullptr;
        d_previous_y_g = d_previous_y_f_0 = d_previous_y_f_dyn = nullptr;
        d_previous_xi_g = d_S_work = d_rho_bar_work = d_theta_inv_work = nullptr;
        d_finite_flags = d_iter_counts = nullptr;
        h_converged_flag = h_rho_changed_flag = nullptr;
        d_converged_flag = d_rho_changed_flag = nullptr;
        initialized = false;
        analysis_done = false;
        structure_written = false;
        Nb = T = n = nx = n0 = m = dtype_bytes = 0;
    }

    ~AdmmCudssWorkspace() noexcept = default;
};

static std::mutex g_workspace_registry_mutex;
static std::unordered_map<uint64_t, std::unique_ptr<AdmmCudssWorkspace>> g_workspace_registry;
static uint64_t g_next_workspace_handle = 1;
static thread_local std::string g_last_error;

uint64_t CreateAdmmCudssWorkspace() {
    try {
        auto workspace = std::make_unique<AdmmCudssWorkspace>();
        std::lock_guard<std::mutex> lock(g_workspace_registry_mutex);
        const uint64_t handle = g_next_workspace_handle++;
        g_workspace_registry.emplace(handle, std::move(workspace));
        return handle;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 0;
    }
}

int DestroyAdmmCudssWorkspace(turbompc::FfiDeviceAllocator& allocator, uint64_t handle) {
    try {
        std::unique_ptr<AdmmCudssWorkspace> workspace;
        std::unique_lock<std::mutex> registry_lock(g_workspace_registry_mutex);
        auto position = g_workspace_registry.find(handle);
        if (position == g_workspace_registry.end())
            throw std::invalid_argument("ADMM cuDSS workspace handle is not live");
        std::unique_lock<std::mutex> execution_lock(position->second->entry_mutex);
        workspace = std::move(position->second);
        g_workspace_registry.erase(position);
        registry_lock.unlock();
        execution_lock.unlock();
        workspace->release_capacity(allocator);
        if (workspace->completion_event) cudaEventDestroy(workspace->completion_event);
        workspace.reset();
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    }
}

const char* AdmmCudssLastError() {
    return g_last_error.c_str();
}

template<typename T>
static void ensureWorkspaceCapacity(turbompc::FfiDeviceAllocator& allocator,
                                    AdmmCudssWorkspace* entry, int32_t Nb,
                                    int32_t T_blocks, int32_t n, int32_t nx,
                                    int32_t n0, int32_t m) {
    if (Nb < 1 || T_blocks < 1 || n < 1 || nx < 1 || n0 < 0 || m < 0)
        throw std::invalid_argument("invalid ADMM cuDSS workspace dimensions");

    if (entry->initialized && entry->Nb == Nb && entry->T == T_blocks &&
        entry->n == n && entry->nx == nx && entry->n0 == n0 &&
        entry->m == m && entry->dtype_bytes == sizeof(T))
        return;

    entry->release_capacity(allocator);
    if (!entry->completion_event) {
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&entry->completion_event, cudaEventDisableTiming));
        CUDA_CHECK_THROW(cudaEventRecord(entry->completion_event, nullptr));
    }
    entry->Nb = Nb;
    entry->T = T_blocks;
    entry->n = n;
    entry->nx = nx;
    entry->n0 = n0;
    entry->m = m;
    entry->dtype_bytes = sizeof(T);

    const int N = T_blocks - 1;
    const int total_tn = T_blocks * n;
    const int total_Tm = T_blocks * m;
    const int total_Nnx = N * nx;
    const int total = Nb * total_tn;
    const size_t nnz_per_sys = (T_blocks == 1)
        ? (size_t)n * n
        : (size_t)(3 * T_blocks - 2) * n * n;
    const size_t nnz = (size_t)Nb * nnz_per_sys;
    const cudssDataType_t valueType = std::is_same_v<T, float>
        ? CUDSS_R_32F : CUDSS_R_64F;

    turbompc::FfiAllocate(allocator, &entry->d_x, (size_t)Nb * total_tn * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_z_g, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_y_g, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_y_f_0, (size_t)Nb * n0 * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_y_f_dyn, (size_t)Nb * total_Nnx * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_xi_g, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_gamma, (size_t)Nb * total_tn * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_x_old, (size_t)Nb * total_tn * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_Gx, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_Cx0, (size_t)Nb * n0 * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_Cx_dyn, (size_t)Nb * total_Nnx * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_z_g_old, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_residuals, (size_t)Nb * 4 * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_finite_flags, (size_t)Nb * sizeof(uint32_t));
    turbompc::FfiAllocate(allocator, &entry->d_infeasibility_metrics, (size_t)Nb * 7 * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_previous_y_g, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_previous_y_f_0, (size_t)Nb * n0 * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_previous_y_f_dyn, (size_t)Nb * total_Nnx * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_previous_xi_g, (size_t)Nb * total_Tm * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_iter_counts, (size_t)Nb * sizeof(uint32_t));
    turbompc::FfiAllocate(allocator, &entry->d_S_work, (size_t)Nb * T_blocks * n * 3 * n * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_rho_bar_work, (size_t)Nb * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->d_theta_inv_work, (size_t)Nb * T_blocks * n * n * sizeof(T));
    turbompc::FfiAllocate(allocator, &entry->csr_indptr, (total + 1) * sizeof(int32_t));
    turbompc::FfiAllocate(allocator, &entry->csr_indices, nnz * sizeof(int32_t));
    turbompc::FfiAllocate(allocator, &entry->csr_data, nnz * sizeof(T));

    CUDA_CHECK_THROW(cudaHostAlloc((void**)&entry->h_converged_flag, sizeof(int),
                                   cudaHostAllocMapped));
    CUDA_CHECK_THROW(cudaHostGetDevicePointer(&entry->d_converged_flag,
                                               (int*)entry->h_converged_flag, 0));
    CUDA_CHECK_THROW(cudaHostAlloc((void**)&entry->h_rho_changed_flag, sizeof(int),
                                   cudaHostAllocMapped));
    CUDA_CHECK_THROW(cudaHostGetDevicePointer(&entry->d_rho_changed_flag,
                                               (int*)entry->h_rho_changed_flag, 0));

    CUDSS_CHECK(cudssCreate(&entry->handle));
    entry->cudss_allocator = allocator;
    CUDSS_CHECK(turbompc::SetCudssDeviceMemory(entry->handle,
                                               entry->cudss_allocator));
    CUDSS_CHECK(cudssConfigCreate(&entry->config));
    CUDSS_CHECK(cudssDataCreate(entry->handle, &entry->data));
    cudssReorderingAlg_t reorder_alg = CUDSS_REORDERING_ALG_DEFAULT;
    CUDSS_CHECK(cudssConfigSet(entry->config, CUDSS_CONFIG_REORDERING_ALG,
                               &reorder_alg, sizeof(reorder_alg)));
    CUDSS_CHECK(cudssMatrixCreateCsr(
        &entry->matA, total, total, nnz, entry->csr_indptr, nullptr,
        entry->csr_indices, (T*)entry->csr_data, CUDSS_R_32I, CUDSS_R_32I,
        valueType, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&entry->matX, total, 1, total,
                                    (T*)entry->d_x, valueType,
                                    CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&entry->matB, total, 1, total,
                                    (T*)entry->d_gamma, valueType,
                                    CUDSS_LAYOUT_COL_MAJOR));
    entry->initialized = true;
}

int EnsureAdmmCudssWorkspaceCapacity(
    turbompc::FfiDeviceAllocator& allocator, uint64_t workspace_handle, int32_t Nb, int32_t T, int32_t n,
    int32_t nx, int32_t n0, int32_t m, int32_t dtype_bytes) {
    try {
        std::unique_lock<std::mutex> registry_lock(g_workspace_registry_mutex);
        auto position = g_workspace_registry.find(workspace_handle);
        if (position == g_workspace_registry.end())
            throw std::invalid_argument("ADMM cuDSS workspace handle is not live");
        AdmmCudssWorkspace* entry = position->second.get();
        std::unique_lock<std::mutex> entry_lock(entry->entry_mutex);
        registry_lock.unlock();
        if (dtype_bytes == 4)
            ensureWorkspaceCapacity<float>(allocator, entry, Nb, T, n, nx, n0, m);
        else if (dtype_bytes == 8)
            ensureWorkspaceCapacity<double>(allocator, entry, Nb, T, n, nx, n0, m);
        else
            throw std::invalid_argument("workspace dtype must be 4 or 8 bytes");
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    }
}

template<typename T>
static void launchADMMCudss(
    turbompc::FfiDeviceAllocator& allocator, uint64_t workspace_handle,
    cudaStream_t stream,
    uint32_t* iters_out,
    T* x_blocks_out, T* z_g_out, T* y_g_out,
    T* y_f_0_out, T* y_f_dyn_out, T* xi_g_out, T* rho_bar_out,
    T* kernel_ns_out, uint32_t* status_out, T* terminal_residuals_out,
    const T* S, const T* D, const T* E, const T* q,
    const T* A0, const T* A_minus, const T* A_plus, const T* G,
    const T* l_bounds, const T* u_bounds, const T* c0, const T* c_dyn,
    const T* x0, const T* z_g0, const T* y_g0,
    const T* y_f_0_init, const T* y_f_dyn_init, const T* xi_g0,
    const T* rho_bar_init,
    const T* slack_weight_init,
    int32_t T_blocks, int32_t n, int32_t nx, int32_t n0, int32_t m, int32_t Nb,
    ADMMCudssConfig cfg)
{
    int N = T_blocks - 1;
    int total_tn  = T_blocks * n;
    int total_Tm  = T_blocks * m;
    int total_Nnx = N * nx;

    // Thread count for kernels
    int max_elem = total_tn;
    if (total_Tm > max_elem) max_elem = total_Tm;
    if (total_Nnx > max_elem) max_elem = total_Nnx;
    int threads = ((max_elem + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    // Cap at 768 to stay within the 65536 register file limit.
    // All loops use strided patterns (i += blockDim.x) so fewer
    // threads than elements is correct.
    if (threads > 768) threads = 768;

    std::unique_lock<std::mutex> registry_lock(g_workspace_registry_mutex);
    auto position = g_workspace_registry.find(workspace_handle);
    if (position == g_workspace_registry.end())
    throw std::invalid_argument("ADMM cuDSS workspace handle is not live");
    AdmmCudssWorkspace* entry = position->second.get();
    std::unique_lock<std::mutex> entry_lock(entry->entry_mutex);
    registry_lock.unlock();
    ensureWorkspaceCapacity<T>(allocator, entry, Nb, T_blocks, n, nx, n0, m);
    CUDA_CHECK_THROW(cudaStreamWaitEvent(stream, entry->completion_event, 0));

    // Update stream (may change between calls)
    entry->cudss_allocator = allocator;
    CUDSS_CHECK(cudssSetStream(entry->handle, stream));

    // Cast workspace-owned void* to typed pointers
    T* d_x        = (T*)entry->d_x;
    T* d_z_g      = (T*)entry->d_z_g;
    T* d_y_g      = (T*)entry->d_y_g;
    T* d_y_f_0    = (T*)entry->d_y_f_0;
    T* d_y_f_dyn  = (T*)entry->d_y_f_dyn;
    T* d_xi_g     = (T*)entry->d_xi_g;
    T* d_gamma    = (T*)entry->d_gamma;
    T* d_x_old    = (T*)entry->d_x_old;
    T* d_Gx       = (T*)entry->d_Gx;
    T* d_Cx0      = (T*)entry->d_Cx0;
    T* d_Cx_dyn   = (T*)entry->d_Cx_dyn;
    T* d_z_g_old  = (T*)entry->d_z_g_old;
    T* d_residuals = (T*)entry->d_residuals;
    uint32_t* d_finite_flags = entry->d_finite_flags;
    T* d_infeasibility_metrics = (T*)entry->d_infeasibility_metrics;
    T* d_previous_y_g = (T*)entry->d_previous_y_g;
    T* d_previous_y_f_0 = (T*)entry->d_previous_y_f_0;
    T* d_previous_y_f_dyn = (T*)entry->d_previous_y_f_dyn;
    T* d_previous_xi_g = (T*)entry->d_previous_xi_g;
    uint32_t* d_iter_counts = entry->d_iter_counts;
    T* d_S_work       = (T*)entry->d_S_work;
    T* d_rho_bar_work = (T*)entry->d_rho_bar_work;

    // ---- Stage inputs into writable workspaces ----
    // S → d_S_work, rho_bar_init → d_rho_bar_work. The body kernels and CSR
    // conversion all operate on the workspaces from here on, so the adapt-rho
    // refactor (Phase C) only has to mutate the workspaces.
    cudaMemcpyAsync(d_S_work, S,
                    (size_t)Nb * T_blocks * n * 3 * n * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_rho_bar_work, rho_bar_init,
                    (size_t)Nb * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    // ---- Init state kernel ----
    initStateKernel<T><<<Nb, threads, 0, stream>>>(
        d_x, d_z_g, d_y_g, d_y_f_0, d_y_f_dyn, d_xi_g,
        x0, z_g0, y_g0, y_f_0_init, y_f_dyn_init, xi_g0,
        total_tn, total_Tm, n0, total_Nnx,
        status_out, d_iter_counts, cfg.max_iter);

    // ---- CSR assembly: full on first call, values-only on subsequent ----
    if (!entry->structure_written) {
        if (Nb == 1) {
            if constexpr (std::is_same_v<T, float>) {
                blkTridiToCSR(d_S_work, entry->csr_indptr, entry->csr_indices,
                              (float*)entry->csr_data, T_blocks, n, stream);
            } else {
                blkTridiToCSR_f64(d_S_work, entry->csr_indptr, entry->csr_indices,
                                  (double*)entry->csr_data, T_blocks, n, stream);
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                batchedBlkTridiToCSR(d_S_work, entry->csr_indptr, entry->csr_indices,
                                     (float*)entry->csr_data, Nb, T_blocks, n, stream);
            } else {
                batchedBlkTridiToCSR_f64(d_S_work, entry->csr_indptr, entry->csr_indices,
                                          (double*)entry->csr_data, Nb, T_blocks, n, stream);
            }
        }
        entry->structure_written = true;
    } else {
        // Values-only update — CSR structure (indptr/indices) is unchanged
        if (Nb == 1) {
            if constexpr (std::is_same_v<T, float>) {
                blkTridiToCSR_data_only(d_S_work, (float*)entry->csr_data, T_blocks, n, stream);
            } else {
                blkTridiToCSR_data_only_f64(d_S_work, (double*)entry->csr_data, T_blocks, n, stream);
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                batchedBlkTridiToCSR_data_only(d_S_work, (float*)entry->csr_data, Nb, T_blocks, n, stream);
            } else {
                batchedBlkTridiToCSR_data_only_f64(d_S_work, (double*)entry->csr_data, Nb, T_blocks, n, stream);
            }
        }
    }

  // ---- cuDSS: analysis (reused) + factorization (per-call, S changes between SQP iters) ----
    if (!entry->analysis_done) {
        CUDSS_CHECK(cudssExecute(entry->handle, CUDSS_PHASE_ANALYSIS, entry->config,
                                 entry->data, entry->matA, entry->matX, entry->matB));
        entry->analysis_done = true;
    }
    CUDSS_CHECK(cudssExecute(entry->handle, CUDSS_PHASE_FACTORIZATION, entry->config,
                             entry->data, entry->matA, entry->matX, entry->matB));

    // ---- Reset convergence flag (host-side, visible to host immediately) ----
    *entry->h_converged_flag = -1;

    T sigma_k       = static_cast<T>(cfg.sigma);
    T alpha_k       = static_cast<T>(cfg.alpha);
    T eps_abs_k     = static_cast<T>(cfg.eps_abs);
    T eps_rel_k     = static_cast<T>(cfg.eps_rel);
    T eps_prim_inf_k = static_cast<T>(cfg.eps_prim_inf);
    T eps_dual_inf_k = static_cast<T>(cfg.eps_dual_inf);
    T rho_f_factor  = static_cast<T>(cfg.rho_f_factor);
    T rho_min_k     = static_cast<T>(cfg.rho_min);
    T rho_max_k     = static_cast<T>(cfg.rho_max);
    T rho_tol_k     = static_cast<T>(cfg.adaptive_rho_tolerance);
    T rho_factor_k  = static_cast<T>(cfg.rho_update_max_factor);
    T log_barrier_kappa_k = static_cast<T>(cfg.log_barrier_kappa);

    const bool log_barrier = cfg.constraint_handling == ConstraintHandling::LogBarrier;
    // Both instantiations share one signature, so the selection is a pointer.
    auto postSolve = log_barrier
        ? postSolveKernel<T, ConstraintHandling::LogBarrier>
        : postSolveKernel<T, ConstraintHandling::Projection>;

    for (int it = 0; it < cfg.max_iter; ++it) {
        bool termination_check_fires =
            (cfg.check_termination_every > 0
             && ((it + 1) % cfg.check_termination_every == 0))
            || it + 1 == cfg.max_iter;
        // Hard-QP infeasibility certificates only apply to projection; a
        // barrier solve reports SOLVED, MAX_ITER, or NONFINITE.
        bool infeasibility_check_fires = !log_barrier
            && ((cfg.check_infeasibility_every > 0
                 && ((it + 1) % cfg.check_infeasibility_every == 0))
                || it + 1 == cfg.max_iter);
        bool rho_check_fires = cfg.adapt_rho_every > 0
            && ((it + 1) % cfg.adapt_rho_every == 0);
        if (infeasibility_check_fires) {
            snapshotCertificateDualsKernel<T><<<Nb, threads, 0, stream>>>(
                d_previous_y_g, d_previous_y_f_0, d_previous_y_f_dyn,
                d_previous_xi_g, d_y_g, d_y_f_0, d_y_f_dyn, d_xi_g,
                total_Tm, n0, total_Nnx);
        }
        // 1. Compute gamma (RHS) and save x_old
        computeGammaAndSaveXOldKernel<T><<<Nb, threads, 0, stream>>>(
            d_gamma, d_x_old, d_x, d_z_g, d_y_g, d_y_f_0, d_y_f_dyn,
            q, A0, A_minus, A_plus, G, c0, c_dyn,
            sigma_k, rho_f_factor, d_rho_bar_work,
            T_blocks, n, nx, n0, m, status_out);

        // 2. cuDSS solve-only: S @ x_new = gamma
        //    matX points directly to d_x — solution lands in d_x, no D→D memcpy
        CUDSS_CHECK(cudssMatrixSetValues(entry->matB, (void*)d_gamma));
        CUDSS_CHECK(cudssExecute(entry->handle, CUDSS_PHASE_SOLVE, entry->config,
                                 entry->data, entry->matA, entry->matX, entry->matB));

        // 3. Post-solve: constraints + over-relax + inequality proximal + dual
        postSolve<<<Nb, threads, 0, stream>>>(
            d_x, d_z_g, d_y_g, d_y_f_0, d_y_f_dyn,
            d_Gx, d_Cx0, d_Cx_dyn, d_z_g_old, d_x_old,
            A0, A_minus, A_plus, G, l_bounds, u_bounds, c0, c_dyn,
            alpha_k, rho_f_factor, d_rho_bar_work,
            T_blocks, n, nx, n0, m,
            slack_weight_init, cfg.use_slack, log_barrier_kappa_k, d_xi_g, status_out);

        // 4. Residual, status, infeasibility, and rho checks.
        if (termination_check_fires || infeasibility_check_fires
            || rho_check_fires) {
            size_t res_smem = ((threads + 31) / 32) * sizeof(T);
            computeResidualsKernel<T><<<Nb, threads, res_smem, stream>>>(
                d_residuals, d_finite_flags,
                d_x, d_z_g, d_y_g, d_y_f_0, d_y_f_dyn,
                D, E, q, A0, A_minus, A_plus, G, c0, c_dyn,
                T_blocks, n, nx, n0, m,
                d_xi_g, d_rho_bar_work, slack_weight_init, cfg.use_slack);
            if (infeasibility_check_fires) {
                size_t certificate_smem = 7 * ((threads + 31) / 32) * sizeof(T);
                computeInfeasibilityMetricsKernel<T><<<
                    Nb, threads, certificate_smem, stream>>>(
                    d_infeasibility_metrics,
                    d_x, d_y_g, d_y_f_0, d_y_f_dyn, d_xi_g,
                    d_x_old, d_previous_y_g, d_previous_y_f_0,
                    d_previous_y_f_dyn, d_previous_xi_g,
                    D, E, q, A0, A_minus, A_plus, G,
                    l_bounds, u_bounds, c0, c_dyn,
                    slack_weight_init, cfg.use_slack,
                    T_blocks, n, nx, n0, m);
            }
            if (termination_check_fires || infeasibility_check_fires) {
                checkConvergenceSignalKernel<T><<<1, 1, 0, stream>>>(
                    d_residuals, d_infeasibility_metrics, d_finite_flags,
                    status_out, d_iter_counts, terminal_residuals_out,
                    entry->d_converged_flag, eps_abs_k, eps_rel_k,
                    eps_prim_inf_k, eps_dual_inf_k,
                    Nb, it + 1, termination_check_fires,
                    infeasibility_check_fires, it + 1 == cfg.max_iter);
                cudaStreamSynchronize(stream);
                if (*entry->h_converged_flag >= 0) {
                    break;
                }
            }

            // Update rho after any requested convergence check has completed.
            // This may mutate d_rho_bar_work and trigger refactorization.
            if (rho_check_fires) {
                // Reset the rho-changed flag, launch the per-batch update.
                *entry->h_rho_changed_flag = 0;
                int rho_threads = 32;
                int rho_blocks  = (Nb + rho_threads - 1) / rho_threads;
                rhoUpdateKernel<T><<<rho_blocks, rho_threads, 0, stream>>>(
                    d_residuals, d_rho_bar_work, entry->d_rho_changed_flag,
                    rho_factor_k, rho_min_k, rho_max_k, rho_tol_k, Nb, status_out);
                cudaStreamSynchronize(stream);
            }

            if (rho_check_fires && *entry->h_rho_changed_flag != 0) {
                // Some batch's rho moved beyond tolerance: rebuild d_S_work,
                // refresh the CSR data buffer, and re-factorise via cuDSS.
                // want_theta_inv=false writes S straight to global d_S_work
                // with no shared scratch -> zero dynamic shared memory.
                recomputeSKernel<T><<<Nb, threads, 0, stream>>>(
                    d_S_work, D, E, A0, A_minus, A_plus, G,
                    (T*)entry->d_theta_inv_work, d_rho_bar_work,
                    rho_f_factor, sigma_k,
                    T_blocks, n, nx, n0, m);
                // A failed launch (e.g. dynamic shared memory > device
                // limit) must NOT silently leave d_S_work stale and let the
                // solve "converge" to the rho-independent QP minimum.
                CUDA_CHECK_THROW(cudaGetLastError());

                if (Nb == 1) {
                    if constexpr (std::is_same_v<T, float>) {
                        blkTridiToCSR_data_only(d_S_work, (float*)entry->csr_data,
                                                T_blocks, n, stream);
                    } else {
                        blkTridiToCSR_data_only_f64(d_S_work, (double*)entry->csr_data,
                                                    T_blocks, n, stream);
                    }
                } else {
                    if constexpr (std::is_same_v<T, float>) {
                        batchedBlkTridiToCSR_data_only(d_S_work, (float*)entry->csr_data,
                                                       Nb, T_blocks, n, stream);
                    } else {
                        batchedBlkTridiToCSR_data_only_f64(d_S_work, (double*)entry->csr_data,
                                                           Nb, T_blocks, n, stream);
                    }
                }

                CUDSS_CHECK(cudssExecute(entry->handle, CUDSS_PHASE_FACTORIZATION, entry->config,
                                         entry->data, entry->matA, entry->matX, entry->matB));
            }
        }
    }

    // ---- Copy outputs ----
    // rho_bar_out matches each row's final ADMM iterate.
    copyOutputsKernel<T><<<Nb, threads, 0, stream>>>(
        iters_out, x_blocks_out, z_g_out, y_g_out,
        y_f_0_out, y_f_dyn_out, xi_g_out, rho_bar_out,
        d_x, d_z_g, d_y_g, d_y_f_0, d_y_f_dyn, d_xi_g,
        d_rho_bar_work, d_iter_counts,
        total_tn, total_Tm, n0, total_Nnx);

    // Timing: write zero — kernel-level timing removed to avoid cudaEventSynchronize
    // which would drain the GPU pipeline. Wall-clock timing at the Python level is
    // the correct way to measure; this field existed only for debugging.
    {
        T zero_ns = T(0);
        std::vector<T> h_ns(Nb, zero_ns);
        cudaMemcpyAsync(kernel_ns_out, h_ns.data(), Nb * sizeof(T),
                        cudaMemcpyHostToDevice, stream);
    }
    CUDA_CHECK_THROW(cudaEventRecord(entry->completion_event, stream));
}

// ---- Explicit instantiations ----

void LaunchADMMCudssF32(
    turbompc::FfiDeviceAllocator& allocator,
    uint64_t workspace_handle,
    cudaStream_t stream,
    uint32_t* iters_out,
    float* x_blocks_out, float* z_g_out, float* y_g_out,
    float* y_f_0_out, float* y_f_dyn_out, float* xi_g_out, float* rho_bar_out,
    float* kernel_ns_out, uint32_t* status_out, float* terminal_residuals_out,
    const float* S, const float* D, const float* E, const float* q,
    const float* A0, const float* A_minus, const float* A_plus, const float* G,
    const float* l_bounds, const float* u_bounds, const float* c0, const float* c_dyn,
    const float* x0, const float* z_g0, const float* y_g0,
    const float* y_f_0_init, const float* y_f_dyn_init, const float* xi_g0,
    const float* rho_bar_init,
    const float* slack_weight_init,
    int32_t T, int32_t n, int32_t nx, int32_t n0, int32_t m, int32_t Nb,
    ADMMCudssConfig cfg)
{
  launchADMMCudss<float>(
      allocator, workspace_handle, stream, iters_out, x_blocks_out, z_g_out, y_g_out,
      y_f_0_out, y_f_dyn_out, xi_g_out, rho_bar_out, kernel_ns_out, status_out,
      terminal_residuals_out, S, D, E, q, A0, A_minus, A_plus, G, l_bounds,
      u_bounds, c0, c_dyn, x0, z_g0, y_g0, y_f_0_init, y_f_dyn_init, xi_g0,
      rho_bar_init, slack_weight_init, T, n, nx, n0, m, Nb, cfg);
}

void LaunchADMMCudssF64(
    turbompc::FfiDeviceAllocator& allocator,
    uint64_t workspace_handle,
    cudaStream_t stream,
    uint32_t* iters_out,
    double* x_blocks_out, double* z_g_out, double* y_g_out,
    double* y_f_0_out, double* y_f_dyn_out, double* xi_g_out, double* rho_bar_out,
    double* kernel_ns_out, uint32_t* status_out, double* terminal_residuals_out,
    const double* S, const double* D, const double* E, const double* q,
    const double* A0, const double* A_minus, const double* A_plus, const double* G,
    const double* l_bounds, const double* u_bounds, const double* c0, const double* c_dyn,
    const double* x0, const double* z_g0, const double* y_g0,
    const double* y_f_0_init, const double* y_f_dyn_init, const double* xi_g0,
    const double* rho_bar_init,
    const double* slack_weight_init,
    int32_t T, int32_t n, int32_t nx, int32_t n0, int32_t m, int32_t Nb,
    ADMMCudssConfig cfg)
{
    launchADMMCudss<double>(allocator, workspace_handle, stream,
        iters_out,
        x_blocks_out, z_g_out, y_g_out, y_f_0_out, y_f_dyn_out, xi_g_out, rho_bar_out,
        kernel_ns_out, status_out, terminal_residuals_out,
        S, D, E, q,
        A0, A_minus, A_plus, G, l_bounds, u_bounds, c0, c_dyn,
        x0, z_g0, y_g0, y_f_0_init, y_f_dyn_init, xi_g0, rho_bar_init,
        slack_weight_init,
        T, n, nx, n0, m, Nb, cfg);
}
