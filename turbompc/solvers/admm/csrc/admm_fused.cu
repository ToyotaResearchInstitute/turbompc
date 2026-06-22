#include "admm_fused.cuh"
#include "admm_math.cuh"
#include "admm_reductions.cuh"
#include <cmath>
#include <cfloat>
#include <cstdio>  // for device printf (used by ADMM_PROFILE)

// Block-tridiagonal matvec: y = M @ v
// M is in global memory (T_blocks*n rows, 3n cols, row-major).
// v and y are in shared memory, length = total = T_blocks * n.
template<typename T>
__device__ void blkTridiMatvecShared_admm(
    T* __restrict__ y,
    const T* __restrict__ M,
    const T* __restrict__ v,
    int T_blocks, int n, int total)
{
    int tid = threadIdx.x;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        int t = idx / n;
        int i = idx % n;
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
        y[idx] = sum;
    }
    __syncthreads();
}

// blockDotShared_admm, blockInfNormShared → admm_reductions.cuh

// invertSmallMatrixDevice and recomputeSchurDevice (+ its split:
// recomputeSDevice / recomputePhiinvFromSDevice) live in admm_math.cuh
// so the cuDSS-loop backend can reuse them. computeGammaDevice and the
// rho-update helper computeRhoCandidateDevice are also there.

// ============================================================
// Device function: PCG solve (adapted from pcgBlkTridiSingleBlock)
// Solves S @ x = gamma, starting from x0 (current x_blocks).
// All vectors in shared memory; S and Phiinv in global memory.
// Result is written to x_blocks in shared memory.
// ============================================================
template<typename T>
__device__ void pcgSolveDevice(
    T* __restrict__ x_blocks,    // [T*n] in shared memory (initial guess and output)
    const T* __restrict__ gamma, // [T*n] in shared memory (RHS)
    // PCG work vectors in shared memory
    T* __restrict__ pcg_r,       // [T*n]
    T* __restrict__ pcg_z,       // [T*n]
    T* __restrict__ pcg_p,       // [T*n]
    T* __restrict__ pcg_Sp,      // [T*n]
    T* scalars,     // [16] — no __restrict__: shared across threads via __syncthreads
    T* warp_buf,    // [32] — no __restrict__: cross-thread reduction buffer
    // Global memory matrices (batch-offset already applied)
    const T* __restrict__ S_b,
    const T* __restrict__ Phiinv_b,
    int T_blocks, int n,
    int pcg_max_iter, T pcg_eps)
{
    int tid = threadIdx.x;
    int total = T_blocks * n;

    // Sp = S @ x
    blkTridiMatvecShared_admm(pcg_Sp, S_b, x_blocks, T_blocks, n, total);

    // r = gamma - Sp
    for (int i = tid; i < total; i += blockDim.x)
        pcg_r[i] = gamma[i] - pcg_Sp[i];
    __syncthreads();

    // z = Phiinv @ r
    blkTridiMatvecShared_admm(pcg_z, Phiinv_b, pcg_r, T_blocks, n, total);

    // p = z
    for (int i = tid; i < total; i += blockDim.x)
        pcg_p[i] = pcg_z[i];
    __syncthreads();

    // rho = r . z
    {
        T dot = blockDotShared_admm(pcg_r, pcg_z, warp_buf, total);
        if (tid == 0) {
            scalars[0] = dot;          // rho
            scalars[1] = fabs(dot);    // rho_init
            scalars[2] = T(0);        // pcg_done flag
        }
    }
    __syncthreads();

    const T abs_tol = T(1e-12);

    for (int iter = 0; iter < pcg_max_iter; ++iter) {
        // Convergence check — set flag but do NOT break (all threads must
        // continue to hit __syncthreads barriers in the loop body)
        if (tid == 0) {
            if (fabs(scalars[0]) < abs_tol + pcg_eps * scalars[1])
                scalars[2] = T(1);  // done
            scalars[12] = T(iter);  // track PCG iteration count
        }
        __syncthreads();
        if (scalars[2] > T(0.5))
            break;  // safe: all threads see same shared flag after __syncthreads

        // Sp = S @ p
        blkTridiMatvecShared_admm(pcg_Sp, S_b, pcg_p, T_blocks, n, total);

        // denom = p . Sp -> alpha = rho / denom
        {
            T dot = blockDotShared_admm(pcg_p, pcg_Sp, warp_buf, total);
            if (tid == 0) {
                T d = dot;
                if (fabs(d) < T(1e-30)) d = T(1e-30);
                scalars[3] = scalars[0] / d;  // alpha = rho / denom
            }
        }
        __syncthreads();

        // x += alpha*p, r -= alpha*Sp
        {
            T alpha_pcg = scalars[3];
            for (int i = tid; i < total; i += blockDim.x) {
                x_blocks[i] += alpha_pcg * pcg_p[i];
                pcg_r[i] -= alpha_pcg * pcg_Sp[i];
            }
        }
        __syncthreads();

        // z = Phiinv @ r
        blkTridiMatvecShared_admm(pcg_z, Phiinv_b, pcg_r, T_blocks, n, total);

        // rho_new = r . z -> beta = rho_new / rho
        {
            T dot = blockDotShared_admm(pcg_r, pcg_z, warp_buf, total);
            if (tid == 0) {
                T rho_old = scalars[0];
                scalars[5] = dot / rho_old;  // beta
                scalars[0] = dot;            // rho = rho_new
            }
        }
        __syncthreads();

        // p = z + beta * p
        {
            T beta = scalars[5];
            for (int i = tid; i < total; i += blockDim.x)
                pcg_p[i] = pcg_z[i] + beta * pcg_p[i];
        }
        __syncthreads();
    }
    // x_blocks now contains the PCG solution
}

// applyConstraintsDevice, overRelaxDevice, slackUpdateDevice, dualUpdateDevice → admm_math.cuh

// ============================================================
// Device function: compute residuals for convergence check
// Primal: max(||Cx - c||_inf, ||Gx - z_g||_inf)
// Dual:   ||Px + q + C^T*y_f + G^T*y_g||_inf
// ============================================================
template<typename T>
__device__ void computeResidualsDevice(
    T* scalars,                       // scalars[6..14] — no __restrict__: shared across threads
    // Shared memory vectors
    T* __restrict__ x_blocks,         // [T*n]
    T* __restrict__ z_g,              // [T*m]
    T* __restrict__ y_g,              // [T*m]
    T* __restrict__ y_f_0,            // [nx]
    T* __restrict__ y_f_dyn,          // [N*nx]
    T* __restrict__ scratch,          // work area: [T*n] for computing temp vectors
    T* warp_buf,                      // [32] — no __restrict__: cross-thread reduction buffer
    // Global memory (batch-offset applied)
    const T* __restrict__ S_b,        // [T, n, 3n] -- the full S (=cost block tridiag + regularization)
    const T* __restrict__ D_g,        // [T, n, n]  cost D blocks
    const T* __restrict__ E_g,        // [N, n, n]  cost E blocks
    const T* __restrict__ q_g,        // [T, n]
    const T* __restrict__ A0_g,
    const T* __restrict__ Aminus_g,
    const T* __restrict__ Aplus_g,
    const T* __restrict__ G_g,
    const T* __restrict__ c0_g,
    const T* __restrict__ cdyn_g,
    int T_blocks, int n, int nx, int n0, int m,
    T* __restrict__ xi_g, T slack_weight, bool use_slack)
{
    int tid = threadIdx.x;
    int N = T_blocks - 1;
    int total_tn = T_blocks * n;

    // ---- Primal residual: need Cx - c and Gx - z_g ----
    // We reuse the scratch buffer. First compute Cx0 and Cx_dyn residuals.
    // We'll compute these in scratch and then do inf-norm reductions.

    // Step 1: Compute Cx0 - c0 into scratch[0..n0-1]
    if (tid < n0) {
        T sum = T(0);
        for (int k = 0; k < n; ++k)
            sum += A0_g[tid * n + k] * x_blocks[k];
        scratch[tid] = sum - c0_g[tid];
    }
    __syncthreads();

    T prim_res_eq0 = blockInfNormShared(scratch, warp_buf, n0);
    T primal_res;
    if (tid == 0) primal_res = prim_res_eq0;

    // Step 2: Compute Cx_dyn - c_dyn into scratch[0..N*nx-1]
    int total_Nnx = N * nx;
    if (tid < total_Nnx) {
        int t = tid / nx;
        int j = tid % nx;
        T sum = T(0);
        const T* Am_row = Aminus_g + t * nx * n + j * n;
        const T* Ap_row = Aplus_g  + t * nx * n + j * n;
        const T* x_t  = x_blocks + t * n;
        const T* x_tp = x_blocks + (t + 1) * n;
        for (int k = 0; k < n; ++k)
            sum += Am_row[k] * x_t[k] + Ap_row[k] * x_tp[k];
        scratch[tid] = sum - cdyn_g[tid];
    }
    __syncthreads();

    T prim_res_dyn = blockInfNormShared(scratch, warp_buf, total_Nnx);
    if (tid == 0) {
        primal_res = (prim_res_dyn > primal_res) ? prim_res_dyn : primal_res;
    }

    // Step 3: Compute Gx - z_g into scratch[0..T*m-1]
    if (m > 0) {
        int total_Tm = T_blocks * m;
        for (int i = tid; i < total_Tm; i += blockDim.x) {
            int t = i / m;
            int j = i % m;
            T sum = T(0);
            const T* G_row = G_g + t * m * n + j * n;
            const T* x_t = x_blocks + t * n;
            for (int k = 0; k < n; ++k)
                sum += G_row[k] * x_t[k];
            scratch[i] = sum - z_g[i];
        }
        __syncthreads();

        T prim_res_ineq = blockInfNormShared(scratch, warp_buf, total_Tm);
        if (tid == 0) {
            primal_res = (prim_res_ineq > primal_res) ? prim_res_ineq : primal_res;
        }
    }

    if (tid == 0) scalars[6] = primal_res;
    __syncthreads();

    // ---- Dual residual: ||Px + q + C^T*y_f + G^T*y_g||_inf ----
    // Compute into scratch[0..T*n-1]
    // Px = D @ x (block-tridiag with cost blocks D, E)
    // We compute Px element by element.
    for (int idx = tid; idx < total_tn; idx += blockDim.x) {
        int t = idx / n;
        int j = idx % n;

        T px_val = T(0);
        const T* D_t = D_g + t * n * n;
        for (int k = 0; k < n; ++k)
            px_val += D_t[j * n + k] * x_blocks[t * n + k];
        if (t > 0) {
            const T* E_tm1 = E_g + (t - 1) * n * n;
            for (int k = 0; k < n; ++k)
                px_val += E_tm1[j * n + k] * x_blocks[(t - 1) * n + k];
        }
        if (t < N) {
            const T* E_t = E_g + t * n * n;
            for (int k = 0; k < n; ++k)
                px_val += E_t[k * n + j] * x_blocks[(t + 1) * n + k];
        }

        T ct_yf = T(0);
        if (t == 0) {
            for (int k = 0; k < n0; ++k)
                ct_yf += A0_g[k * n + j] * y_f_0[k];
            if (N > 0)
                for (int k = 0; k < nx; ++k)
                    ct_yf += Aminus_g[k * n + j] * y_f_dyn[k];
        } else if (t < N) {
            const T* Aplus_tm1 = Aplus_g + (t - 1) * nx * n;
            const T* Aminus_t = Aminus_g + t * nx * n;
            for (int k = 0; k < nx; ++k)
                ct_yf += Aplus_tm1[k * n + j] * y_f_dyn[(t - 1) * nx + k];
            for (int k = 0; k < nx; ++k)
                ct_yf += Aminus_t[k * n + j] * y_f_dyn[t * nx + k];
        } else {
            if (N > 0) {
                const T* Aplus_Nm1 = Aplus_g + (N - 1) * nx * n;
                for (int k = 0; k < nx; ++k)
                    ct_yf += Aplus_Nm1[k * n + j] * y_f_dyn[(N - 1) * nx + k];
            }
        }

        T gt_yg = T(0);
        if (m > 0) {
            const T* G_t = G_g + t * m * n;
            for (int k = 0; k < m; ++k)
                gt_yg += G_t[k * n + j] * y_g[t * m + k];
        }

        scratch[idx] = px_val + q_g[idx] + ct_yf + gt_yg;
    }
    __syncthreads();

    T dual_res = blockInfNormShared(scratch, warp_buf, total_tn);

    // Slack dual residual: ||slack_weight * xi_g + y_g||_inf
    if (use_slack && m > 0) {
        int total_Tm = T_blocks * m;
        for (int i = tid; i < total_Tm; i += blockDim.x)
            scratch[i] = fabs(slack_weight * xi_g[i] + y_g[i]);
        __syncthreads();
        T slack_dual_res = blockInfNormShared(scratch, warp_buf, total_Tm);
        if (tid == 0) {
            if (slack_dual_res > dual_res) dual_res = slack_dual_res;
        }
    }

    if (tid == 0) scalars[7] = dual_res;
    __syncthreads();

    // ---- Norm terms for normalized residuals (adaptive rho) ----
    // primal_norm_term = max(||Cx||_inf, ||c||_inf, ||Gx||_inf, ||z_g||_inf)
    // dual_norm_term   = max(||Px||_inf, ||q||_inf, ||C^T y_f + G^T y_g||_inf)
    // These are computed by reusing scratch; cost is negligible (runs every check_every iters).

    T primal_norm_term = T(0);
    T dual_norm_term = T(0);

    // --- Primal norm terms ---
    // ||Cx0||_inf: A0 @ x[0]
    if (tid < nx) {
        T sum = T(0);
        for (int k = 0; k < n; ++k)
            sum += A0_g[tid * n + k] * x_blocks[k];
        scratch[tid] = sum;
    }
    __syncthreads();
    {
        T norm_cx0 = blockInfNormShared(scratch, warp_buf, nx);
        if (tid == 0) primal_norm_term = norm_cx0;
    }

    // ||Cx_dyn||_inf: Am[t] @ x[t] + Ap[t] @ x[t+1]
    if (tid < total_Nnx) {
        int t = tid / nx;
        int j = tid % nx;
        T sum = T(0);
        const T* Am_row = Aminus_g + t * nx * n + j * n;
        const T* Ap_row = Aplus_g  + t * nx * n + j * n;
        for (int k = 0; k < n; ++k)
            sum += Am_row[k] * x_blocks[t * n + k] + Ap_row[k] * x_blocks[(t + 1) * n + k];
        scratch[tid] = sum;
    }
    __syncthreads();
    {
        T norm_cx_dyn = blockInfNormShared(scratch, warp_buf, total_Nnx);
        if (tid == 0) { if (norm_cx_dyn > primal_norm_term) primal_norm_term = norm_cx_dyn; }
    }

    // ||c||_inf: max(||c0||, ||c_dyn||)
    if (tid < nx)
        scratch[tid] = c0_g[tid];
    __syncthreads();
    {
        T norm_c0 = blockInfNormShared(scratch, warp_buf, nx);
        if (tid == 0) { if (norm_c0 > primal_norm_term) primal_norm_term = norm_c0; }
    }
    if (tid < total_Nnx)
        scratch[tid] = cdyn_g[tid];
    __syncthreads();
    {
        T norm_cdyn = blockInfNormShared(scratch, warp_buf, total_Nnx);
        if (tid == 0) { if (norm_cdyn > primal_norm_term) primal_norm_term = norm_cdyn; }
    }

    // ||Gx||_inf and ||z_g||_inf
    if (m > 0) {
        int total_Tm = T_blocks * m;
        // Gx
        for (int i = tid; i < total_Tm; i += blockDim.x) {
            int t = i / m;
            int j = i % m;
            T sum = T(0);
            const T* G_row = G_g + t * m * n + j * n;
            for (int k = 0; k < n; ++k)
                sum += G_row[k] * x_blocks[t * n + k];
            scratch[i] = sum;
        }
        __syncthreads();
        {
            T norm_gx = blockInfNormShared(scratch, warp_buf, total_Tm);
            if (tid == 0) { if (norm_gx > primal_norm_term) primal_norm_term = norm_gx; }
        }
        // z_g
        {
            T norm_zg = blockInfNormShared(z_g, warp_buf, total_Tm);
            if (tid == 0) { if (norm_zg > primal_norm_term) primal_norm_term = norm_zg; }
        }
    }

    // --- Dual norm terms ---
    // ||Px||_inf: block-tridiag(D, E) @ x
    for (int idx = tid; idx < total_tn; idx += blockDim.x) {
        int t = idx / n;
        int j = idx % n;
        T px_val = T(0);
        const T* D_t = D_g + t * n * n;
        for (int k = 0; k < n; ++k)
            px_val += D_t[j * n + k] * x_blocks[t * n + k];
        if (t > 0) {
            const T* E_tm1 = E_g + (t - 1) * n * n;
            for (int k = 0; k < n; ++k)
                px_val += E_tm1[j * n + k] * x_blocks[(t - 1) * n + k];
        }
        if (t < N) {
            const T* E_t = E_g + t * n * n;
            for (int k = 0; k < n; ++k)
                px_val += E_t[k * n + j] * x_blocks[(t + 1) * n + k];
        }
        scratch[idx] = px_val;
    }
    __syncthreads();
    {
        T norm_px = blockInfNormShared(scratch, warp_buf, total_tn);
        if (tid == 0) dual_norm_term = norm_px;
    }

    // ||q||_inf
    for (int i = tid; i < total_tn; i += blockDim.x)
        scratch[i] = q_g[i];
    __syncthreads();
    {
        T norm_q = blockInfNormShared(scratch, warp_buf, total_tn);
        if (tid == 0) { if (norm_q > dual_norm_term) dual_norm_term = norm_q; }
    }

    // ||C^T y_f + G^T y_g||_inf (the Aty vector)
    for (int idx = tid; idx < total_tn; idx += blockDim.x) {
        int t = idx / n;
        int j = idx % n;
        T ct_yf = T(0);
        if (t == 0) {
            for (int k = 0; k < nx; ++k)
                ct_yf += A0_g[k * n + j] * y_f_0[k];
            if (N > 0)
                for (int k = 0; k < nx; ++k)
                    ct_yf += Aminus_g[k * n + j] * y_f_dyn[k];
        } else if (t < N) {
            const T* Aplus_tm1 = Aplus_g + (t - 1) * nx * n;
            const T* Aminus_t = Aminus_g + t * nx * n;
            for (int k = 0; k < nx; ++k)
                ct_yf += Aplus_tm1[k * n + j] * y_f_dyn[(t - 1) * nx + k];
            for (int k = 0; k < nx; ++k)
                ct_yf += Aminus_t[k * n + j] * y_f_dyn[t * nx + k];
        } else {
            if (N > 0) {
                const T* Aplus_Nm1 = Aplus_g + (N - 1) * nx * n;
                for (int k = 0; k < nx; ++k)
                    ct_yf += Aplus_Nm1[k * n + j] * y_f_dyn[(N - 1) * nx + k];
            }
        }
        T gt_yg = T(0);
        if (m > 0) {
            const T* G_t = G_g + t * m * n;
            for (int k = 0; k < m; ++k)
                gt_yg += G_t[k * n + j] * y_g[t * m + k];
        }
        scratch[idx] = ct_yf + gt_yg;
    }
    __syncthreads();
    {
        T norm_aty = blockInfNormShared(scratch, warp_buf, total_tn);
        if (tid == 0) { if (norm_aty > dual_norm_term) dual_norm_term = norm_aty; }
    }

    // Store norm terms and normalized residuals
    if (tid == 0) {
        scalars[10] = primal_norm_term;
        scalars[11] = dual_norm_term;
        scalars[13] = scalars[6] / (T(1e-10) + primal_norm_term);
        scalars[14] = scalars[7] / (T(1e-10) + dual_norm_term);
    }
    __syncthreads();
}


// ============================================================
// Main fused ADMM kernel
// ============================================================
template<typename T>
__global__ void __launch_bounds__(768) admmFusedKernel(
    // Outputs
    T*        __restrict__ x_out,
    uint32_t* __restrict__ iters_out,
    T*        __restrict__ x_blocks_out,
    T*        __restrict__ z_g_out,
    T*        __restrict__ y_g_out,
    T*        __restrict__ y_f_0_out,
    T*        __restrict__ y_f_dyn_out,
    T*        __restrict__ xi_g_out,
    T*        __restrict__ rho_bar_out,
    T*        __restrict__ kernel_ns_out,
    // Workspace (writable global memory for adaptive rho)
    T*        __restrict__ S_work_global,       // [Nb, T, n, 3n]
    T*        __restrict__ Phiinv_work_global,  // [Nb, T, n, 3n]
    // QP data (read-only)
    const T* __restrict__ S_global,
    const T* __restrict__ Phiinv_global,
    const T* __restrict__ D_global,
    const T* __restrict__ E_global,
    const T* __restrict__ q_global,
    const T* __restrict__ A0_global,
    const T* __restrict__ Aminus_global,
    const T* __restrict__ Aplus_global,
    const T* __restrict__ G_global,
    const T* __restrict__ l_global,
    const T* __restrict__ u_global,
    const T* __restrict__ c0_global,
    const T* __restrict__ cdyn_global,
    // Warm-start (read-only)
    const T* __restrict__ x0_global,
    const T* __restrict__ z_g0_global,
    const T* __restrict__ y_g0_global,
    const T* __restrict__ y_f_0_init_global,
    const T* __restrict__ y_f_dyn_init_global,
    const T* __restrict__ xi_g0_global,
    const T* __restrict__ rho_bar_init_global,
    const T* __restrict__ slack_weight_global,  // [Nb] runtime slack weight (JAX-traceable)
    // Dimensions
    int T_blocks, int n, int nx, int n0, int m, int Nb,
    // Config
    int max_iter, int pcg_max_iter, int check_every,
    T eps_abs, T eps_rel, T sigma, T rho_f_factor, T alpha_relax, T pcg_eps,
    bool use_slack,
    // Adaptive rho config
    int adapt_rho_every, T adaptive_rho_tolerance, T rho_min, T rho_max)
{
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    const int bid = blockIdx.x;
    if (bid >= Nb) return;

    // Record kernel start time for profiling
    long long t_kernel_start = clock64();

    const int tid = threadIdx.x;
    const int N   = T_blocks - 1;

    const int total_tn = T_blocks * n;
    const int total_Tm = T_blocks * m;
    const int total_Nnx = N * nx;

    // ---- Compute batch offsets for global memory ----
    const int S_stride       = T_blocks * n * 3 * n;
    // S_work has extra theta_inv space: T*n*(3n+n) = T*n*4n per batch
    const int S_work_stride  = T_blocks * n * 4 * n;
    const int D_stride       = T_blocks * n * n;
    const int E_stride       = N * n * n;
    const int q_stride       = total_tn;
    const int A0_stride      = n0 * n;
    const int Am_stride      = N * nx * n;
    const int Ap_stride      = N * nx * n;
    const int G_stride       = T_blocks * m * n;
    const int l_stride       = total_Tm;
    const int u_stride       = total_Tm;
    const int c0_stride      = n0;
    const int cdyn_stride    = N * nx;

    const T* S_b       = S_global       + bid * S_stride;
    const T* Phiinv_b  = Phiinv_global  + bid * S_stride;
    T* S_work_b        = S_work_global  + bid * S_work_stride;
    T* Phiinv_work_b   = Phiinv_work_global + bid * S_stride;
    const T* D_b       = D_global       + bid * D_stride;
    const T* E_b       = E_global       + bid * E_stride;
    const T* q_b       = q_global       + bid * q_stride;
    const T* A0_b      = A0_global      + bid * A0_stride;
    const T* Am_b      = Aminus_global  + bid * Am_stride;
    const T* Ap_b      = Aplus_global   + bid * Ap_stride;
    const T* G_b       = G_global       + bid * G_stride;
    const T* l_b       = l_global       + bid * l_stride;
    const T* u_b       = u_global       + bid * u_stride;
    const T* c0_b      = c0_global      + bid * c0_stride;
    const T* cdyn_b    = cdyn_global    + bid * cdyn_stride;

    const T* x0_b      = x0_global          + bid * q_stride;
    const T* z_g0_b    = z_g0_global        + bid * l_stride;
    const T* y_g0_b    = y_g0_global        + bid * l_stride;
    const T* yf0_b     = y_f_0_init_global  + bid * c0_stride;
    const T* yfdyn_b   = y_f_dyn_init_global + bid * cdyn_stride;
    const T* xi_g0_b   = xi_g0_global       + bid * l_stride;

    // ---- Shared memory layout ----
    // We lay out arrays sequentially:
    //   x_blocks:    [T*n]
    //   y_f_0:       [nx]
    //   y_f_dyn:     [N*nx]
    //   gamma:       [T*n]
    //   pcg_r:       [T*n]
    //   pcg_z:       [T*n]
    //   pcg_p:       [T*n]
    //   pcg_Sp:      [T*n]
    //   scalars:     [16]
    //   warp_buf:    [32]
    //   scratch:     max(T*m, T*n, N*nx) for constraint eval / residuals / z_g_old
    //
    // NOTE: z_g, y_g, xi_g are stored in global memory (using the output buffers
    // as working storage) to reduce shared memory usage. These arrays are only
    // accessed once per ADMM iteration (not in the hot PCG inner loop), so the
    // performance impact is minimal. This allows H=100+ with n=12 under the
    // 99 KB opt-in shared memory limit.

    T* sh_x_blocks = smem;
    T* sh_y_f_0    = sh_x_blocks + total_tn;
    T* sh_y_f_dyn  = sh_y_f_0    + n0;
    T* sh_gamma    = sh_y_f_dyn  + total_Nnx;
    T* sh_pcg_r    = sh_gamma    + total_tn;
    T* sh_pcg_z    = sh_pcg_r    + total_tn;
    T* sh_pcg_p    = sh_pcg_z    + total_tn;
    T* sh_pcg_Sp   = sh_pcg_p    + total_tn;
    T* sh_scalars  = sh_pcg_Sp   + total_tn;
    T* sh_warp_buf = sh_scalars  + 16;
    T* sh_scratch  = sh_warp_buf + 32;

    // z_g, y_g, xi_g use the output buffers as in-place workspace (global memory).
    // This avoids 3*T*m elements in shared memory.
    T* sh_z_g  = z_g_out  + bid * l_stride;
    T* sh_y_g  = y_g_out  + bid * l_stride;
    T* sh_xi_g = xi_g_out + bid * l_stride;

    // ---- 1. Load warm-start from global memory to shared memory ----
    for (int i = tid; i < total_tn; i += blockDim.x)
        sh_x_blocks[i] = x0_b[i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        sh_z_g[i] = z_g0_b[i];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        sh_y_g[i] = y_g0_b[i];
    if (tid < n0)
        sh_y_f_0[tid] = yf0_b[tid];
    if (tid < total_Nnx)
        sh_y_f_dyn[tid] = yfdyn_b[tid];
    for (int i = tid; i < total_Tm; i += blockDim.x)
        sh_xi_g[i] = xi_g0_b[i];

    // Load rho_bar and slack_weight from global buffers (JAX-traceable)
    T rho_bar;
    T slack_weight;
    if (tid == 0) {
        rho_bar = rho_bar_init_global[bid];
        slack_weight = slack_weight_global[bid];
        sh_scalars[8] = rho_bar;   // store rho_bar in scalars[8]
        sh_scalars[9] = T(0);      // convergence flag
    }
    __syncthreads();

    rho_bar = sh_scalars[8];
    slack_weight = slack_weight_global[bid];

    // Initialize scalars for residual tracking
    if (tid == 0) {
        sh_scalars[6] = T(1e30);   // primal residual (large initial)
        sh_scalars[7] = T(1e30);   // dual residual (large initial)
        sh_scalars[10] = T(1);     // primal_norm_term
        sh_scalars[11] = T(1);     // dual_norm_term
        sh_scalars[15] = T(0);    // rho update flag
    }
    __syncthreads();

    // ---- Copy S → S_work and Phiinv → Phiinv_work (workspace init) ----
    for (int i = tid; i < S_stride; i += blockDim.x) {
        S_work_b[i] = S_b[i];
        Phiinv_work_b[i] = Phiinv_b[i];
    }
    __syncthreads();

    // ---- 2. Main ADMM loop ----
    uint32_t final_iter = (uint32_t)max_iter;

#ifdef ADMM_PROFILE
    // Profiling: clock64() timestamps per stage, printed from bid==0, tid==0
    // Stages: 0=gamma, 1=save_xold, 2=pcg, 3=constraints, 4=overrelax,
    //         5=save_zgold, 6=slack, 7=dual, 8=residuals
    const bool do_profile = (bid == 0);
    if (do_profile && tid == 0)
        printf("ADMM_PROFILE: iter,gamma,save_xold,pcg,constraints,overrelax,save_zgold,slack,dual,residuals,total,pcg_iters (GPU cycles)\n");
#endif

    for (int it = 0; it < max_iter; ++it) {
        rho_bar = sh_scalars[8];
        T rho_f = rho_bar * rho_f_factor;

#ifdef ADMM_PROFILE
        unsigned long long t_iter_start = 0, t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        unsigned long long t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0;
        if (do_profile && tid == 0) t_iter_start = clock64();
#endif

        // 2a. Compute gamma (RHS for PCG)
        computeGammaDevice(
            sh_gamma, sh_x_blocks, sh_z_g, sh_y_g, sh_y_f_0, sh_y_f_dyn,
            q_b, A0_b, Am_b, Ap_b, G_b, c0_b, cdyn_b,
            sigma, rho_f, rho_bar,
            T_blocks, n, nx, n0, m);

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t0 = clock64();
#endif

        // Save x_old into scratch for over-relaxation (uses scratch[0..T*n-1])
        for (int i = tid; i < total_tn; i += blockDim.x)
            sh_scratch[i] = sh_x_blocks[i];
        __syncthreads();

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t1 = clock64();
#endif

        // 2b. PCG solve: S @ x_new = gamma
        pcgSolveDevice(
            sh_x_blocks, sh_gamma,
            sh_pcg_r, sh_pcg_z, sh_pcg_p, sh_pcg_Sp,
            sh_scalars, sh_warp_buf,
            S_work_b, Phiinv_work_b,
            T_blocks, n, pcg_max_iter, pcg_eps);

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t2 = clock64();
#endif

        // Evaluate constraints (C, G) on the PRE-relaxation PCG result; z_tilde,
        // y_f, and y_g consume these. Over-relaxation of x happens AFTER.
        T* scratch_Cx0     = sh_scratch + total_tn;
        T* scratch_Cx_dyn  = scratch_Cx0 + n0;
        T* scratch_Gx      = sh_gamma;

        applyConstraintsDevice(
            scratch_Cx0, scratch_Cx_dyn, scratch_Gx,
            sh_x_blocks,
            A0_b, Am_b, Ap_b, G_b,
            T_blocks, n, nx, n0, m);

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t3 = clock64();
#endif

        // 2d. Over-relaxation: x = alpha * x_new + (1-alpha) * x_old
        // x_old is in sh_scratch[0..total_tn-1]
        overRelaxDevice(sh_x_blocks, sh_scratch, alpha_relax, total_tn);

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t4 = clock64();
#endif

        // 2e. Save z_g_old for dual update (into pcg_r which is free now)
        T* z_g_old = sh_pcg_r;  // reuse PCG vector
        for (int i = tid; i < total_Tm; i += blockDim.x)
            z_g_old[i] = sh_z_g[i];
        __syncthreads();

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t5 = clock64();
#endif

        // Slack update: z_g = clamp(z_tilde, l, u)
        slackUpdateDevice(
            sh_z_g, scratch_Gx, sh_y_g,
            l_b, u_b,
            alpha_relax, rho_bar,
            T_blocks, m,
            slack_weight, use_slack, sh_xi_g);

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t6 = clock64();
#endif

        // 2f. Dual update
        dualUpdateDevice(
            sh_y_f_0, sh_y_f_dyn, sh_y_g,
            scratch_Cx0, scratch_Cx_dyn, scratch_Gx,
            sh_z_g, z_g_old,
            c0_b, cdyn_b,
            alpha_relax, rho_f, rho_bar,
            T_blocks, n, nx, n0, m);

#ifdef ADMM_PROFILE
        if (do_profile && tid == 0) t7 = clock64();
#endif

        // 2g. Convergence check every check_every iterations
        if (check_every > 0 && (it % check_every == 0)) {
            // Use sh_scratch for residual computation work area
            computeResidualsDevice(
                sh_scalars,
                sh_x_blocks, sh_z_g, sh_y_g, sh_y_f_0, sh_y_f_dyn,
                sh_scratch, sh_warp_buf,
                S_b, D_b, E_b, q_b,
                A0_b, Am_b, Ap_b, G_b,
                c0_b, cdyn_b,
                T_blocks, n, nx, n0, m,
                sh_xi_g, slack_weight, use_slack);

            // Check convergence using normalized residuals
            // primal_res in scalars[6], dual_res in scalars[7]
            // norm terms in scalars[10], scalars[11]
            if (tid == 0) {
                T p_res = sh_scalars[6];
                T d_res = sh_scalars[7];
                T p_margin = eps_abs + eps_rel * sh_scalars[10];
                T d_margin = eps_abs + eps_rel * sh_scalars[11];
                bool converged = (p_res < p_margin) && (d_res < d_margin);
#ifdef ADMM_TRACE
                if (bid == 0) {
                    printf("ADMM_TRACE: iter=%d p_res=%.6e d_res=%.6e p_norm=%.6e d_norm=%.6e p_margin=%.6e d_margin=%.6e conv=%d\n",
                           it, (double)p_res, (double)d_res,
                           (double)sh_scalars[10], (double)sh_scalars[11],
                           (double)p_margin, (double)d_margin, (int)converged);
                }
#endif
                if (converged) {
                    sh_scalars[9] = T(1);  // convergence flag
                }
            }
            __syncthreads();

#ifdef ADMM_PROFILE
            if (do_profile && tid == 0) t8 = clock64();
#endif

            // If converged, break
            if (sh_scalars[9] > T(0.5)) {
                final_iter = (uint32_t)(it + 1);
#ifdef ADMM_PROFILE
                if (do_profile && tid == 0) {
                    unsigned long long total = t8 - t_iter_start;
                    int pcg_iters = (int)sh_scalars[12] + 1;
                    printf("ADMM_PROFILE: %d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d\n",
                           it, t0 - t_iter_start, t1 - t0, t2 - t1, t3 - t2,
                           t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, total, pcg_iters);
                }
#endif
                break;
            }

            // ---- Adaptive rho daemon ----
            // Match JAX: it % adapt_rho_every == 0 && it >= 2 (0-based)
            if (adapt_rho_every > 0 && it >= 2 && (it % adapt_rho_every == 0)) {
                // Thread 0 decides if rho should change
                if (tid == 0) {
                    T p_norm = sh_scalars[13]; // primal_residual_normalized
                    T d_norm = sh_scalars[14]; // dual_residual_normalized
                    T ratio = sqrt(p_norm / (d_norm + T(1e-30)));
                    T rho_candidate = sh_scalars[8] * ratio;
                    if (rho_candidate < rho_min) rho_candidate = rho_min;
                    if (rho_candidate > rho_max) rho_candidate = rho_max;

                    T rhos_ratio = rho_candidate / sh_scalars[8];
                    if (rhos_ratio < T(1)) rhos_ratio = T(1) / rhos_ratio;

                    bool admm_converged = (sh_scalars[9] > T(0.5));
                    bool should_update = (rhos_ratio >= adaptive_rho_tolerance) && !admm_converged;

                    if (should_update) {
                        sh_scalars[8] = rho_candidate;  // update rho_bar
                        sh_scalars[15] = T(1);          // flag: need Schur recompute
                    } else {
                        sh_scalars[15] = T(0);
                    }
                }
                __syncthreads();

                // If rho changed, recompute S_work and Phiinv_work
                if (sh_scalars[15] > T(0.5)) {
                    T rho_bar_new = sh_scalars[8];
                    T rho_f_new = rho_bar_new * rho_f_factor;

                    // theta_inv needs T*n*n elements.  Store it AFTER the
                    // S_work data in the same buffer (S_work is allocated
                    // with extra space: T*n*(3n+n) = T*n*4n total).
                    // dtilde and gauss_work use shared memory (PCG vectors).
                    int nn = n * n;
                    T* theta_inv_gm = S_work_b + S_stride;  // [T*n*n] after S data
                    T* dtilde_sh    = sh_gamma;              // [n*n] shared memory
                    T* gauss_sh     = sh_gamma + nn;         // [n*2n] shared memory
                    // Total shared: nn + 2*nn = 3*nn = 108 << 780 available

                    recomputeSchurDevice(
                        S_work_b, Phiinv_work_b,
                        D_b, E_b, A0_b, Am_b, Ap_b, G_b,
                        theta_inv_gm, dtilde_sh, gauss_sh,
                        rho_bar_new, rho_f_new, sigma,
                        T_blocks, n, nx, n0, m);

                    // Reset flag
                    if (tid == 0) sh_scalars[15] = T(0);
                    __syncthreads();
                }
            }
        }

#ifdef ADMM_PROFILE
        // Print timing for this iteration (use t7 as end if no residual check)
        if (do_profile && tid == 0) {
            unsigned long long t_end_iter = (check_every > 0 && ((it + 1) % check_every == 0)) ? t8 : t7;
            unsigned long long resid = (check_every > 0 && ((it + 1) % check_every == 0)) ? (t8 - t7) : 0;
            unsigned long long total = t_end_iter - t_iter_start;
            int pcg_iters = (int)sh_scalars[12] + 1;
            printf("ADMM_PROFILE: %d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d\n",
                   it, t0 - t_iter_start, t1 - t0, t2 - t1, t3 - t2,
                   t4 - t3, t5 - t4, t6 - t5, t7 - t6, resid, total, pcg_iters);
        }
#endif
    }

    // ---- 3. Store results to global memory ----
    rho_bar = sh_scalars[8];

    // x_out and x_blocks_out
    T* xo = x_out + bid * q_stride;
    T* xbo = x_blocks_out + bid * q_stride;
    for (int i = tid; i < total_tn; i += blockDim.x) {
        xo[i] = sh_x_blocks[i];
        xbo[i] = sh_x_blocks[i];
    }

    // z_g_out and y_g_out: already in-place (sh_z_g = z_g_out + bid*l_stride, etc.)

    // y_f_0_out
    T* yf0o = y_f_0_out + bid * c0_stride;
    if (tid < n0)
        yf0o[tid] = sh_y_f_0[tid];

    // y_f_dyn_out
    T* yfdyno = y_f_dyn_out + bid * cdyn_stride;
    if (tid < total_Nnx)
        yfdyno[tid] = sh_y_f_dyn[tid];

    // xi_g_out: already in-place (sh_xi_g = xi_g_out + bid*l_stride)

    // rho_bar_out
    if (tid == 0) {
        rho_bar_out[bid] = rho_bar;
        iters_out[bid] = final_iter;

        // Record kernel time in nanoseconds
        // clock64() returns SM clock cycles; convert to ns using 1e9/clockRate
        // We store raw cycles here and let the host convert using cudaDevAttrClockRate
        long long t_kernel_end = clock64();
        long long elapsed_cycles = t_kernel_end - t_kernel_start;
        kernel_ns_out[bid] = static_cast<T>(elapsed_cycles);
    }
}


// ============================================================
// Launcher functions
// ============================================================

template<typename T>
static cudaError_t launchADMMFused(
    cudaStream_t stream,
    T* x_out, uint32_t* iters_out,
    T* x_blocks_out, T* z_g_out, T* y_g_out,
    T* y_f_0_out, T* y_f_dyn_out, T* xi_g_out, T* rho_bar_out,
    T* kernel_ns_out,
    T* S_work, T* Phiinv_work,
    const T* S, const T* Phiinv, const T* D, const T* E, const T* q,
    const T* A0, const T* A_minus, const T* A_plus, const T* G,
    const T* l_bounds, const T* u_bounds, const T* c0, const T* c_dyn,
    const T* x0, const T* z_g0, const T* y_g0,
    const T* y_f_0_init, const T* y_f_dyn_init, const T* xi_g0,
    const T* rho_bar_init,
    const T* slack_weight_init,  // [Nb] runtime slack weight (JAX-traceable)
    int32_t T_blocks, int32_t n, int32_t nx, int32_t n0, int32_t m, int32_t Nb,
    ADMMConfig cfg)
{
    int N = T_blocks - 1;
    int total_tn  = T_blocks * n;
    int total_Tm  = T_blocks * m;
    int total_Nnx = N * nx;

    // Thread count: max of all array index spaces, rounded up to warp
    int max_elem = total_tn;
    if (total_Tm > max_elem) max_elem = total_Tm;
    if (total_Nnx > max_elem) max_elem = total_Nnx;
    int threads = ((max_elem + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    // Cap at 768 to stay within the 65536 register file limit.
    // All loops use `for (i = tid; i < N; i += blockDim.x)` so
    // fewer threads than elements is correct — each thread processes
    // multiple elements per iteration.
    if (threads > 768) threads = 768;

    // Shared memory calculation
    // Fixed arrays:
    //   x_blocks[T*n] + y_f_0[n0] + y_f_dyn[N*nx]
    //   + gamma[T*n]
    //   + pcg_r[T*n] + pcg_z[T*n] + pcg_p[T*n] + pcg_Sp[T*n]
    //   + scalars[16] + warp_buf[32]
    //   + scratch: x_old (T*n) + Cx0 (n0) + Cx_dyn (N*nx)
    // NOTE: z_g, y_g, xi_g are in global memory (output buffers used as workspace).
    int scratch_size = total_tn + n0 + total_Nnx;

    size_t smem_elems =
        total_tn          // x_blocks
        + n0              // y_f_0
        + total_Nnx       // y_f_dyn
        + total_tn        // gamma
        + total_tn        // pcg_r
        + total_tn        // pcg_z
        + total_tn        // pcg_p
        + total_tn        // pcg_Sp
        + 16              // scalars
        + 32              // warp_buf
        + scratch_size;   // scratch

    size_t smem_bytes = smem_elems * sizeof(T);

    // Opt-in to extended shared memory if needed (>48 KB).
    // This is required for large horizons (H>=40 with n=12, F64).
    if (smem_bytes > 48 * 1024) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            admmFusedKernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        if (attr_err != cudaSuccess) {
            fprintf(stderr, "ADMM fused: cudaFuncSetAttribute failed for %zu bytes: %s\n",
                    smem_bytes, cudaGetErrorString(attr_err));
            // Do NOT launch the kernel — it would corrupt the CUDA context.
            cudaGetLastError();  // consume sticky error before reporting through FFI
            return attr_err;
        }
    }

    // Convert config to kernel-type values
    T eps_abs_k   = static_cast<T>(cfg.eps_abs);
    T eps_rel_k   = static_cast<T>(cfg.eps_rel);
    T sigma_k     = static_cast<T>(cfg.sigma);
    T rho_f_k     = static_cast<T>(cfg.rho_f_factor);
    T alpha_k     = static_cast<T>(cfg.alpha);
    T pcg_eps_k   = static_cast<T>(cfg.pcg_eps);
    bool use_sl   = cfg.use_slack;
    T rho_tol_k   = static_cast<T>(cfg.adaptive_rho_tolerance);
    T rho_min_k   = static_cast<T>(cfg.rho_min);
    T rho_max_k   = static_cast<T>(cfg.rho_max);

    admmFusedKernel<T><<<Nb, threads, smem_bytes, stream>>>(
        x_out, iters_out,
        x_blocks_out, z_g_out, y_g_out, y_f_0_out, y_f_dyn_out, xi_g_out, rho_bar_out,
        kernel_ns_out,
        S_work, Phiinv_work,
        S, Phiinv, D, E, q,
        A0, A_minus, A_plus, G, l_bounds, u_bounds, c0, c_dyn,
        x0, z_g0, y_g0, y_f_0_init, y_f_dyn_init, xi_g0, rho_bar_init,
        slack_weight_init,
        T_blocks, n, nx, n0, m, Nb,
        cfg.max_iter, cfg.pcg_max_iter, cfg.check_every,
        eps_abs_k, eps_rel_k, sigma_k, rho_f_k, alpha_k, pcg_eps_k,
        use_sl,
        cfg.adapt_rho_every, rho_tol_k, rho_min_k, rho_max_k);
    return cudaPeekAtLastError();
}

// ---- Explicit instantiations ----

cudaError_t LaunchADMMFusedF32(
    cudaStream_t stream,
    float* x_out, uint32_t* iters_out,
    float* x_blocks_out, float* z_g_out, float* y_g_out,
    float* y_f_0_out, float* y_f_dyn_out, float* xi_g_out, float* rho_bar_out,
    float* kernel_ns_out,
    float* S_work, float* Phiinv_work,
    const float* S, const float* Phiinv, const float* D, const float* E, const float* q,
    const float* A0, const float* A_minus, const float* A_plus, const float* G,
    const float* l_bounds, const float* u_bounds, const float* c0, const float* c_dyn,
    const float* x0, const float* z_g0, const float* y_g0,
    const float* y_f_0_init, const float* y_f_dyn_init, const float* xi_g0,
    const float* rho_bar_init, const float* slack_weight_init,
    int32_t T, int32_t n, int32_t nx, int32_t n0, int32_t m, int32_t Nb,
    ADMMConfig cfg)
{
    return launchADMMFused<float>(stream,
        x_out, iters_out,
        x_blocks_out, z_g_out, y_g_out, y_f_0_out, y_f_dyn_out, xi_g_out, rho_bar_out,
        kernel_ns_out,
        S_work, Phiinv_work,
        S, Phiinv, D, E, q,
        A0, A_minus, A_plus, G, l_bounds, u_bounds, c0, c_dyn,
        x0, z_g0, y_g0, y_f_0_init, y_f_dyn_init, xi_g0, rho_bar_init,
        slack_weight_init,
        T, n, nx, n0, m, Nb, cfg);
}

cudaError_t LaunchADMMFusedF64(
    cudaStream_t stream,
    double* x_out, uint32_t* iters_out,
    double* x_blocks_out, double* z_g_out, double* y_g_out,
    double* y_f_0_out, double* y_f_dyn_out, double* xi_g_out, double* rho_bar_out,
    double* kernel_ns_out,
    double* S_work, double* Phiinv_work,
    const double* S, const double* Phiinv, const double* D, const double* E, const double* q,
    const double* A0, const double* A_minus, const double* A_plus, const double* G,
    const double* l_bounds, const double* u_bounds, const double* c0, const double* c_dyn,
    const double* x0, const double* z_g0, const double* y_g0,
    const double* y_f_0_init, const double* y_f_dyn_init, const double* xi_g0,
    const double* rho_bar_init, const double* slack_weight_init,
    int32_t T, int32_t n, int32_t nx, int32_t n0, int32_t m, int32_t Nb,
    ADMMConfig cfg)
{
    return launchADMMFused<double>(stream,
        x_out, iters_out,
        x_blocks_out, z_g_out, y_g_out, y_f_0_out, y_f_dyn_out, xi_g_out, rho_bar_out,
        kernel_ns_out,
        S_work, Phiinv_work,
        S, Phiinv, D, E, q,
        A0, A_minus, A_plus, G, l_bounds, u_bounds, c0, c_dyn,
        x0, z_g0, y_g0, y_f_0_init, y_f_dyn_init, xi_g0, rho_bar_init,
        slack_weight_init,
        T, n, nx, n0, m, Nb, cfg);
}
