#pragma once
// Shared __device__ helpers used by admm_fused.cu and admm_cudss.cu so both
// backends compute identical ADMM math.

// gamma = sigma * x - q + C^T(rho_f*c - y_f) + G^T(rho*z_g - y_g)
template<typename T>
__device__ void computeGammaDevice(
    T* __restrict__ gamma,           // [T*n] output
    const T* __restrict__ x_blocks,  // [T*n] primal variable
    const T* __restrict__ z_g,       // [T*m] inequality slack
    const T* __restrict__ y_g,       // [T*m] inequality dual
    const T* __restrict__ y_f_0,     // [n0]  equality dual (initial)
    const T* __restrict__ y_f_dyn,   // [N*nx] equality dual (dynamics)
    const T* __restrict__ q_g,       // [T*n] cost linear term
    const T* __restrict__ A0_g,      // [n0, n] initial constraint
    const T* __restrict__ Aminus_g,  // [N, nx, n] dynamics (current)
    const T* __restrict__ Aplus_g,   // [N, nx, n] dynamics (next)
    const T* __restrict__ G_g,       // [T, m, n] inequality constraint
    const T* __restrict__ c0_g,      // [n0] initial constraint RHS
    const T* __restrict__ cdyn_g,    // [N, nx] dynamics constraint RHS
    T sigma, T rho_f, T rho_bar,
    int T_blocks, int n, int nx, int n0, int m)
{
    int tid = threadIdx.x;
    int total_tn = T_blocks * n;
    int N = T_blocks - 1;

    for (int idx = tid; idx < total_tn; idx += blockDim.x) {
        int t = idx / n;
        int j = idx % n;

        // Base: sigma * x - q
        T val = sigma * x_blocks[idx] - q_g[idx];

        // --- C^T(rho_f * c - y_f) contribution ---
        T eq_term = T(0);

        if (t == 0) {
            for (int k = 0; k < n0; ++k) {
                T c0_tilde_k = rho_f * c0_g[k] - y_f_0[k];
                eq_term += A0_g[k * n + j] * c0_tilde_k;
            }
            if (N > 0) {
                for (int k = 0; k < nx; ++k) {
                    T ct_k = rho_f * cdyn_g[k] - y_f_dyn[k];
                    eq_term += Aminus_g[k * n + j] * ct_k;
                }
            }
        } else if (t < N) {
            const T* Aplus_tm1 = Aplus_g + (t - 1) * nx * n;
            const T* Aminus_t = Aminus_g + t * nx * n;
            for (int k = 0; k < nx; ++k) {
                T ct_tm1 = rho_f * cdyn_g[(t - 1) * nx + k] - y_f_dyn[(t - 1) * nx + k];
                eq_term += Aplus_tm1[k * n + j] * ct_tm1;
            }
            for (int k = 0; k < nx; ++k) {
                T ct_t = rho_f * cdyn_g[t * nx + k] - y_f_dyn[t * nx + k];
                eq_term += Aminus_t[k * n + j] * ct_t;
            }
        } else {
            if (N > 0) {
                const T* Aplus_Nm1 = Aplus_g + (N - 1) * nx * n;
                for (int k = 0; k < nx; ++k) {
                    T ct_Nm1 = rho_f * cdyn_g[(N - 1) * nx + k] - y_f_dyn[(N - 1) * nx + k];
                    eq_term += Aplus_Nm1[k * n + j] * ct_Nm1;
                }
            }
        }

        val += eq_term;

        // --- G^T(rho_bar * z_g - y_g) contribution ---
        if (m > 0) {
            T ineq_term = T(0);
            const T* G_t = G_g + t * m * n;
            for (int k = 0; k < m; ++k) {
                T v = rho_bar * z_g[t * m + k] - y_g[t * m + k];
                ineq_term += G_t[k * n + j] * v;
            }
            val += ineq_term;
        }

        gamma[idx] = val;
    }
    __syncthreads();
}


// Cx0[j]      = A0[j,:] @ x[0,:]                       (j < n0)
// Cx_dyn[t,j] = Am[t,j,:] @ x[t,:] + Ap[t,j,:] @ x[t+1,:]
// Gx[t,j]     = G[t,j,:] @ x[t,:]                      (j < m)
template<typename T>
__device__ void applyConstraintsDevice(
    T* __restrict__ scratch_Cx0,     // [n0] output
    T* __restrict__ scratch_Cx_dyn,  // [N*nx] output
    T* __restrict__ scratch_Gx,      // [T*m] output
    const T* __restrict__ x_blocks,  // [T*n] primal variable
    const T* __restrict__ A0_g,      // [n0, n]
    const T* __restrict__ Aminus_g,  // [N, nx, n]
    const T* __restrict__ Aplus_g,   // [N, nx, n]
    const T* __restrict__ G_g,       // [T, m, n]
    int T_blocks, int n, int nx, int n0, int m)
{
    int tid = threadIdx.x;
    int N = T_blocks - 1;

    // Cx0[j] = A0[j,:] @ x[0,:]
    // Strided so n0 > blockDim.x is handled correctly.
    for (int idx = tid; idx < n0; idx += blockDim.x) {
        T sum = T(0);
        for (int k = 0; k < n; ++k)
            sum += A0_g[idx * n + k] * x_blocks[k];
        scratch_Cx0[idx] = sum;
    }

    // Cx_dyn[t*nx+j] = Am[t,j,:] @ x[t,:] + Ap[t,j,:] @ x[t+1,:]
    // Strided so N*nx > blockDim.x is handled correctly (long horizons —
    // e.g. H=128, nx=8 → total_Nnx=1024 > 768 thread cap).
    int total_Nnx = N * nx;
    for (int idx = tid; idx < total_Nnx; idx += blockDim.x) {
        int t = idx / nx;
        int j = idx % nx;
        T sum = T(0);
        const T* Am_row = Aminus_g + t * nx * n + j * n;
        const T* Ap_row = Aplus_g  + t * nx * n + j * n;
        const T* x_t  = x_blocks + t * n;
        const T* x_tp = x_blocks + (t + 1) * n;
        for (int k = 0; k < n; ++k)
            sum += Am_row[k] * x_t[k] + Ap_row[k] * x_tp[k];
        scratch_Cx_dyn[idx] = sum;
    }

    // Gx[t*m+j] = G[t,j,:] @ x[t,:]
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
            scratch_Gx[i] = sum;
        }
    }
    __syncthreads();
}


// x = alpha * x_new + (1 - alpha) * x_old
template<typename T>
__device__ void overRelaxDevice(
    T* __restrict__ x_blocks,    // [T*n] modified in-place
    const T* __restrict__ x_old, // [T*n] saved copy
    T alpha, int total_tn)
{
    int tid = threadIdx.x;
    for (int i = tid; i < total_tn; i += blockDim.x)
        x_blocks[i] = alpha * x_blocks[i] + (T(1) - alpha) * x_old[i];
    __syncthreads();
}


// z_tilde = alpha * Gx + (1-alpha) * z_old + y_g / rho_bar
// z_g     = clamp(z_tilde, l, u)        (with optional slack penalty)
template<typename T>
__device__ void slackUpdateDevice(
    T* __restrict__ z_g,              // [T*m] updated in-place
    const T* __restrict__ scratch_Gx, // [T*m] inequality values
    const T* __restrict__ y_g,        // [T*m] inequality dual
    const T* __restrict__ l_g,        // [T*m] lower bounds
    const T* __restrict__ u_g,        // [T*m] upper bounds
    T alpha, T rho_bar,
    int T_blocks, int m,
    T slack_weight, bool use_slack, T* __restrict__ xi_g)
{
    int tid = threadIdx.x;
    int total_Tm = T_blocks * m;

    if (m > 0) {
        for (int i = tid; i < total_Tm; i += blockDim.x) {
            T z_old = z_g[i];
            T z_tilde = alpha * scratch_Gx[i] + (T(1) - alpha) * z_old + y_g[i] / rho_bar;
            T lb = l_g[i];
            T ub = u_g[i];
            T clamped = z_tilde;
            if (clamped < lb) clamped = lb;
            if (clamped > ub) clamped = ub;
            if (use_slack) {
                T frac = slack_weight / (slack_weight + rho_bar);
                z_g[i] = (T(1) - frac) * z_tilde + frac * clamped;
                xi_g[i] = (rho_bar / (slack_weight + rho_bar)) * (clamped - z_tilde);
            } else {
                z_g[i] = clamped;
            }
        }
    }
    __syncthreads();
}


// y_f_0   += rho_f * alpha * (Cx0 - c0)
// y_f_dyn += rho_f * alpha * (Cx_dyn - c_dyn)
// y_g     += rho_bar * (alpha * Gx + (1-alpha) * z_g_old - z_g)
template<typename T>
__device__ void dualUpdateDevice(
    T* __restrict__ y_f_0,                 // [n0] updated in-place
    T* __restrict__ y_f_dyn,               // [N*nx] updated in-place
    T* __restrict__ y_g,                   // [T*m] updated in-place
    const T* __restrict__ scratch_Cx0,     // [n0]
    const T* __restrict__ scratch_Cx_dyn,  // [N*nx]
    const T* __restrict__ scratch_Gx,      // [T*m]
    const T* __restrict__ z_g,             // [T*m] new z_g
    const T* __restrict__ z_g_old,         // [T*m] old z_g
    const T* __restrict__ c0_g,            // [n0]
    const T* __restrict__ cdyn_g,          // [N*nx]
    T alpha, T rho_f, T rho_bar,
    int T_blocks, int n, int nx, int n0, int m)
{
    int tid = threadIdx.x;
    int N = T_blocks - 1;

    // Strided so n0 > blockDim.x is handled correctly.
    for (int i = tid; i < n0; i += blockDim.x) {
        y_f_0[i] += rho_f * alpha * (scratch_Cx0[i] - c0_g[i]);
    }

    // Strided so N*nx > blockDim.x is handled correctly (long horizons —
    // e.g. H=128, nx=8 → total_Nnx=1024 > 768 thread cap).
    int total_Nnx = N * nx;
    for (int i = tid; i < total_Nnx; i += blockDim.x) {
        y_f_dyn[i] += rho_f * alpha * (scratch_Cx_dyn[i] - cdyn_g[i]);
    }

    if (m > 0) {
        int total_Tm = T_blocks * m;
        for (int i = tid; i < total_Tm; i += blockDim.x) {
            y_g[i] += rho_bar * (alpha * scratch_Gx[i] + (T(1) - alpha) * z_g_old[i] - z_g[i]);
        }
    }
    __syncthreads();
}

// In-place n×n Gauss-Jordan inversion: A → Ainv via n×2n augmented buffer
// in `work`. Thread 0 does the sequential pivoting (n is small, e.g. 6).
template<typename T>
__device__ void invertSmallMatrixDevice(
    T* __restrict__ Ainv,    // [n*n] output (shared memory)
    const T* __restrict__ A, // [n*n] input  (shared memory)
    T* __restrict__ work,    // [n*2n] augmented matrix workspace (shared memory)
    int n)
{
    int tid = threadIdx.x;
    int n2 = 2 * n;

    // Build augmented matrix [A | I] in work (parallel)
    for (int idx = tid; idx < n * n2; idx += blockDim.x) {
        int i = idx / n2;
        int j = idx % n2;
        if (j < n) {
            work[i * n2 + j] = A[i * n + j];
        } else {
            work[i * n2 + j] = (j - n == i) ? T(1) : T(0);
        }
    }
    __syncthreads();

    // Gauss-Jordan elimination (thread 0 only — n is small)
    if (tid == 0) {
        for (int col = 0; col < n; ++col) {
            // Partial pivoting: find max in column
            int max_row = col;
            T max_val = fabs(work[col * n2 + col]);
            for (int r = col + 1; r < n; ++r) {
                T v = fabs(work[r * n2 + col]);
                if (v > max_val) { max_val = v; max_row = r; }
            }
            // Swap rows
            if (max_row != col) {
                for (int j = 0; j < n2; ++j) {
                    T tmp = work[col * n2 + j];
                    work[col * n2 + j] = work[max_row * n2 + j];
                    work[max_row * n2 + j] = tmp;
                }
            }
            // Scale pivot row
            T pivot = work[col * n2 + col];
            T inv_pivot = T(1) / pivot;
            for (int j = 0; j < n2; ++j)
                work[col * n2 + j] *= inv_pivot;
            // Eliminate column in all other rows
            for (int r = 0; r < n; ++r) {
                if (r == col) continue;
                T factor = work[r * n2 + col];
                for (int j = 0; j < n2; ++j)
                    work[r * n2 + j] -= factor * work[col * n2 + j];
            }
        }
    }
    __syncthreads();

    // Extract inverse from right half (parallel)
    for (int idx = tid; idx < n * n; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        Ainv[i * n + j] = work[i * n2 + n + j];
    }
    __syncthreads();
}

// OSQP-style adaptive rho:
//   ratio = sqrt(p_norm / d_norm)
//   rho'  = clip(rho_old * ratio, [rho_min, rho_max])
//   returns max(rho'/rho_old, rho_old/rho') >= tolerance; writes rho' out.
template<typename T>
__device__ inline bool computeRhoCandidateDevice(
    T p_norm, T d_norm, T rho_old,
    T rho_min, T rho_max, T tolerance,
    T* __restrict__ rho_new_out)
{
    T ratio = sqrt(p_norm / (d_norm + T(1e-30)));
    T rho_candidate = rho_old * ratio;
    if (rho_candidate < rho_min) rho_candidate = rho_min;
    if (rho_candidate > rho_max) rho_candidate = rho_max;

    T rhos_ratio = rho_candidate / rho_old;
    if (rhos_ratio < T(1)) rhos_ratio = T(1) / rhos_ratio;

    *rho_new_out = rho_candidate;
    return rhos_ratio >= tolerance;
}

// Rebuild block-tridiagonal Schur S (+ theta_inv) per batch for a new rho.
//   Pass 1: Dtilde[t] = D[t] + sigma*I + rho_bar*GtG[t] + rho_f*A^TA
//           → S[t,:,n:2n] = Dtilde[t]; theta_inv[t] = Dtilde[t]^-1
//   Pass 2: Etilde[t] = E[t] + rho_f * Ap[t]^T @ Am[t]
//           → S[t+1,:,0:n] (lower), S[t,:,2n:3n] (upper)
// Scratch: dtilde_sh [n*n], gauss_work [n*2n] (for invertSmallMatrixDevice).
template<typename T>
__device__ void recomputeSDevice(
    T* __restrict__ S_work_b,       // [T, n, 3n] writable
    const T* __restrict__ D_g,      // [T, n, n]
    const T* __restrict__ E_g,      // [N, n, n]
    const T* __restrict__ A0_g,     // [n0, n]
    const T* __restrict__ Am_g,     // [N, nx, n]
    const T* __restrict__ Ap_g,     // [N, nx, n]
    const T* __restrict__ G_g,      // [T, m, n]
    T* __restrict__ theta_inv_gm,   // [T*n*n] global memory
    T* __restrict__ dtilde_sh,      // [n*n] shared scratch
    T* __restrict__ gauss_work,     // [n*2n] shared scratch
    T rho_bar_new, T rho_f_new, T sigma,
    int T_blocks, int n, int nx, int n0, int m,
    bool want_theta_inv)
{
    int tid = threadIdx.x;
    int N = T_blocks - 1;
    int nn = n * n;
    int n3 = 3 * n;

    // ---- Pass 1: Dtilde[t] → theta_inv[t] + S diagonal ----
    for (int t = 0; t < T_blocks; ++t) {
        for (int idx = tid; idx < nn; idx += blockDim.x) {
            int i = idx / n;
            int j = idx % n;

            T val = D_g[t * nn + i * n + j];
            if (i == j) val += sigma;

            if (m > 0) {
                T gtg = T(0);
                const T* G_t = G_g + t * m * n;
                for (int k = 0; k < m; ++k)
                    gtg += G_t[k * n + i] * G_t[k * n + j];
                val += rho_bar_new * gtg;
            }

            T ata = T(0);
            if (t == 0) {
                for (int k = 0; k < n0; ++k)
                    ata += A0_g[k * n + i] * A0_g[k * n + j];
                if (N > 0) {
                    const T* Am0 = Am_g;
                    for (int k = 0; k < nx; ++k)
                        ata += Am0[k * n + i] * Am0[k * n + j];
                }
            } else if (t < N) {
                const T* Ap_tm1 = Ap_g + (t - 1) * nx * n;
                for (int k = 0; k < nx; ++k)
                    ata += Ap_tm1[k * n + i] * Ap_tm1[k * n + j];
                const T* Am_t = Am_g + t * nx * n;
                for (int k = 0; k < nx; ++k)
                    ata += Am_t[k * n + i] * Am_t[k * n + j];
            } else {
                if (N > 0) {
                    const T* Ap_Nm1 = Ap_g + (N - 1) * nx * n;
                    for (int k = 0; k < nx; ++k)
                        ata += Ap_Nm1[k * n + i] * Ap_Nm1[k * n + j];
                }
            }
            val += rho_f_new * ata;

            if (want_theta_inv) {
                dtilde_sh[idx] = val;
            } else {
                // cuDSS path: S is factorised directly; theta_inv is never
                // read, so write Dtilde straight to the S diagonal and skip
                // both the shared staging buffer and the Gauss-Jordan
                // inversion. The kernel then needs zero dynamic shared
                // memory and works at any dimension.
                S_work_b[t * n * n3 + i * n3 + n + j] = val;
            }
        }

        if (want_theta_inv) {
            __syncthreads();

            for (int idx = tid; idx < nn; idx += blockDim.x) {
                int i = idx / n;
                int j = idx % n;
                S_work_b[t * n * n3 + i * n3 + n + j] = dtilde_sh[idx];
            }

            invertSmallMatrixDevice(theta_inv_gm + t * nn, dtilde_sh, gauss_work, n);
            // __syncthreads() is inside invertSmallMatrixDevice
        }
    }

    // ---- Pass 2: Etilde → S off-diagonals ----
    for (int idx = tid; idx < nn; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        S_work_b[i * n3 + j] = T(0);
    }
    for (int idx = tid; idx < nn; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        S_work_b[(T_blocks - 1) * n * n3 + i * n3 + 2 * n + j] = T(0);
    }
    __syncthreads();

    for (int t = 0; t < N; ++t) {
        for (int idx = tid; idx < nn; idx += blockDim.x) {
            int i = idx / n;
            int j = idx % n;

            T val = E_g[t * nn + i * n + j];

            T ap_am = T(0);
            const T* Ap_t = Ap_g + t * nx * n;
            const T* Am_t = Am_g + t * nx * n;
            for (int k = 0; k < nx; ++k)
                ap_am += Ap_t[k * n + i] * Am_t[k * n + j];
            val += rho_f_new * ap_am;

            S_work_b[(t + 1) * n * n3 + i * n3 + j] = val;
        }
        __syncthreads();
    }

    for (int t = 0; t < N; ++t) {
        for (int idx = tid; idx < nn; idx += blockDim.x) {
            int i = idx / n;
            int j = idx % n;
            T etilde_ji = S_work_b[(t + 1) * n * n3 + j * n3 + i];
            S_work_b[t * n * n3 + i * n3 + 2 * n + j] = etilde_ji;
        }
        __syncthreads();
    }
}

// Pass 3 of the Schur recompute: build the PCG block-Jacobi preconditioner
// Phi-inverse from S and theta_inv. cuDSS does not need this pass.
template<typename T>
__device__ void recomputePhiinvFromSDevice(
    T* __restrict__ Phiinv_work_b,         // [T, n, 3n] writable
    const T* __restrict__ S_work_b,        // [T, n, 3n] (Etilde in lower off-diags)
    const T* __restrict__ theta_inv_gm,    // [T*n*n]
    int T_blocks, int n)
{
    int tid = threadIdx.x;
    int N = T_blocks - 1;
    int nn = n * n;
    int n3 = 3 * n;

    for (int t = 0; t < T_blocks; ++t) {
        const T* ti_t = theta_inv_gm + t * nn;

        for (int idx = tid; idx < n * n3; idx += blockDim.x) {
            int i = idx / n3;
            int j = idx % n3;

            T val = T(0);

            if (j < n) {
                if (t > 0) {
                    const T* ti_tm1 = theta_inv_gm + (t - 1) * nn;
                    for (int p = 0; p < n; ++p) {
                        T ti_ip = ti_t[i * n + p];
                        for (int q = 0; q < n; ++q) {
                            T etilde_pq = S_work_b[t * n * n3 + p * n3 + q];
                            val += ti_ip * etilde_pq * ti_tm1[q * n + j];
                        }
                    }
                }
            } else if (j < 2 * n) {
                int jj = j - n;
                val = -ti_t[i * n + jj];
            } else {
                int jj = j - 2 * n;
                if (t < N) {
                    const T* ti_tp1 = theta_inv_gm + (t + 1) * nn;
                    for (int p = 0; p < n; ++p) {
                        T ti_ip = ti_t[i * n + p];
                        for (int q = 0; q < n; ++q) {
                            T etilde_qp = S_work_b[(t + 1) * n * n3 + q * n3 + p];
                            val += ti_ip * etilde_qp * ti_tp1[q * n + jj];
                        }
                    }
                }
            }

            Phiinv_work_b[t * n * n3 + i * n3 + j] = val;
        }
        __syncthreads();
    }
}

// Full Schur + Phi-inverse rebuild for the fused PCG path.
// cuDSS-loop callers should call recomputeSDevice directly to skip Pass 3.
template<typename T>
__device__ void recomputeSchurDevice(
    T* __restrict__ S_work_b,
    T* __restrict__ Phiinv_work_b,
    const T* __restrict__ D_g,
    const T* __restrict__ E_g,
    const T* __restrict__ A0_g,
    const T* __restrict__ Am_g,
    const T* __restrict__ Ap_g,
    const T* __restrict__ G_g,
    T* __restrict__ theta_inv_gm,
    T* __restrict__ dtilde_sh,
    T* __restrict__ gauss_work,
    T rho_bar_new, T rho_f_new, T sigma,
    int T_blocks, int n, int nx, int n0, int m)
{
    recomputeSDevice<T>(
        S_work_b, D_g, E_g, A0_g, Am_g, Ap_g, G_g,
        theta_inv_gm, dtilde_sh, gauss_work,
        rho_bar_new, rho_f_new, sigma,
        T_blocks, n, nx, n0, m,
        /*want_theta_inv=*/true);
    recomputePhiinvFromSDevice<T>(
        Phiinv_work_b, S_work_b, theta_inv_gm, T_blocks, n);
}
