#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Configuration for the fused ADMM kernel.
struct ADMMConfig {
    int32_t max_iter;        // Maximum ADMM iterations
    int32_t pcg_max_iter;    // Maximum PCG iterations per ADMM step
    int32_t check_every;     // Check convergence every N iterations
    double  eps_abs;         // Absolute convergence tolerance
    double  eps_rel;         // Relative convergence tolerance
    double  sigma;           // Proximal regularization
    double  rho_f_factor;    // rho_f = rho_bar * rho_f_factor
    double  alpha;           // Over-relaxation parameter
    double  pcg_eps;         // PCG convergence tolerance
    bool    adapt_rho;       // Whether to adapt rho (legacy, use adapt_rho_every > 0)
    bool    use_slack;       // use_slack_variables flag
    int32_t adapt_rho_every;         // Adapt rho every N iters (0 = disabled)
    double  adaptive_rho_tolerance;  // Min ratio to trigger update (e.g. 5.0)
    double  rho_min;                 // Lower bound on rho_bar (e.g. 1e-6)
    double  rho_max;                 // Upper bound on rho_bar (e.g. 1e6)
};

// ---- f32 launcher ----
cudaError_t LaunchADMMFusedF32(
    cudaStream_t stream,
    // Outputs
    float*    x_out,           // [Nb, T, n]
    uint32_t* iters_out,       // [Nb]
    float*    x_blocks_out,    // [Nb, T, n]   warm-start output
    float*    z_g_out,         // [Nb, T, m]
    float*    y_g_out,         // [Nb, T, m]
    float*    y_f_0_out,       // [Nb, n0]
    float*    y_f_dyn_out,     // [Nb, N, nx]
    float*    xi_g_out,        // [Nb, T, m]
    float*    rho_bar_out,     // [Nb]
    float*    kernel_ns_out,   // [Nb]  GPU kernel time in nanoseconds
    float*    S_work,          // [Nb, T, n, 4n]  writable workspace for S + theta_inv
    float*    Phiinv_work,     // [Nb, T, n, 3n]  writable workspace for Phiinv
    // QP data (read-only)
    const float* S,            // [Nb, T, n, 3n]
    const float* Phiinv,       // [Nb, T, n, 3n]
    const float* D,            // [Nb, T, n, n]
    const float* E,            // [Nb, N, n, n]
    const float* q,            // [Nb, T, n]
    const float* A0,           // [Nb, n0, n]
    const float* A_minus,      // [Nb, N, nx, n]
    const float* A_plus,       // [Nb, N, nx, n]
    const float* G,            // [Nb, T, m, n]
    const float* l_bounds,     // [Nb, T, m]
    const float* u_bounds,     // [Nb, T, m]
    const float* c0,           // [Nb, n0]
    const float* c_dyn,        // [Nb, N, nx]
    // Warm-start (read-only)
    const float* x0,           // [Nb, T, n]
    const float* z_g0,         // [Nb, T, m]
    const float* y_g0,         // [Nb, T, m]
    const float* y_f_0_init,   // [Nb, n0]
    const float* y_f_dyn_init, // [Nb, N, nx]
    const float* xi_g0,        // [Nb, T, m]
    const float* rho_bar_init, // [Nb]
    const float* slack_weight_init, // [Nb] runtime (JAX-traceable)
    // Dimensions
    int32_t T,
    int32_t n,
    int32_t nx,
    int32_t n0,
    int32_t m,
    int32_t Nb,
    // Config
    ADMMConfig cfg);

// ---- f64 launcher ----
cudaError_t LaunchADMMFusedF64(
    cudaStream_t stream,
    // Outputs
    double*   x_out,
    uint32_t* iters_out,
    double*   x_blocks_out,
    double*   z_g_out,
    double*   y_g_out,
    double*   y_f_0_out,
    double*   y_f_dyn_out,
    double*   xi_g_out,
    double*   rho_bar_out,
    double*   kernel_ns_out,   // [Nb]  GPU kernel time in nanoseconds
    double*   S_work,          // [Nb, T, n, 4n]  writable workspace for S + theta_inv
    double*   Phiinv_work,     // [Nb, T, n, 3n]  writable workspace for Phiinv
    // QP data (read-only)
    const double* S,
    const double* Phiinv,
    const double* D,
    const double* E,
    const double* q,
    const double* A0,
    const double* A_minus,
    const double* A_plus,
    const double* G,
    const double* l_bounds,
    const double* u_bounds,
    const double* c0,
    const double* c_dyn,
    // Warm-start (read-only)
    const double* x0,
    const double* z_g0,
    const double* y_g0,
    const double* y_f_0_init,
    const double* y_f_dyn_init,
    const double* xi_g0,
    const double* rho_bar_init,
    const double* slack_weight_init,
    // Dimensions
    int32_t T,
    int32_t n,
    int32_t nx,
    int32_t n0,
    int32_t m,
    int32_t Nb,
    // Config
    ADMMConfig cfg);
