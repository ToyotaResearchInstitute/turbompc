#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Configuration for the host-loop ADMM + cuDSS direct-solve kernel.
struct ADMMCudssConfig {
    int32_t max_iter;            // Maximum ADMM iterations
    int32_t check_every;         // Check convergence every N iterations
    int32_t adapt_rho_every;     // OSQP-style rho adaptation period (0 disables)
    double  eps_abs;             // Absolute convergence tolerance
    double  eps_rel;             // Relative convergence tolerance
    double  sigma;               // Proximal regularization
    double  rho_f_factor;        // rho_f = rho_bar * rho_f_factor
    double  alpha;               // Over-relaxation parameter
    double  adaptive_rho_tolerance;  // OSQP rho-update gate (>= this triggers refactor)
    double  rho_min;             // Clamp for adaptive rho
    double  rho_max;             // Clamp for adaptive rho
    bool    use_slack;           // use_slack_variables flag
};

// ---- f32 launcher ----
void LaunchADMMCudssF32(
    cudaStream_t stream,
    // Outputs (10 buffers)
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
    // QP data
    const float* S,            // [Nb, T, n, 3n]
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
    // Warm-start (7 read-only buffers)
    const float* x0,           // [Nb, T, n]
    const float* z_g0,         // [Nb, T, m]
    const float* y_g0,         // [Nb, T, m]
    const float* y_f_0_init,   // [Nb, n0]
    const float* y_f_dyn_init, // [Nb, N, nx]
    const float* xi_g0,        // [Nb, T, m]
    const float* rho_bar_init, // [Nb]
    const float* slack_weight_init, // [1] scalar on device
    // Dimensions
    int32_t T,
    int32_t n,
    int32_t nx,
    int32_t n0,
    int32_t m,
    int32_t Nb,
    // Config
    ADMMCudssConfig cfg);

// ---- f64 launcher ----
void LaunchADMMCudssF64(
    cudaStream_t stream,
    // Outputs
    double*   x_out,
    uint32_t* iters_out,
    double*   x_blocks_out,
    double*   z_g_out,
    double*   y_g_out,
    double*   y_f_0_out,       // [Nb, n0]
    double*   y_f_dyn_out,     // [Nb, N, nx]
    double*   xi_g_out,
    double*   rho_bar_out,
    double*   kernel_ns_out,   // [Nb]  GPU kernel time in nanoseconds
    // QP data
    const double* S,
    const double* D,
    const double* E,
    const double* q,
    const double* A0,          // [Nb, n0, n]
    const double* A_minus,
    const double* A_plus,
    const double* G,
    const double* l_bounds,
    const double* u_bounds,
    const double* c0,          // [Nb, n0]
    const double* c_dyn,
    // Warm-start
    const double* x0,
    const double* z_g0,
    const double* y_g0,
    const double* y_f_0_init,  // [Nb, n0]
    const double* y_f_dyn_init,
    const double* xi_g0,
    const double* rho_bar_init,
    const double* slack_weight_init, // [1] scalar on device
    // Dimensions
    int32_t T,
    int32_t n,
    int32_t nx,
    int32_t n0,
    int32_t m,
    int32_t Nb,
    // Config
    ADMMCudssConfig cfg);
