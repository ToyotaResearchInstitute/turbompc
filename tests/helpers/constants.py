from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend
from turbompc.solvers.turbompc_solver import ForwardBackend

ADMM_EPS_BY_BACKEND = {
    SchurSolverBackend.PCG: 1.0e-10,
    SchurSolverBackend.PCG_FFI: 1.0e-7,
    SchurSolverBackend.CUDSS_FFI: 1.0e-10,
}

# Forward-backend-level tolerances (fused kernels use their own convergence)
ADMM_EPS_BY_FORWARD_BACKEND = {
    ForwardBackend.ADMM_JAX_LOOP_PCG: 1.0e-6,
    ForwardBackend.ADMM_JAX_LOOP_PCG_FFI: 1.0e-6,
    ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI: 1.0e-10,
    ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE: 1.0e-10,
    ForwardBackend.ADMM_FUSED_PCG: 1.0e-4,
    ForwardBackend.ADMM_FUSED_CUDSS: 1.0e-4,
}

OSQP_EPS = 1.0e-10
EQ_TOL = 1.0e-4
Z_TOL = 2.0e-3
COST_TOL = 1.0e-4
# Fused/cuDSS-loop backends use looser ADMM tolerances, so solution
# accuracy is lower. Use relaxed thresholds for these.
Z_TOL_FUSED = 5.0e-3
COST_TOL_FUSED = 2.0e-3
