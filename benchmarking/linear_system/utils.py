"""Utilities"""
import numpy as np

# Problem dimensions
N_STATE = 8
N_CTRL = 4

# Randomness
INITIAL_STATE_STDEV = 5.0

# Solver settings
# mpc.pytorch
MPC_LQR_ITER = 1  # mpc.pytorch iterations
MPC_TOL = 1e-3  # mpc.pytorch tolerance
MPC_MAX_LINES = 1  # mpc.pytorch max linesearch iterations
# turbompc
TURBOMPC_SQP_ITER = 1  # turbompc SQP iterations
# trajax
TRAJAX_LQR_ITER = 1  # trajax iLQR iterations
TRAJAX_ALPHA_MIN = 0.99  # trajax iLQR alpha min
# theseus
THESEUS_MAX_ITER = 1  # theseus max iterations

# Debug flag for printing gradient values
DEBUGD = False


def generate_problem_data(N_BATCH, seed, n_state=None, n_ctrl=None):
    """Generate problem data (matrices and initial states) for benchmarking"""
    nx = n_state if n_state is not None else N_STATE
    nu = n_ctrl if n_ctrl is not None else N_CTRL

    # Set seeds for reproducibility
    np.random.seed(seed)

    # Cost matrices (identity)
    Q = np.eye(nx, dtype=np.double)
    R = np.eye(nu, dtype=np.double)

    # Create a stable A matrix with eigenvalues strictly inside unit circle
    A_base = np.eye(nx) + 0.1 * np.random.randn(nx, nx)

    # Restrict eigenvalues of A to be less than 1. Based on
    # https://github.com/osqp/osqp_benchmarks/blob/master/problem_classes/control.py.
    lambda_values, V = np.linalg.eig(A_base)
    abs_lambda_values = np.abs(lambda_values)
    for i in range(len(lambda_values)):
        lambda_values[i] = (
            lambda_values[i]
            if abs_lambda_values[i] < 1 - 1e-02
            else lambda_values[i] / (abs_lambda_values[i] + 1e-02)
        )
    # Reconstruct A = V * Lambda * V^{-1}
    A_base = (V @ np.diag(lambda_values) @ np.linalg.inv(V)).real

    B_base = np.random.randn(nx, nu)
    b_base = 0.01 * np.random.randn(nx)

    # Initial states (batch)
    x0 = np.random.randn(N_BATCH, nx) * INITIAL_STATE_STDEV

    return Q, R, A_base, B_base, b_base, x0
