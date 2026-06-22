import jax.numpy as jnp
from tests.helpers.problem_fixtures import make_spacecraft_params
from tests.helpers.solver_fixtures import turbompc_solver_params
from turbompc.dynamics.spacecraft_dynamics import SpacecraftDynamics
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.admm.admm import compute_S_Phiinv
from turbompc.solvers.turbompc_solver import TurboMPCSolver


def make_spacecraft_schur_system(horizon: int = 10):
    params = make_spacecraft_params(horizon=horizon)
    ocp = OptimalControlProblem(dynamics=SpacecraftDynamics(), params=params)
    solver = TurboMPCSolver(ocp, params=turbompc_solver_params())
    guess = solver.initial_guess(params)
    qp_data = solver._build_qp_data(guess.states, guess.controls, params)
    schur = compute_S_Phiinv(qp_data, rho_f=1.0, sigma=1e-6, rho_ineq=0.1)
    T = horizon + 1
    n = ocp.num_state_variables + ocp.num_control_variables
    gamma = jnp.ones((T, n))
    return schur, gamma
