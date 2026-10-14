"""Unit tests for OCPProcessor and SolverProcessor.

Covers:
- Default construction of specs
- __post_init__ validation with helpful error messages
- enumerate() cartesian product
- build() produces working instances
"""
import pytest

from tests.cuda.helpers.processors import (
    OCPProcessor,
    OCPSpec,
    SolverProcessor,
    SolverSpec,
)
from tests.helpers.backend_utils import backend_available
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.linesearch import LinesearchMethod
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    HessianApproximation,
    TurboMPCSolver,
)


class TestOCPSpecValidation:
    def test_defaults_valid(self):
        spec = OCPSpec()
        assert spec.dynamics == "spacecraft"
        assert spec.horizon == 10

    def test_invalid_ocp_variant(self):
        with pytest.raises(ValueError, match="Unknown ocp_variant"):
            OCPSpec(ocp_variant="typo")

    def test_invalid_dynamics(self):
        with pytest.raises(ValueError, match="Unknown dynamics"):
            OCPSpec(dynamics="invalid_dyn")

    def test_invalid_discretization(self):
        with pytest.raises(ValueError, match="Unknown discretization"):
            OCPSpec(discretization="verlet")

    def test_obstacle_requires_drone(self):
        with pytest.raises(ValueError, match="requires dynamics='drone'"):
            OCPSpec(
                ocp_variant="obstacle",
                dynamics="spacecraft",
                obstacle_centers=((0.0, 0.0),),
                obstacle_radii=(0.1,),
            )

    def test_obstacle_requires_non_empty_centers(self):
        with pytest.raises(ValueError, match="obstacle_centers"):
            OCPSpec(ocp_variant="obstacle", dynamics="drone")

    def test_rescaling_mode_requires_rescale(self):
        with pytest.raises(ValueError, match="rescaling_mode"):
            OCPSpec(rescaling_mode="unit", rescale=False)

    def test_rescale_requires_mode(self):
        with pytest.raises(ValueError, match="rescale=True"):
            OCPSpec(rescale=True, rescaling_mode="none")

    def test_horizon_too_small(self):
        with pytest.raises(ValueError, match="horizon"):
            OCPSpec(horizon=1)

    def test_pytest_id_readable(self):
        spec = OCPSpec(
            dynamics="spacecraft",
            horizon=10,
            discretization="euler",
            bounds_mode="control_only",
        )
        assert spec.pytest_id() == "spacecraft-H10-euler-b-control_only"


class TestSolverSpecValidation:
    def test_defaults_valid(self):
        spec = SolverSpec()
        assert spec.forward_backend == ForwardBackend.ADMM_JAX_LOOP_PCG
        assert spec.effective_backward == BackwardBackend.ADMM_JAX_LOOP_PCG
        assert spec.linesearch_method == LinesearchMethod.NONE

    def test_wrong_forward_type(self):
        with pytest.raises(ValueError, match="forward_backend must be ForwardBackend"):
            SolverSpec(forward_backend="admm_jax_loop_pcg")

    def test_wrong_linesearch_type(self):
        with pytest.raises(ValueError, match="linesearch_method"):
            SolverSpec(linesearch_method="filter")

    def test_linesearch_method_in_pytest_id(self):
        spec = SolverSpec(linesearch_method=LinesearchMethod.FILTER)
        assert spec.pytest_id().endswith("ls-filter")

    def test_full_hessian_requires_direct_backward(self):
        # default backward for ADMM_JAX_LOOP_PCG is ADMM_JAX_LOOP_PCG (not direct)
        with pytest.raises(ValueError, match="FULL_HESSIAN backward requires"):
            SolverSpec(
                forward_backend=ForwardBackend.ADMM_JAX_LOOP_PCG,
                backward_hessian=HessianApproximation.FULL_HESSIAN,
            )

    def test_full_hessian_with_direct_cudss_ok(self):
        SolverSpec(
            forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
            backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
            backward_hessian=HessianApproximation.FULL_HESSIAN,
        )  # no error

    def test_pytest_id_includes_hessian_modes(self):
        spec = SolverSpec(
            forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
            backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
            forward_hessian=HessianApproximation.FULL_HESSIAN_PROJECTED,
            backward_hessian=HessianApproximation.FULL_HESSIAN,
        )
        assert (
            spec.pytest_id()
            == "ADMM_FUSED_CUDSS-fwh-FULL_HESSIAN_PROJECTED-bwh-FULL_HESSIAN"
        )

    def test_rho_range(self):
        with pytest.raises(ValueError, match="admm_rho_min"):
            SolverSpec(admm_rho_min=1.0, admm_rho_max=0.5)

    def test_sigma_positive(self):
        with pytest.raises(ValueError, match="admm_sigma"):
            SolverSpec(admm_sigma=0.0)

    def test_eps_positive(self):
        with pytest.raises(ValueError, match="admm_eps"):
            SolverSpec(admm_eps_abs=-1e-4)


class TestEnumerate:
    def test_single_filter(self):
        specs = OCPProcessor.enumerate(horizon=[5, 10, 25])
        assert len(specs) == 3
        assert [s.horizon for s in specs] == [5, 10, 25]

    def test_cartesian(self):
        specs = OCPProcessor.enumerate(
            dynamics=["spacecraft", "linear"],
            horizon=[5, 10],
        )
        assert len(specs) == 4

    def test_plural_alias(self):
        specs = OCPProcessor.enumerate(horizons=[5, 10])
        assert len(specs) == 2
        assert specs[0].horizon == 5

    def test_unknown_filter_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            OCPProcessor.enumerate(bogus_field=[1, 2])

    def test_scalar_value(self):
        specs = OCPProcessor.enumerate(dynamics="linear", horizon=10)
        assert len(specs) == 1
        assert specs[0].dynamics == "linear"


class TestOCPBuild:
    def test_spacecraft_default(self):
        ocp, params = OCPProcessor.build(OCPSpec(dynamics="spacecraft", horizon=5))
        assert isinstance(ocp, OptimalControlProblem)
        assert params["horizon"] == 5
        assert params["discretization_scheme"] == 0  # euler

    def test_linear_implicit(self):
        ocp, params = OCPProcessor.build(
            OCPSpec(dynamics="linear", horizon=5, discretization="implicit")
        )
        assert isinstance(ocp, OptimalControlProblem)
        assert params["discretization_scheme"] == 10

    def test_obstacle_variant_has_no_solver_settings(self):
        ocp, params = OCPProcessor.build(
            OCPSpec(
                dynamics="drone",
                horizon=5,
                ocp_variant="obstacle",
                obstacle_centers=((-1.4, -0.1),),
                obstacle_radii=(0.3,),
            )
        )
        assert isinstance(ocp, OptimalControlProblemObstacle)
        assert params["use_slack_variables"] is False
        assert params["slack_penalization_weight"] == 0.0

    def test_control_only_bounds(self):
        ocp, params = OCPProcessor.build(
            OCPSpec(dynamics="spacecraft", horizon=5, bounds_mode="control_only")
        )
        # When bounds_mode='control_only', state bounds should be None/wide
        assert ocp is not None


class TestSolverBuild:
    def test_default_solver(self):
        ocp, params = OCPProcessor.build(OCPSpec(horizon=5))
        solver = SolverProcessor.build(ocp, params, SolverSpec())
        assert isinstance(solver, TurboMPCSolver)

    def test_fused_cudss_with_full_hessian(self):
        if not backend_available(ForwardBackend.ADMM_FUSED_CUDSS):
            pytest.skip("ADMM_FUSED_CUDSS not built/available")
        if not backend_available(BackwardBackend.DIRECT_CUDSS_FFI):
            pytest.skip("DIRECT_CUDSS_FFI not built/available")
        ocp, params = OCPProcessor.build(OCPSpec(horizon=5))
        spec = SolverSpec(
            forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
            backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
            backward_hessian=HessianApproximation.FULL_HESSIAN,
        )
        solver = SolverProcessor.build(ocp, params, spec)
        assert solver._backward_hessian == HessianApproximation.FULL_HESSIAN


class TestParametrize:
    def test_ocp_parametrize_returns_params(self):
        params = OCPProcessor.parametrize(horizon=[5, 10])
        assert len(params) == 2
        # Each is a pytest.param with an id
        assert all(hasattr(p, "id") for p in params)

    def test_solver_parametrize_auto_skip(self):
        # Unavailable backend should be marked as skip, not raise
        params = SolverProcessor.parametrize(
            forward_backend=[ForwardBackend.ADMM_JAX_LOOP_PCG]
        )
        assert len(params) == 1


@pytest.mark.parametrize(
    "ocp_spec",
    OCPProcessor.parametrize(
        dynamics=["spacecraft", "linear"],
        horizon=[5, 10],
    ),
)
def test_processor_parametrize_works_in_test(ocp_spec):
    """Sanity check that parametrize produces usable OCPSpec instances."""
    assert isinstance(ocp_spec, OCPSpec)
    ocp, params = OCPProcessor.build(ocp_spec)
    assert params["horizon"] == ocp_spec.horizon
