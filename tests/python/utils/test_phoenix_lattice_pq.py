"""Unit tests for zero-float p/q and Laser Phoenix Lattice Locking in TurboMPC."""

import unittest

from turbompc.utils.phoenix_lattice_pq import (
    Q16_ONE,
    PhoenixInterferenceState,
    PhoenixPhaseLatticePllPq,
    classify_phoenix_lattice_z32,
    isqrt_pq,
    phoenix_phase_from_ratio_z32,
    phoenix_shortest_delta_z32,
    restore_mpc_trajectory_dropouts_pq,
    update_admm_rho_phoenix_pq,
)


class TestPhoenixLatticePq(unittest.TestCase):
    def test_isqrt_pq_exact(self) -> None:
        for n in [0, 1, 2, 3, 4, 15, 16, 17, 65535, 65536, 10**12]:
            r = isqrt_pq(n)
            self.assertLessEqual(r * r, n)
            self.assertGreater((r + 1) * (r + 1), n)

    def test_phoenix_lattice_states_and_seam_wrapping(self) -> None:
        p0 = phoenix_phase_from_ratio_z32(0, 360)
        p45 = phoenix_phase_from_ratio_z32(45, 360)
        p90 = phoenix_phase_from_ratio_z32(90, 360)
        p135 = phoenix_phase_from_ratio_z32(135, 360)
        p180 = phoenix_phase_from_ratio_z32(180, 360)
        p359 = phoenix_phase_from_ratio_z32(359, 360)
        p1 = phoenix_phase_from_ratio_z32(1, 360)

        self.assertEqual(
            classify_phoenix_lattice_z32(phoenix_shortest_delta_z32(p0, p0)),
            PhoenixInterferenceState.CONSTRUCTIVE,
        )
        self.assertEqual(
            classify_phoenix_lattice_z32(phoenix_shortest_delta_z32(p45, p0)),
            PhoenixInterferenceState.PARTIAL_CONSTRUCTIVE,
        )
        self.assertEqual(
            classify_phoenix_lattice_z32(phoenix_shortest_delta_z32(p90, p0)),
            PhoenixInterferenceState.QUADRATURE,
        )
        self.assertEqual(
            classify_phoenix_lattice_z32(phoenix_shortest_delta_z32(p135, p0)),
            PhoenixInterferenceState.PARTIAL_DESTRUCTIVE,
        )
        self.assertEqual(
            classify_phoenix_lattice_z32(phoenix_shortest_delta_z32(p180, p0)),
            PhoenixInterferenceState.DESTRUCTIVE,
        )

        # Across the 359 deg -> 1 deg seam, shortest delta is +2 deg (CONSTRUCTIVE)
        delta_seam = phoenix_shortest_delta_z32(p1, p359)
        self.assertGreater(delta_seam, 0)
        self.assertEqual(
            classify_phoenix_lattice_z32(delta_seam),
            PhoenixInterferenceState.CONSTRUCTIVE,
        )

    def test_phoenix_pll_tracks_across_pi_seam(self) -> None:
        pll = PhoenixPhaseLatticePllPq()
        for deg in [170, 175, 179, -177, -173, -169]:
            z32 = phoenix_phase_from_ratio_z32(deg, 360)
            locked, state = pll.step(z32)
            self.assertEqual(state, PhoenixInterferenceState.CONSTRUCTIVE)
            residual = abs(phoenix_shortest_delta_z32(z32, locked))
            self.assertLess(residual, phoenix_phase_from_ratio_z32(5, 360))

    def test_update_admm_rho_phoenix_pq(self) -> None:
        # r_p == 4 * r_d => sqrt(r_p / r_d) == 2 => rho doubles
        rho_new = update_admm_rho_phoenix_pq(
            rho_bar_q16=Q16_ONE,
            primal_res_norm_q16=4 * Q16_ONE,
            dual_res_norm_q16=Q16_ONE,
        )
        self.assertEqual(rho_new, 2 * Q16_ONE)

        # Zero dual residual clips safely to rho_max without ZeroDivisionError / Inf
        rho_clipped = update_admm_rho_phoenix_pq(
            rho_bar_q16=Q16_ONE,
            primal_res_norm_q16=Q16_ONE,
            dual_res_norm_q16=0,
            rho_min_q16=10,
            rho_max_q16=500000,
        )
        self.assertEqual(rho_clipped, 500000)

    def test_restore_mpc_trajectory_dropouts_exact_quadratic(self) -> None:
        # 2D quadratic state trajectory x(t) = [2*t^2 - 5*t + 100, -3*t^2 + 11*t - 20]
        true_traj = [
            [2 * t * t - 5 * t + 100, -3 * t * t + 11 * t - 20] for t in range(12)
        ]
        corrupted = [list(row) for row in true_traj]
        valid_mask = [True] * 12
        for dropped in [4, 5, 6]:
            corrupted[dropped] = [-999999, 999999]
            valid_mask[dropped] = False

        restored = restore_mpc_trajectory_dropouts_pq(corrupted, valid_mask)
        self.assertEqual(restored, true_traj)


if __name__ == "__main__":
    unittest.main()
