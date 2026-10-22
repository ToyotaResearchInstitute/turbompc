"""Zero-float p/q ("Counting Divided") and Laser Phoenix Lattice Locking for TurboMPC.

Provides deterministic, pure-integer (zero-FPU) primitives for Model Predictive
Control (MPC) and ADMM/SQP trajectory optimization:
1. ``isqrt_pq``: Exact integer floor square root ``floor(sqrt(n))``.
2. ``PhoenixPhaseLatticePllPq``: Z_{2^32} modular phase-locked loop with the
   5-state Laser Phoenix Interference Lattice ({0 deg, 45 deg, 90 deg, 135 deg,
   180 deg}), Marine phase-jitter salience gate, and golden-ratio harmonic
   damping (phi = 4181 / 2584) for rotational MPC states (e.g. CartPole theta,
   Quadrotor/Spacecraft yaw/attitude phase) across the +/-pi seam.
3. ``update_admm_rho_phoenix_pq``: Zero-float p/q ADMM penalty (rho) adaptation
   via exact integer square-root ratio ``sqrt(r_p / r_d)`` with zero NaN/Inf risk.
4. ``restore_mpc_trajectory_dropouts_pq``: Exact non-uniform Bessel-Overhauser
   p/q Catmull-Rom cubic Hermite warm-start / sensor-dropout trajectory
   reconstruction across (horizon + 1, dim) state or control matrices (0 error
   on linear and quadratic trajectories).
"""

from __future__ import annotations

import enum
from typing import List, Sequence, Tuple

MOD32_MASK: int = 0xFFFFFFFF
HALF_TURN_Z32: int = 0x80000000
FULL_TURN_Z32: int = 0x100000000

# Golden-ratio harmonic constants (phi ~= 4181 / 2584)
PHOENIX_PHI_NUM_PQ: int = 4181
PHOENIX_PHI_DEN_PQ: int = 2584

# Q16 fixed-point scale (65536 == 1.0)
Q16_ONE: int = 65536


class PhoenixInterferenceState(enum.IntEnum):
    """5-state Laser Phoenix Interference Lattice on Z_{2^32}."""

    DESTRUCTIVE = 0          # ~180 deg opposition (null gain)
    PARTIAL_DESTRUCTIVE = 1  # ~135 deg partial opposition
    QUADRATURE = 2           # ~90 deg orthogonal quadrature
    PARTIAL_CONSTRUCTIVE = 3 # ~45 deg partial alignment
    CONSTRUCTIVE = 4         # ~0 deg phase lock (full alignment)


PHOENIX_LATTICE_GAIN_Q16: Tuple[int, int, int, int, int] = (
    0,      # DESTRUCTIVE
    19220,  # PARTIAL_DESTRUCTIVE
    46341,  # QUADRATURE
    60547,  # PARTIAL_CONSTRUCTIVE
    65536,  # CONSTRUCTIVE
)


def isqrt_pq(n: int) -> int:
    """Compute exact integer floor square root floor(sqrt(n)) using binary shifts."""
    if n < 0:
        raise ValueError(f"isqrt_pq requires non-negative integer, got {n}")
    if n < 2:
        return n
    bit = 1 << ((n.bit_length() - 1) & ~1)
    res = 0
    rem = n
    while bit != 0:
        cand = res + bit
        if rem >= cand:
            rem -= cand
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def div_round_nearest_pq(num: int, den: int) -> int:
    """Deterministic integer division with round-half-away-from-zero."""
    if den <= 0:
        raise ValueError(f"Denominator must be positive, got {den}")
    if num >= 0:
        return (num + (den >> 1)) // den
    return -((-num + (den >> 1)) // den)


def phoenix_phase_from_ratio_z32(numerator: int, denominator: int) -> int:
    """Map a rational turn fraction (numerator / denominator) to Z_{2^32}."""
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    rem = numerator % denominator
    scaled = (rem << 32) + (denominator >> 1)
    return (scaled // denominator) & MOD32_MASK


def phoenix_shortest_delta_z32(target_z32: int, current_z32: int) -> int:
    """Compute shortest signed angular residual (target - current) on Z_{2^32}."""
    diff = (target_z32 - current_z32) & MOD32_MASK
    if diff >= HALF_TURN_Z32:
        return diff - FULL_TURN_Z32
    return diff


def classify_phoenix_lattice_z32(delta_z32: int) -> PhoenixInterferenceState:
    """Classify a signed Z_{2^32} phase residual into the 5-state Phoenix Lattice."""
    abs_delta = abs(delta_z32)
    if abs_delta < 0x10000000:
        return PhoenixInterferenceState.CONSTRUCTIVE
    if abs_delta < 0x30000000:
        return PhoenixInterferenceState.PARTIAL_CONSTRUCTIVE
    if abs_delta < 0x50000000:
        return PhoenixInterferenceState.QUADRATURE
    if abs_delta < 0x70000000:
        return PhoenixInterferenceState.PARTIAL_DESTRUCTIVE
    return PhoenixInterferenceState.DESTRUCTIVE


class PhoenixPhaseLatticePllPq:
    """Z_{2^32} Laser Phoenix Lattice Phase-Locked Loop for rotational MPC states.

    Tracks rotational states across the +/-pi seam in Z_{2^32} with Marine
    phase-jitter salience gating and phi = 4181/2584 harmonic damping.
    """

    def __init__(
        self,
        kp_num: int = 1,
        kp_den: int = 2,
        ki_num: int = 1,
        ki_den: int = 16,
        jitter_gate_z32: int = 0x20000000,
    ) -> None:
        self.kp_num = int(kp_num)
        self.kp_den = int(kp_den)
        self.ki_num = int(ki_num)
        self.ki_den = int(ki_den)
        self.jitter_gate_z32 = int(jitter_gate_z32)
        self.phase_z32: int = 0
        self.omega_z32: int = 0
        self.initialized: bool = False

    def step(self, measured_z32: int) -> Tuple[int, PhoenixInterferenceState]:
        """Advance PLL by one MPC step and return (locked_phase_z32, lattice_state)."""
        measured = measured_z32 & MOD32_MASK
        if not self.initialized:
            self.phase_z32 = measured
            self.omega_z32 = 0
            self.initialized = True
            return self.phase_z32, PhoenixInterferenceState.CONSTRUCTIVE

        predicted = (self.phase_z32 + self.omega_z32) & MOD32_MASK
        err = phoenix_shortest_delta_z32(measured, predicted)
        state = classify_phoenix_lattice_z32(err)
        gain_q16 = PHOENIX_LATTICE_GAIN_Q16[int(state)]

        # Marine encoder/gyro jitter gate: attenuate impulses beyond jitter_gate_z32
        if abs(err) > self.jitter_gate_z32:
            err_gated = div_round_nearest_pq(
                err * PHOENIX_PHI_DEN_PQ, PHOENIX_PHI_NUM_PQ
            )
        else:
            err_gated = err

        p_step = div_round_nearest_pq(
            err_gated * self.kp_num * gain_q16, self.kp_den * Q16_ONE
        )
        i_step = div_round_nearest_pq(
            err_gated * self.ki_num * gain_q16, self.ki_den * Q16_ONE
        )

        self.omega_z32 = div_round_nearest_pq(
            (self.omega_z32 + i_step) * PHOENIX_PHI_NUM_PQ,
            PHOENIX_PHI_NUM_PQ + (PHOENIX_PHI_DEN_PQ >> 5),
        )
        self.phase_z32 = (predicted + p_step) & MOD32_MASK
        return self.phase_z32, state


def update_admm_rho_phoenix_pq(
    rho_bar_q16: int,
    primal_res_norm_q16: int,
    dual_res_norm_q16: int,
    rho_min_q16: int = 1,
    rho_max_q16: int = 65536 * 1000000,
) -> int:
    """Deterministic zero-float p/q ADMM penalty (rho) adaptation.

    Computes ``clip(rho_bar * sqrt(r_p / r_d), rho_min, rho_max)`` in Q16 fixed-point
    using exact integer square roots (`isqrt_pq`), avoiding float division by zero.
    """
    if primal_res_norm_q16 <= 0 and dual_res_norm_q16 <= 0:
        return max(rho_min_q16, min(rho_max_q16, rho_bar_q16))
    if dual_res_norm_q16 <= 0:
        return rho_max_q16
    if primal_res_norm_q16 <= 0:
        return rho_min_q16

    # Scale ratio by 2^32 before integer square root so sqrt_ratio is in Q16 (2^16 == 1.0)
    ratio_q32 = (primal_res_norm_q16 << 32) // dual_res_norm_q16
    sqrt_ratio_q16 = isqrt_pq(ratio_q32)
    rho_candidate_q16 = div_round_nearest_pq(rho_bar_q16 * sqrt_ratio_q16, Q16_ONE)
    return max(rho_min_q16, min(rho_max_q16, rho_candidate_q16))


def catmull_rom_bessel_pq(
    y0: int,
    y1: int,
    y2: int,
    y3: int,
    h0: int,
    h1: int,
    h2: int,
    step_num: int,
) -> int:
    """Evaluate exact non-uniform Bessel-Overhauser p/q Catmull-Rom cubic Hermite spline."""
    if h0 <= 0 or h1 <= 0 or h2 <= 0:
        raise ValueError(f"Step intervals must be positive, got ({h0}, {h1}, {h2})")
    if step_num < 0 or step_num > h1:
        raise ValueError(f"step_num {step_num} must be in [0, {h1}]")

    m1_num = (h1 * h1 * (y1 - y0)) + (h0 * h0 * (y2 - y1))
    m1_den = h0 * (h0 + h1)
    m1 = m1_num // m1_den

    m2_num = ((h2 * h2 * (y2 - y1)) + (h1 * h1 * (y3 - y2))) * h1
    m2_den = h2 * h1 * (h1 + h2)
    m2 = m2_num // m2_den

    p0 = y2 - 2 * m1
    p1 = y1
    p2 = y2
    p3 = y1 + 2 * m2

    a0 = 2 * p1
    a1 = p2 - p0
    a2 = 2 * p0 - 5 * p1 + 4 * p2 - p3
    a3 = -p0 + 3 * p1 - 3 * p2 + p3

    q = h1
    q2 = q * q
    q3 = q2 * q
    p = step_num
    p2_step = p * p
    p3_step = p2_step * p

    num = a0 * q3 + a1 * p * q2 + a2 * p2_step * q + a3 * p3_step
    den = 2 * q3
    return div_round_nearest_pq(num, den)


def restore_mpc_trajectory_dropouts_pq(
    trajectory_q: Sequence[Sequence[int]],
    valid_mask: Sequence[bool],
) -> List[List[int]]:
    """Restore missing steps in an integer-quantized (horizon + 1, dim) MPC trajectory.

    Uses exact non-uniform Bessel-Overhauser p/q Catmull-Rom cubic Hermite splines
    across valid anchor timesteps, reproducing linear and quadratic trajectories with
    0 integer error.
    """
    num_steps = len(trajectory_q)
    if num_steps == 0:
        return []
    if len(valid_mask) != num_steps:
        raise ValueError(
            f"valid_mask length {len(valid_mask)} != trajectory length {num_steps}"
        )

    dim = len(trajectory_q[0])
    out = [[int(v) for v in row] for row in trajectory_q]
    valid_indices = [idx for idx, ok in enumerate(valid_mask) if ok]
    if not valid_indices:
        return out

    for t in range(num_steps):
        if valid_mask[t]:
            continue
        if t < valid_indices[0]:
            out[t] = list(out[valid_indices[0]])
            continue
        if t > valid_indices[-1]:
            out[t] = list(out[valid_indices[-1]])
            continue

        # Locate surrounding valid anchors: idx1 < t < idx2
        pos = 0
        while pos < len(valid_indices) and valid_indices[pos] < t:
            pos += 1
        idx1 = valid_indices[pos - 1]
        idx2 = valid_indices[pos]
        idx0 = valid_indices[pos - 2] if pos >= 2 else idx1
        idx3 = valid_indices[pos + 1] if (pos + 1) < len(valid_indices) else idx2

        h1 = idx2 - idx1
        h0 = (idx1 - idx0) if idx0 < idx1 else h1
        h2 = (idx3 - idx2) if idx3 > idx2 else h1
        step_num = t - idx1

        for d in range(dim):
            y1 = out[idx1][d]
            y2 = out[idx2][d]
            y0 = out[idx0][d] if idx0 < idx1 else (2 * y1 - y2)
            y3 = out[idx3][d] if idx3 > idx2 else (2 * y2 - y1)
            out[t][d] = catmull_rom_bessel_pq(y0, y1, y2, y3, h0, h1, h2, step_num)

    return out
