"""Backward cuDSS-FFI chunking: collision regression, exactness, compile-time
guard, and the large-dim completion check.

The regression coverage here protects against three prior failure modes:
static-matrix batch slicing, cuDSS fill-in budget underestimation, and
compile-time blowup from unrolled chunk loops.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.helpers.backend_utils import backend_available
from turbompc.solvers.turbompc_solver import BackwardBackend

jax.config.update("jax_enable_x64", True)

from turbompc.solvers.qp_data import (
    QPCostBlocks,
    QPData,
    QPEqualityBlocks,
    QPInequalityBlocks,
)
from turbompc.solvers.qp_utils import ZShape

pytestmark = pytest.mark.skipif(
    not backend_available(BackwardBackend.DIRECT_CUDSS_FFI),
    reason="DIRECT_CUDSS_FFI not built/available",
)


def _backward_cudss_module():
    import turbompc.solvers.backward.backward_kkt_cudss_ffi as bk

    return bk


def _build_backward_qp(nx, nu, H, *, m_g=0, seed=0):
    rng = np.random.default_rng(seed)
    n = nx + nu
    N = H
    D = jnp.broadcast_to(2.0 * jnp.eye(n), (N + 1, n, n))
    E = jnp.zeros((N, n, n))
    q = jnp.asarray(rng.standard_normal((N + 1, n)))
    one = np.hstack([np.eye(nx), np.zeros((nx, nu))])
    A0 = jnp.asarray(one)
    c0 = jnp.asarray(rng.standard_normal((nx,)))
    A_minus = jnp.broadcast_to(jnp.asarray(-one), (N, nx, n))
    A_plus = jnp.broadcast_to(jnp.asarray(one), (N, nx, n))
    c = jnp.asarray(rng.standard_normal((N, nx)))
    G = jnp.zeros((N + 1, m_g, n))
    qp = QPData(
        cost=QPCostBlocks(D=D, E=E, q=q),
        eq=QPEqualityBlocks(A0=A0, A_minus=A_minus, A_plus=A_plus, c0=c0, c=c),
        ineq=QPInequalityBlocks(
            G=G, l=jnp.zeros((N + 1, m_g)), u=jnp.zeros((N + 1, m_g))
        ),
    )
    return qp, ZShape(horizon=N, num_states=nx, num_controls=nu)


def _batch_cost_only(qp, Nb, seed=1):
    rng = np.random.default_rng(seed)

    def bc(a):
        return jnp.broadcast_to(a, (Nb, *a.shape))

    return QPData(
        cost=QPCostBlocks(
            D=bc(qp.cost.D),
            E=bc(qp.cost.E),
            q=jnp.asarray(
                np.asarray(qp.cost.q)[None]
                + rng.standard_normal((Nb, *qp.cost.q.shape))
            ),
        ),
        eq=qp.eq,
        ineq=qp.ineq,
    )


_COST_ONLY_IN_AXES = QPData(
    cost=QPCostBlocks(D=0, E=0, q=0),
    eq=QPEqualityBlocks(A0=None, A_minus=None, A_plus=None, c0=None, c=None),
    ineq=QPInequalityBlocks(G=None, l=None, u=None, slack_penalization_weight=None),
)


def _max_rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1.0)))


def _run(qp, zshape, in_axes):
    bk = _backward_cudss_module()
    fn = jax.jit(
        jax.vmap(bk.solve_backward_kkt_cudss_ffi, in_axes=(in_axes, None)),
        static_argnums=1,
    )
    (x, y), m = fn(qp, zshape)
    jax.block_until_ready(x)
    return np.asarray(x), np.asarray(y), np.asarray(m)


def test_chunk_slice_uses_in_batched_not_shape():
    """nx == Nb with STATIC eq blocks must not mis-slice A0 (the #37 bug)."""
    bk = _backward_cudss_module()
    nx, nu, H = 8, 4, 5
    Nb = nx
    qp0, zshape = _build_backward_qp(nx, nu, H)
    qp = _batch_cost_only(qp0, Nb)
    saved = bk._safe_cudss_chunk
    try:
        bk._safe_cudss_chunk = lambda *a, **k: 10**9  # huge → single call
        xr, yr, mr = _run(qp, zshape, _COST_ONLY_IN_AXES)
        bk._safe_cudss_chunk = lambda *a, **k: 1  # chunk=1 → Nb chunks
        xc, yc, mc = _run(qp, zshape, _COST_ONLY_IN_AXES)
    finally:
        bk._safe_cudss_chunk = saved
    assert xc.shape == xr.shape and mc.shape == mr.shape, (
        f"shapes diverged (A0 mis-slice): x {xc.shape} vs {xr.shape}, "
        f"mult {mc.shape} vs {mr.shape}"
    )
    rel = max(_max_rel(xc, xr), _max_rel(yc, yr), _max_rel(mc, mr))
    assert rel < 1e-9, f"chunking changed the solution: rel={rel:.2e}"


def test_backward_chunking_is_numerically_exact():
    """All-batched uniform: chunk=1 must match the single solve (rel<1e-9)."""
    bk = _backward_cudss_module()
    nx, nu, H, Nb = 4, 2, 6, 7
    qp0, zshape = _build_backward_qp(nx, nu, H)

    def bc(a):
        return jnp.broadcast_to(a, (Nb, *a.shape))

    rng = np.random.default_rng(1)
    qp = QPData(
        cost=QPCostBlocks(
            D=bc(qp0.cost.D),
            E=bc(qp0.cost.E),
            q=jnp.asarray(
                np.asarray(qp0.cost.q)[None]
                + rng.standard_normal((Nb, *qp0.cost.q.shape))
            ),
        ),
        eq=QPEqualityBlocks(
            A0=bc(qp0.eq.A0),
            A_minus=bc(qp0.eq.A_minus),
            A_plus=bc(qp0.eq.A_plus),
            c0=bc(qp0.eq.c0),
            c=bc(qp0.eq.c),
        ),
        ineq=QPInequalityBlocks(
            G=bc(qp0.ineq.G),
            l=bc(qp0.ineq.l),
            u=bc(qp0.ineq.u),
            slack_penalization_weight=jnp.zeros((Nb,)),
        ),
    )
    all_b = jax.tree.map(lambda _: 0, qp)
    saved = bk._safe_cudss_chunk
    try:
        bk._safe_cudss_chunk = lambda *a, **k: 10**9
        xr, yr, mr = _run(qp, zshape, all_b)
        bk._safe_cudss_chunk = lambda *a, **k: 1
        xc, yc, mc = _run(qp, zshape, all_b)
    finally:
        bk._safe_cudss_chunk = saved
    rel = max(_max_rel(xc, xr), _max_rel(yc, yr), _max_rel(mc, mr))
    assert rel < 1e-9, f"chunking changed the solution: rel={rel:.2e}"


def test_chunk_loop_compiles_in_constant_time():
    """Regression guard: chunked-path compile must not blow up with n_chunks.

    chunk=1 over Nb=48 ⇒ 48 chunks. NOTE: at these tiny dims even the
    (current) unrolled Python for-loop compiles fast, so this guard only
    becomes load-bearing once the non-unrolled lax.map lands and at
    realistic dims; the decisive proof is an end-to-end large-shape run in a
    later task. Kept here as a cheap structural regression canary.
    """
    bk = _backward_cudss_module()
    nx, nu, H, Nb = 4, 2, 5, 48
    qp0, zshape = _build_backward_qp(nx, nu, H)

    def bc(a):
        return jnp.broadcast_to(a, (Nb, *a.shape))

    qp = QPData(
        cost=QPCostBlocks(D=bc(qp0.cost.D), E=bc(qp0.cost.E), q=bc(qp0.cost.q)),
        eq=qp0.eq,
        ineq=qp0.ineq,
    )
    saved = bk._safe_cudss_chunk
    try:
        bk._safe_cudss_chunk = lambda *a, **k: 1  # 48 chunks
        t0 = time.time()
        fn = jax.jit(
            jax.vmap(
                bk.solve_backward_kkt_cudss_ffi, in_axes=(_COST_ONLY_IN_AXES, None)
            ),
            static_argnums=1,
        )
        (x, _), _ = fn(qp, zshape)
        jax.block_until_ready(x)
        dt = time.time() - t0
    finally:
        bk._safe_cudss_chunk = saved
    assert np.isfinite(np.asarray(x)).all(), "non-finite output from chunked solve"
    assert dt < 90.0, f"compile scaled with n_chunks (unrolled?): {dt:.1f}s"


def test_singular_solve_raises_not_aborts():
    """A cuDSS failure must surface as a CATCHABLE Python exception, not
    abort() the process.

    Subprocess-isolated: forcing a cuDSS error can leave the CUDA context
    unusable, so we run the failing solve in a disposable child. Pre-fix
    the child dies via abort() inside CUDSS_CHECK (no "CAUGHT", non-zero
    rc); post-fix the child catches a Python exception and exits 0.
    """
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    worktree = str(Path(__file__).resolve().parents[3])  # .../<worktree root>
    child = textwrap.dedent(
        """
        import re, sys
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import turbompc.solvers.backward.backward_kkt_cudss_ffi as bk
        # Confirm we loaded the worktree copy (fail loudly otherwise).
        assert "%s" in bk.__file__, "child imported wrong turbompc: " + bk.__file__
        rp, ci, pat = bk._get_or_build_pattern(4, 4, 2, 2, 0)
        n = int(pat.N_kkt)
        rp_bad = rp.at[-1].set(int(pat.nnz) + 100)   # inconsistent CSR -> cuDSS error
        vals = jnp.ones((int(pat.nnz),), jnp.float64)
        rhs = jnp.ones((n,), jnp.float64)
        try:
            out = jax.jit(lambda v, r: bk._call_ffi(rp_bad, ci, v, r, n, jnp.float64))(vals, rhs)
            jax.block_until_ready(out)
            # os._exit (not sys.exit) bypasses Python interpreter teardown
            # + atexit; we need it because the forced cuDSS failure corrupted
            # the CUDA context, and JAX's PJRT buffer destructor (jax >= 0.10)
            # treats a CUDA error during cleanup as fatal — aborting the child
            # with SIGABRT after the Python-level catch already succeeded.
            # `flush=True` is essential before _exit: it skips stdio flushing.
            print("NORAISE", flush=True)
            import os as _os
            _os._exit(2)
        except Exception as e:
            msg = str(e)
            print("CAUGHT:" + msg[:200], flush=True)
            import os as _os
            _os._exit(0 if re.search(r"(?i)cudss|factor|singular|internal", msg) else 3)
        """
        % worktree
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = worktree + os.pathsep + env.get("PYTHONPATH", "")
    # Don't let JAX grab 75% VRAM in the child (caused OOM/SIGABRT under memory
    # pressure from larger tests running before this one).  Do NOT set
    # XLA_PYTHON_CLIENT_MEM_FRACTION: a small fraction (e.g. 0.10) causes
    # "no supported devices found for platform CUDA" when the GPU is already
    # under pressure from the parent test suite; PREALLOCATE=false alone is
    # sufficient to prevent the child from over-committing device memory.
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    r = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-1000:]
    assert r.returncode == 0, (
        f"cuDSS failure was not caught (rc={r.returncode}; "
        f"pre-fix abort()/SIGABRT is rc<0):\n{tail}"
    )
    assert "CAUGHT:" in r.stdout, f"no catchable exception raised:\n{tail}"


def test_probe_returns_positive_block_and_free_bytes():
    """The eager cuDSS probe returns (M_block>0, free>0) for a real shape,
    and a bigger problem needs more per-block memory.

    Subprocess-isolated: cudaMemGetInfo delta measures physical VRAM pool
    expansion, which only shows nonzero on the *first* large-enough probe in
    a fresh CUDA context. Running after other GPU tests would see 0 because
    the pool was already expanded. Each probe runs in a disposable child.

    Shapes (20,20,10,10,10) and (40,96,64,64,64) are chosen so the first
    exceeds the cuDSS pool's initial reservation (~2 MB page) and the second
    is strictly larger.
    """
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    worktree = str(Path(__file__).resolve().parents[3])

    def _run_probe(params):
        child = textwrap.dedent(
            f"""
        import turbompc.solvers.backward.backward_kkt_cudss_ffi as _bk
        r = _bk._probe_cudss_device_bytes(_bk._single_pattern(*{params}))
        print(r[0], r[1])
        """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = worktree + os.pathsep + env.get("PYTHONPATH", "")
        # Don't let JAX grab 75% VRAM in the child; PREALLOCATE=false alone is
        # sufficient and avoids the "no supported devices found for platform CUDA"
        # error that occurs when XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 is set while
        # the GPU is under memory pressure from the parent test suite.
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        r = subprocess.run(
            [sys.executable, "-c", child],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert r.returncode == 0, f"probe child failed:\n{r.stderr[-500:]}"
        vals = list(map(int, r.stdout.strip().split()))
        return (vals[0], vals[1])

    small = _run_probe((20, 20, 10, 10, 10))
    big = _run_probe((40, 96, 64, 64, 64))
    assert small[0] > 0 and small[1] > 0, f"small probe failed: {small}"
    assert big[0] > 0 and big[1] > 0, f"big probe failed: {big}"
    assert big[0] > small[0], "bigger problem must need more per-block memory"


def test_safe_chunk_calibrated_and_cached():
    """safe_chunk = max(1, free*safety // per_block); cached per shape;
    probe failure ⇒ conservative chunk=1."""
    import turbompc.solvers.backward.backward_kkt_cudss_ffi as _bk

    _bk._SAFE_CHUNK_CACHE.clear()
    saved = _bk._probe_cudss_device_bytes
    calls = []
    try:
        _bk._probe_cudss_device_bytes = lambda pat: (
            calls.append(1) or (1_000_000, 800_000_000)
        )
        c1 = _bk._safe_cudss_chunk(10, 6, 4, 4, 0)
        c1b = _bk._safe_cudss_chunk(10, 6, 4, 4, 0)  # cached: no 2nd probe
        assert c1 == max(1, int(800_000_000 * _bk._CUDSS_BUDGET_SAFETY) // 1_000_000)
        assert c1b == c1
        assert len(calls) == 1, f"probe not cached: {len(calls)} calls"
        _bk._probe_cudss_device_bytes = lambda pat: (0, 0)  # probe failure
        c_fail = _bk._safe_cudss_chunk(7, 5, 3, 3, 0)
        assert c_fail == 1, f"probe failure must fall back to chunk=1, got {c_fail}"
    finally:
        _bk._probe_cudss_device_bytes = saved
        _bk._SAFE_CHUNK_CACHE.clear()


@pytest.mark.extended
def test_backward_large_dim_completes():
    """Production-like large shape: dim=64+32 H=40 nx==batch==64, static eq,
    real calibrated chunk. Must complete finite (pre-fix: ALLOC_FAILED).

    NOTE: this test allocates large GPU memory (cuDSS for nx=64 batch=64
    H=40). It is placed last in the file so that subprocess-isolated tests
    (test_singular_solve_raises_not_aborts, test_probe_returns_positive_block_and_free_bytes)
    run first and are not affected by the GPU memory pressure from this test.
    """
    nx, nu, H = 64, 32, 40
    Nb = nx
    qp0, zshape = _build_backward_qp(nx, nu, H, m_g=64, seed=2)
    qp = _batch_cost_only(qp0, Nb, seed=3)
    x, y, m = _run(qp, zshape, _COST_ONLY_IN_AXES)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(m).all()
    assert x.shape[0] == Nb and m.shape[0] == Nb
