import numpy as np

from sparseprop_v11 import (
    LIFParams,
    SparseConnectivity,
    V11Config,
    hybrid_event_gradient,
    simulate,
)


def first_spike_or_horizon(tape, neuron=0, horizon=1.0):
    for ev in tape.events:
        if ev.kind == "spike" and ev.spiker == neuron:
            return ev.t
    return horizon


def test_v11_creates_gradient_across_spike_birth_boundary():
    # v_inf below threshold => without enough input this neuron is truly silent.
    p = LIFParams(tau=1.0, v_inf=0.0, v_thresh=1.0, v_reset=0.0)
    conn = SparseConnectivity.from_edges(1, 1, [0], [0], [0.99])
    inputs = [(0.10, 0)]
    cfg = V11Config(
        risk_threshold=0.1,
        threshold_scale=0.05,
        order_scale=0.02,
        epsilon_min=1e-4,
        epsilon_max=0.1,
        boundary_overshoot=1.25,
        min_probe_pairs=1,
        max_probe_pairs=3,
        max_events=4,
        max_time=1.0,
    )
    tape = simulate(p, conn, inputs, max_events=cfg.max_events, max_time=cfg.max_time)
    assert first_spike_or_horizon(tape) == 1.0

    # No output spike exists in the fixed event graph, so exact event credit is zero.
    credit = np.zeros(len(tape.events), dtype=np.float64)
    result = hybrid_event_gradient(
        p, conn, inputs, credit,
        lambda t: first_spike_or_horizon(t),
        base_tape=tape,
        config=cfg,
    )

    assert result.exact_gradient[0] == 0.0
    assert result.candidate_edges[0]
    assert result.signature_change_edges[0]
    assert result.corrected_edges[0]
    assert np.isfinite(result.gradient[0])
    assert result.gradient[0] < 0.0  # gradient descent therefore increases w
    assert result.probe_simulations <= 2 * cfg.max_probe_pairs


def test_inactive_edges_are_never_probed():
    p = LIFParams(tau=1.0, v_inf=0.0, v_thresh=1.0, v_reset=0.0)
    # Edge 1 is never traversed because input channel 1 never fires.
    conn = SparseConnectivity.from_edges(2, 1, [0, 1], [0, 0], [0.2, 0.99])
    inputs = [(0.1, 0)]
    tape = simulate(p, conn, inputs, max_events=4, max_time=1.0)
    credit = np.zeros(len(tape.events))
    cfg = V11Config(
        risk_threshold=0.1,
        threshold_scale=0.05,
        max_events=4,
        max_time=1.0,
    )
    result = hybrid_event_gradient(
        p, conn, inputs, credit,
        lambda t: first_spike_or_horizon(t),
        base_tape=tape,
        config=cfg,
    )
    assert not result.candidate_edges[1]
