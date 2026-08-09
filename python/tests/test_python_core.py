import numpy as np

from sparseprop_v11 import (
    LIFParams,
    SparseConnectivity,
    exact_event_gradient,
    finite_difference_gradient,
    first_spike_events,
    simulate,
)
from sparseprop_v11.training import latency_cross_entropy


def test_exact_gradient_with_external_spikes():
    p = LIFParams(tau=1.0, v_inf=1.3, v_thresh=1.0, v_reset=0.0)
    conn = SparseConnectivity.from_edges(
        2, 2,
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0.035, 0.055, 0.060, 0.030],
    )
    input_events = [(0.2, 0), (0.8, 1)]
    tape = simulate(p, conn, input_events, max_events=10)
    idx, times = first_spike_events(tape, [0, 1])
    _, dldt, _ = latency_cross_entropy(times, target=0, temperature=0.2)
    credit = np.zeros(len(tape.events))
    for k, event_idx in enumerate(idx):
        credit[event_idx] = dldt[k]
    exact = exact_event_gradient(tape, conn, credit)

    def loss_fn(t):
        _, ts = first_spike_events(t, [0, 1])
        return latency_cross_entropy(ts, target=0, temperature=0.2)[0]

    fd, changed = finite_difference_gradient(
        p, conn, input_events, loss_fn, epsilon=1e-6, max_events=10
    )
    assert not changed.any()
    assert np.max(np.abs(exact - fd)) < 2e-6


def test_event_work_tracks_edges_not_population_size():
    p = LIFParams()
    n = 100_000
    pre = [0] * 8
    post = list(range(8))
    conn = SparseConnectivity.from_edges(1, n, pre, post, [0.01] * 8)
    tape = simulate(p, conn, [(0.1, 0), (0.2, 0)], max_events=2)
    assert tape.synaptic_updates == 16
