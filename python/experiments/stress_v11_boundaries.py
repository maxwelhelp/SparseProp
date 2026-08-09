import numpy as np

from sparseprop_v11 import LIFParams, SparseConnectivity, V11Config, hybrid_event_gradient, simulate
from sparseprop_v11.training import Adam


def first_spike_or_horizon(tape, neuron=0, horizon=1.0):
    for ev in tape.events:
        if ev.kind == "spike" and ev.spiker == neuron:
            return ev.t
    return horizon


def run(mode: str, steps: int = 6, lr: float = 0.02):
    p = LIFParams(tau=1.0, v_inf=0.0, v_thresh=1.0, v_reset=0.0)
    conn = SparseConnectivity.from_edges(1, 1, [0], [0], [0.95])
    opt = Adam(len(conn.weight), lr=lr)
    inputs = [(0.10, 0)]
    cfg = V11Config(
        risk_threshold=0.1,
        threshold_scale=0.08,
        epsilon_min=1e-4,
        epsilon_max=0.15,
        boundary_overshoot=1.25,
        min_probe_pairs=1,
        max_probe_pairs=3,
        max_events=4,
        max_time=1.0,
    )

    print(f"\n[{mode}]")
    for step in range(steps):
        tape = simulate(p, conn, inputs, max_events=cfg.max_events, max_time=cfg.max_time)
        loss = first_spike_or_horizon(tape)
        credit = np.zeros(len(tape.events), dtype=np.float64)
        result = hybrid_event_gradient(
            p, conn, inputs, credit,
            lambda t: first_spike_or_horizon(t),
            base_tape=tape,
            config=cfg,
        )
        grad = result.exact_gradient if mode == "exact" else result.gradient
        print(
            f"step={step:02d} w={conn.weight[0]:.6f} loss={loss:.6f} "
            f"exact={result.exact_gradient[0]: .6f} "
            f"hybrid={result.gradient[0]: .6f} "
            f"candidate={bool(result.candidate_edges[0])} "
            f"boundary_crossed={bool(result.signature_change_edges[0])} "
            f"probe_pairs={result.probe_pairs}"
        )
        opt.step(conn.weight, grad)

    final = simulate(p, conn, inputs, max_events=cfg.max_events, max_time=cfg.max_time)
    print(f"final w={conn.weight[0]:.6f} first_spike={first_spike_or_horizon(final):.6f}")


if __name__ == "__main__":
    run("exact")
    run("hybrid")
