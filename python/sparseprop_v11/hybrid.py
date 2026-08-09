from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from .core import EventTape, LIFParams, SparseConnectivity, exact_event_gradient, simulate


@dataclass(frozen=True)
class V11Config:
    """Controls sparse boundary-aware zeroth-order correction.

    V11 never probes every parameter. It first restricts itself to edges that
    were actually traversed by the event-driven forward pass, then scores only
    edges close to a spike-birth/death or event-order boundary.
    """

    risk_threshold: float = 0.20
    threshold_scale: float = 0.05
    order_scale: float = 0.02
    epsilon_min: float = 1.0e-5
    epsilon_max: float = 0.10
    boundary_overshoot: float = 1.25
    min_probe_pairs: int = 1
    max_probe_pairs: int = 3
    preboundary_blend: float = 0.25
    preboundary_risk: float = 0.80
    gradient_floor: float = 0.05
    max_events: int = 100
    max_time: float = math.inf

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError("risk_threshold must be in [0, 1]")
        if self.threshold_scale <= 0.0 or self.order_scale <= 0.0:
            raise ValueError("boundary scales must be > 0")
        if self.epsilon_min <= 0.0 or self.epsilon_max < self.epsilon_min:
            raise ValueError("invalid epsilon range")
        if self.min_probe_pairs < 1 or self.max_probe_pairs < self.min_probe_pairs:
            raise ValueError("invalid probe-pair range")


@dataclass(frozen=True)
class EdgeBoundaryInfo:
    edge: int
    usage: int
    immediate_hits: int
    min_threshold_delta: float
    min_order_delta: float
    risk: float
    score: float
    epsilon: float
    probe_pairs: int = 0


@dataclass
class HybridGradientResult:
    gradient: np.ndarray
    exact_gradient: np.ndarray
    zo_gradient: np.ndarray
    candidate_edges: np.ndarray
    corrected_edges: np.ndarray
    signature_change_edges: np.ndarray
    boundary_info: list[EdgeBoundaryInfo]
    probe_pairs: int
    probe_simulations: int

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "candidate_edges": int(self.candidate_edges.sum()),
            "corrected_edges": int(self.corrected_edges.sum()),
            "signature_change_edges": int(self.signature_change_edges.sum()),
            "probe_pairs": int(self.probe_pairs),
            "probe_simulations": int(self.probe_simulations),
        }


def event_signature(tape: EventTape) -> tuple[tuple[str, int | None, int | None], ...]:
    return tuple((ev.kind, ev.spiker, ev.input_channel) for ev in tape.events)


def _future_two_spikes(tape: EventTape) -> tuple[np.ndarray, np.ndarray]:
    """First and second network-spike times strictly after each tape event."""
    n = len(tape.events)
    first = np.full(n, np.inf, dtype=np.float64)
    second = np.full(n, np.inf, dtype=np.float64)
    next1 = math.inf
    next2 = math.inf
    for m in range(n - 1, -1, -1):
        first[m] = next1
        second[m] = next2
        ev = tape.events[m]
        if ev.kind == "spike":
            next2 = next1
            next1 = ev.t
    return first, second


def analyse_event_boundaries(
    tape: EventTape,
    p: LIFParams,
    conn: SparseConnectivity,
    exact_gradient: Sequence[float],
    config: V11Config = V11Config(),
) -> list[EdgeBoundaryInfo]:
    """Estimate weight-space distance to the nearest discrete event boundary.

    Two boundaries are tracked:
      * threshold boundary: a weight move can create/delete an immediate spike;
      * ordering boundary: a scheduled spike can overtake a competing event.

    The resulting distance directly sets the adaptive perturbation scale.
    """
    grad = np.asarray(exact_gradient, dtype=np.float64)
    if grad.shape != conn.weight.shape:
        raise ValueError("exact_gradient shape mismatch")

    n_edges = len(conn.weight)
    usage = np.zeros(n_edges, dtype=np.int64)
    immediate_hits = np.zeros(n_edges, dtype=np.int64)
    min_threshold = np.full(n_edges, np.inf, dtype=np.float64)
    min_order = np.full(n_edges, np.inf, dtype=np.float64)
    first_future, second_future = _future_two_spikes(tape)

    for m, ev in enumerate(tape.events):
        next1 = float(first_future[m])
        next2 = float(second_future[m])
        for op in ev.post_updates:
            e = int(op.edge)
            usage[e] += 1
            node = tape.nodes[op.new_node]

            # Incoming weight is an additive voltage jump: dV/dw = 1.
            threshold_delta = abs(p.v_thresh - node.v)
            min_threshold[e] = min(min_threshold[e], threshold_delta)
            if op.immediate:
                immediate_hits[e] += 1

            # schedule_dv = d(next_spike_time)/dV, and dV/dw = 1.
            sens = abs(op.schedule_dv)
            if sens > 1.0e-14 and math.isfinite(node.scheduled_t) and math.isfinite(next1):
                if abs(node.scheduled_t - next1) <= 1.0e-10 and math.isfinite(next2):
                    time_margin = abs(next2 - next1)
                else:
                    time_margin = abs(node.scheduled_t - next1)
                min_order[e] = min(min_order[e], time_margin / sens)

    raw: list[EdgeBoundaryInfo] = []
    for e in range(n_edges):
        if usage[e] == 0:
            continue

        dth = float(min_threshold[e])
        dor = float(min_order[e])
        rth = math.exp(-dth / config.threshold_scale) if math.isfinite(dth) else 0.0
        ror = math.exp(-dor / config.order_scale) if math.isfinite(dor) else 0.0
        rimmediate = 1.0 if immediate_hits[e] else 0.0
        risk = max(rth, ror, rimmediate)

        finite_deltas = [x for x in (dth, dor) if math.isfinite(x)]
        boundary_delta = min(finite_deltas) if finite_deltas else config.epsilon_max
        epsilon = float(np.clip(
            config.boundary_overshoot * max(boundary_delta, config.epsilon_min),
            config.epsilon_min,
            config.epsilon_max,
        ))

        # V10-style soft relevance. The floor is essential at spike-birth/death
        # surfaces where the exact fixed-order gradient can be exactly zero.
        credit = abs(float(grad[e])) + config.gradient_floor
        score = risk * credit * math.sqrt(float(usage[e]))
        raw.append(EdgeBoundaryInfo(
            edge=e,
            usage=int(usage[e]),
            immediate_hits=int(immediate_hits[e]),
            min_threshold_delta=dth,
            min_order_delta=dor,
            risk=risk,
            score=score,
            epsilon=epsilon,
        ))

    candidates = [x for x in raw if x.risk >= config.risk_threshold]
    if not candidates:
        return raw

    max_score = max(max(x.score for x in candidates), 1.0e-30)
    budget_by_edge: dict[int, int] = {}
    span = config.max_probe_pairs - config.min_probe_pairs
    for info in candidates:
        soft = math.sqrt(info.score / max_score)
        pairs = config.min_probe_pairs + int(round(span * soft))
        budget_by_edge[info.edge] = int(np.clip(
            pairs, config.min_probe_pairs, config.max_probe_pairs
        ))

    return [
        EdgeBoundaryInfo(
            edge=x.edge,
            usage=x.usage,
            immediate_hits=x.immediate_hits,
            min_threshold_delta=x.min_threshold_delta,
            min_order_delta=x.min_order_delta,
            risk=x.risk,
            score=x.score,
            epsilon=x.epsilon,
            probe_pairs=budget_by_edge.get(x.edge, 0),
        )
        for x in raw
    ]


def _probe_scales(n: int) -> tuple[float, ...]:
    # Extra budget brackets the estimated boundary from both sides.
    order = (1.0, 0.5, 2.0, 0.25, 4.0, 0.125, 8.0)
    if n > len(order):
        raise ValueError("max_probe_pairs is too large for available scales")
    return order[:n]


def hybrid_event_gradient(
    p: LIFParams,
    conn: SparseConnectivity,
    input_events: Sequence[tuple[float, int]],
    dloss_dt_event: Sequence[float],
    loss_fn: Callable[[EventTape], float],
    *,
    base_tape: EventTape | None = None,
    config: V11Config = V11Config(),
) -> HybridGradientResult:
    """Exact sparse gradient plus selective V11 correction at event boundaries.

    Stable regions use exact reverse-mode gradients with zero extra probes.
    Only active edges near a discrete boundary receive hard W+eps/W-eps
    simulations. If a probe changes the event signature, its zeroth-order
    estimate replaces the invalid fixed-order derivative.
    """
    if base_tape is None:
        base_tape = simulate(
            p, conn, input_events,
            max_events=config.max_events, max_time=config.max_time,
        )

    exact = exact_event_gradient(base_tape, conn, dloss_dt_event)
    info = analyse_event_boundaries(base_tape, p, conn, exact, config)
    hybrid = exact.copy()
    zo = np.full_like(exact, np.nan)
    candidates = np.zeros(len(conn.weight), dtype=bool)
    corrected = np.zeros(len(conn.weight), dtype=bool)
    changed_edges = np.zeros(len(conn.weight), dtype=bool)
    base_sig = event_signature(base_tape)
    probe_pairs = 0

    for edge_info in info:
        if edge_info.probe_pairs <= 0:
            continue
        e = edge_info.edge
        candidates[e] = True
        estimates: list[float] = []
        changed_estimates: list[float] = []

        for scale in _probe_scales(edge_info.probe_pairs):
            eps = float(np.clip(
                edge_info.epsilon * scale,
                config.epsilon_min,
                config.epsilon_max,
            ))
            wp = conn.weight.copy(); wp[e] += eps
            wm = conn.weight.copy(); wm[e] -= eps
            tp = simulate(
                p, conn.with_weights(wp), input_events,
                max_events=config.max_events, max_time=config.max_time,
            )
            tm = simulate(
                p, conn.with_weights(wm), input_events,
                max_events=config.max_events, max_time=config.max_time,
            )
            lp = float(loss_fn(tp))
            lm = float(loss_fn(tm))
            estimate = (lp - lm) / (2.0 * eps)
            if math.isfinite(estimate):
                estimates.append(estimate)

            changed = event_signature(tp) != base_sig or event_signature(tm) != base_sig
            if changed and math.isfinite(estimate):
                changed_estimates.append(estimate)
            changed_edges[e] |= changed
            probe_pairs += 1

        if not estimates:
            continue

        if changed_estimates:
            z = float(np.median(changed_estimates))
            hybrid[e] = z
            corrected[e] = True
        else:
            z = float(np.median(estimates))
            if edge_info.immediate_hits > 0:
                hybrid[e] = z
                corrected[e] = True
            elif edge_info.risk >= config.preboundary_risk:
                alpha = min(1.0, config.preboundary_blend * edge_info.risk)
                hybrid[e] = (1.0 - alpha) * exact[e] + alpha * z
                corrected[e] = True
        zo[e] = z

    return HybridGradientResult(
        gradient=hybrid,
        exact_gradient=exact,
        zo_gradient=zo,
        candidate_edges=candidates,
        corrected_edges=corrected,
        signature_change_edges=changed_edges,
        boundary_info=info,
        probe_pairs=probe_pairs,
        probe_simulations=2 * probe_pairs,
    )
