from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from .core import EventTape, LIFParams, SparseConnectivity, exact_event_gradient, simulate


@dataclass(frozen=True)
class V11Config:
    """Sparse event-boundary zeroth-order correction.

    Exact reverse-mode is the default. Hard W+eps/W-eps probes are spent only
    on active edges near a discrete event boundary. A probe is allowed to
    override the exact gradient only when it provides a *descent certificate*:
    at least one perturbed hard-spike trajectory has lower loss than the
    unperturbed trajectory. This prevents V11 from continuing to push a weight
    merely because a distant probe can cross back over an already-solved
    boundary.
    """

    risk_threshold: float = 0.20
    threshold_scale: float = 0.05
    order_scale: float = 0.02
    epsilon_min: float = 1.0e-5
    epsilon_max: float = 0.10
    boundary_overshoot: float = 1.25
    min_probe_pairs: int = 1
    max_probe_pairs: int = 3
    max_total_probe_pairs: int = 24
    preboundary_blend: float = 0.25
    preboundary_risk: float = 0.80
    gradient_floor: float = 0.05
    improvement_tol: float = 1.0e-9
    zo_gradient_clip: float = 5.0
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
        if self.max_total_probe_pairs < 1:
            raise ValueError("max_total_probe_pairs must be >= 1")
        if self.zo_gradient_clip <= 0.0:
            raise ValueError("zo_gradient_clip must be > 0")


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
    improving_probe_edges: np.ndarray
    boundary_info: list[EdgeBoundaryInfo]
    probe_pairs: int
    probe_simulations: int
    base_loss: float

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "candidate_edges": int(self.candidate_edges.sum()),
            "corrected_edges": int(self.corrected_edges.sum()),
            "signature_change_edges": int(self.signature_change_edges.sum()),
            "improving_probe_edges": int(self.improving_probe_edges.sum()),
            "probe_pairs": int(self.probe_pairs),
            "probe_simulations": int(self.probe_simulations),
            "base_loss": float(self.base_loss),
        }


def event_signature(tape: EventTape) -> tuple[tuple[str, int | None, int | None], ...]:
    return tuple((ev.kind, ev.spiker, ev.input_channel) for ev in tape.events)


def _future_two_spikes(tape: EventTape) -> tuple[np.ndarray, np.ndarray]:
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


def _allocate_soft_budget(
    raw: list[EdgeBoundaryInfo], config: V11Config
) -> dict[int, int]:
    """Allocate a global discrete probe budget approximately proportional to score."""
    candidates = [x for x in raw if x.risk >= config.risk_threshold and x.score > 0.0]
    if not candidates:
        return {}

    cap = config.max_total_probe_pairs
    min_total = config.min_probe_pairs * len(candidates)
    budget: dict[int, int] = {x.edge: 0 for x in candidates}

    if min_total <= cap:
        for x in candidates:
            budget[x.edge] = config.min_probe_pairs
        remaining = cap - min_total
        capacities = np.array(
            [config.max_probe_pairs - config.min_probe_pairs for _ in candidates],
            dtype=np.int64,
        )
    else:
        remaining = cap
        capacities = np.full(len(candidates), config.max_probe_pairs, dtype=np.int64)

    if remaining <= 0:
        return budget

    scores = np.array([x.score for x in candidates], dtype=np.float64)
    if not np.isfinite(scores).all() or float(scores.sum()) <= 0.0:
        scores[:] = 1.0
    scores /= scores.sum()

    ideal = remaining * scores
    floor = np.minimum(np.floor(ideal).astype(np.int64), capacities)
    for i, x in enumerate(candidates):
        budget[x.edge] += int(floor[i])
    remaining -= int(floor.sum())
    capacities -= floor

    remainder = ideal - floor
    while remaining > 0 and np.any(capacities > 0):
        idx = int(np.argmax(np.where(capacities > 0, remainder, -np.inf)))
        budget[candidates[idx].edge] += 1
        capacities[idx] -= 1
        remainder[idx] = 0.0
        remaining -= 1

    return budget


def analyse_event_boundaries(
    tape: EventTape,
    p: LIFParams,
    conn: SparseConnectivity,
    exact_gradient: Sequence[float],
    config: V11Config = V11Config(),
) -> list[EdgeBoundaryInfo]:
    """Estimate weight-space distance to spike birth/death and order boundaries."""
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

            dth = abs(p.v_thresh - node.v)
            min_threshold[e] = min(min_threshold[e], dth)
            if op.immediate:
                immediate_hits[e] += 1

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

        finite = [x for x in (dth, dor) if math.isfinite(x)]
        boundary_delta = min(finite) if finite else config.epsilon_max
        epsilon = float(np.clip(
            config.boundary_overshoot * max(boundary_delta, config.epsilon_min),
            config.epsilon_min,
            config.epsilon_max,
        ))
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

    budget_by_edge = _allocate_soft_budget(raw, config)
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
    order = (1.0, 0.5, 2.0, 0.25, 4.0, 0.125, 8.0)
    if n > len(order):
        raise ValueError("max_probe_pairs is too large for available scales")
    return order[:n]


def _descent_certificate(
    base_loss: float, plus_loss: float, minus_loss: float, eps: float, tol: float
) -> tuple[list[float], bool]:
    """Return pseudo-gradient directions certified to reduce hard-trajectory loss."""
    estimates: list[float] = []
    if plus_loss < base_loss - tol:
        estimates.append(-(base_loss - plus_loss) / eps)
    if minus_loss < base_loss - tol:
        estimates.append(+(base_loss - minus_loss) / eps)
    return estimates, bool(estimates)


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
    """Exact sparse gradient plus selective hard-spike boundary correction."""
    if base_tape is None:
        base_tape = simulate(
            p, conn, input_events,
            max_events=config.max_events, max_time=config.max_time,
        )

    base_loss = float(loss_fn(base_tape))
    exact = exact_event_gradient(base_tape, conn, dloss_dt_event)
    info = analyse_event_boundaries(base_tape, p, conn, exact, config)
    hybrid = exact.copy()
    zo = np.full_like(exact, np.nan)
    candidates = np.zeros(len(conn.weight), dtype=bool)
    corrected = np.zeros(len(conn.weight), dtype=bool)
    changed_edges = np.zeros(len(conn.weight), dtype=bool)
    improving_edges = np.zeros(len(conn.weight), dtype=bool)
    base_sig = event_signature(base_tape)
    probe_pairs = 0

    for edge_info in info:
        if edge_info.probe_pairs <= 0:
            continue
        e = edge_info.edge
        candidates[e] = True
        improving_estimates: list[float] = []
        changed_improving_estimates: list[float] = []

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
            estimates, improves = _descent_certificate(
                base_loss, lp, lm, eps, config.improvement_tol
            )
            if improves:
                improving_edges[e] = True
                improving_estimates.extend(estimates)

            changed = event_signature(tp) != base_sig or event_signature(tm) != base_sig
            if changed:
                changed_edges[e] = True
                if improves:
                    changed_improving_estimates.extend(estimates)
            probe_pairs += 1

        if not improving_estimates:
            continue

        pool = changed_improving_estimates or improving_estimates
        z = float(np.median(pool))
        z = float(np.clip(z, -config.zo_gradient_clip, config.zo_gradient_clip))
        zo[e] = z

        if changed_improving_estimates or edge_info.immediate_hits > 0:
            hybrid[e] = z
            corrected[e] = True
        elif edge_info.risk >= config.preboundary_risk:
            alpha = min(1.0, config.preboundary_blend * edge_info.risk)
            hybrid[e] = (1.0 - alpha) * exact[e] + alpha * z
            corrected[e] = True

    return HybridGradientResult(
        gradient=hybrid,
        exact_gradient=exact,
        zo_gradient=zo,
        candidate_edges=candidates,
        corrected_edges=corrected,
        signature_change_edges=changed_edges,
        improving_probe_edges=improving_edges,
        boundary_info=info,
        probe_pairs=probe_pairs,
        probe_simulations=2 * probe_pairs,
        base_loss=base_loss,
    )
