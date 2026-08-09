from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LIFParams:
    tau: float = 1.0
    v_inf: float = 1.3
    v_thresh: float = 1.0
    v_reset: float = 0.0

    def __post_init__(self) -> None:
        if self.tau <= 0.0:
            raise ValueError("tau must be > 0")
        if self.v_reset >= self.v_thresh:
            raise ValueError("v_reset must be below v_thresh")


@dataclass
class SparseConnectivity:
    n_inputs: int
    n_neurons: int
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    outgoing: list[list[int]]

    @classmethod
    def from_edges(
        cls,
        n_inputs: int,
        n_neurons: int,
        pre: Sequence[int],
        post: Sequence[int],
        weight: Sequence[float],
    ) -> "SparseConnectivity":
        pre_a = np.asarray(pre, dtype=np.int64)
        post_a = np.asarray(post, dtype=np.int64)
        w_a = np.asarray(weight, dtype=np.float64).copy()
        if not (len(pre_a) == len(post_a) == len(w_a)):
            raise ValueError("pre, post and weight must have equal lengths")
        if n_inputs < 0 or n_neurons <= 0:
            raise ValueError("invalid population sizes")
        n_sources = n_inputs + n_neurons
        outgoing = [[] for _ in range(n_sources)]
        for e, (src, dst) in enumerate(zip(pre_a, post_a)):
            if not (0 <= src < n_sources):
                raise ValueError(f"pre[{e}] out of range")
            if not (0 <= dst < n_neurons):
                raise ValueError(f"post[{e}] out of range")
            outgoing[int(src)].append(e)
        return cls(n_inputs, n_neurons, pre_a, post_a, w_a, outgoing)

    def with_weights(self, weight: Sequence[float]) -> "SparseConnectivity":
        weight_a = np.asarray(weight, dtype=np.float64)
        if weight_a.shape != self.weight.shape:
            raise ValueError("replacement weight vector has wrong shape")
        return SparseConnectivity.from_edges(
            self.n_inputs, self.n_neurons, self.pre, self.post, weight_a
        )


@dataclass(frozen=True)
class StateNode:
    v: float
    t: float
    scheduled_t: float


@dataclass(frozen=True)
class PostUpdate:
    prev_node: int
    new_node: int
    edge: int
    flow_a: float
    flow_slope: float
    schedule_dv: float
    immediate: bool


@dataclass(frozen=True)
class EventRecord:
    kind: str
    t: float
    source_node: int | None
    spiker: int | None
    input_channel: int | None
    reset_new_node: int | None
    post_updates: tuple[PostUpdate, ...]


@dataclass
class EventTape:
    nodes: list[StateNode]
    events: list[EventRecord]
    spike_neurons: list[int]
    spike_times: list[float]
    synaptic_updates: int
    immediate_schedules: int

    def operation_stats(self) -> dict[str, int]:
        return {
            "events": len(self.events),
            "network_spikes": len(self.spike_times),
            "synaptic_updates": self.synaptic_updates,
            "state_nodes": len(self.nodes),
            "immediate_schedules": self.immediate_schedules,
        }


def _flow(v: float, dt: float, p: LIFParams) -> tuple[float, float, float]:
    a = math.exp(-dt / p.tau)
    u = p.v_inf + (v - p.v_inf) * a
    slope = (p.v_inf - u) / p.tau
    return u, a, slope


def _schedule(v: float, t: float, p: LIFParams) -> tuple[float, float, bool]:
    if v >= p.v_thresh:
        return t, 0.0, True
    if p.v_inf <= p.v_thresh:
        return math.inf, 0.0, False
    denom = p.v_inf - v
    numer = p.v_inf - p.v_thresh
    if denom <= 0.0 or numer <= 0.0:
        return math.inf, 0.0, False
    dt = -p.tau * math.log(numer / denom)
    return t + dt, -p.tau / denom, False


def simulate(
    p: LIFParams,
    conn: SparseConnectivity,
    input_events: Iterable[tuple[float, int]] = (),
    *,
    initial_v: Sequence[float] | None = None,
    t0: float = 0.0,
    max_events: int = 100,
    max_time: float = math.inf,
) -> EventTape:
    if max_events < 0:
        raise ValueError("max_events must be non-negative")
    if initial_v is None:
        initial_v_a = np.zeros(conn.n_neurons, dtype=np.float64)
    else:
        initial_v_a = np.asarray(initial_v, dtype=np.float64)
        if initial_v_a.shape != (conn.n_neurons,):
            raise ValueError("initial_v has wrong shape")

    inputs = sorted((float(t), int(ch)) for t, ch in input_events)
    for _, ch in inputs:
        if not (0 <= ch < conn.n_inputs):
            raise ValueError("input channel out of range")

    nodes: list[StateNode] = []
    current_node = np.empty(conn.n_neurons, dtype=np.int64)
    scheduled = np.empty(conn.n_neurons, dtype=np.float64)
    version = np.zeros(conn.n_neurons, dtype=np.int64)
    heap: list[tuple[float, int, int]] = []

    for i, v in enumerate(initial_v_a):
        s, _, _ = _schedule(float(v), float(t0), p)
        nodes.append(StateNode(float(v), float(t0), s))
        current_node[i] = len(nodes) - 1
        scheduled[i] = s
        heapq.heappush(heap, (s, i, 0))

    events: list[EventRecord] = []
    spike_neurons: list[int] = []
    spike_times: list[float] = []
    synaptic_updates = 0
    immediate_schedules = 0
    input_pos = 0

    while len(events) < max_events:
        while heap:
            ht, hi, hv = heap[0]
            if hv != version[hi] or ht != scheduled[hi]:
                heapq.heappop(heap)
                continue
            break

        next_spike_t = heap[0][0] if heap else math.inf
        next_input_t = inputs[input_pos][0] if input_pos < len(inputs) else math.inf
        t = min(next_input_t, next_spike_t)
        if not math.isfinite(t) or t > max_time:
            break

        if next_input_t <= next_spike_t:
            t, channel = inputs[input_pos]
            input_pos += 1
            kind = "input"
            source_global = channel
            source_node = None
            spiker = None
            reset_new_node = None
            input_channel = channel
        else:
            t, j, _ = heapq.heappop(heap)
            kind = "spike"
            source_global = conn.n_inputs + j
            source_node = int(current_node[j])
            spiker = j
            input_channel = None

            reset_s, _, reset_immediate = _schedule(p.v_reset, t, p)
            immediate_schedules += int(reset_immediate)
            nodes.append(StateNode(p.v_reset, t, reset_s))
            reset_new_node = len(nodes) - 1
            current_node[j] = reset_new_node
            version[j] += 1
            scheduled[j] = reset_s
            heapq.heappush(heap, (reset_s, j, int(version[j])))
            spike_neurons.append(j)
            spike_times.append(t)

        post_updates: list[PostUpdate] = []
        for edge in conn.outgoing[source_global]:
            i = int(conn.post[edge])
            prev_node = int(current_node[i])
            prev = nodes[prev_node]
            dt = max(t - prev.t, 0.0)
            u, a, slope = _flow(prev.v, dt, p)
            new_v = u + float(conn.weight[edge])
            new_s, schedule_dv, immediate = _schedule(new_v, t, p)
            immediate_schedules += int(immediate)

            nodes.append(StateNode(new_v, t, new_s))
            new_node = len(nodes) - 1
            current_node[i] = new_node
            version[i] += 1
            scheduled[i] = new_s
            heapq.heappush(heap, (new_s, i, int(version[i])))
            post_updates.append(
                PostUpdate(prev_node, new_node, edge, a, slope, schedule_dv, immediate)
            )
            synaptic_updates += 1

        events.append(
            EventRecord(kind, t, source_node, spiker, input_channel,
                        reset_new_node, tuple(post_updates))
        )

    return EventTape(nodes, events, spike_neurons, spike_times,
                     synaptic_updates, immediate_schedules)


def exact_event_gradient(
    tape: EventTape,
    conn: SparseConnectivity,
    dloss_dt_event: Sequence[float],
) -> np.ndarray:
    dloss = np.asarray(dloss_dt_event, dtype=np.float64)
    if dloss.shape != (len(tape.events),):
        raise ValueError("dloss_dt_event must match recorded events")

    adj_v = np.zeros(len(tape.nodes), dtype=np.float64)
    adj_t = np.zeros(len(tape.nodes), dtype=np.float64)
    adj_s = np.zeros(len(tape.nodes), dtype=np.float64)
    grad_w = np.zeros_like(conn.weight)

    for m in range(len(tape.events) - 1, -1, -1):
        ev = tape.events[m]
        adj_event_t = float(dloss[m])
        for op in reversed(ev.post_updates):
            lambda_v = adj_v[op.new_node]
            lambda_t = adj_t[op.new_node]
            lambda_s = adj_s[op.new_node]
            adj_new_v = lambda_v + lambda_s * op.schedule_dv
            grad_w[op.edge] += adj_new_v
            adj_v[op.prev_node] += adj_new_v * op.flow_a
            adj_t[op.prev_node] -= adj_new_v * op.flow_slope
            adj_event_t += lambda_t + lambda_s + adj_new_v * op.flow_slope

        if ev.kind == "spike":
            assert ev.reset_new_node is not None and ev.source_node is not None
            adj_event_t += adj_t[ev.reset_new_node] + adj_s[ev.reset_new_node]
            adj_s[ev.source_node] += adj_event_t

    return grad_w


def first_spike_events(
    tape: EventTape, neuron_ids: Sequence[int]
) -> tuple[list[int | None], np.ndarray]:
    wanted = {int(n): k for k, n in enumerate(neuron_ids)}
    indices: list[int | None] = [None] * len(neuron_ids)
    times = np.full(len(neuron_ids), np.inf, dtype=np.float64)
    for m, ev in enumerate(tape.events):
        if ev.kind != "spike" or ev.spiker not in wanted:
            continue
        k = wanted[int(ev.spiker)]
        if indices[k] is None:
            indices[k] = m
            times[k] = ev.t
    return indices, times


def finite_difference_gradient(
    p: LIFParams,
    conn: SparseConnectivity,
    input_events: Sequence[tuple[float, int]],
    loss_fn,
    *,
    epsilon: float = 1e-6,
    max_events: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    base = simulate(p, conn, input_events, max_events=max_events)
    base_signature = [(e.kind, e.spiker, e.input_channel) for e in base.events]
    grad = np.full_like(conn.weight, np.nan)
    changed = np.zeros(len(conn.weight), dtype=bool)
    for e in range(len(conn.weight)):
        wp = conn.weight.copy(); wp[e] += epsilon
        wm = conn.weight.copy(); wm[e] -= epsilon
        tp = simulate(p, conn.with_weights(wp), input_events, max_events=max_events)
        tm = simulate(p, conn.with_weights(wm), input_events, max_events=max_events)
        sigp = [(x.kind, x.spiker, x.input_channel) for x in tp.events]
        sigm = [(x.kind, x.spiker, x.input_channel) for x in tm.events]
        if sigp != base_signature or sigm != base_signature:
            changed[e] = True
            continue
        grad[e] = (loss_fn(tp) - loss_fn(tm)) / (2.0 * epsilon)
    return grad, changed
