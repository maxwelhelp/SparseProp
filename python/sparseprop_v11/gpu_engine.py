from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from .core import LIFParams, SparseConnectivity


@dataclass
class TorchConnectivity:
    n_inputs: int
    n_neurons: int
    pre: torch.Tensor
    post: torch.Tensor
    weight: torch.Tensor
    outgoing: torch.Tensor

    @classmethod
    def from_numpy(cls, conn: SparseConnectivity, device: torch.device | str):
        device = torch.device(device)
        max_fanout = max((len(x) for x in conn.outgoing), default=0)
        max_fanout = max(max_fanout, 1)
        out = torch.full((len(conn.outgoing), max_fanout), -1, dtype=torch.long, device=device)
        for src, edges in enumerate(conn.outgoing):
            if edges:
                out[src, : len(edges)] = torch.as_tensor(edges, dtype=torch.long, device=device)
        return cls(
            n_inputs=conn.n_inputs,
            n_neurons=conn.n_neurons,
            pre=torch.as_tensor(conn.pre, dtype=torch.long, device=device),
            post=torch.as_tensor(conn.post, dtype=torch.long, device=device),
            weight=torch.as_tensor(conn.weight, dtype=torch.float32, device=device).clone(),
            outgoing=out,
        )

    @property
    def device(self):
        return self.weight.device

    @property
    def max_fanout(self) -> int:
        return int(self.outgoing.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.weight.numel())


@dataclass
class GPUTape:
    event_time: torch.Tensor
    event_active: torch.Tensor
    event_is_spike: torch.Tensor
    event_spiker: torch.Tensor
    event_input_channel: torch.Tensor
    event_source_node: torch.Tensor
    event_reset_node: torch.Tensor
    post_valid: torch.Tensor
    post_edge: torch.Tensor
    post_prev_node: torch.Tensor
    post_new_node: torch.Tensor
    post_flow_a: torch.Tensor
    post_flow_slope: torch.Tensor
    post_schedule_dv: torch.Tensor
    post_new_v: torch.Tensor
    post_scheduled_t: torch.Tensor
    post_immediate: torch.Tensor
    node_v: torch.Tensor
    node_t: torch.Tensor
    node_scheduled_t: torch.Tensor
    synaptic_updates: torch.Tensor
    input_event_count: torch.Tensor

    @property
    def steps(self) -> int:
        return int(self.event_time.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.event_time.shape[1])


def _schedule(v: torch.Tensor, t: torch.Tensor, p: LIFParams):
    thresh = float(p.v_thresh)
    vinf = float(p.v_inf)
    tau = float(p.tau)
    immediate = v >= thresh
    if vinf <= thresh:
        sched = torch.where(immediate, t, torch.full_like(t, float("inf")))
        return sched, torch.zeros_like(v), immediate
    denom = vinf - v
    numer = vinf - thresh
    safe = (denom > 0.0) & (~immediate)
    dt = torch.where(
        safe,
        -tau * torch.log(torch.as_tensor(numer, device=v.device, dtype=v.dtype) / denom),
        torch.zeros_like(v),
    )
    sched = torch.where(immediate, t, torch.where(safe, t + dt, torch.full_like(t, float("inf"))))
    deriv = torch.where(safe, -tau / denom, torch.zeros_like(v))
    return sched, deriv, immediate


def _flow(v: torch.Tensor, dt: torch.Tensor, p: LIFParams):
    a = torch.exp(-dt / float(p.tau))
    u = float(p.v_inf) + (v - float(p.v_inf)) * a
    slope = (float(p.v_inf) - u) / float(p.tau)
    return u, a, slope


def _build_heap(scheduled: torch.Tensor):
    order = torch.argsort(scheduled, dim=1)
    values = torch.gather(scheduled, 1, order).clone()
    handles = order.clone()
    b, n = scheduled.shape
    pos = torch.empty_like(order)
    positions = torch.arange(n, device=scheduled.device, dtype=torch.long).expand(b, n)
    pos.scatter_(1, order, positions)
    return values, handles, pos


def _heap_update(values: torch.Tensor, handles: torch.Tensor, pos: torch.Tensor,
                 target: torch.Tensor, new_value: torch.Tensor, active: torch.Tensor):
    """One mutable-heap handle update per batch row: O(log N), not O(N)."""
    bsz, n = values.shape
    rows = torch.arange(bsz, device=values.device)
    target_safe = torch.where(active, target, torch.zeros_like(target))
    pidx = pos[rows, target_safe]
    old = values[rows, pidx]
    values[rows, pidx] = torch.where(active, new_value, old)
    levels = max(1, int(math.ceil(math.log2(max(n, 2)))) + 1)

    moving = active & (new_value < old)
    for _ in range(levels):
        parent = torch.clamp((pidx - 1) // 2, min=0)
        cv, pv = values[rows, pidx], values[rows, parent]
        ch, ph = handles[rows, pidx], handles[rows, parent]
        swap = moving & (pidx > 0) & (cv < pv)
        values[rows, pidx] = torch.where(swap, pv, cv)
        values[rows, parent] = torch.where(swap, cv, pv)
        handles[rows, pidx] = torch.where(swap, ph, ch)
        handles[rows, parent] = torch.where(swap, ch, ph)
        pos[rows, ch] = torch.where(swap, parent, pidx)
        pos[rows, ph] = torch.where(swap, pidx, parent)
        pidx = torch.where(swap, parent, pidx)
        moving &= swap

    moving = active & ~(new_value < old)
    for _ in range(levels):
        left = 2 * pidx + 1
        has_left = left < n
        left_safe = torch.clamp(left, max=n - 1)
        right = left + 1
        has_right = right < n
        right_safe = torch.clamp(right, max=n - 1)
        lv, rv = values[rows, left_safe], values[rows, right_safe]
        child = torch.where(has_right & (rv < lv), right_safe, left_safe)
        child_v, cur_v = values[rows, child], values[rows, pidx]
        child_h, cur_h = handles[rows, child], handles[rows, pidx]
        swap = moving & has_left & (child_v < cur_v)
        values[rows, pidx] = torch.where(swap, child_v, cur_v)
        values[rows, child] = torch.where(swap, cur_v, child_v)
        handles[rows, pidx] = torch.where(swap, child_h, cur_h)
        handles[rows, child] = torch.where(swap, cur_h, child_h)
        pos[rows, cur_h] = torch.where(swap, child, pidx)
        pos[rows, child_h] = torch.where(swap, pidx, child)
        pidx = torch.where(swap, child, pidx)
        moving &= swap


def simulate_batch_gpu(
    p: LIFParams,
    conn: TorchConnectivity,
    input_times: torch.Tensor,
    input_channels: torch.Tensor,
    *,
    max_events: int,
    max_time: float,
    initial_v: Optional[torch.Tensor] = None,
    override_edge: Optional[torch.Tensor] = None,
    override_delta: Optional[torch.Tensor] = None,
) -> GPUTape:
    """Batched sparse event simulation using torch CUDA tensors.

    Each sample owns a mutable binary heap. An event touches only the firing
    source and that source's outgoing sparse edges. There is no dense
    neuron-by-timestep update.
    """
    device = conn.device
    input_times = input_times.to(device=device, dtype=torch.float32)
    input_channels = input_channels.to(device=device, dtype=torch.long)
    bsz, nin_events = input_times.shape
    n, kmax = conn.n_neurons, conn.max_fanout
    rows = torch.arange(bsz, device=device)
    init_v = torch.zeros((bsz, n), dtype=torch.float32, device=device) if initial_v is None else initial_v.to(device=device, dtype=torch.float32)
    t0 = torch.zeros((bsz, n), dtype=torch.float32, device=device)
    init_sched, _, _ = _schedule(init_v, t0, p)
    heap_values, heap_handles, heap_pos = _build_heap(init_sched)
    current_node = torch.arange(n, device=device, dtype=torch.long).expand(bsz, n).clone()

    max_nodes = n + max_events * (1 + kmax)
    node_v = torch.zeros((bsz, max_nodes), dtype=torch.float32, device=device)
    node_t = torch.zeros_like(node_v)
    node_sched = torch.full_like(node_v, float("inf"))
    node_v[:, :n], node_sched[:, :n] = init_v, init_sched

    eshape = (max_events, bsz)
    pshape = (max_events, bsz, kmax)
    event_time = torch.full(eshape, float("inf"), dtype=torch.float32, device=device)
    event_active = torch.zeros(eshape, dtype=torch.bool, device=device)
    event_is_spike = torch.zeros(eshape, dtype=torch.bool, device=device)
    event_spiker = torch.full(eshape, -1, dtype=torch.long, device=device)
    event_input_channel = torch.full(eshape, -1, dtype=torch.long, device=device)
    event_source_node = torch.full(eshape, -1, dtype=torch.long, device=device)
    event_reset_node = torch.full(eshape, -1, dtype=torch.long, device=device)
    post_valid = torch.zeros(pshape, dtype=torch.bool, device=device)
    post_edge = torch.full(pshape, -1, dtype=torch.long, device=device)
    post_prev_node = torch.full(pshape, -1, dtype=torch.long, device=device)
    post_new_node = torch.full(pshape, -1, dtype=torch.long, device=device)
    post_flow_a = torch.zeros(pshape, dtype=torch.float32, device=device)
    post_flow_slope = torch.zeros(pshape, dtype=torch.float32, device=device)
    post_schedule_dv = torch.zeros(pshape, dtype=torch.float32, device=device)
    post_new_v = torch.zeros(pshape, dtype=torch.float32, device=device)
    post_scheduled_t = torch.full(pshape, float("inf"), dtype=torch.float32, device=device)
    post_immediate = torch.zeros(pshape, dtype=torch.bool, device=device)
    input_ptr = torch.zeros(bsz, dtype=torch.long, device=device)
    syn_updates = torch.zeros(bsz, dtype=torch.long, device=device)
    input_count = torch.isfinite(input_times).sum(dim=1).to(torch.long)

    if override_edge is None:
        override_edge = torch.full((bsz,), -1, dtype=torch.long, device=device)
        override_delta = torch.zeros((bsz,), dtype=torch.float32, device=device)
    else:
        override_edge = override_edge.to(device=device, dtype=torch.long)
        override_delta = override_delta.to(device=device, dtype=torch.float32)

    actual_steps = max_events
    for step in range(max_events):
        if nin_events:
            ptr_safe = torch.clamp(input_ptr, max=nin_events - 1)
            next_input_t = input_times[rows, ptr_safe]
            next_input_ch = input_channels[rows, ptr_safe]
            next_input_t = torch.where(input_ptr < nin_events, next_input_t, torch.full_like(next_input_t, float("inf")))
        else:
            next_input_t = torch.full((bsz,), float("inf"), device=device)
            next_input_ch = torch.zeros((bsz,), dtype=torch.long, device=device)
        next_spike_t, spiker = heap_values[:, 0], heap_handles[:, 0]
        choose_input = next_input_t <= next_spike_t
        t = torch.minimum(next_input_t, next_spike_t)
        active = torch.isfinite(t) & (t <= float(max_time))
        if not bool(active.any()):
            actual_steps = step
            break

        is_input, is_spike = active & choose_input, active & (~choose_input)
        event_time[step], event_active[step], event_is_spike[step] = t, active, is_spike
        event_spiker[step] = torch.where(is_spike, spiker, torch.full_like(spiker, -1))
        event_input_channel[step] = torch.where(is_input, next_input_ch, torch.full_like(next_input_ch, -1))
        input_ptr += is_input.to(torch.long)
        source_node = current_node[rows, spiker]
        event_source_node[step] = torch.where(is_spike, source_node, torch.full_like(source_node, -1))

        reset_idx = n + step * (1 + kmax)
        reset_v = torch.full((bsz,), float(p.v_reset), dtype=torch.float32, device=device)
        reset_sched, _, _ = _schedule(reset_v, t, p)
        node_v[:, reset_idx] = torch.where(is_spike, reset_v, node_v[:, reset_idx])
        node_t[:, reset_idx] = torch.where(is_spike, t, node_t[:, reset_idx])
        node_sched[:, reset_idx] = torch.where(is_spike, reset_sched, node_sched[:, reset_idx])
        old_cur = current_node[rows, spiker]
        current_node[rows, spiker] = torch.where(is_spike, torch.full_like(old_cur, reset_idx), old_cur)
        event_reset_node[step] = torch.where(is_spike, torch.full_like(spiker, reset_idx), torch.full_like(spiker, -1))
        _heap_update(heap_values, heap_handles, heap_pos, spiker, reset_sched, is_spike)

        source_global = torch.where(is_input, next_input_ch, conn.n_inputs + spiker)
        source_safe = torch.where(active, source_global, torch.zeros_like(source_global))
        edge_block = conn.outgoing[source_safe]
        for kk in range(kmax):
            edge = edge_block[:, kk]
            valid = active & (edge >= 0)
            edge_safe = torch.where(valid, edge, torch.zeros_like(edge))
            post = conn.post[edge_safe]
            prev_node = current_node[rows, post]
            prev_v, prev_t = node_v[rows, prev_node], node_t[rows, prev_node]
            u, a, slope = _flow(prev_v, torch.clamp(t - prev_t, min=0.0), p)
            w = conn.weight[edge_safe]
            w += torch.where(edge_safe == override_edge, override_delta, torch.zeros_like(w))
            new_v = u + w
            new_sched, schedule_dv, immediate = _schedule(new_v, t, p)
            new_idx = reset_idx + 1 + kk
            node_v[:, new_idx] = torch.where(valid, new_v, node_v[:, new_idx])
            node_t[:, new_idx] = torch.where(valid, t, node_t[:, new_idx])
            node_sched[:, new_idx] = torch.where(valid, new_sched, node_sched[:, new_idx])
            old_cur = current_node[rows, post]
            current_node[rows, post] = torch.where(valid, torch.full_like(old_cur, new_idx), old_cur)
            _heap_update(heap_values, heap_handles, heap_pos, post, new_sched, valid)

            post_valid[step, :, kk] = valid
            post_edge[step, :, kk] = torch.where(valid, edge, torch.full_like(edge, -1))
            post_prev_node[step, :, kk] = torch.where(valid, prev_node, torch.full_like(prev_node, -1))
            post_new_node[step, :, kk] = torch.where(valid, torch.full_like(prev_node, new_idx), torch.full_like(prev_node, -1))
            post_flow_a[step, :, kk] = torch.where(valid, a, torch.zeros_like(a))
            post_flow_slope[step, :, kk] = torch.where(valid, slope, torch.zeros_like(slope))
            post_schedule_dv[step, :, kk] = torch.where(valid, schedule_dv, torch.zeros_like(schedule_dv))
            post_new_v[step, :, kk] = torch.where(valid, new_v, torch.zeros_like(new_v))
            post_scheduled_t[step, :, kk] = torch.where(valid, new_sched, torch.full_like(new_sched, float("inf")))
            post_immediate[step, :, kk] = valid & immediate
            syn_updates += valid.to(torch.long)

    sl = slice(0, actual_steps)
    return GPUTape(
        event_time[sl], event_active[sl], event_is_spike[sl], event_spiker[sl], event_input_channel[sl],
        event_source_node[sl], event_reset_node[sl], post_valid[sl], post_edge[sl], post_prev_node[sl],
        post_new_node[sl], post_flow_a[sl], post_flow_slope[sl], post_schedule_dv[sl], post_new_v[sl],
        post_scheduled_t[sl], post_immediate[sl], node_v, node_t, node_sched, syn_updates, input_count,
    )


def exact_event_gradient_gpu(tape: GPUTape, conn: TorchConnectivity, event_credit: torch.Tensor):
    """Per-sample exact fixed-event-graph gradient on the same torch device."""
    device = conn.device
    credit = event_credit.to(device=device, dtype=torch.float32)
    s_count, bsz = tape.event_time.shape
    if credit.shape != (s_count, bsz):
        raise ValueError("event_credit shape mismatch")
    rows = torch.arange(bsz, device=device)
    adj_v, adj_t, adj_s = torch.zeros_like(tape.node_v), torch.zeros_like(tape.node_t), torch.zeros_like(tape.node_scheduled_t)
    grad = torch.zeros((bsz, conn.n_edges), dtype=torch.float32, device=device)
    for s in range(s_count - 1, -1, -1):
        adj_event_t = credit[s].clone()
        for kk in range(conn.max_fanout - 1, -1, -1):
            valid = tape.post_valid[s, :, kk]
            edge = torch.where(valid, tape.post_edge[s, :, kk], torch.zeros_like(tape.post_edge[s, :, kk]))
            prev = torch.where(valid, tape.post_prev_node[s, :, kk], torch.zeros_like(tape.post_prev_node[s, :, kk]))
            new = torch.where(valid, tape.post_new_node[s, :, kk], torch.zeros_like(tape.post_new_node[s, :, kk]))
            lv, lt, ls = adj_v[rows, new], adj_t[rows, new], adj_s[rows, new]
            anv = lv + ls * tape.post_schedule_dv[s, :, kk]
            val = torch.where(valid, anv, torch.zeros_like(anv))
            grad.scatter_add_(1, edge[:, None], val[:, None])
            adj_v.scatter_add_(1, prev[:, None], (val * tape.post_flow_a[s, :, kk])[:, None])
            adj_t.scatter_add_(1, prev[:, None], (-val * tape.post_flow_slope[s, :, kk])[:, None])
            adj_event_t += torch.where(valid, lt + ls + anv * tape.post_flow_slope[s, :, kk], torch.zeros_like(adj_event_t))
        spike = tape.event_active[s] & tape.event_is_spike[s]
        reset = torch.where(spike, tape.event_reset_node[s], torch.zeros_like(tape.event_reset_node[s]))
        source = torch.where(spike, tape.event_source_node[s], torch.zeros_like(tape.event_source_node[s]))
        adj_event_t += torch.where(spike, adj_t[rows, reset] + adj_s[rows, reset], torch.zeros_like(adj_event_t))
        adj_s.scatter_add_(1, source[:, None], torch.where(spike, adj_event_t, torch.zeros_like(adj_event_t))[:, None])
    return grad


def first_spike_loss_gpu(tape: GPUTape, output_start: int, n_outputs: int,
                         labels: torch.Tensor, horizon: float, temperature: float):
    device = tape.event_time.device
    s_count, bsz = tape.event_time.shape
    rows = torch.arange(bsz, device=device)
    first_t = torch.full((bsz, n_outputs), float(horizon), dtype=torch.float32, device=device)
    first_idx = torch.full((bsz, n_outputs), -1, dtype=torch.long, device=device)
    for s in range(s_count):
        sp = tape.event_spiker[s]
        valid = tape.event_active[s] & tape.event_is_spike[s] & (sp >= output_start) & (sp < output_start + n_outputs)
        cls = torch.clamp(sp - output_start, min=0, max=n_outputs - 1)
        old = first_idx[rows, cls]
        take = valid & (old < 0)
        first_idx[rows, cls] = torch.where(take, torch.full_like(old, s), old)
        first_t[rows, cls] = torch.where(take, tape.event_time[s], first_t[rows, cls])
    labels = labels.to(device=device, dtype=torch.long)
    logits = -first_t / float(temperature)
    log_probs = torch.log_softmax(logits, dim=1)
    losses = -log_probs[rows, labels]
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmin(first_t, dim=1)
    missing = (first_idx < 0).sum(dim=1)
    dldt = -probs / float(temperature)
    dldt[rows, labels] += 1.0 / float(temperature)
    credit = torch.zeros((s_count, bsz), dtype=torch.float32, device=device)
    for cls in range(n_outputs):
        idx = first_idx[:, cls]
        valid = idx >= 0
        idx_safe = torch.where(valid, idx, torch.zeros_like(idx))
        credit[idx_safe, rows] += torch.where(valid, dldt[:, cls], torch.zeros_like(dldt[:, cls]))
    return losses, credit, pred, first_t, first_idx, missing


def boundary_stats_gpu(tape: GPUTape, conn: TorchConnectivity, p: LIFParams,
                       exact_grad: torch.Tensor, *, threshold_scale: float,
                       order_scale: float, gradient_floor: float):
    """Boundary distances are accumulated only from executed sparse transitions."""
    s_count, bsz, kmax = tape.post_valid.shape
    ecount, device = conn.n_edges, conn.device
    rows3 = torch.arange(bsz, device=device)[None, :, None].expand(s_count, bsz, kmax)
    edge = torch.where(tape.post_valid, tape.post_edge, torch.zeros_like(tape.post_edge))
    flat_idx = (rows3 * ecount + edge).reshape(-1)
    valid = tape.post_valid.reshape(-1)
    usage = torch.zeros(bsz * ecount, dtype=torch.float32, device=device)
    usage.scatter_add_(0, flat_idx, valid.float())
    min_th = torch.full((bsz * ecount,), float("inf"), dtype=torch.float32, device=device)
    dth = torch.abs(float(p.v_thresh) - tape.post_new_v).reshape(-1)
    min_th.scatter_reduce_(0, flat_idx, torch.where(valid, dth, torch.full_like(dth, float("inf"))), reduce="amin", include_self=True)
    imm = torch.zeros(bsz * ecount, dtype=torch.float32, device=device)
    imm.scatter_reduce_(0, flat_idx, (tape.post_immediate.reshape(-1) & valid).float(), reduce="amax", include_self=True)

    next1 = torch.full((s_count, bsz), float("inf"), dtype=torch.float32, device=device)
    next2 = torch.full_like(next1, float("inf"))
    if s_count > 1: next1[:-1] = tape.event_time[1:]
    if s_count > 2: next2[:-2] = tape.event_time[2:]
    n1, n2 = next1[:, :, None].expand_as(tape.post_scheduled_t), next2[:, :, None].expand_as(tape.post_scheduled_t)
    sens = torch.abs(tape.post_schedule_dv)
    same = torch.abs(tape.post_scheduled_t - n1) <= 1e-7
    margin = torch.where(same & torch.isfinite(n2), torch.abs(n2 - n1), torch.abs(tape.post_scheduled_t - n1))
    dorder = margin / torch.clamp(sens, min=1e-12)
    order_valid = tape.post_valid & torch.isfinite(tape.post_scheduled_t) & torch.isfinite(n1) & (sens > 1e-12)
    min_order = torch.full((bsz * ecount,), float("inf"), dtype=torch.float32, device=device)
    dov = dorder.reshape(-1)
    min_order.scatter_reduce_(0, flat_idx, torch.where(order_valid.reshape(-1), dov, torch.full_like(dov, float("inf"))), reduce="amin", include_self=True)

    usage, min_th, min_order, imm = [x.view(bsz, ecount) for x in (usage, min_th, min_order, imm)]
    rth = torch.where(torch.isfinite(min_th), torch.exp(-min_th / float(threshold_scale)), torch.zeros_like(min_th))
    ror = torch.where(torch.isfinite(min_order), torch.exp(-min_order / float(order_scale)), torch.zeros_like(min_order))
    risk = torch.maximum(torch.maximum(rth, ror), imm)
    score = risk * (torch.abs(exact_grad) + float(gradient_floor)) * torch.sqrt(torch.clamp(usage, min=0.0))
    return usage, min_th, min_order, risk, score


def event_signature_changed(base: GPUTape, probe: GPUTape, origin: torch.Tensor):
    """Hard event identity changed under a probe."""
    device = probe.event_time.device
    max_s = max(base.steps, probe.steps)
    def pad(t):
        b = t.batch_size
        active = torch.zeros((max_s, b), dtype=torch.bool, device=device)
        spike = torch.zeros((max_s, b), dtype=torch.bool, device=device)
        spiker = torch.full((max_s, b), -1, dtype=torch.long, device=device)
        inp = torch.full((max_s, b), -1, dtype=torch.long, device=device)
        active[:t.steps], spike[:t.steps] = t.event_active, t.event_is_spike
        spiker[:t.steps], inp[:t.steps] = t.event_spiker, t.event_input_channel
        return active, spike, spiker, inp
    ba, bs, bj, bi = pad(base)
    pa, ps, pj, pi = pad(probe)
    return ((pa != ba[:, origin]) | (ps != bs[:, origin]) | (pj != bj[:, origin]) | (pi != bi[:, origin])).any(dim=0)
