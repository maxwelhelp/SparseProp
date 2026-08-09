from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .core import LIFParams
from .cifar100 import CIFAR100EventConfig, make_cifar100_network
from .gpu_engine import (
    TorchConnectivity,
    boundary_stats_gpu,
    event_signature_changed,
    exact_event_gradient_gpu,
    first_spike_loss_gpu,
    simulate_batch_gpu,
)


@dataclass(frozen=True)
class GPUV11Config:
    risk_threshold: float = 0.20
    threshold_scale: float = 0.05
    order_scale: float = 0.02
    epsilon_min: float = 1e-4
    epsilon_max: float = 0.10
    boundary_overshoot: float = 1.25
    max_probe_pairs_per_sample: int = 12
    max_probe_pairs_per_edge: int = 3
    gradient_floor: float = 0.05
    improvement_tol: float = 1e-7
    zo_gradient_clip: float = 5.0


@dataclass
class GPUTrainStats:
    loss: float
    accuracy: float
    input_events: float
    network_events: float
    synaptic_updates: float
    missing_outputs: float
    candidates: float = 0.0
    corrected: float = 0.0
    crossings: float = 0.0
    probe_pairs: float = 0.0


def encode_batch_events_gpu(images: torch.Tensor, cfg: CIFAR100EventConfig):
    x = images.float() / 255.0
    pooled = F.avg_pool2d(x, kernel_size=cfg.patch_size, stride=cfg.patch_size)
    flat = pooled.flatten(1)
    active = flat >= float(cfg.event_threshold)
    times = float(cfg.t_min) + (1.0 - flat) * (float(cfg.t_max) - float(cfg.t_min))
    times = torch.where(active, times, torch.full_like(times, float("inf")))
    channels = torch.arange(flat.shape[1], device=images.device, dtype=torch.long).expand_as(flat)
    order = torch.argsort(times, dim=1)
    return torch.gather(times, 1, order), torch.gather(channels, 1, order), active.sum(dim=1)


def make_gpu_network(cfg: CIFAR100EventConfig, seed: int, device: torch.device | str):
    return TorchConnectivity.from_numpy(make_cifar100_network(cfg, seed=seed), device)


def _allocate_probe_edges(score: torch.Tensor, risk: torch.Tensor, cfg: GPUV11Config):
    """Soft stochastic budget, rather than deterministic hard top-k."""
    bsz, ecount = score.shape
    selected = torch.full((bsz, cfg.max_probe_pairs_per_sample), -1, dtype=torch.long, device=score.device)
    counts = torch.zeros((bsz, ecount), dtype=torch.long, device=score.device)
    for b in range(bsz):
        probs = torch.where(risk[b] >= cfg.risk_threshold, score[b], torch.zeros_like(score[b]))
        if float(probs.sum().item()) <= 0.0:
            continue
        probs = probs / probs.sum()
        for slot in range(cfg.max_probe_pairs_per_sample):
            allowed = counts[b] < cfg.max_probe_pairs_per_edge
            p = torch.where(allowed, probs, torch.zeros_like(probs))
            if float(p.sum().item()) <= 0.0:
                break
            edge = torch.multinomial(p, 1, replacement=True)[0]
            selected[b, slot] = edge
            counts[b, edge] += 1
    return selected


def hybrid_gradient_gpu(
    p: LIFParams,
    conn: TorchConnectivity,
    input_times: torch.Tensor,
    input_channels: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_start: int,
    n_outputs: int,
    horizon: float,
    temperature: float,
    max_events: int,
    cfg: GPUV11Config,
):
    base = simulate_batch_gpu(p, conn, input_times, input_channels, max_events=max_events, max_time=horizon)
    losses, credit, pred, _, _, missing = first_spike_loss_gpu(base, output_start, n_outputs, labels, horizon, temperature)
    exact = exact_event_gradient_gpu(base, conn, credit)
    usage, dth, dorder, risk, score = boundary_stats_gpu(
        base, conn, p, exact,
        threshold_scale=cfg.threshold_scale,
        order_scale=cfg.order_scale,
        gradient_floor=cfg.gradient_floor,
    )
    selected = _allocate_probe_edges(score, risk, cfg)

    origins, edges, deltas, epsilons, signs = [], [], [], [], []
    scales = (1.0, 0.5, 2.0)
    bsz = labels.numel()
    for b in range(bsz):
        seen = {}
        for slot in range(cfg.max_probe_pairs_per_sample):
            e = int(selected[b, slot].item())
            if e < 0:
                continue
            occ = seen.get(e, 0)
            seen[e] = occ + 1
            boundary = torch.minimum(dth[b, e], dorder[b, e])
            if not bool(torch.isfinite(boundary)):
                boundary = torch.as_tensor(cfg.epsilon_max, device=conn.device)
            eps = float(torch.clamp(
                cfg.boundary_overshoot * torch.clamp(boundary, min=cfg.epsilon_min),
                min=cfg.epsilon_min, max=cfg.epsilon_max,
            ).item()) * scales[min(occ, len(scales) - 1)]
            eps = float(np.clip(eps, cfg.epsilon_min, cfg.epsilon_max))
            for sign in (+1.0, -1.0):
                origins.append(b); edges.append(e); deltas.append(sign * eps); epsilons.append(eps); signs.append(sign)

    if not origins:
        zero = torch.zeros(bsz, device=conn.device)
        return exact, losses, pred, missing, base, {
            "candidate_edges": ((risk >= cfg.risk_threshold) & (usage > 0)).sum(dim=1).float(),
            "corrected_edges": zero, "crossings": zero, "probe_pairs": zero,
        }

    origin = torch.tensor(origins, dtype=torch.long, device=conn.device)
    edge_t = torch.tensor(edges, dtype=torch.long, device=conn.device)
    delta_t = torch.tensor(deltas, dtype=torch.float32, device=conn.device)
    eps_t = torch.tensor(epsilons, dtype=torch.float32, device=conn.device)
    sign_t = torch.tensor(signs, dtype=torch.float32, device=conn.device)
    probe = simulate_batch_gpu(
        p, conn, input_times[origin], input_channels[origin],
        max_events=max_events, max_time=horizon,
        override_edge=edge_t, override_delta=delta_t,
    )
    probe_losses, _, _, _, _, _ = first_spike_loss_gpu(
        probe, output_start, n_outputs, labels[origin], horizon, temperature
    )
    changed = event_signature_changed(base, probe, origin)
    base_loss = losses[origin]
    certified = (probe_losses < base_loss - cfg.improvement_tol) & changed
    pseudo = -sign_t * (base_loss - probe_losses) / eps_t
    pseudo = torch.where(certified, torch.clamp(pseudo, -cfg.zo_gradient_clip, cfg.zo_gradient_clip), torch.zeros_like(pseudo))

    flat = origin * conn.n_edges + edge_t
    sum_grad = torch.zeros(bsz * conn.n_edges, dtype=torch.float32, device=conn.device)
    count = torch.zeros_like(sum_grad)
    sum_grad.scatter_add_(0, flat, pseudo)
    count.scatter_add_(0, flat, certified.float())
    z = sum_grad.view(bsz, conn.n_edges) / torch.clamp(count.view(bsz, conn.n_edges), min=1.0)
    corrected = count.view(bsz, conn.n_edges) > 0
    hybrid = torch.where(corrected, z, exact)

    crossings = torch.zeros(bsz, dtype=torch.float32, device=conn.device)
    crossings.scatter_add_(0, origin, changed.float())
    probe_pairs = torch.zeros(bsz, dtype=torch.float32, device=conn.device)
    probe_pairs.scatter_add_(0, origin, torch.full_like(origin, 0.5, dtype=torch.float32))
    return hybrid, losses, pred, missing, base, {
        "candidate_edges": ((risk >= cfg.risk_threshold) & (usage > 0)).sum(dim=1).float(),
        "corrected_edges": corrected.sum(dim=1).float(),
        "crossings": crossings,
        "probe_pairs": probe_pairs,
    }


def train_batch_gpu(
    p: LIFParams,
    conn: TorchConnectivity,
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: CIFAR100EventConfig,
    *,
    mode: Literal["exact", "hybrid"],
    v11: GPUV11Config,
):
    input_times, input_channels, input_count = encode_batch_events_gpu(images, cfg)
    if mode == "hybrid":
        grad_ps, losses, pred, missing, tape, diag = hybrid_gradient_gpu(
            p, conn, input_times, input_channels, labels,
            output_start=cfg.hidden, n_outputs=100,
            horizon=cfg.horizon, temperature=cfg.temperature,
            max_events=cfg.max_events, cfg=v11,
        )
    else:
        tape = simulate_batch_gpu(p, conn, input_times, input_channels, max_events=cfg.max_events, max_time=cfg.horizon)
        losses, credit, pred, _, _, missing = first_spike_loss_gpu(tape, cfg.hidden, 100, labels, cfg.horizon, cfg.temperature)
        grad_ps = exact_event_gradient_gpu(tape, conn, credit)
        z = torch.zeros(labels.numel(), device=conn.device)
        diag = {"candidate_edges": z, "corrected_edges": z, "crossings": z, "probe_pairs": z}

    grad = grad_ps.mean(dim=0)
    stats = GPUTrainStats(
        loss=float(losses.mean().item()),
        accuracy=float((pred == labels).float().mean().item()),
        input_events=float(input_count.float().mean().item()),
        network_events=float(tape.event_is_spike.float().sum(dim=0).mean().item()),
        synaptic_updates=float(tape.synaptic_updates.float().mean().item()),
        missing_outputs=float(missing.float().mean().item()),
        candidates=float(diag["candidate_edges"].mean().item()),
        corrected=float(diag["corrected_edges"].mean().item()),
        crossings=float(diag["crossings"].mean().item()),
        probe_pairs=float(diag["probe_pairs"].mean().item()),
    )
    return grad, stats
