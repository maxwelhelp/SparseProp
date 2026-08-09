from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Literal

import numpy as np

from .core import LIFParams, SparseConnectivity, exact_event_gradient, first_spike_events, simulate
from .hybrid import V11Config, hybrid_event_gradient
from .training import Adam


@dataclass(frozen=True)
class CIFAR100EventConfig:
    patch_size: int = 4
    event_threshold: float = 0.10
    t_min: float = 0.05
    t_max: float = 0.95
    hidden: int = 96
    input_fanout: int = 12
    hidden_fanout: int = 40
    input_weight_mean: float = 0.085
    input_weight_sd: float = 0.015
    output_weight_mean: float = 0.060
    output_weight_sd: float = 0.012
    tau: float = 0.55
    v_thresh: float = 1.0
    v_reset: float = 0.0
    horizon: float = 1.8
    temperature: float = 0.20
    max_events: int = 512
    weight_min: float = -0.25
    weight_max: float = 0.35

    def __post_init__(self) -> None:
        if 32 % self.patch_size != 0:
            raise ValueError("patch_size must divide 32")
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be > t_min")
        if self.horizon <= self.t_max:
            raise ValueError("horizon must be after the input-event window")

    @property
    def grid(self) -> int:
        return 32 // self.patch_size

    @property
    def n_inputs(self) -> int:
        return 3 * self.grid * self.grid

    @property
    def n_neurons(self) -> int:
        return self.hidden + 100

    @property
    def output_ids(self) -> list[int]:
        return list(range(self.hidden, self.hidden + 100))


@dataclass
class CIFARStepStats:
    loss: float = 0.0
    correct: int = 0
    samples: int = 0
    input_events: int = 0
    network_spikes: int = 0
    synaptic_updates: int = 0
    missing_outputs: int = 0
    candidates: int = 0
    corrected: int = 0
    signature_changes: int = 0
    improving_edges: int = 0
    probe_pairs: int = 0
    probe_simulations: int = 0

    def add(self, other: "CIFARStepStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def summary(self) -> dict[str, float]:
        n = max(self.samples, 1)
        return {
            "loss": self.loss / n,
            "accuracy": self.correct / n,
            "input_events_per_sample": self.input_events / n,
            "network_spikes_per_sample": self.network_spikes / n,
            "synaptic_updates_per_sample": self.synaptic_updates / n,
            "missing_outputs_per_sample": self.missing_outputs / n,
            "candidate_edges_per_sample": self.candidates / n,
            "corrected_edges_per_sample": self.corrected / n,
            "signature_change_edges_per_sample": self.signature_changes / n,
            "improving_edges_per_sample": self.improving_edges / n,
            "probe_pairs_per_sample": self.probe_pairs / n,
            "probe_simulations_per_sample": self.probe_simulations / n,
        }


def load_cifar100_python(root: str | Path, split: Literal["train", "test"] = "train"):
    path = Path(root) / split
    if not path.exists():
        raise FileNotFoundError(f"CIFAR-100 file not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="bytes")
    data = np.asarray(obj[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(obj[b"fine_labels"], dtype=np.int64)
    return data, labels


def encode_image_events(image: np.ndarray, cfg: CIFAR100EventConfig) -> list[tuple[float, int]]:
    x = np.asarray(image, dtype=np.float32)
    if x.shape != (3, 32, 32):
        raise ValueError("image must have shape (3,32,32)")
    x = x / 255.0
    p = cfg.patch_size
    g = cfg.grid
    patches = x.reshape(3, g, p, g, p).mean(axis=(2, 4)).reshape(-1)
    active = np.flatnonzero(patches >= cfg.event_threshold)
    if active.size == 0:
        active = np.array([int(np.argmax(patches))], dtype=np.int64)
    intensities = patches[active]
    times = cfg.t_min + (1.0 - intensities) * (cfg.t_max - cfg.t_min)
    events = [(float(t), int(ch)) for t, ch in zip(times, active)]
    events.sort(key=lambda z: (z[0], z[1]))
    return events


def make_cifar100_network(cfg: CIFAR100EventConfig, seed: int = 0) -> SparseConnectivity:
    rng = np.random.default_rng(seed)
    pre: list[int] = []
    post: list[int] = []
    weight: list[float] = []

    fan_in = min(cfg.input_fanout, cfg.hidden)
    for src in range(cfg.n_inputs):
        targets = rng.choice(cfg.hidden, size=fan_in, replace=False)
        for dst in targets:
            pre.append(src)
            post.append(int(dst))
            weight.append(float(rng.normal(cfg.input_weight_mean, cfg.input_weight_sd)))

    fan_out = min(cfg.hidden_fanout, 100)
    for h in range(cfg.hidden):
        classes = rng.choice(100, size=fan_out, replace=False)
        source_global = cfg.n_inputs + h
        for cls in classes:
            pre.append(source_global)
            post.append(cfg.hidden + int(cls))
            weight.append(float(rng.normal(cfg.output_weight_mean, cfg.output_weight_sd)))

    conn = SparseConnectivity.from_edges(cfg.n_inputs, cfg.n_neurons, pre, post, weight)
    np.clip(conn.weight, cfg.weight_min, cfg.weight_max, out=conn.weight)
    return conn


def cifar_latency_loss(tape, cfg: CIFAR100EventConfig, target: int):
    indices, times = first_spike_events(tape, cfg.output_ids)
    finite = np.isfinite(times)
    effective = np.where(finite, times, cfg.horizon)
    logits = -effective / cfg.temperature
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    loss = -math.log(float(probs[target]) + 1e-12)

    dloss_dt = -probs / cfg.temperature
    dloss_dt[target] += 1.0 / cfg.temperature
    credit = np.zeros(len(tape.events), dtype=np.float64)
    for k, event_idx in enumerate(indices):
        if event_idx is not None:
            credit[int(event_idx)] = dloss_dt[k]

    pred = int(np.argmin(effective))
    return loss, credit, pred, probs, int((~finite).sum())


def sample_cifar_loss_and_grad(
    p: LIFParams,
    conn: SparseConnectivity,
    image: np.ndarray,
    target: int,
    cfg: CIFAR100EventConfig,
    *,
    mode: Literal["exact", "hybrid"] = "exact",
    v11: V11Config | None = None,
):
    input_events = encode_image_events(image, cfg)
    tape = simulate(
        p, conn, input_events,
        max_events=cfg.max_events, max_time=cfg.horizon,
    )
    loss, credit, pred, probs, missing = cifar_latency_loss(tape, cfg, target)

    if mode == "exact":
        grad = exact_event_gradient(tape, conn, credit)
        diag = None
    elif mode == "hybrid":
        if v11 is None:
            v11 = V11Config(max_events=cfg.max_events, max_time=cfg.horizon)
        result = hybrid_event_gradient(
            p, conn, input_events, credit,
            lambda tt: cifar_latency_loss(tt, cfg, target)[0],
            base_tape=tape, config=v11,
        )
        grad = result.gradient
        diag = result
    else:
        raise ValueError("mode must be exact or hybrid")

    stats = CIFARStepStats(
        loss=float(loss),
        correct=int(pred == target),
        samples=1,
        input_events=len(input_events),
        network_spikes=len(tape.spike_times),
        synaptic_updates=tape.synaptic_updates,
        missing_outputs=missing,
    )
    if diag is not None:
        d = diag.diagnostics()
        stats.candidates = int(d["candidate_edges"])
        stats.corrected = int(d["corrected_edges"])
        stats.signature_changes = int(d["signature_change_edges"])
        stats.improving_edges = int(d["improving_probe_edges"])
        stats.probe_pairs = int(d["probe_pairs"])
        stats.probe_simulations = int(d["probe_simulations"])
    return loss, grad, pred, probs, tape, stats


def evaluate_cifar100(
    p: LIFParams,
    conn: SparseConnectivity,
    images: np.ndarray,
    labels: np.ndarray,
    cfg: CIFAR100EventConfig,
    *,
    limit: int | None = None,
):
    n = len(images) if limit is None else min(int(limit), len(images))
    total = CIFARStepStats()
    for i in range(n):
        events = encode_image_events(images[i], cfg)
        tape = simulate(p, conn, events, max_events=cfg.max_events, max_time=cfg.horizon)
        loss, _, pred, _, missing = cifar_latency_loss(tape, cfg, int(labels[i]))
        total.add(CIFARStepStats(
            loss=float(loss), correct=int(pred == int(labels[i])), samples=1,
            input_events=len(events), network_spikes=len(tape.spike_times),
            synaptic_updates=tape.synaptic_updates, missing_outputs=missing,
        ))
    return total.summary()


def train_cifar100(
    images: np.ndarray,
    labels: np.ndarray,
    cfg: CIFAR100EventConfig,
    *,
    mode: Literal["exact", "hybrid"] = "exact",
    epochs: int = 3,
    train_samples: int = 1000,
    batch_size: int = 8,
    lr: float = 2e-3,
    seed: int = 0,
    v11: V11Config | None = None,
    initial_weights: np.ndarray | None = None,
):
    p = LIFParams(
        tau=cfg.tau, v_inf=0.0, v_thresh=cfg.v_thresh, v_reset=cfg.v_reset
    )
    conn = make_cifar100_network(cfg, seed=seed)
    if initial_weights is not None:
        if initial_weights.shape != conn.weight.shape:
            raise ValueError("initial_weights shape mismatch")
        conn.weight[:] = initial_weights
    opt = Adam(len(conn.weight), lr=lr)
    rng = np.random.default_rng(seed)
    n = min(int(train_samples), len(images))
    history: list[dict[str, float]] = []

    if v11 is None:
        v11 = V11Config(
            risk_threshold=0.35,
            threshold_scale=0.04,
            order_scale=0.015,
            epsilon_min=5e-4,
            epsilon_max=0.08,
            boundary_overshoot=1.2,
            min_probe_pairs=1,
            max_probe_pairs=2,
            max_total_probe_pairs=2,
            zo_gradient_clip=3.0,
            max_events=cfg.max_events,
            max_time=cfg.horizon,
        )

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(images))[:n]
        total = CIFARStepStats()
        batch_grad = np.zeros_like(conn.weight)
        batch_count = 0

        for index in order:
            _, grad, _, _, _, stats = sample_cifar_loss_and_grad(
                p, conn, images[index], int(labels[index]), cfg,
                mode=mode, v11=v11,
            )
            batch_grad += grad
            batch_count += 1
            total.add(stats)

            if batch_count == batch_size:
                batch_grad /= batch_count
                opt.step(conn.weight, batch_grad)
                np.clip(conn.weight, cfg.weight_min, cfg.weight_max, out=conn.weight)
                batch_grad.fill(0.0)
                batch_count = 0

        if batch_count:
            batch_grad /= batch_count
            opt.step(conn.weight, batch_grad)
            np.clip(conn.weight, cfg.weight_min, cfg.weight_max, out=conn.weight)

        row = total.summary()
        row["epoch"] = float(epoch)
        history.append(row)

    return p, conn, history
