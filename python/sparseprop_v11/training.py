from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core import LIFParams, SparseConnectivity, exact_event_gradient, first_spike_events, simulate


def latency_cross_entropy(times: np.ndarray, target: int, temperature: float = 0.2):
    if not np.all(np.isfinite(times)):
        raise RuntimeError("all readout neurons must spike for latency loss")
    logits = -times / temperature
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= probs.sum()
    loss = -math.log(float(probs[target]) + 1e-12)
    dloss_dt = -probs / temperature
    dloss_dt[target] += 1.0 / temperature
    return loss, dloss_dt, probs


@dataclass
class Adam:
    size: int
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        self.m = np.zeros(self.size, dtype=np.float64)
        self.v = np.zeros(self.size, dtype=np.float64)
        self.step_index = 0

    def step(self, w: np.ndarray, grad: np.ndarray) -> None:
        self.step_index += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        mhat = self.m / (1.0 - self.beta1 ** self.step_index)
        vhat = self.v / (1.0 - self.beta2 ** self.step_index)
        w -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def temporal_earlier_channel_sample(rng: np.random.Generator, jitter: float = 0.03):
    target = int(rng.integers(0, 2))
    early = 0.2 + float(rng.normal(0.0, jitter))
    late = 0.8 + float(rng.normal(0.0, jitter))
    times = [late, late]
    times[target] = early
    return sorted([(times[0], 0), (times[1], 1)]), target


def make_two_readout_network(seed: int = 0) -> SparseConnectivity:
    rng = np.random.default_rng(seed)
    pre = [0, 0, 1, 1]
    post = [0, 1, 0, 1]
    weight = rng.normal(0.04, 0.01, 4)
    return SparseConnectivity.from_edges(2, 2, pre, post, weight)


def sample_loss_and_grad(
    p: LIFParams,
    conn: SparseConnectivity,
    input_events,
    target: int,
    *,
    temperature: float = 0.2,
    max_events: int = 12,
):
    tape = simulate(p, conn, input_events, max_events=max_events)
    event_index, times = first_spike_events(tape, [0, 1])
    if any(i is None for i in event_index):
        raise RuntimeError("readout failed to spike")
    loss, dloss_dt, probs = latency_cross_entropy(times, target, temperature)
    event_credit = np.zeros(len(tape.events), dtype=np.float64)
    for k, idx in enumerate(event_index):
        event_credit[int(idx)] = dloss_dt[k]
    grad = exact_event_gradient(tape, conn, event_credit)
    pred = int(np.argmin(times))
    return loss, grad, pred, probs, tape


def evaluate(p: LIFParams, conn: SparseConnectivity, *, n: int = 1000, seed: int = 1234):
    rng = np.random.default_rng(seed)
    correct = 0
    loss_sum = 0.0
    syn_updates = 0
    for _ in range(n):
        events, target = temporal_earlier_channel_sample(rng)
        loss, _, pred, _, tape = sample_loss_and_grad(p, conn, events, target)
        loss_sum += loss
        correct += int(pred == target)
        syn_updates += tape.synaptic_updates
    return {
        "loss": loss_sum / n,
        "accuracy": correct / n,
        "synaptic_updates_per_sample": syn_updates / n,
    }


def train_temporal_classifier(
    *,
    epochs: int = 20,
    steps_per_epoch: int = 50,
    batch_size: int = 32,
    seed: int = 0,
    lr: float = 1e-2,
):
    rng = np.random.default_rng(seed)
    p = LIFParams(tau=1.0, v_inf=1.3, v_thresh=1.0, v_reset=0.0)
    conn = make_two_readout_network(seed)
    opt = Adam(len(conn.weight), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_correct = 0
        running_syn = 0
        for _ in range(steps_per_epoch):
            batch_grad = np.zeros_like(conn.weight)
            for _ in range(batch_size):
                events, target = temporal_earlier_channel_sample(rng)
                loss, grad, pred, _, tape = sample_loss_and_grad(p, conn, events, target)
                batch_grad += grad
                running_loss += loss
                running_correct += int(pred == target)
                running_syn += tape.synaptic_updates
            batch_grad /= batch_size
            opt.step(conn.weight, batch_grad)
            # Phase-1 trainer intentionally stays in the fixed-order regime.
            # Boundary crossings are handled by the next V11 layer rather than hidden.
            np.clip(conn.weight, -0.2, 0.3, out=conn.weight)

        samples = steps_per_epoch * batch_size
        history.append({
            "epoch": epoch,
            "loss": running_loss / samples,
            "accuracy": running_correct / samples,
            "synaptic_updates_per_sample": running_syn / samples,
        })

    return p, conn, history
