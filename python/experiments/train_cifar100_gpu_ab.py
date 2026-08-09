from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from sparseprop_v11.core import LIFParams
from sparseprop_v11.cifar100 import CIFAR100EventConfig, load_cifar100_python
from sparseprop_v11.cifar100_gpu import GPUV11Config, make_gpu_network, train_batch_gpu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--mode", choices=["exact", "hybrid", "both"], default="both")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--max-events", type=int, default=384)
    return p.parse_args()


def run(mode, args, images, labels):
    device = torch.device("cuda")
    cfg = CIFAR100EventConfig(
        patch_size=args.patch_size,
        hidden=args.hidden,
        max_events=args.max_events,
        tau=0.55,
        horizon=1.6,
    )
    # Slow analytic background crossing: exact event-time gradients remain
    # meaningful, but many outputs are beyond the finite horizon until input
    # spikes accelerate them. Hybrid V11 can then correct horizon/birth/order
    # boundaries without giving exact-only an artificially zero-gradient model.
    p = LIFParams(tau=cfg.tau, v_inf=1.015, v_thresh=cfg.v_thresh, v_reset=cfg.v_reset)
    conn = make_gpu_network(cfg, args.seed, device)
    v11 = GPUV11Config(max_probe_pairs_per_sample=8)
    m = torch.zeros_like(conn.weight)
    v = torch.zeros_like(conn.weight)
    step = 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(min(args.train_samples, len(images)))
        sums = None
        nb = 0
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x = torch.as_tensor(images[idx], dtype=torch.uint8, device=device)
            y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
            grad, st = train_batch_gpu(p, conn, x, y, cfg, mode=mode, v11=v11)
            step += 1
            m.mul_(0.9).add_(grad, alpha=0.1)
            v.mul_(0.999).addcmul_(grad, grad, value=0.001)
            mhat = m / (1.0 - 0.9 ** step)
            vhat = v / (1.0 - 0.999 ** step)
            conn.weight.addcdiv_(mhat, torch.sqrt(vhat) + 1e-8, value=-args.lr)
            conn.weight.clamp_(-0.25, 0.35)
            vals = np.array([
                st.loss, st.accuracy, st.input_events, st.network_events,
                st.synaptic_updates, st.missing_outputs, st.candidates,
                st.corrected, st.crossings, st.probe_pairs,
            ], dtype=np.float64)
            sums = vals if sums is None else sums + vals
            nb += 1
        torch.cuda.synchronize()
        avg = sums / max(nb, 1)
        dt = time.perf_counter() - t0
        print(
            f"[{mode}] e{epoch:02d} loss={avg[0]:.4f} acc={100*avg[1]:.2f}% "
            f"in={avg[2]:.1f} spikes={avg[3]:.1f} syn={avg[4]:.0f} missing={avg[5]:.1f} "
            f"cand={avg[6]:.1f} corr={avg[7]:.1f} cross={avg[8]:.1f} probes={avg[9]:.1f} "
            f"time={dt:.1f}s mem={torch.cuda.max_memory_allocated()/2**20:.0f}MiB"
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment; CPU fallback is intentionally disabled")
    print(f"[device] {torch.cuda.get_device_name(0)} | torch={torch.__version__} | cuda={torch.version.cuda}")
    images, labels = load_cifar100_python(args.dataset_root, "train")
    modes = ["exact", "hybrid"] if args.mode == "both" else [args.mode]
    for mode in modes:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        run(mode, args, images, labels)


if __name__ == "__main__":
    main()
