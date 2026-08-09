import argparse
from pathlib import Path
import time

from sparseprop_v11.cifar100 import (
    CIFAR100EventConfig,
    evaluate_cifar100,
    load_cifar100_python,
    make_cifar100_network,
    train_cifar100,
)


def fmt(prefix, row):
    extras = ""
    if row.get("probe_pairs_per_sample", 0.0) > 0.0:
        extras = (
            f" cand={row['candidate_edges_per_sample']:.2f}"
            f" corr={row['corrected_edges_per_sample']:.2f}"
            f" cross={row['signature_change_edges_per_sample']:.2f}"
            f" improve={row['improving_edges_per_sample']:.2f}"
            f" probes={row['probe_pairs_per_sample']:.2f}"
        )
    print(
        f"{prefix} loss={row['loss']:.4f} acc={100*row['accuracy']:.2f}%"
        f" in_ev={row['input_events_per_sample']:.1f}"
        f" spikes={row['network_spikes_per_sample']:.1f}"
        f" syn={row['synaptic_updates_per_sample']:.1f}"
        f" missing_out={row['missing_outputs_per_sample']:.1f}{extras}"
    )


def run_mode(mode, train_x, train_y, test_x, test_y, cfg, args, initial_weights):
    start = time.perf_counter()
    p, conn, history = train_cifar100(
        train_x, train_y, cfg,
        mode=mode,
        epochs=args.epochs,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        initial_weights=initial_weights,
    )
    seconds = time.perf_counter() - start
    for row in history:
        fmt(f"[{mode} e{int(row['epoch']):02d}]", row)
    eval_row = evaluate_cifar100(
        p, conn, test_x, test_y, cfg, limit=args.test_samples
    )
    fmt(f"[{mode} eval]", eval_row)
    print(f"[{mode}] train_seconds={seconds:.2f}")
    return history, eval_row, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--mode", choices=["exact", "hybrid", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-samples", type=int, default=1000)
    ap.add_argument("--test-samples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=96)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    train_x, train_y = load_cifar100_python(root, "train")
    test_x, test_y = load_cifar100_python(root, "test")
    cfg = CIFAR100EventConfig(patch_size=args.patch_size, hidden=args.hidden)

    initial = make_cifar100_network(cfg, seed=args.seed).weight.copy()
    print(
        f"dataset train={len(train_x)} test={len(test_x)} "
        f"inputs={cfg.n_inputs} hidden={cfg.hidden} outputs=100 "
        f"weights={len(initial)}"
    )

    if args.mode in ("exact", "both"):
        run_mode("exact", train_x, train_y, test_x, test_y, cfg, args, initial.copy())
    if args.mode in ("hybrid", "both"):
        run_mode("hybrid", train_x, train_y, test_x, test_y, cfg, args, initial.copy())


if __name__ == "__main__":
    main()
