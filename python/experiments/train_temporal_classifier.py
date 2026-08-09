from sparseprop_v11 import evaluate, train_temporal_classifier


if __name__ == "__main__":
    p, conn, history = train_temporal_classifier(
        epochs=12, steps_per_epoch=25, batch_size=32, seed=0, lr=1e-2
    )
    for h in history:
        print(
            f"epoch={h['epoch']:02d} loss={h['loss']:.5f} "
            f"acc={100*h['accuracy']:.2f}% "
            f"syn_updates/sample={h['synaptic_updates_per_sample']:.1f}"
        )
    metrics = evaluate(p, conn, n=1000, seed=999)
    print("final weights:", conn.weight.tolist())
    print(
        f"eval loss={metrics['loss']:.5f} "
        f"accuracy={100*metrics['accuracy']:.2f}% "
        f"syn_updates/sample={metrics['synaptic_updates_per_sample']:.1f}"
    )
