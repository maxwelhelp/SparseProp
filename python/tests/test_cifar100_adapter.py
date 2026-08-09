import pickle

import numpy as np

from sparseprop_v11.cifar100 import (
    CIFAR100EventConfig,
    encode_image_events,
    load_cifar100_python,
    make_cifar100_network,
    sample_cifar_loss_and_grad,
)
from sparseprop_v11.core import LIFParams
from sparseprop_v11.hybrid import V11Config


def test_cifar_pickle_loader_and_event_encoder(tmp_path):
    x = np.zeros((2, 3072), dtype=np.uint8)
    x[0, :] = 255
    obj = {b"data": x, b"fine_labels": [3, 7]}
    with (tmp_path / "train").open("wb") as f:
        pickle.dump(obj, f)
    data, labels = load_cifar100_python(tmp_path, "train")
    assert data.shape == (2, 3, 32, 32)
    assert labels.tolist() == [3, 7]
    cfg = CIFAR100EventConfig(patch_size=8, hidden=8, input_fanout=2, hidden_fanout=5)
    events = encode_image_events(data[0], cfg)
    assert len(events) == cfg.n_inputs
    assert all(cfg.t_min <= t <= cfg.t_max for t, _ in events)


def test_cifar_sample_exact_and_hybrid_run():
    cfg = CIFAR100EventConfig(
        patch_size=8, hidden=12, input_fanout=3, hidden_fanout=12,
        max_events=180, horizon=1.8,
    )
    conn = make_cifar100_network(cfg, seed=4)
    p = LIFParams(tau=cfg.tau, v_inf=0.0, v_thresh=1.0, v_reset=0.0)
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(3, 32, 32), dtype=np.uint8)
    loss, grad, _, _, tape, stats = sample_cifar_loss_and_grad(
        p, conn, image, 2, cfg, mode="exact"
    )
    assert np.isfinite(loss)
    assert grad.shape == conn.weight.shape
    assert tape.synaptic_updates == stats.synaptic_updates

    v11 = V11Config(
        max_events=cfg.max_events, max_time=cfg.horizon,
        max_probe_pairs=1, max_total_probe_pairs=1,
    )
    loss2, grad2, _, _, _, stats2 = sample_cifar_loss_and_grad(
        p, conn, image, 2, cfg, mode="hybrid", v11=v11
    )
    assert np.isfinite(loss2)
    assert grad2.shape == conn.weight.shape
    assert stats2.probe_pairs <= 1
