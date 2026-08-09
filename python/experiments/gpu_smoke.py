import numpy as np
import torch

from sparseprop_v11 import LIFParams, SparseConnectivity, exact_event_gradient, first_spike_events, simulate
from sparseprop_v11.gpu_engine import TorchConnectivity, exact_event_gradient_gpu, first_spike_loss_gpu, simulate_batch_gpu
from sparseprop_v11.training import latency_cross_entropy


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this smoke intentionally has no CPU fallback")
    dev = torch.device("cuda")
    print(f"[device] {torch.cuda.get_device_name(0)} | torch={torch.__version__} | cuda={torch.version.cuda}")
    p = LIFParams(tau=1.0, v_inf=1.3, v_thresh=1.0, v_reset=0.0)
    conn_np = SparseConnectivity.from_edges(
        2, 2, [0, 0, 1, 1], [0, 1, 0, 1], [0.035, 0.055, 0.060, 0.030]
    )
    inputs = [(0.2, 0), (0.8, 1)]
    ref = simulate(p, conn_np, inputs, max_events=10, max_time=3.0)
    idx, rt = first_spike_events(ref, [0, 1])
    _, dldt, _ = latency_cross_entropy(rt, 0, 0.2)
    credit = np.zeros(len(ref.events), dtype=np.float64)
    for k, event_idx in enumerate(idx):
        credit[event_idx] = dldt[k]
    ref_grad = exact_event_gradient(ref, conn_np, credit)

    conn = TorchConnectivity.from_numpy(conn_np, dev)
    times = torch.tensor([[0.2, 0.8]], dtype=torch.float32, device=dev)
    channels = torch.tensor([[0, 1]], dtype=torch.long, device=dev)
    tape = simulate_batch_gpu(p, conn, times, channels, max_events=10, max_time=3.0)
    labels = torch.tensor([0], dtype=torch.long, device=dev)
    _, tcredit, _, _, _, _ = first_spike_loss_gpu(tape, 0, 2, labels, 3.0, 0.2)
    grad = exact_event_gradient_gpu(tape, conn, tcredit)[0]
    torch.cuda.synchronize()

    seq = []
    ts = []
    for s in range(tape.steps):
        if bool(tape.event_active[s, 0].item()) and bool(tape.event_is_spike[s, 0].item()):
            seq.append(int(tape.event_spiker[s, 0].item()))
            ts.append(float(tape.event_time[s, 0].item()))
    max_time_err = float(np.max(np.abs(np.asarray(ts) - np.asarray(ref.spike_times))))
    max_grad_err = float(torch.max(torch.abs(grad - torch.as_tensor(ref_grad, device=dev, dtype=grad.dtype))).item())
    print("cpu spike sequence:", ref.spike_neurons)
    print("gpu spike sequence:", seq)
    print(f"max |time_gpu-time_cpu| = {max_time_err:.3e}")
    print(f"max |grad_gpu-grad_cpu| = {max_grad_err:.3e}")
    print("gpu synaptic updates:", tape.synaptic_updates.tolist())
    print("peak GPU memory MiB:", round(torch.cuda.max_memory_allocated() / 2**20, 2))
    assert seq == ref.spike_neurons
    assert max_time_err < 2e-5
    assert max_grad_err < 2e-4
    print("GPU_SMOKE_OK")


if __name__ == "__main__":
    main()
