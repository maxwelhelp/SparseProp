# SparseProp-V11 training architecture

## Goal

Keep the defining SNN advantage intact: no event means no synaptic work. Stable event regions use an exact sparse reverse-mode gradient. Zeroth-order probing is reserved only for the places where the discrete event graph itself can change and the fixed-order derivative becomes incomplete.

## Core invariant

- Forward touches only emitted input/network events and their outgoing sparse edges.
- Backward touches only state nodes and edges present on the event tape.
- V11 never probes inactive edges.
- If an edge is far from a discrete event boundary, V11 adds zero extra forward simulations.

## Exact sparse path

For a fixed spike identity/order, the event-time map is differentiable. `exact_event_gradient` propagates loss credit backwards through only recorded event transitions. This is the default gradient and remains untouched in stable regions.

## Boundary detector

The hybrid path estimates distance to two failure surfaces in weight space.

### Spike birth/death boundary

For an additive synaptic jump, `dV/dw = 1`, therefore

`delta_threshold = abs(V_thresh - V_after_event)`

estimates the weight move required to create/delete an immediate spike.

### Event-order boundary

A transition records `schedule_dv = d(next_spike_time)/dV`. With `dV/dw = 1`,

`delta_order ~= time_margin / abs(schedule_dv)`.

The smaller boundary distance controls the local perturbation scale.

## Membrane/event-aware adaptive perturbation

V11 uses

`epsilon_e = clamp(boundary_overshoot * min(delta_threshold, delta_order), epsilon_min, epsilon_max)`.

The perturbation is therefore tied to the actual local event geometry rather than one global epsilon.

## Global V10-style soft budget

Boundary risk, exact-gradient credit and actual edge usage produce a score. Probe pairs are distributed approximately proportionally to score under a global budget `max_total_probe_pairs`; this prevents CIFAR-sized samples from launching probes for every near-threshold edge.

## Hard boundary correction with descent certificate

For a selected edge V11 evaluates real hard-spike trajectories at `W+epsilon` and `W-epsilon`. A changed event signature alone is not enough to modify the gradient.

A V11 correction is accepted only if at least one hard probe has lower loss than the current unperturbed hard trajectory. Across a discontinuity the finite-difference magnitude is not a true derivative, so V11 uses the probe as directional descent evidence and clips the pseudo-gradient magnitude.

This fixes the first stress-test failure where, after the desired spike had already appeared, a distant probe crossing back to the bad side kept pushing the weight farther. In the corrected implementation the boundary can still be detected, but `improves=False` causes zero V11 correction.

## What is solved now

1. **Stable hard-spike regions:** exact sparse reverse-mode agrees with finite differences to numerical precision in controlled tests.
2. **Spike birth/death on an active incoming edge:** exact fixed-order credit can be zero while V11 crosses the threshold boundary and finds a useful descent direction.
3. **Post-boundary overdrive:** fixed by the loss-improvement gate; V11 stops once crossing the boundary no longer improves the actual hard loss.
4. **Runtime sparsity:** forward/backward state is stored only for actual events and traversed sparse edges.

## What is NOT solved yet

1. **Completely unreachable silent structure.** V11 currently probes only edges traversed by the current event graph. If no upstream event ever reaches a useful silent neuron/path, there is no active edge to probe. The planned sparse pseudospike/frontier mechanism is needed for that case.
2. **Multi-edge cooperative boundaries.** A useful spike may require several weights to move together although no single-edge perturbation helps. Grouped/guided low-dimensional probes are needed here.
3. **Dense clusters of simultaneous event-order changes.** The current detector estimates pairwise local margins. Several coupled order swaps can still make the local model unreliable.
4. **ZO magnitude is not a mathematical derivative at a discontinuity.** We deliberately treat it as a bounded descent correction, not an exact gradient.
5. **GPU execution is not implemented yet.** The current Python event engine is for learning-correctness A/B experiments; GPU kernels must preserve the same event sparsity instead of reverting to dense timestep updates.

## CIFAR-100 A/B

`python/sparseprop_v11/cifar100.py` loads the original CIFAR-100 Python pickles directly. Images are converted into sparse patch-latency events, then processed by a sparse input -> hidden -> 100-readout LIF network with `v_inf=0`, so neurons do not spontaneously fire without event input.

The readout uses censored first-spike cross entropy: missing output spikes are assigned a finite horizon for the loss, but they have no exact event credit because the event does not exist. This intentionally exposes the discrete-gradient failure instead of hiding it with a surrogate derivative. V11 can then create missing useful output/hidden spikes by probing active near-threshold edges.

`python/experiments/train_cifar100_ab.py` starts exact-only and hybrid runs from the identical topology, weights, seed, encoding and optimiser and reports accuracy plus event/synaptic/probe diagnostics.

## Next stages after CIFAR A/B

1. Measure exact-only vs hybrid on CIFAR-100 and inspect whether corrections correlate with falling `missing_outputs` and better validation accuracy.
2. Add grouped/guided probes for cooperative boundaries so a neuron with many incoming weights is searched in a small local subspace instead of edge-by-edge.
3. Add sparse pseudospike frontier candidates from the event heap for useful silent neurons not reached by the current event graph.
4. Batch independent samples and move event queue/update/backward hot paths to GPU/CUDA without introducing dense timestep computation.
