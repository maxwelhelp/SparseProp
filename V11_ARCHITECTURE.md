# SparseProp-V11 training architecture

## Goal

Keep the defining SNN advantage intact: no event means no synaptic work. Stable event regions use an exact sparse reverse-mode gradient. Zeroth-order probing is reserved only for the places where the discrete event graph itself can change and the fixed-order derivative becomes incomplete.

## Core invariant

- Forward touches only emitted input/network events and their outgoing sparse edges.
- Backward touches only state nodes and edges present on the event tape.
- V11 never probes inactive edges.
- If an edge is far from a discrete event boundary, V11 adds zero extra forward simulations.

## 1. Exact sparse path

For a fixed spike identity/order, the event-time map is differentiable. `exact_event_gradient` propagates loss credit backwards through only recorded event transitions. This is the default gradient and remains untouched in stable regions.

## 2. Boundary detector

The hybrid path estimates distance to two failure surfaces in weight space.

### Spike birth/death boundary

For an additive synaptic jump, dV/dw = 1. Therefore

`delta_threshold = abs(V_thresh - V_after_event)`

is directly an estimate of how much the active weight must move to create/delete an immediate spike.

### Event-order boundary

A postsynaptic transition records `schedule_dv = d(next_spike_time)/dV`. With dV/dw = 1, a time margin to a competing event becomes

`delta_order ~= time_margin / abs(schedule_dv)`.

The smaller of the two deltas sets the local boundary scale.

## 3. Membrane/event-aware adaptive perturbation

V11 does not use one global epsilon. For active edge e:

`epsilon_e = clamp(boundary_overshoot * min(delta_threshold, delta_order), epsilon_min, epsilon_max)`.

The overshoot intentionally places the hard W+epsilon / W-epsilon probes on opposite sides of the estimated event boundary when possible.

## 4. V10 soft probe budget

Only boundary candidates receive probes. Their score combines:

- boundary risk;
- exact error credit magnitude;
- actual edge usage on the event tape.

A small credit floor is retained because spike-birth/death boundaries can have an exact fixed-order gradient of exactly zero. Probe-pair count grows smoothly with this score rather than using a hard top-k selection.

## 5. Hard boundary correction

For every selected edge V11 executes real hard-spike simulations with W+epsilon and W-epsilon. No surrogate spike derivative is introduced.

If either probe changes the event signature, the exact fixed-order gradient on that edge is replaced by the robust median zeroth-order estimate across allocated probe scales. If the event signature stays fixed, exact gradient remains primary; only very high pre-boundary risk receives a small blend.

## 6. Why this addresses the learning failure

A silent neuron can have no output spike in the current event graph. Exact reverse-mode credit through that absent event is zero. The boundary detector can still see an active incoming edge near threshold, select it, choose epsilon from its membrane distance, and probe across the missing/present spike boundary. This creates a useful update direction without making the whole network dense or smoothing the hard spike.

`python/experiments/stress_v11_boundaries.py` is the minimal proof: exact training remains stuck below threshold while the hybrid path obtains a non-zero update and creates the missing spike.

## Next stages

1. Validate threshold birth/death and event-order swaps on small controlled networks.
2. Add grouped/directional probes so a problematic neuron with many incoming weights is searched in a small guided subspace rather than edge-by-edge.
3. Add a sparse pseudospike frontier from the event heap for silent neurons that are close to becoming useful.
4. Add a real dataset trainer. CIFAR-100 will use the existing Python event engine first for learning-quality A/B tests: exact-only vs hybrid V11 under identical sparse topology and encoding.
5. Only after learning correctness is established, batch independent event streams and move hot kernels to GPU/CUDA while preserving event sparsity.
