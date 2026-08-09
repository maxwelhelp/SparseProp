# SparseProp-V11 development branch

This branch keeps the original SparseProp scripts untouched and builds a
trainable event-driven core beside them.

## Milestone 1: exact sparse event engine

Implemented:

- explicit sparse directed connectivity with one trainable weight per edge;
- continuous-time LIF dynamics with analytic next-spike scheduling;
- mutable time heap: the event loop touches only the spiking neuron and its
  outgoing edges;
- sparse event tape: no full-network state is materialised per event;
- exact reverse-mode derivative of an arbitrary event-time loss with respect
  to edge weights, assuming the discrete spike ordering stays fixed;
- central finite-difference checker which marks weight directions that change
  event ordering instead of pretending a smooth derivative exists;
- operation counters and scaling benchmark.

The exact backward equations are intentionally implemented before any V11
heuristic. This gives us a clean baseline and exposes the precise places where
hard event discreteness breaks differentiability.

## Next milestones

1. Boundary detector
   - immediate threshold crossings;
   - spike birth/death;
   - event-order swaps under tiny perturbations;
   - large spike-time sensitivity.

2. V11 correction only on flagged boundaries
   - adaptive perturbation size from event margin;
   - hard-spike re-evaluation, never a surrogate sigmoid;
   - soft probe budget from boundary risk x error credit.

3. Sparse pseudospike frontier
   - use only the nearest future heap candidates to give silent neurons credit;
   - avoid dense pseudospikes over all neurons.

4. Guided low-dimensional probes
   - exact adjoint/eligibility direction as the first probe;
   - a small orthogonal local subspace for additional probes;
   - never project the whole network by default.

5. Structural sparsity
   - edge usage and error-credit statistics;
   - prune persistently useless edges;
   - create weak new edges only where residual error remains high.

6. GPU path
   - only after event-time gradients and boundary handling pass CPU numerical
     checks;
   - benchmark actual synaptic/event updates, wall time and memory rather than
     theoretical SNN FLOPs.

## Core invariant

No emitted spike -> no outgoing synaptic work.
No flagged boundary -> no V11 zeroth-order probe.
