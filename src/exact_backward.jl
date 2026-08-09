"""Linear event-time loss used by the exact-gradient tests and examples."""
function event_time_loss(tape::EventTape, dloss_dt::AbstractVector{<:Real})
    length(dloss_dt) == length(tape.spike_times) ||
        throw(ArgumentError("dloss_dt must match number of recorded events"))
    total = 0.0
    @inbounds for i in eachindex(tape.spike_times)
        total += Float64(dloss_dt[i]) * tape.spike_times[i]
    end
    return total
end

"""
Exact reverse-mode derivative with respect to sparse edge weights for a fixed
spike ordering.

`dloss_dt[m]` is the derivative of an arbitrary downstream loss with respect
to event time `t_m`. The reverse pass then propagates this credit only through
state nodes and synapses that actually participated in events.

The derivative is exact as long as infinitesimal parameter changes do not
change the discrete event ordering or create/delete a spike. Those boundary
cases are intentionally exposed rather than hidden; they are the targets for
the later V11 correction layer.
"""
function exact_event_time_gradient(tape::EventTape,
                                   conn::SparseConnectivity,
                                   dloss_dt::AbstractVector{<:Real})
    nevents = length(tape.events)
    length(dloss_dt) == nevents ||
        throw(ArgumentError("dloss_dt must match number of recorded events"))

    nnodes = length(tape.nodes)
    adj_v = zeros(Float64, nnodes)
    adj_t = zeros(Float64, nnodes)
    adj_s = zeros(Float64, nnodes)
    grad_w = zeros(Float64, length(conn.weight))

    @inbounds for m in nevents:-1:1
        ev = tape.events[m]
        adj_event_t = Float64(dloss_dt[m])

        # Reverse postsynaptic transitions in the opposite order in which they
        # were applied. This also correctly handles duplicate targets/autapses.
        for uidx in length(ev.post_updates):-1:1
            op = ev.post_updates[uidx]

            lambda_v = adj_v[op.new_node]
            lambda_t = adj_t[op.new_node]
            lambda_s = adj_s[op.new_node]

            # scheduled_t = event_t + g(new_v)
            adj_new_v = lambda_v + lambda_s * op.schedule_dv

            # new_v = flow(prev_v, event_t - prev_t) + weight[e]
            grad_w[op.edge] += adj_new_v
            adj_v[op.prev_node] += adj_new_v * op.flow_a
            adj_t[op.prev_node] -= adj_new_v * op.flow_slope
            adj_event_t += lambda_t + lambda_s +
                           adj_new_v * op.flow_slope
        end

        # Reset node has constant voltage; only its local time and next
        # scheduled time depend on the current event time.
        adj_event_t += adj_t[ev.reset_new_node] + adj_s[ev.reset_new_node]

        # The current event time is exactly the scheduled time stored in the
        # source neuron's previous local state node.
        adj_s[ev.source_node] += adj_event_t
    end

    # Initial nodes are the first conn.n nodes by construction. Returning their
    # voltage adjoints is useful for future input encoders and gradient checks.
    grad_initial_v = copy(@view adj_v[1:conn.n])
    return (weight=grad_w, initial_v=grad_initial_v)
end

"""
Central finite-difference reference for the event-time gradient.

If a perturbation changes the spike identity sequence, the corresponding edge
is marked `NaN`. That is not a numerical failure: it identifies exactly the
non-differentiable event boundary that V11 will later handle selectively.
"""
function finite_difference_event_time_gradient(p::LIFParams,
                                               conn::SparseConnectivity,
                                               dloss_dt::AbstractVector{<:Real};
                                               initial_v::AbstractVector{<:Real}=zeros(conn.n),
                                               t0::Real=0.0,
                                               n_events::Integer=length(dloss_dt),
                                               epsilon::Real=1e-6)
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0"))

    base = simulate(p, conn; initial_v=initial_v, t0=t0,
                    n_events=n_events)
    length(base.events) == n_events ||
        throw(ArgumentError("baseline produced fewer events than requested"))
    base_seq = spike_sequence(base)

    grad = fill(NaN, length(conn.weight))
    changed_order = falses(length(conn.weight))

    for e in eachindex(conn.weight)
        wp = copy(conn.weight)
        wm = copy(conn.weight)
        wp[e] += epsilon
        wm[e] -= epsilon

        tp = simulate(p, _with_weights(conn, wp);
                      initial_v=initial_v, t0=t0, n_events=n_events)
        tm = simulate(p, _with_weights(conn, wm);
                      initial_v=initial_v, t0=t0, n_events=n_events)

        if length(tp.events) != n_events || length(tm.events) != n_events ||
           spike_sequence(tp) != base_seq || spike_sequence(tm) != base_seq
            changed_order[e] = true
            continue
        end

        lp = event_time_loss(tp, dloss_dt)
        lm = event_time_loss(tm, dloss_dt)
        grad[e] = (lp - lm) / (2.0 * epsilon)
    end

    return (weight=grad, changed_order=changed_order, baseline=base)
end
