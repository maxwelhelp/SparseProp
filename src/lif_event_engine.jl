"""
Continuous-time LIF parameters.

Between events the membrane follows

    dV/dt = (v_inf - V) / tau

and an incoming spike applies an instantaneous voltage jump equal to the edge
weight. If `v_inf > v_thresh`, an untouched neuron has an analytic next-spike
time. Otherwise it remains unscheduled (`Inf`) until an input jump reaches the
threshold.
"""
struct LIFParams
    tau::Float64
    v_inf::Float64
    v_thresh::Float64
    v_reset::Float64
end

function LIFParams(; tau::Real=1.0,
                     v_inf::Real=1.5,
                     v_thresh::Real=1.0,
                     v_reset::Real=0.0)
    tau > 0 || throw(ArgumentError("tau must be > 0"))
    v_reset < v_thresh || throw(ArgumentError("v_reset must be below v_thresh"))
    return LIFParams(Float64(tau), Float64(v_inf),
                     Float64(v_thresh), Float64(v_reset))
end

struct StateNode
    v::Float64
    t::Float64
    scheduled_t::Float64
end

"""One sparse postsynaptic state transition caused by one emitted spike."""
struct PostUpdateRecord
    prev_node::Int
    new_node::Int
    edge::Int
    flow_a::Float64
    flow_slope::Float64
    schedule_dv::Float64
    immediate::Bool
end

"""One network spike and all state transitions directly caused by it."""
struct EventRecord
    source_node::Int
    spiker::Int
    t::Float64
    reset_new_node::Int
    post_updates::Vector{PostUpdateRecord}
end

"""
Sparse event tape.

The tape contains one reset node per emitted spike and one state node per
actually traversed synapse. It deliberately does not materialise states for
neurons that were unaffected by an event.
"""
struct EventTape
    nodes::Vector{StateNode}
    events::Vector{EventRecord}
    spike_neurons::Vector{Int}
    spike_times::Vector{Float64}
    synaptic_updates::Int
    immediate_schedules::Int
end

@inline function _flow(v::Float64, dt::Float64, p::LIFParams)
    a = exp(-dt / p.tau)
    u = p.v_inf + (v - p.v_inf) * a
    # d flow / d(current event time)
    slope = (p.v_inf - u) / p.tau
    return u, a, slope
end

"""
Return `(next_time, dnext_dv, immediate)`.

The derivative is exact inside a fixed event-order region. At an instantaneous
threshold crossing (`v >= v_thresh`) the map is genuinely non-smooth; the
baseline derivative is therefore zero and the event is flagged for the later
V11 boundary-correction path.
"""
@inline function _schedule(v::Float64, t::Float64, p::LIFParams)
    if v >= p.v_thresh
        return t, 0.0, true
    elseif p.v_inf <= p.v_thresh
        return Inf, 0.0, false
    end

    denom = p.v_inf - v
    numer = p.v_inf - p.v_thresh
    if denom <= 0.0 || numer <= 0.0
        return Inf, 0.0, false
    end

    ratio = numer / denom
    dt = -p.tau * log(ratio)
    dtdv = -p.tau / denom
    return t + dt, dtdv, false
end

"""
Run a true event-driven LIF simulation.

Only the next spiking neuron and the outgoing edges of that neuron are touched
inside the event loop. Unaffected neurons are represented implicitly by their
existing heap entry and last local state node.
"""
function simulate(p::LIFParams,
                  conn::SparseConnectivity;
                  initial_v::AbstractVector{<:Real}=zeros(conn.n),
                  t0::Real=0.0,
                  n_events::Integer=100)
    length(initial_v) == conn.n ||
        throw(ArgumentError("initial_v must have length conn.n"))
    n_events >= 0 || throw(ArgumentError("n_events must be non-negative"))

    t0f = Float64(t0)
    nodes = StateNode[]
    current_node = Vector{Int}(undef, conn.n)
    initial_schedule = Vector{Float64}(undef, conn.n)

    for i in 1:conn.n
        v = Float64(initial_v[i])
        s, _, _ = _schedule(v, t0f, p)
        push!(nodes, StateNode(v, t0f, s))
        current_node[i] = length(nodes)
        initial_schedule[i] = s
    end

    # MutableBinaryHeap keeps stable handles. Because entries are inserted in
    # neuron order and never deleted/reinserted, handle i is neuron i.
    heap = MutableBinaryHeap{Float64,DataStructures.FasterForward}(initial_schedule)

    events = EventRecord[]
    spike_neurons = Int[]
    spike_times = Float64[]
    synaptic_updates = 0
    immediate_schedules = 0

    for _ in 1:n_events
        t, j = top_with_handle(heap)
        isfinite(t) || break

        source_node = current_node[j]
        post_updates = PostUpdateRecord[]

        # Reset only the neuron that emitted the event.
        reset_s, _, reset_immediate = _schedule(p.v_reset, t, p)
        reset_immediate && (immediate_schedules += 1)
        push!(nodes, StateNode(p.v_reset, t, reset_s))
        reset_new_node = length(nodes)
        current_node[j] = reset_new_node
        update!(heap, j, reset_s)

        # Traverse only outgoing edges of the spiking neuron.
        for e in conn.outgoing[j]
            i = conn.post[e]
            prev_node = current_node[i]
            prev = nodes[prev_node]
            dt = t - prev.t
            dt >= -1e-12 || error("event time moved backwards")
            dt = max(dt, 0.0)

            u, a, slope = _flow(prev.v, dt, p)
            new_v = u + conn.weight[e]
            new_s, schedule_dv, immediate = _schedule(new_v, t, p)
            immediate && (immediate_schedules += 1)

            push!(nodes, StateNode(new_v, t, new_s))
            new_node = length(nodes)
            current_node[i] = new_node
            update!(heap, i, new_s)

            push!(post_updates,
                  PostUpdateRecord(prev_node, new_node, e,
                                   a, slope, schedule_dv, immediate))
            synaptic_updates += 1
        end

        push!(events, EventRecord(source_node, j, t,
                                  reset_new_node, post_updates))
        push!(spike_neurons, j)
        push!(spike_times, t)
    end

    return EventTape(nodes, events, spike_neurons, spike_times,
                     synaptic_updates, immediate_schedules)
end

spike_sequence(tape::EventTape) = tape.spike_neurons

function operation_stats(tape::EventTape)
    return (events=length(tape.events),
            synaptic_updates=tape.synaptic_updates,
            state_nodes=length(tape.nodes),
            immediate_schedules=tape.immediate_schedules)
end
