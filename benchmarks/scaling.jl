using SparsePropV11

function ring_network(n, k)
    pre = Int[]
    post = Int[]
    weight = Float64[]
    for j in 1:n
        for d in 1:k
            push!(pre, j)
            push!(post, mod1(j + d, n))
            push!(weight, -0.02 / d)
        end
    end
    return SparseConnectivity(n, pre, post, weight)
end

p = LIFParams(tau=1.0, v_inf=1.5, v_thresh=1.0, v_reset=0.0)
k = 8
n_events = 10_000

println("N,K,events,synaptic_updates,state_nodes,seconds")
for n in (1_000, 10_000, 100_000)
    conn = ring_network(n, k)
    initial_v = [0.05 + 0.8 * (i - 1) / max(n - 1, 1) for i in 1:n]

    # One warmup for Julia compilation at the smallest size.
    if n == 1_000
        simulate(p, conn; initial_v=initial_v, n_events=10)
    end

    elapsed = @elapsed tape = simulate(
        p, conn; initial_v=initial_v, n_events=n_events)
    s = operation_stats(tape)
    println("$n,$k,$(s.events),$(s.synaptic_updates),$(s.state_nodes),$elapsed")
end
