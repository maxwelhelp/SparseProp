@testset "work scales with emitted edges, not all neurons" begin
    n = 128
    k = 2
    pre = Int[]
    post = Int[]
    weight = Float64[]

    # Deterministic ring connectivity with exactly k outgoing edges/neuron.
    for j in 1:n
        for d in 1:k
            push!(pre, j)
            push!(post, mod1(j + d, n))
            push!(weight, -0.02 / d)
        end
    end

    conn = SparseConnectivity(n, pre, post, weight)
    p = LIFParams(tau=1.0, v_inf=1.5, v_thresh=1.0, v_reset=0.0)
    initial_v = [0.05 + 0.8 * (i - 1) / (n - 1) for i in 1:n]
    requested_events = 200

    tape = simulate(p, conn; initial_v=initial_v,
                    n_events=requested_events)
    stats = operation_stats(tape)

    @test stats.events == requested_events
    @test stats.synaptic_updates == k * requested_events
    # Initial N nodes + one reset node/event + one node/traversed synapse.
    @test stats.state_nodes == n + stats.events + stats.synaptic_updates
end
