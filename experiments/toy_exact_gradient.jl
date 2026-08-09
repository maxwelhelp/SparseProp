using SparsePropV11

n = 5
pre  = [1, 1, 2, 3, 4, 5, 2, 3]
post = [2, 3, 4, 4, 5, 1, 5, 1]
w = [-0.05, -0.03, -0.04, -0.06, -0.02, -0.03, -0.01, -0.02]
conn = SparseConnectivity(n, pre, post, w)
p = LIFParams(tau=1.0, v_inf=1.5, v_thresh=1.0, v_reset=0.0)
initial_v = collect(0.1:0.1:0.5)
n_events = 20

tape = simulate(p, conn; initial_v=initial_v, n_events=n_events)
dloss_dt = collect(range(0.1, 1.0; length=n_events))
exact = exact_event_time_gradient(tape, conn, dloss_dt)
fd = finite_difference_event_time_gradient(
    p, conn, dloss_dt; initial_v=initial_v,
    n_events=n_events, epsilon=1e-6)

println("spikes:            ", tape.spike_neurons)
println("first spike times: ", tape.spike_times[1:min(end, 8)])
println("operation stats:   ", operation_stats(tape))
println("exact dL/dw:       ", exact.weight)
println("finite diff dL/dw: ", fd.weight)
println("changed order:     ", fd.changed_order)
println("max |exact-fd|:    ", maximum(abs.(exact.weight .- fd.weight)))
