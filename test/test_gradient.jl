@testset "exact event-time gradient" begin
    n = 5
    pre  = [1, 1, 2, 3, 4, 5, 2, 3]
    post = [2, 3, 4, 4, 5, 1, 5, 1]
    w = [-0.05, -0.03, -0.04, -0.06, -0.02, -0.03, -0.01, -0.02]
    conn = SparseConnectivity(n, pre, post, w)
    p = LIFParams(tau=1.0, v_inf=1.5, v_thresh=1.0, v_reset=0.0)
    initial_v = collect(0.1:0.1:0.5)
    n_events = 20

    tape = simulate(p, conn; initial_v=initial_v, n_events=n_events)
    @test length(tape.events) == n_events
    @test tape.immediate_schedules == 0

    dloss_dt = collect(range(0.1, 1.0; length=n_events))
    exact = exact_event_time_gradient(tape, conn, dloss_dt)
    fd = finite_difference_event_time_gradient(
        p, conn, dloss_dt; initial_v=initial_v,
        n_events=n_events, epsilon=1e-6)

    @test !any(fd.changed_order)
    @test all(isfinite, fd.weight)
    @test maximum(abs.(exact.weight .- fd.weight)) < 1e-5
end
