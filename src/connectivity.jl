"""
Sparse directed connectivity with one trainable scalar weight per edge.

`pre[e] -> post[e]` is edge `e`. `outgoing[j]` stores only the edge IDs that
must be touched when neuron `j` spikes. This is the key representation needed
to keep synaptic work proportional to emitted events rather than `N^2`.
"""
struct SparseConnectivity
    n::Int
    pre::Vector{Int}
    post::Vector{Int}
    weight::Vector{Float64}
    outgoing::Vector{Vector{Int}}
end

function SparseConnectivity(n::Integer,
                            pre::AbstractVector{<:Integer},
                            post::AbstractVector{<:Integer},
                            weight::AbstractVector{<:Real})
    n > 0 || throw(ArgumentError("n must be positive"))
    length(pre) == length(post) == length(weight) ||
        throw(ArgumentError("pre, post and weight must have equal lengths"))

    pre_i = Int.(pre)
    post_i = Int.(post)
    weight_f = Float64.(weight)
    outgoing = [Int[] for _ in 1:n]

    for e in eachindex(weight_f)
        1 <= pre_i[e] <= n || throw(ArgumentError("pre[$e] is out of range"))
        1 <= post_i[e] <= n || throw(ArgumentError("post[$e] is out of range"))
        push!(outgoing[pre_i[e]], e)
    end

    return SparseConnectivity(Int(n), pre_i, post_i, weight_f, outgoing)
end

function _with_weights(conn::SparseConnectivity, weight::AbstractVector{<:Real})
    length(weight) == length(conn.weight) ||
        throw(ArgumentError("replacement weight vector has wrong length"))
    return SparseConnectivity(conn.n, conn.pre, conn.post, weight)
end
