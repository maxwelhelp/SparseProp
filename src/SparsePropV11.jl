module SparsePropV11

using DataStructures

export LIFParams,
       SparseConnectivity,
       EventTape,
       simulate,
       exact_event_time_gradient,
       finite_difference_event_time_gradient,
       event_time_loss,
       spike_sequence,
       operation_stats

include("connectivity.jl")
include("lif_event_engine.jl")
include("exact_backward.jl")

end
