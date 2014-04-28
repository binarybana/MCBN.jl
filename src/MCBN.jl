module MCBN

    import Base: show
    using DAI
    using Graphs
    using Distributions
    import Graphs: vertex_index, make_vertex, add_edge!
    #import OBC: Sampler, energy, propose, reject, sample
    #import SAMC: SAMCRecord, set_energy_limits

    include("bayesnet.jl")

end # module
