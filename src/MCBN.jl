module MCBN

    import Base: show
    using DAI
    using Distributions
    #using Graphs
    #import Graphs: vertex_index, make_vertex, add_edge!
    #import OBC: Sampler, energy, propose!, reject, sample
    #import SAMC: SAMCRecord, set_energy_limits

    include("bayesnet.jl")

end # module
