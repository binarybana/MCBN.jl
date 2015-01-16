module MCBN

    import Base: show, length
    using DAI
    using Distributions
    using Graphs
    import Graphs: add_edge!

    using StatsBase
    import StatsBase: sample
    #using Graphs
    #import Graphs: vertex_index, make_vertex, add_edge!
    
    using SAMC
    import SAMC: propose!, energy, reject!, save!, record, Sampler

    include("bayesnet.jl")

end # module
