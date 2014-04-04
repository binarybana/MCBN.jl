module MCBN

    using DAI
    import OBC: Sampler, energy, propose, reject, SAMCRecord, sample, set_energy_limits

    include("bayesnet.jl")

end # module
