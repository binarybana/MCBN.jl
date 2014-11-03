reload("SAMC")
#using SAMC
reload("MCBN")
#using MCBN


D = 8
template = eye(Bool, D)
cpds = MCBN.BayesNetDAI(D)

function factor_klds(bnd1::MCBN.BayesNetDAI, bnd2::MCBN.BayesNetDAI)
    0 #FIXME
end

function prior(bns::MCBN.BayesNetSampler)
    p_cpds = 0.2
    p_structural = 0.2

    #p_cpds::Float64
    #cpds::BayesNetDAI
    #p_structural::Float64
    #template::Matrix{Bool}
    #limparent::Int
    #limit number of parents
    if any(sum(bns.mat, 2) .> bns.limparent + 1) # +1 -> diagonal # FIXME: this is actually pretty slow
        return 1e10
    end

    # energy contribution from structural
    adj_diff = sum(abs(bns.mat[bns.x,bns.x] - template)) 
    e_struct = p_structural * adj_diff

    # energy contribution from cpds
    cpd_diff = factor_klds(bns.bnd, cpds)
    e_cpd = p_cpds * cpd_diff
    return e_cpd + e_struct
end

bns = MCBN.BayesNetSampler(D, rand(1:2, 60,D), prior)
samc = SAMC.set_energy_limits(bns)
@time SAMC.sample(samc, 10000)
