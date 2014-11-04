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
    d = size(bns.mat,1)
    tot = 0
    for i=1:d
        for j=1:d
            tot += bns.mat[i,j]
        end
        if tot > bns.limparent + 1
            return 1e10
        end
        tot = 0
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
samc = SAMC.set_energy_limits(bns, refden_power=1.0)
samc.thin = 1000
samc.stepscale = 3000.0
@time SAMC.sample(samc, 300_000)
