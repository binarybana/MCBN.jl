reload("SAMC")
#using SAMC
reload("MCBN")
#using MCBN


D = 8
template = zeros(Bool,D,D)
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

preseed = rand(Uint)
srand(123)
#data = rand(1:2, D, 60)
gold = MCBN.random_net(D) 
data = MCBN.draw_data(gold,5)
srand(preseed)

#bns = MCBN.BayesNetSampler(D, data, prior)
#samc1 = SAMC.set_energy_limits(bns, refden_power=0.0)
#samc1.thin = 1000
#samc1.stepscale = 20.0
#samc1.burn = 40_000
#@time SAMC.sample(samc1, 8_00_000, beta=0.6)

## PopSAMC
bngen = x->MCBN.BayesNetSampler(D, data, prior)
samc2 = SAMC.set_energy_limits(bngen, 10, refden_power=0.0)
samc2.thin = 100
samc2.stepscale = 10.0
samc2.burn = 1_000
@profile SAMC.sample(samc2, 10_000, beta=0.6)

function edge_prob(bns::MCBN.BayesNetSampler)
    ord = sortperm(bns.x)
    return bns.mat[ord,ord]
end

kld_gold(bns::MCBN.BayesNetSampler) = MCBN.kld(gold, bns.bnd)

post_edge = SAMC.posterior_e(edge_prob,samc2)
post_kld = SAMC.posterior_e(kld_gold,samc2)
