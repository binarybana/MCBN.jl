##################################################
# Random net and data draw from nets
##################################################

nets = [MCBN.random_net_weights(x) for x=[5,10,40]]
graphs = map(MCBN.net2graph, nets)
draws = map(x->MCBN.draw_data(x,10), nets)
#klds = Float64[]

global g1,g2,d1,d2,bns1,bns2,bns1c,bns2c
for i=1:100
    g1 = MCBN.random_net_weights(15)
    g2 = MCBN.random_net_weights(15)

    d1 = MCBN.draw_data(g1,50)
    d2 = MCBN.draw_data(g2,50)

    bns1 = MCBN.BayesNetSampler(g1,d1)
    bns1c = MCBN.BayesNetSampler(g1,d2)

    bns2 = MCBN.BayesNetSampler(g2,d2)
    bns2c = MCBN.BayesNetSampler(g2,d1)

    diff1 = MCBN.energy(bns1) - MCBN.energy(bns1c)
    diff2 = MCBN.energy(bns2) - MCBN.energy(bns2c)
    @test (diff1 + diff2) < 0
    # We do it this way because sometimes both networks will actually favor the
    # same dataset, but usually the 'wrong' favoritism is much smaller
    # magnitude than the correct one. So summing them up will make this
    # apparent (and prevent failures)

    #push!(klds, MCBN.kld(g1,g2))
    i%10==0 && print("\rSuccessfully completed round $i of energy checks")
end
println("")

D = 4
template = zeros(Int,D,D)
cpds = MCBN.BayesNetDAI(D)

function factor_klds(bnd1::MCBN.BayesNetDAI, bnd2::MCBN.BayesNetDAI)
    0 #FIXME
end

function prior(bns::MCBN.BayesNetSampler)
    p_cpds=0.2
    p_structural=0.2
    limparent=4
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

    adj_diff = sum(abs(bns.mat[bns.x,bns.x] - template))
    e_struct = p_structural * adj_diff

    # energy contribution from cpds
    cpd_diff = factor_klds(bns.bnd, cpds)
    e_cpd = p_cpds * cpd_diff
    return e_cpd + e_struct
end

gold = MCBN.random_net_weights(D)
data = MCBN.draw_data(gold,10)
samc2 = SAMC.MHRecord(MCBN.BayesNetSampler(D, data, prior), burn = 100, thin=20)
MCBN.check_bns(bns)
for i=1:1000
  SAMC.sample!(samc2, 10, verbose=0)
  MCBN.check_bns(bns)
end
println("Passed 1000 check_bns points every 10 samples")

# TODO: Test factor_klds

