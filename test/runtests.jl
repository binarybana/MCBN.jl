reload("SAMC")
reload("MCBN")
using Base.Test

# write your own tests here
bnd = MCBN.BayesNetDAI(5)
bnd2 = MCBN.BayesNetDAI(5)
bnd3 = MCBN.BayesNetDAI(5)

##################################################
# Entropy and KLD Tests
##################################################

@test_approx_eq MCBN.entropy(bnd) 5.0
@test_approx_eq MCBN.naive_entropy(bnd) 5.0
@test_approx_eq MCBN.kld(bnd,bnd2) 0.0

function test_net(bnd)
    MCBN.check_bnd(bnd)
    @test_approx_eq MCBN.naive_entropy(bnd) MCBN.entropy(bnd)
    @test_approx_eq MCBN.kld(bnd,bnd) 0.0
end

function test_kld(bnd1, bnd2)
    kldval = MCBN.kld(bnd1, bnd2)
    kldvalrev = MCBN.kld(bnd2, bnd1)
    @test kldval >= 0
    @test kldvalrev >= 0
end

test_net(bnd)
test_kld(bnd, bnd2)

MCBN.add_edge!(bnd3, 1,2)
#MCBN.check_bnd(bnd3)

MCBN.set_factor!(bnd3, 2, [1.0, 0.5, 0.0, 0.5])
MCBN.set_factor!(bnd3, 1, [0.8, 0.2])
MCBN.check_bnd(bnd3)

@test_approx_eq MCBN.entropy(bnd3) 3.9219280948873623
@test_approx_eq MCBN.entropy(bnd3) MCBN.naive_entropy(bnd3)

test_kld(bnd, bnd3)

MCBN.restoreFactors!(bnd3.fg)
bnd3.dirty = true

@test_approx_eq MCBN.entropy(bnd3) MCBN.naive_entropy(bnd3)
@test_approx_eq MCBN.entropy(bnd3) 5.0

@test_approx_eq MCBN.kld(bnd,bnd3) 0.0
@test_approx_eq MCBN.kld(bnd3,bnd) 0.0

test_net(bnd3)
test_kld(bnd, bnd3)

##################################################
# Entropy and KLD Tests
##################################################

bns = MCBN.BayesNetSampler(3, rand(1:2, 5,3))
origE = MCBN.energy(bns)
origfvalue = copy(bns.fvalue)

@time for i=1:1000
    scheme = MCBN.propose!(bns)
    MCBN.check_bns(bns)
    MCBN.reject!(bns)
    MCBN.check_bns(bns)
    E = MCBN.energy(bns)
    @test_approx_eq bns.fvalue origfvalue
    @test_approx_eq E origE
end

@time for i=1:1000
    s = MCBN.propose!(bns)
    MCBN.check_bns(bns)
    E1 = MCBN.energy(bns)
    f1 = copy(bns.fvalue)
    bns.changelist = [1:3]
    E2 = MCBN.energy(bns)
    f2 = copy(bns.fvalue)
    @test_approx_eq f1 f2
    @test_approx_eq E1 E2
end

##################################################
# Random net and data draw from nets
##################################################

nets = [MCBN.random_net(x) for x=[5,10,40]]
graphs = map(MCBN.net2graph, nets)
draws = map(x->MCBN.draw_data(x,10), nets)
#klds = Float64[]

global g1,g2,d1,d2,bns1,bns2,bns1c,bns2c
for i=1:100
    g1 = MCBN.random_net(15)
    g2 = MCBN.random_net(15)

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

gold = MCBN.random_net(D)
data = MCBN.draw_data(gold,10)
samc2 = SAMC.MHRecord(MCBN.BayesNetSampler(D, data, prior), burn = 100, thin=20)
MCBN.check_bns(bns)
for i=1:1000
  SAMC.sample!(samc2, 10, verbose=0)
  MCBN.check_bns(bns)
end
println("Passed 1000 check_bns points every 10 samples")

    #empty = BayesNetCPD(np.array([2,2,2]))
    #assert empty.entropy() == 3.0
    #assert empty.naive_entropy() == 3.0

    #connected = BayesNetCPD(np.array([2,2,2]))
    #connected.adjust_factor(0,[1],[])
    #connected.set_factor(0, [0.0,1.0,0.0,1.0])
    #connected.adjust_factor(1,[2],[])
    #connected.set_factor(1, [0.0,1.0,0.0,1.0])

    #assert connected.entropy() == 1.0
    #assert connected.kld(empty) == 2.0

    ##np.random.seed(1234)
    ##con = TreeNet(4)
    ##rcon = TreeNet(4)
    ##for i in range(1000):
        ##kl1 = rcon.kld(con)
        ##kl2 = con.kld(rcon)
        ##assert (kl1 >= 0.0)
        ##assert (kl2 >= 0.0)
        ##rcon.propose!()
        ##con.propose!()

    #t = BayesNetCPD(np.ones(6,dtype=np.int32)*2)
    #for i in range(500):
        #t.mutate()

    #return empty, connected, t
