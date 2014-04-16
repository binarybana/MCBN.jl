reload("MCBN")
using Base.Test  

# write your own tests here
bnd = MCBN.BayesNetDAI(5)
bnd2 = MCBN.BayesNetDAI(5)
bnd3 = MCBN.BayesNetDAI(5)


@test_approx_eq MCBN.entropy(bnd) 5.0
@test_approx_eq MCBN.naive_entropy(bnd) 5.0
@test_approx_eq MCBN.kld(bnd,bnd2) 0.0

function test_net(bnd)
    MCBN.check_cpds(bnd)
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
#MCBN.check_cpds(bnd3)

MCBN.set_factor!(bnd3, 2, [1.0, 0.5, 0.0, 0.5])
MCBN.set_factor!(bnd3, 1, [0.8, 0.2])
MCBN.check_cpds(bnd3)

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
        ##rcon.propose()
        ##con.propose()

    #t = BayesNetCPD(np.ones(6,dtype=np.int32)*2)
    #for i in range(500):
        #t.mutate()

    #return empty, connected, t
