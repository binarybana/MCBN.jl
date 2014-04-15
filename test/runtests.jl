reload("MCBN")
using Base.Test  

# write your own tests here
#g = MCBN.BayesNet(5)
bnd = MCBN.BayesNetDAI(5)
bnd2 = MCBN.BayesNetDAI(5)
bnd3 = MCBN.BayesNetDAI(5)

@test_approx_eq MCBN.entropy(bnd) 5.0
@test_approx_eq MCBN.naive_entropy(bnd) 5.0
@test_approx_eq MCBN.naive_entropy(bnd) MCBN.entropy(bnd)

@show k1 = MCBN.kld(bnd,bnd2)
@show k2 = MCBN.faster_kld(bnd,bnd2)

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
