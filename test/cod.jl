##################################################
# Test zero factor marginalization
##################################################
bnd = MCBN.BayesNetDAI(3)
MCBN.add_edge!(bnd, 1, 2)
MCBN.add_edge!(bnd, 1, 3)
DAI.clearBackups!(bnd)
n = 0.2
d = 0.1
MCBN.set_factor!(bnd, 1, [0.0, 1.0])
MCBN.set_factor!(bnd, 2, [1-n, d, n, 1-d])
MCBN.set_factor!(bnd, 3, [1-n, d, n, 1-d])
DAI.clearBackups!(bnd)

allvars = DAI.VarSet(DAI.Var(1,2), DAI.Var(2,2))
MCBN.setupIA!(bnd)
jointfac = DAI.marginal(bnd.ia, allvars, true)
margfac = DAI.marginal(bnd.ia, DAI.VarSet(DAI.Var(2,2)), true)
belfac = DAI.belief(bnd.ia, DAI.VarSet(DAI.Var(2,2)))

@test_approx_eq jointfac[1] 0.0
@test_approx_eq jointfac[2] 0.1
@test_approx_eq jointfac[3] 0.0
@test_approx_eq jointfac[4] 0.9

##################################################
# Test CoD from Zhao et al 2011
##################################################
bnd = MCBN.BayesNetDAI(3)
MCBN.add_edge!(bnd, 1, 2)
MCBN.add_edge!(bnd, 1, 3)
DAI.clearBackups!(bnd)
c10 = 0.4
c11 = 1. - c10
n13 = n12 = 0.2
d12 = d13 = 0.1
MCBN.set_factor!(bnd, 1, [c10, c11])
MCBN.set_factor!(bnd, 2, [1-n12, d12, n12, 1-d12])
MCBN.set_factor!(bnd, 3, [1-n13, d13, n13, 1-d13])

MCBN.check_bnd(bnd)
@test_approx_eq MCBN.cod(bnd, 1, [2,3]) 0.675
@test_approx_eq MCBN.cod(bnd, 1, [2]) 0.65
@test_approx_eq MCBN.cod(bnd, 2, [1]) 0.631578947368421

codbnd = bnd
bns = MCBN.BayesNetSampler(codbnd, MCBN.draw_data(codbnd, 50))

@show MCBN.ting_cod(bns, 1, [2,3])
@show MCBN.ting_cod(bns, 1, [2])
@show MCBN.ting_cod(bns, 2, [1])


##################################################
# Test Controllability
##################################################
@test_approx_eq MCBN.controllability(bnd, [2], [1], 1) 0.42
@test_approx_eq MCBN.controllability(bnd, [2], [2], 1) 0.28
@test_approx_eq MCBN.controllability(bnd, [2], [2], 3) 0.0
@test_approx_eq MCBN.controllability(bnd, [1], [2], 2) 0.0
@test_approx_eq MCBN.controllability(bnd, [1,2], [2,2], 3)+1 1.0 # Not much accuracy here
@test_approx_eq MCBN.controllability(bnd, [2,3], [2,2], 1) 0.308
