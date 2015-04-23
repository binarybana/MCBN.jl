
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
@show true_cod1 = MCBN.cod(bnd, 1, [2,3])
@show true_cod2 = MCBN.cod(bnd, 1, [2])
@show true_cod3 = MCBN.cod(bnd, 2, [1])

codbnd = bnd
bn = MCBN.BayesNetSampler(codbnd, MCBN.draw_data(codbnd, 20))

@show MCBN.ting_cod(bn, 1, [2,3])
@show MCBN.ting_cod(bn, 1, [2])
@show MCBN.ting_cod(bn, 2, [1])

mh = SAMC.set_energy_limits(bn)
mh.stepscale = 10
mh.thin = 100
mh.burn = 1000
SAMC.sample!(mh, 100000)

char_cod1(bnd::MCBN.BayesNetDAI) = MCBN.cod(bnd, 1, [2,3])
char_cod2(bnd::MCBN.BayesNetDAI) = MCBN.cod(bnd, 1, [2])
char_cod3(bnd::MCBN.BayesNetDAI) = MCBN.cod(bnd, 2, [1])

@show SAMC.posterior_e(char_cod1,mh)
@show SAMC.posterior_e(char_cod2,mh)
@show SAMC.posterior_e(char_cod3,mh)

@show map_cod = char_cod1(SAMC.mapvalue(mh))
@show map_cod = char_cod2(SAMC.mapvalue(mh))
@show map_cod = char_cod3(SAMC.mapvalue(mh))

chains = 10
burn = 500
thin = 100
iters = 100000
mh = SAMC.MHRecord[SAMC.MHRecord(deepcopy(bn), burn=burn, thin=thin) for i=1:chains]
SAMC.sample!(mh, div(iters,chains))

@show SAMC.posterior_e(char_cod1,mh)
@show SAMC.posterior_e(char_cod2,mh)
@show SAMC.posterior_e(char_cod3,mh)

@show map_cod = char_cod1(SAMC.mapvalue(mh))
@show map_cod = char_cod2(SAMC.mapvalue(mh))
@show map_cod = char_cod3(SAMC.mapvalue(mh))

