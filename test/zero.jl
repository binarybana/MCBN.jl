using DAI

bnd = MCBN.BayesNetDAI(3)
MCBN.add_edge!(bnd, 2, 1)
MCBN.add_edge!(bnd, 3, 1)
MCBN.add_edge!(bnd, 3, 2)
DAI.clearBackups!(bnd)

MCBN.set_factor!(bnd, 1, [0.9717959704448,
                  0.02820402955519996,
                  0.4884633419082751,
                  0.5115366580917249,
                  0.4671602284144756,
                  0.5328397715855244,
                  0.09147912958666593,
                  0.9085208704133341])
MCBN.set_factor!(bnd, 2, [0.6038715782985324,
                          0.39612842170146756,
                          0.16877438390015786,
                          0.8312256160998421])
MCBN.set_factor!(bnd, 3, [0.3137903917753795, 0.6862096082246205])

MCBN.check_bnd(bnd)
@show true_cod1 = MCBN.cod(bnd, 1, [2,3])
@show true_cod2 = MCBN.cod(bnd, 1, [2])
@show true_cod3 = MCBN.cod(bnd, 2, [1])

MCBN.setupIA!(bnd)
@show marginal(bnd.ia, VarSet(bnd.verts[1], bnd.verts[2]))
@show marginal(bnd.ia, VarSet(bnd.verts[1]))
